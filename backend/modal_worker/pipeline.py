"""Low-memory CellExLink branch for Modal GPU or local CPU execution.

Only cell-type recognition and normalization run here.  PubTator3 executes on
the Railway controller in parallel.  Modal publishes a text-free cell branch
artifact and returns immediately, allowing the T4 to shut down without waiting
for NCBI requests or the final merge.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlparse

import requests

from backend.cellexlink_lite.resources import (
    DEFAULT_ABBREVIATIONS_PATH,
    DEFAULT_ONTOLOGY_PATH,
)
from backend.pipeline.cell_annotation_worker import run_nen, run_ner, utc_now
from backend.pipeline.entity_artifacts import (
    CELL_BRANCH_FILENAME,
    CELL_BRANCH_SCHEMA,
    build_cell_branch,
    sha256_path,
    split_bundle,
)

logger = logging.getLogger(__name__)
_ONE_MIB = 1024 * 1024


def _silence_model_output() -> None:
    """Disable model progress bars and non-error library output."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    for name in ("transformers", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass


def _resource_version(path: Path) -> str:
    return f"{path.stem}-{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"


ONTOLOGY_VERSION = _resource_version(DEFAULT_ONTOLOGY_PATH)
ABBREVIATION_VERSION = _resource_version(DEFAULT_ABBREVIATIONS_PATH)

# Backward-compatible private alias used by tests and local tooling.
_sha256_path = sha256_path


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    if parsed.scheme == "file":
        source = Path(unquote(parsed.path))
        if os.name == "nt" and re.match(r"^/[A-Za-z]:/", str(source)):
            source = Path(str(source)[1:])
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, destination)
        return

    with requests.get(url, stream=True, timeout=(20, 600)) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for block in response.iter_content(chunk_size=_ONE_MIB):
                if block:
                    output.write(block)
            output.flush()
            os.fsync(output.fileno())


def _upload(
    url: str,
    path: Path,
    *,
    content_type: str,
    content_encoding: str | None = None,
) -> None:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        destination = Path(unquote(parsed.path))
        if os.name == "nt" and re.match(r"^/[A-Za-z]:/", str(destination)):
            destination = Path(str(destination)[1:])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        shutil.copyfile(path, temporary)
        os.replace(temporary, destination)
        return

    headers = {"Content-Type": content_type}
    if content_encoding:
        headers["Content-Encoding"] = content_encoding
    with path.open("rb") as handle:
        response = requests.put(
            url,
            data=handle,
            headers=headers,
            timeout=(20, 900),
        )
    response.raise_for_status()


def _post_callback(callback: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    url = str(callback.get("url") or "")
    token = str(callback.get("token") or "")
    if not url or not token:
        return
    try:
        response = requests.post(
            url,
            json=dict(payload),
            headers={"X-Annotation-Token": token},
            timeout=(10, 30),
        )
        response.raise_for_status()
    except Exception as exc:  # callback loss must not destroy completed GPU work
        logger.warning("Could not report cell-annotation progress: %s", exc)


class CallbackProgress:
    """Adapt the cell worker progress interface to Railway callbacks."""

    def __init__(
        self,
        callback: Mapping[str, Any],
        *,
        start: float,
        end: float,
        base_stats: Mapping[str, Any] | None = None,
    ) -> None:
        self.callback = callback
        self.start = float(start)
        self.end = float(end)
        self.base_stats = dict(base_stats or {})
        self.last_post = 0.0
        self.last_progress = -1

    def emit(
        self,
        *,
        stage: str,
        percent: float,
        message: str,
        stats: Mapping[str, Any],
        force: bool = False,
    ) -> None:
        overall = int(
            round(self.start + (self.end - self.start) * max(0, min(100, percent)) / 100)
        )
        now = time.monotonic()
        should_post = (
            force and overall in {int(self.start), int(self.end)}
        ) or overall >= self.last_progress + 1 or now - self.last_post >= 1.0
        if not should_post:
            return
        merged = dict(self.base_stats)
        merged.update(dict(stats))
        _post_callback(
            self.callback,
            {
                "status": "processing",
                "stage": stage,
                "progress": overall,
                "message": message,
                "stats": merged,
            },
        )
        self.last_post = now
        self.last_progress = overall


def ensure_model_snapshot(
    *,
    repo_id: str,
    revision: str | None,
    cache_root: Path,
) -> tuple[Path, bool]:
    """Return a cached Hugging Face snapshot, downloading only when absent."""

    from huggingface_hub import snapshot_download

    cache_root.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "cache_dir": str(cache_root),
    }
    try:
        path = snapshot_download(local_files_only=True, **kwargs)
        return Path(path), False
    except Exception:
        path = snapshot_download(local_files_only=False, **kwargs)
        return Path(path), True


def warm_model_cache(
    *,
    ner_model: str,
    ner_revision: str | None,
    nen_model: str,
    nen_revision: str | None,
    model_cache_root: Path = Path("/model-cache/huggingface"),
    commit_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    ner_path, ner_downloaded = ensure_model_snapshot(
        repo_id=ner_model,
        revision=ner_revision,
        cache_root=model_cache_root,
    )
    nen_path, nen_downloaded = ensure_model_snapshot(
        repo_id=nen_model,
        revision=nen_revision,
        cache_root=model_cache_root,
    )
    if commit_callback is not None and (ner_downloaded or nen_downloaded):
        commit_callback()
    return {
        "ner_snapshot": str(ner_path),
        "nen_snapshot": str(nen_path),
        "ner_downloaded": ner_downloaded,
        "nen_downloaded": nen_downloaded,
    }


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def run_annotation_bundle(
    payload: Mapping[str, Any],
    *,
    model_cache_root: Path = Path("/model-cache"),
    commit_callback: Callable[[], None] | None = None,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Run the CellExLink branch and publish one aligned cell artifact.

    Production Modal calls keep ``require_cuda=True``. Local development uses
    the same bounded branch with ``require_cuda=False`` so PyTorch selects CPU
    when CUDA is unavailable.
    """

    _silence_model_output()
    started = time.monotonic()
    callback = (
        payload.get("callback")
        if isinstance(payload.get("callback"), Mapping)
        else {}
    )
    input_spec = (
        payload.get("input") if isinstance(payload.get("input"), Mapping) else {}
    )
    output_spec = (
        payload.get("output") if isinstance(payload.get("output"), Mapping) else {}
    )
    models = (
        payload.get("models") if isinstance(payload.get("models"), Mapping) else {}
    )
    options = (
        payload.get("options") if isinstance(payload.get("options"), Mapping) else {}
    )
    source_stats = (
        payload.get("source_stats")
        if isinstance(payload.get("source_stats"), Mapping)
        else {}
    )

    job_id = str(payload.get("job_id") or "annotation-job")
    ner_repo = str(models.get("ner") or "almire/CellExLink-bioformer16L")
    nen_repo = str(models.get("nen") or "almire/CellExLink-Sapbert")
    pipeline_version = str(payload.get("pipeline_version") or "unknown")
    model_signature = str(payload.get("model_signature") or "")

    runtime_name = "Modal T4" if require_cuda else "local worker"
    preparation_stage = "preparing_gpu" if require_cuda else "preparing_local"
    _post_callback(
        callback,
        {
            "status": "processing",
            "stage": preparation_stage,
            "progress": 3,
            "message": f"The {runtime_name} started the CellExLink branch.",
            "stats": {
                **dict(source_stats),
                "cell_branch_status": "running",
                "cell_branch_runtime": runtime_name,
            },
        },
    )

    try:
        import torch

        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("Modal started the cell-annotation function without CUDA.")
        uses_cuda = bool(torch.cuda.is_available())
        compute_device = torch.cuda.get_device_name(0) if uses_cuda else "CPU"

        with tempfile.TemporaryDirectory(prefix=f"ovarian-cell-{job_id}-") as temp_name:
            temp_root = Path(temp_name)
            input_path = temp_root / "chunks.jsonl.gz"
            _download(str(input_spec.get("url") or ""), input_path)
            expected_sha = str(input_spec.get("sha256") or "")
            actual_sha = sha256_path(input_path)
            if expected_sha and actual_sha != expected_sha:
                raise ValueError(
                    "The downloaded retrieval artifact failed its SHA-256 check."
                )

            entries, chunk_count = split_bundle(input_path, temp_root / "papers")
            base_stats = {
                "paper_count": len(entries),
                "chunk_count": chunk_count,
                "source_artifact_sha256": actual_sha,
                "cell_branch_status": "running",
                "cell_branch_runtime": runtime_name,
            }
            _post_callback(
                callback,
                {
                    "status": "processing",
                    "stage": preparation_stage,
                    "progress": 7,
                    "message": (
                        f"Prepared {chunk_count:,} chunks from {len(entries):,} papers "
                        "for CellExLink."
                    ),
                    "stats": base_stats,
                },
            )

            hf_cache = model_cache_root / "huggingface"
            ner_snapshot, ner_downloaded = ensure_model_snapshot(
                repo_id=ner_repo,
                revision=(
                    str(models.get("ner_revision"))
                    if models.get("ner_revision")
                    else None
                ),
                cache_root=hf_cache,
            )
            if ner_downloaded and commit_callback is not None:
                commit_callback()

            manifest = {
                "pipeline_version": pipeline_version,
                "ner_model": ner_repo,
                "nen_model": nen_repo,
                "ontology_version": ONTOLOGY_VERSION,
                "abbreviation_version": ABBREVIATION_VERSION,
                "abbreviations_enabled": not bool(
                    options.get("disable_abbreviations", False)
                ),
                "entries": entries,
            }

            ner_args = SimpleNamespace(
                model=str(ner_snapshot),
                model_label=ner_repo,
                model_cache_dir=str(hf_cache),
                max_seq_length=None,
                doc_stride=128,
                window_batch_size=max(
                    1, int(options.get("ner_window_batch_size") or 4)
                ),
                text_batch_size=max(1, int(options.get("ner_text_batch_size") or 8)),
                cpu_threads=max(1, int(options.get("cpu_threads") or 4)),
            )
            ner_stats = run_ner(
                ner_args,
                manifest,
                CallbackProgress(callback, start=8, end=40, base_stats=base_stats),
            )
            _release_cuda()
            _post_callback(
                callback,
                {
                    "status": "processing",
                    "stage": "releasing_recognition",
                    "progress": 42,
                    "message": (
                        "Recognition finished; the NER model was released from "
                        f"{('GPU' if uses_cuda else 'system')} memory."
                    ),
                    "stats": {**base_stats, **ner_stats},
                },
            )

            nen_snapshot, nen_downloaded = ensure_model_snapshot(
                repo_id=nen_repo,
                revision=(
                    str(models.get("nen_revision"))
                    if models.get("nen_revision")
                    else None
                ),
                cache_root=hf_cache,
            )
            if nen_downloaded and commit_callback is not None:
                commit_callback()

            nen_args = SimpleNamespace(
                model=str(nen_snapshot),
                model_label=nen_repo,
                model_cache_dir=str(hf_cache),
                embedding_cache_dir=str(model_cache_root / "ontology-embeddings"),
                ontology_path=DEFAULT_ONTOLOGY_PATH,
                abbreviations_path=DEFAULT_ABBREVIATIONS_PATH,
                disable_abbreviations=bool(
                    options.get("disable_abbreviations", False)
                ),
                batch_size=max(1, int(options.get("nen_batch_size") or 64)),
                request_batch_size=max(
                    1, int(options.get("nen_request_batch_size") or 128)
                ),
                cpu_threads=max(1, int(options.get("cpu_threads") or 4)),
                work_database=temp_root / "normalization.sqlite",
                keep_ner_intermediates=False,
            )
            nen_stats = run_nen(
                nen_args,
                manifest,
                CallbackProgress(
                    callback,
                    start=44,
                    end=72,
                    base_stats={**base_stats, **ner_stats},
                ),
            )
            _release_cuda()

            # Persist checkpoint downloads and ontology embeddings only.  There
            # is no PubTator3 cache on Modal anymore.
            if commit_callback is not None:
                commit_callback()

            _post_callback(
                callback,
                {
                    "status": "processing",
                    "stage": "building_cell_branch",
                    "progress": 74,
                    "message": "Building the text-free CellExLink branch artifact.",
                    "stats": {**base_stats, **ner_stats, **nen_stats},
                },
            )

            cell_output = temp_root / CELL_BRANCH_FILENAME
            output_chunk_count, entity_counts = build_cell_branch(entries, cell_output)
            if output_chunk_count != chunk_count:
                raise RuntimeError(
                    "The CellExLink branch did not preserve every Stage 1 chunk."
                )

            cell_output_sha = sha256_path(cell_output)
            cell_elapsed = round(time.monotonic() - started, 2)
            cell_count = int(entity_counts["cell"])
            stats = {
                **base_stats,
                **ner_stats,
                **nen_stats,
                "cell_branch_status": "completed",
                "cell_branch_schema": CELL_BRANCH_SCHEMA,
                "cell_output_chunk_count": output_chunk_count,
                "mention_count": cell_count,
                "cell_count": cell_count,
                "normalized_count": int(
                    nen_stats.get("normalized_occurrences") or 0
                ),
                "unresolved_count": int(
                    nen_stats.get("unresolved_occurrences") or 0
                ),
                "unique_mentions": int(nen_stats.get("unique_mentions") or 0),
                "normalization_rate": round(
                    100.0
                    * int(nen_stats.get("normalized_occurrences") or 0)
                    / max(1, cell_count),
                    2,
                ),
                "cell_branch_elapsed_seconds": cell_elapsed,
                "compute_device": compute_device,
                "gpu": compute_device if uses_cuda else "",
                "ner_model": ner_repo,
                "nen_model": nen_repo,
                "model_signature": model_signature,
                "cell_output_sha256": cell_output_sha,
                "cell_output_bytes": cell_output.stat().st_size,
            }

            cell_annotations_url = str(
                output_spec.get("cell_annotations_url")
                or output_spec.get("annotations_url")
                or ""
            )
            cell_annotations_key = (
                output_spec.get("cell_annotations_key")
                or output_spec.get("annotations_key")
            )
            cell_summary_url = str(
                output_spec.get("cell_summary_url")
                or output_spec.get("summary_url")
                or ""
            )
            cell_summary_key = (
                output_spec.get("cell_summary_key")
                or output_spec.get("summary_key")
            )

            _post_callback(
                callback,
                {
                    "status": "processing",
                    "stage": "publishing_cell_branch",
                    "progress": 78,
                    "message": "Publishing the CellExLink branch for Railway.",
                    "stats": stats,
                },
            )
            _upload(
                cell_annotations_url,
                cell_output,
                content_type="application/gzip",
                content_encoding=None,
            )

            message = (
                f"CellExLink finished with {cell_count:,} cell-type annotations. "
                "Railway will merge them with its PubTator3 branch."
            )
            summary = {
                "status": "completed",
                "branch": "cell",
                "message": message,
                "job_id": job_id,
                "pipeline_version": pipeline_version,
                "output_schema": CELL_BRANCH_SCHEMA,
                "model_signature": model_signature,
                "source": {
                    "artifact_key": input_spec.get("key"),
                    "sha256": actual_sha,
                },
                "models": {
                    "cell_recognition": ner_repo,
                    "cell_normalization": nen_repo,
                },
                "stats": stats,
                "files": {
                    "cell_annotations": {
                        "key": cell_annotations_key,
                        "sha256": cell_output_sha,
                        "size_bytes": cell_output.stat().st_size,
                        "content_type": "application/gzip",
                        "content_encoding": None,
                    }
                },
                "completed_at": utc_now(),
            }
            summary_path = temp_root / "cell_summary.json"
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary_sha = sha256_path(summary_path)
            _upload(
                cell_summary_url,
                summary_path,
                content_type="application/json",
            )

            _post_callback(
                callback,
                {
                    "status": "cell_completed",
                    "stage": "cell_branch_completed",
                    "progress": 80,
                    "message": message,
                    "stats": stats,
                    "output_sha256": cell_output_sha,
                    "summary_sha256": summary_sha,
                },
            )
            return {
                "status": "cell_completed",
                "stage": "cell_branch_completed",
                "message": message,
                "stats": stats,
                "cell_annotations_artifact_key": cell_annotations_key,
                "cell_summary_artifact_key": cell_summary_key,
                "completed_at": utc_now(),
            }
    except Exception as exc:
        elapsed = round(time.monotonic() - started, 2)
        logger.error("Cell annotation branch %s failed: %s", job_id, exc)
        _post_callback(
            callback,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 100,
                "message": "CellExLink extraction could not be completed.",
                "error": str(exc),
                "stats": {
                    "cell_branch_status": "failed",
                    "cell_branch_elapsed_seconds": elapsed,
                },
            },
        )
        raise


__all__ = [
    "ensure_model_snapshot",
    "run_annotation_bundle",
    "warm_model_cache",
]
