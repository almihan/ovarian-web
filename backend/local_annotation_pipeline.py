"""Local cell/hormone entity-extraction pipeline used by the Uvicorn app.

The worker downloads the selected Stage 1 bundle, runs one cell/hormone Ab3P
prepass per document, runs chunk-level CellExLink NER and forced-top-1 Cell
Ontology normalization, publishes the local branch, and reports progress to the
FastAPI process. PubTator3 plus HGNC supplies genes/proteins separately.
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
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import requests

from backend.cellexlink_lite.resources import (
    DEFAULT_ABBREVIATIONS_PATH,
    DEFAULT_ONTOLOGY_PATH,
)
from backend.pipeline.cell_annotation_worker import (
    run_abbreviation_prepass,
    run_nen,
    run_ner,
    utc_now,
)
from backend.pipeline.entity_lexicons import (
    DEFAULT_HORMONE_LEXICON_PATH,
    MESH_HORMONE_RESOURCE_VERSION,
)
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
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{path.stem}-{digest}"


ONTOLOGY_VERSION = _resource_version(DEFAULT_ONTOLOGY_PATH)
ABBREVIATION_VERSION = _resource_version(DEFAULT_ABBREVIATIONS_PATH)


def _local_path_from_file_url(url: str) -> Path:
    parsed = urlparse(url)
    path = Path(unquote(parsed.path))
    if os.name == "nt" and re.match(r"^/[A-Za-z]:/", str(path)):
        path = Path(str(path)[1:])
    return path


def _download(url: str, destination: Path) -> None:
    if not url:
        raise ValueError("The local annotation input URL is missing.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    if parsed.scheme == "file":
        source = _local_path_from_file_url(url)
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
    if not url:
        raise ValueError(f"The upload URL for {path.name} is missing.")
    parsed = urlparse(url)
    if parsed.scheme == "file":
        destination = _local_path_from_file_url(url)
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
    local_progress_path = str(callback.get("local_progress_path") or "").strip()
    if local_progress_path:
        destination = Path(local_progress_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            logger.warning("Could not write local CellExLink progress: %s", exc)

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
    except Exception as exc:
        # Callback loss must not discard a locally completed artifact.
        logger.warning("Could not report local cell-annotation progress: %s", exc)


class CallbackProgress:
    """Map worker percentages into the Stage 2 callback interval."""

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
        self._guard = threading.Lock()

    def emit(
        self,
        *,
        stage: str,
        percent: float,
        message: str,
        stats: Mapping[str, Any],
        force: bool = False,
    ) -> None:
        with self._guard:
            bounded = max(0.0, min(100.0, float(percent)))
            overall = int(
                round(self.start + (self.end - self.start) * bounded / 100.0)
            )
            overall = max(self.last_progress, overall)
            now = time.monotonic()
            should_post = (
                force
                or overall >= self.last_progress + 1
                or now - self.last_post >= 1.0
            )
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
    """Return a cached Hugging Face snapshot, downloading it when necessary."""

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


def _snapshot_model_identity(
    *,
    repo_id: str,
    revision: str | None,
    snapshot_path: Path,
) -> str:
    """Return a stable cache identity independent of the project directory."""

    resolved = snapshot_path.expanduser().resolve()
    commit = resolved.name if resolved.parent.name == "snapshots" else ""
    return f"{repo_id}@{commit or revision or 'default'}"


def _release_model_memory() -> None:
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


def run_local_annotation_bundle(
    payload: Mapping[str, Any],
    *,
    model_cache_root: Path,
) -> dict[str, Any]:
    """Run the complete local CellExLink branch for one Stage 2 job."""

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
    requested_device = str(
        options.get("device")
        or os.getenv("CELL_LOCAL_DEVICE")
        or "auto"
    ).strip().casefold()
    if requested_device not in {"auto", "cpu"}:
        requested_device = "auto"

    _post_callback(
        callback,
        {
            "status": "processing",
            "stage": "preparing_local",
            "progress": 3,
            "message": "The local CellExLink worker started.",
            "stats": {
                **dict(source_stats),
                "cell_branch_status": "running",
                "cell_branch_runtime": "local worker",
            },
        },
    )

    try:
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
                "cell_branch_runtime": "local worker",
                "cell_local_device_requested": requested_device,
            }
            _post_callback(
                callback,
                {
                    "status": "processing",
                    "stage": "preparing_local",
                    "progress": 7,
                    "message": (
                        f"Prepared {chunk_count:,} chunks from {len(entries):,} papers "
                        "for local CellExLink extraction."
                    ),
                    "stats": base_stats,
                },
            )

            huggingface_cache = model_cache_root / "huggingface"
            _post_callback(
                callback,
                {
                    "status": "processing",
                    "stage": "loading_cell_recognition_model",
                    "progress": 8,
                    "message": (
                        "Loading the local CellExLink recognition model. "
                        "The first run may populate the Hugging Face cache."
                    ),
                    "stats": base_stats,
                },
            )
            ner_snapshot, _ner_downloaded = ensure_model_snapshot(
                repo_id=ner_repo,
                revision=(
                    str(models.get("ner_revision"))
                    if models.get("ner_revision")
                    else None
                ),
                cache_root=huggingface_cache,
            )

            manifest = {
                "pipeline_version": pipeline_version,
                "ner_model": ner_repo,
                "nen_model": nen_repo,
                "ontology_version": ONTOLOGY_VERSION,
                "abbreviation_version": ABBREVIATION_VERSION,
                "hormone_resource_version": MESH_HORMONE_RESOURCE_VERSION,
                "abbreviations_enabled": not bool(
                    options.get("disable_abbreviations", False)
                ),
                "entries": entries,
            }

            abbreviation_args = SimpleNamespace(
                ontology_path=DEFAULT_ONTOLOGY_PATH,
                hormone_lexicon_path=DEFAULT_HORMONE_LEXICON_PATH,
                disable_abbreviations=bool(
                    options.get("disable_abbreviations", False)
                ),
            )
            abbreviation_stats = run_abbreviation_prepass(
                abbreviation_args,
                manifest,
                CallbackProgress(
                    callback,
                    start=8,
                    end=18,
                    base_stats=base_stats,
                ),
            )

            ner_args = SimpleNamespace(
                model=str(ner_snapshot),
                model_label=ner_repo,
                model_cache_dir=str(huggingface_cache),
                max_seq_length=None,
                doc_stride=128,
                window_batch_size=max(
                    1,
                    int(options.get("ner_window_batch_size") or 4),
                ),
                text_batch_size=max(
                    1,
                    int(options.get("ner_text_batch_size") or 8),
                ),
                cpu_threads=max(1, int(options.get("cpu_threads") or 4)),
                device=requested_device,
            )
            ner_stats = run_ner(
                ner_args,
                manifest,
                CallbackProgress(
                    callback,
                    start=20,
                    end=42,
                    base_stats={**base_stats, **abbreviation_stats},
                ),
            )
            _release_model_memory()
            _post_callback(
                callback,
                {
                    "status": "processing",
                    "stage": "releasing_recognition",
                    "progress": 44,
                    "message": "Recognition finished and its model was released.",
                    "stats": {
                        **base_stats,
                        **abbreviation_stats,
                        **ner_stats,
                    },
                },
            )

            _post_callback(
                callback,
                {
                    "status": "processing",
                    "stage": "loading_cell_normalization_model",
                    "progress": 45,
                    "message": (
                        "Loading the local CellExLink normalization model and "
                        "ontology embeddings."
                    ),
                    "stats": {
                        **base_stats,
                        **abbreviation_stats,
                        **ner_stats,
                    },
                },
            )
            nen_snapshot, _nen_downloaded = ensure_model_snapshot(
                repo_id=nen_repo,
                revision=(
                    str(models.get("nen_revision"))
                    if models.get("nen_revision")
                    else None
                ),
                cache_root=huggingface_cache,
            )
            nen_model_identity = _snapshot_model_identity(
                repo_id=nen_repo,
                revision=(
                    str(models.get("nen_revision"))
                    if models.get("nen_revision")
                    else None
                ),
                snapshot_path=nen_snapshot,
            )
            nen_args = SimpleNamespace(
                model=str(nen_snapshot),
                model_label=nen_repo,
                model_identity=nen_model_identity,
                model_cache_dir=str(huggingface_cache),
                embedding_cache_dir=str(model_cache_root / "ontology-embeddings"),
                ontology_path=DEFAULT_ONTOLOGY_PATH,
                abbreviations_path=DEFAULT_ABBREVIATIONS_PATH,
                disable_abbreviations=bool(
                    options.get("disable_abbreviations", False)
                ),
                normalization_method_log=bool(
                    options.get("normalization_method_log", False)
                ),
                batch_size=max(1, int(options.get("nen_batch_size") or 128)),
                cpu_threads=max(1, int(options.get("cpu_threads") or 4)),
                keep_ner_intermediates=False,
                device=requested_device,
            )
            nen_stats = run_nen(
                nen_args,
                manifest,
                CallbackProgress(
                    callback,
                    start=46,
                    end=72,
                    base_stats={
                        **base_stats,
                        **abbreviation_stats,
                        **ner_stats,
                    },
                ),
            )
            _release_model_memory()

            _post_callback(
                callback,
                {
                    "status": "processing",
                    "stage": "building_cell_branch",
                    "progress": 74,
                    "message": "Building the text-free local CellExLink branch.",
                    "stats": {
                        **base_stats,
                        **abbreviation_stats,
                        **ner_stats,
                        **nen_stats,
                    },
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
            hormone_count = int(entity_counts["hormone"])
            compute_device = str(
                nen_stats.get("normalization_compute_device")
                or ner_stats.get("recognition_compute_device")
                or "CPU"
            )
            uses_accelerator = compute_device != "CPU"
            stats = {
                **base_stats,
                **abbreviation_stats,
                **ner_stats,
                **nen_stats,
                "cell_branch_status": "completed",
                "cell_branch_schema": CELL_BRANCH_SCHEMA,
                "cell_output_chunk_count": output_chunk_count,
                "mention_count": int(entity_counts["total"]),
                "cell_count": cell_count,
                "hormone_count": hormone_count,
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
                "gpu": compute_device if uses_accelerator else "",
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
                    "message": "Publishing the local CellExLink branch.",
                    "stats": stats,
                },
            )
            _upload(
                cell_annotations_url,
                cell_output,
                content_type="application/gzip",
            )

            message = (
                f"The local cell/hormone branch finished with {cell_count:,} "
                f"cell-type and {hormone_count:,} Ab3P hormone annotations. "
                "The controller will merge them with the PubTator3/HGNC branch."
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
        logger.exception("Local cell annotation job %s failed", job_id)
        _post_callback(
            callback,
            {
                "status": "failed",
                "stage": "failed",
                "progress": 100,
                "message": "Local CellExLink extraction could not be completed.",
                "error": str(exc),
                "stats": {
                    "cell_branch_status": "failed",
                    "cell_branch_elapsed_seconds": elapsed,
                },
            },
        )
        raise


__all__ = ["ensure_model_snapshot", "run_local_annotation_bundle"]
