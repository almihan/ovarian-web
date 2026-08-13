"""Railway-side PubTator3 branch and final Stage 2 merge coordinator.

The web service runs PubTator3 as an I/O-bound background task while Modal runs
CellExLink on a T4.  Both branches publish deterministic, text-free artifacts.
Railway merges them only after both are available, then removes the temporary
branch objects to keep persistent storage small.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests

from backend.config import settings
from backend.worker_state import (
    get_annotation_job,
    list_annotation_jobs,
    update_annotation_job,
    utc_now,
)
from backend.pipeline.annotation_contract import (
    ANNOTATION_PIPELINE_VERSION,
    AnnotationArtifactKeys,
    annotation_artifact_keys,
)
from backend.pipeline.entity_artifacts import (
    ANNOTATION_OUTPUT_SCHEMA,
    ENTITY_OUTPUT_FILENAME,
    PUBTATOR_BRANCH_FILENAME,
    PUBTATOR_BRANCH_SCHEMA,
    build_pubtator_branch,
    merge_branch_artifacts,
    sha256_path,
    split_bundle,
)
from backend.pipeline.pubtator3_annotation_worker import run_pubtator3_annotations
from backend.storage.artifacts import ArtifactStore, get_artifact_store

logger = logging.getLogger(__name__)
_ONE_MIB = 1024 * 1024


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _elapsed_since(value: object) -> float:
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def _stats_dict(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


class RailwayAnnotationExecutor:
    """One bounded PubTator3 task plus one bounded finalizer per application."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="railway-stage2",
        )
        self._guard = threading.Lock()
        self._pubtator_jobs: set[str] = set()
        self._finalize_jobs: set[str] = set()

    @staticmethod
    def _keys(job: Mapping[str, Any]) -> AnnotationArtifactKeys:
        source_sha = str(job.get("source_artifact_sha256") or "")
        model_signature = str(job.get("model_signature") or "")
        if not source_sha or not model_signature:
            raise ValueError(
                "The annotation job is missing its source or model fingerprint."
            )
        keys = annotation_artifact_keys(
            source_sha256=source_sha,
            model_signature=model_signature,
        )
        recorded_output = str(job.get("output_artifact_key") or "")
        recorded_summary = str(job.get("summary_artifact_key") or "")
        if recorded_output and recorded_output != keys.final_annotations:
            raise ValueError("The annotation output key does not match its contract.")
        if recorded_summary and recorded_summary != keys.final_summary:
            raise ValueError("The annotation summary key does not match its contract.")
        return keys

    @staticmethod
    def _branch_ready(
        store: ArtifactStore,
        annotations_key: str,
        summary_key: str,
    ) -> bool:
        return (
            store.head(annotations_key) is not None
            and store.head(summary_key) is not None
        )

    @staticmethod
    def _download_artifact(
        store: ArtifactStore,
        key: str,
        destination: Path,
        *,
        expected_sha256: str | None = None,
    ) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        local = store.local_path(key)
        if local is not None:
            shutil.copyfile(local, destination)
        else:
            url = store.presign_get(
                key,
                expires_seconds=settings.artifact_presigned_ttl_seconds,
            )
            with requests.get(url, stream=True, timeout=(20, 900)) as response:
                response.raise_for_status()
                with destination.open("wb") as output:
                    for block in response.iter_content(chunk_size=_ONE_MIB):
                        if block:
                            output.write(block)
                    output.flush()
                    os.fsync(output.fileno())

        actual = sha256_path(destination)
        if expected_sha256 and actual != expected_sha256:
            raise ValueError(f"Artifact {key} failed its SHA-256 check.")
        return actual

    @staticmethod
    def _validate_branch_summary(
        summary: Mapping[str, Any],
        job: Mapping[str, Any],
        *,
        branch: str,
    ) -> None:
        if str(summary.get("branch") or "") != branch:
            raise ValueError(f"Expected the {branch} branch summary.")
        source = summary.get("source")
        source = source if isinstance(source, Mapping) else {}
        if str(source.get("sha256") or "") != str(
            job.get("source_artifact_sha256") or ""
        ):
            raise ValueError(f"The {branch} branch source fingerprint does not match.")
        if str(summary.get("model_signature") or "") != str(
            job.get("model_signature") or ""
        ):
            raise ValueError(f"The {branch} branch model signature does not match.")

    @staticmethod
    def _merge_job_stats(job: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
        merged = _stats_dict(job.get("stats"))
        merged.update(dict(extra))
        return merged

    def _update_processing(
        self,
        job_id: str,
        *,
        stage: str,
        progress: int,
        message: str,
        stats: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        job = get_annotation_job(job_id)
        if job is None or job.get("status") not in {"queued", "processing"}:
            return job
        merged = self._merge_job_stats(job, stats or {})
        return update_annotation_job(
            job_id,
            status="processing",
            stage=stage,
            progress=max(int(job.get("progress") or 0), int(progress)),
            message=message,
            stats=merged,
            elapsed_seconds=max(
                float(job.get("elapsed_seconds") or 0.0),
                _elapsed_since(job.get("started_at")),
            ),
            last_remote_check_at=utc_now(),
        )

    def _fail(
        self,
        job_id: str,
        *,
        message: str,
        error: str,
        stats: Mapping[str, Any] | None = None,
    ) -> None:
        job = get_annotation_job(job_id)
        if job is None or job.get("status") == "completed":
            return
        merged = self._merge_job_stats(job, stats or {})
        update_annotation_job(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            message=message,
            stats=merged,
            elapsed_seconds=max(
                float(job.get("elapsed_seconds") or 0.0),
                _elapsed_since(job.get("started_at")),
            ),
            completed_at=utc_now(),
            last_remote_check_at=utc_now(),
            error=error,
        )

    def submit_pubtator(self, job_id: str) -> bool:
        """Idempotently start or resume the Railway PubTator3 branch."""

        job = get_annotation_job(job_id)
        if job is None or job.get("status") not in {"queued", "processing"}:
            return False
        try:
            keys = self._keys(job)
            store = get_artifact_store()
            if self._branch_ready(
                store,
                keys.pubtator_annotations,
                keys.pubtator_summary,
            ):
                self.schedule_finalize(job_id)
                return False
        except Exception:
            logger.exception("Could not inspect PubTator3 branch for %s", job_id)

        with self._guard:
            if job_id in self._pubtator_jobs:
                return False
            self._pubtator_jobs.add(job_id)
        try:
            future = self._pool.submit(self._run_pubtator_branch, job_id)
        except Exception:
            with self._guard:
                self._pubtator_jobs.discard(job_id)
            raise
        future.add_done_callback(
            lambda completed, current=job_id: self._pubtator_done(current, completed)
        )
        return True

    def _pubtator_done(self, job_id: str, future: Future[None]) -> None:
        with self._guard:
            self._pubtator_jobs.discard(job_id)
        try:
            future.result()
        except Exception:
            logger.exception("Railway PubTator3 branch %s exited with an error", job_id)

    def schedule_finalize(self, job_id: str) -> bool:
        """Queue an idempotent final merge without blocking an HTTP request."""

        job = get_annotation_job(job_id)
        if job is None or job.get("status") not in {"queued", "processing"}:
            return False
        with self._guard:
            if job_id in self._finalize_jobs:
                return False
            self._finalize_jobs.add(job_id)
        try:
            future = self._pool.submit(self._try_finalize, job_id)
        except Exception:
            with self._guard:
                self._finalize_jobs.discard(job_id)
            raise
        future.add_done_callback(
            lambda completed, current=job_id: self._finalize_done(current, completed)
        )
        return True

    def _finalize_done(self, job_id: str, future: Future[None]) -> None:
        with self._guard:
            self._finalize_jobs.discard(job_id)
        try:
            future.result()
        except Exception:
            logger.exception("Railway finalizer %s exited with an error", job_id)

    def ensure_job(self, job_id: str) -> None:
        """Resume both Railway responsibilities after a stale poll or restart."""

        self.submit_pubtator(job_id)
        self.schedule_finalize(job_id)

    def resume_active_jobs(self) -> None:
        for job in list_annotation_jobs(50):
            if job.get("status") in {"queued", "processing"}:
                self.ensure_job(str(job["id"]))

    def _run_pubtator_branch(self, job_id: str) -> None:
        started = time.monotonic()
        job = get_annotation_job(job_id)
        if job is None or job.get("status") not in {"queued", "processing"}:
            return

        try:
            keys = self._keys(job)
            store = get_artifact_store()
            if self._branch_ready(
                store,
                keys.pubtator_annotations,
                keys.pubtator_summary,
            ):
                self.schedule_finalize(job_id)
                return

            self._update_processing(
                job_id,
                stage="running_parallel_branches",
                progress=4,
                message=(
                    "CellExLink is running on Modal while PubTator3 runs on "
                    "Railway CPU."
                ),
                stats={
                    "pubtator_branch_status": "running",
                    "pubtator_branch_runtime": "Railway CPU",
                    "execution_layout": (
                        "modal-cellexlink-plus-railway-pubtator3"
                    ),
                },
            )

            with tempfile.TemporaryDirectory(
                prefix=f"ovarian-pubtator-{job_id}-"
            ) as temp_name:
                temp_root = Path(temp_name)
                source_path = temp_root / "chunks.jsonl.gz"
                source_sha = self._download_artifact(
                    store,
                    str(job["source_artifact_key"]),
                    source_path,
                    expected_sha256=str(job.get("source_artifact_sha256") or ""),
                )
                entries, chunk_count = split_bundle(
                    source_path,
                    temp_root / "papers",
                )

                pubtator_stats = run_pubtator3_annotations(
                    entries,
                    options={
                        "pubtator_batch_size": settings.pubtator3_batch_size,
                        "pubtator_request_timeout": (
                            settings.pubtator3_request_timeout
                        ),
                        "pubtator_required": settings.pubtator3_required,
                        "pubtator_resolve_preferred_labels": (
                            settings.pubtator3_resolve_preferred_labels
                        ),
                        "ncbi_tool": settings.ncbi_tool,
                        "ncbi_email": settings.ncbi_email,
                    },
                    label_cache_path=(
                        settings.cell_model_cache_dir
                        / "pubtator3-preferred-labels.sqlite"
                    ),
                )

                branch_path = temp_root / PUBTATOR_BRANCH_FILENAME
                output_chunk_count, counts = build_pubtator_branch(
                    entries,
                    branch_path,
                )
                if output_chunk_count != chunk_count:
                    raise RuntimeError(
                        "The Railway PubTator3 branch did not preserve every "
                        "Stage 1 chunk."
                    )

                branch_sha = sha256_path(branch_path)
                elapsed = round(time.monotonic() - started, 2)
                stats = {
                    **pubtator_stats,
                    "paper_count": len(entries),
                    "chunk_count": chunk_count,
                    "source_artifact_sha256": source_sha,
                    "pubtator_branch_status": "completed",
                    "pubtator_branch_runtime": "Railway CPU",
                    "pubtator_branch_schema": PUBTATOR_BRANCH_SCHEMA,
                    "pubtator_output_chunk_count": output_chunk_count,
                    "gene_count": int(counts["gene"]),
                    "hormone_count": int(counts["hormone"]),
                    "pubtator_entity_count": int(counts["total"]),
                    "pubtator_branch_elapsed_seconds": elapsed,
                    "pubtator_output_sha256": branch_sha,
                    "pubtator_output_bytes": branch_path.stat().st_size,
                    "model_signature": str(job.get("model_signature") or ""),
                }

                store.put_file(
                    branch_path,
                    key=keys.pubtator_annotations,
                    content_type="application/gzip",
                    content_encoding=None,
                    sha256=branch_sha,
                )

                message = (
                    f"Railway PubTator3 finished with {counts['gene']:,} human gene "
                    f"annotations and {counts['hormone']:,} hormone annotations."
                )
                summary = {
                    "status": "completed",
                    "branch": "pubtator3",
                    "message": message,
                    "job_id": job_id,
                    "pipeline_version": ANNOTATION_PIPELINE_VERSION,
                    "output_schema": PUBTATOR_BRANCH_SCHEMA,
                    "model_signature": job.get("model_signature"),
                    "source": {
                        "artifact_key": job.get("source_artifact_key"),
                        "sha256": source_sha,
                    },
                    "models": {
                        "gene_and_hormone_source": "PubTator3 + human-gene and MeSH hormone filtering",
                    },
                    "execution": {
                        "runtime": "Railway CPU",
                        "concurrent_with_cell_branch": True,
                    },
                    "stats": stats,
                    "files": {
                        "pubtator_annotations": {
                            "key": keys.pubtator_annotations,
                            "sha256": branch_sha,
                            "size_bytes": branch_path.stat().st_size,
                            "content_type": "application/gzip",
                            "content_encoding": None,
                        }
                    },
                    "completed_at": utc_now(),
                }
                summary_path = temp_root / "pubtator3_summary.json"
                _write_json_atomic(summary_path, summary)
                store.put_file(
                    summary_path,
                    key=keys.pubtator_summary,
                    content_type="application/json",
                )

            current = get_annotation_job(job_id)
            if current is None or current.get("status") not in {
                "queued",
                "processing",
            }:
                return
            cell_ready = self._branch_ready(
                store,
                keys.cell_annotations,
                keys.cell_summary,
            )
            self._update_processing(
                job_id,
                stage="merging_entities" if cell_ready else "waiting_for_cell_branch",
                progress=86 if cell_ready else 82,
                message=(
                    "Both branches are ready; Railway is preparing the final merge."
                    if cell_ready
                    else "PubTator3 is complete; Railway is waiting for CellExLink."
                ),
                stats=stats,
            )
            self.schedule_finalize(job_id)
        except Exception as exc:
            logger.exception("Railway PubTator3 branch %s failed", job_id)
            self._fail(
                job_id,
                message="Railway could not complete PubTator3 gene/hormone extraction.",
                error=str(exc),
                stats={
                    "pubtator_branch_status": "failed",
                    "pubtator_branch_runtime": "Railway CPU",
                    "pubtator_branch_elapsed_seconds": round(
                        time.monotonic() - started,
                        2,
                    ),
                },
            )
            raise

    def _complete_from_final_summary(
        self,
        job: Mapping[str, Any],
        summary: Mapping[str, Any],
    ) -> None:
        source = summary.get("source")
        source = source if isinstance(source, Mapping) else {}
        if str(source.get("sha256") or "") != str(
            job.get("source_artifact_sha256") or ""
        ):
            raise ValueError("The final Stage 2 source fingerprint does not match.")
        if str(summary.get("model_signature") or "") != str(
            job.get("model_signature") or ""
        ):
            raise ValueError("The final Stage 2 model signature does not match.")

        stats = _stats_dict(summary.get("stats"))
        update_annotation_job(
            str(job["id"]),
            status="completed",
            stage="completed",
            progress=100,
            message=str(summary.get("message") or "Entity extraction is complete."),
            paper_count=int(stats.get("paper_count") or 0),
            chunk_count=int(stats.get("chunk_count") or 0),
            mention_count=int(stats.get("cell_count") or stats.get("mention_count") or 0),
            normalized_count=int(stats.get("normalized_count") or 0),
            unresolved_count=int(stats.get("unresolved_count") or 0),
            stats=stats,
            elapsed_seconds=float(stats.get("elapsed_seconds") or 0.0),
            completed_at=str(summary.get("completed_at") or utc_now()),
            last_remote_check_at=utc_now(),
            error=None,
        )

    def _try_finalize(self, job_id: str) -> None:
        job = get_annotation_job(job_id)
        if job is None or job.get("status") not in {"queued", "processing"}:
            return

        try:
            keys = self._keys(job)
            store = get_artifact_store()

            if self._branch_ready(
                store,
                keys.final_annotations,
                keys.final_summary,
            ):
                self._complete_from_final_summary(
                    job,
                    store.read_json(keys.final_summary),
                )
                return

            cell_ready = self._branch_ready(
                store,
                keys.cell_annotations,
                keys.cell_summary,
            )
            pubtator_ready = self._branch_ready(
                store,
                keys.pubtator_annotations,
                keys.pubtator_summary,
            )
            if not (cell_ready and pubtator_ready):
                return

            cell_summary = store.read_json(keys.cell_summary)
            pubtator_summary = store.read_json(keys.pubtator_summary)
            self._validate_branch_summary(cell_summary, job, branch="cell")
            self._validate_branch_summary(
                pubtator_summary,
                job,
                branch="pubtator3",
            )

            self._update_processing(
                job_id,
                stage="merging_entities",
                progress=92,
                message=(
                    "CellExLink and PubTator3 are complete; Railway is streaming "
                    "the final merge."
                ),
                stats={
                    "cell_branch_status": "completed",
                    "pubtator_branch_status": "completed",
                },
            )

            merge_started = time.monotonic()
            with tempfile.TemporaryDirectory(
                prefix=f"ovarian-merge-{job_id}-"
            ) as temp_name:
                temp_root = Path(temp_name)
                cell_path = temp_root / "cell_branch.jsonl.gz"
                pubtator_path = temp_root / "pubtator3_branch.jsonl.gz"

                cell_files = cell_summary.get("files")
                cell_files = cell_files if isinstance(cell_files, Mapping) else {}
                cell_file = cell_files.get("cell_annotations")
                cell_file = cell_file if isinstance(cell_file, Mapping) else {}
                pubtator_files = pubtator_summary.get("files")
                pubtator_files = (
                    pubtator_files if isinstance(pubtator_files, Mapping) else {}
                )
                pubtator_file = pubtator_files.get("pubtator_annotations")
                pubtator_file = (
                    pubtator_file if isinstance(pubtator_file, Mapping) else {}
                )

                self._download_artifact(
                    store,
                    keys.cell_annotations,
                    cell_path,
                    expected_sha256=str(cell_file.get("sha256") or "") or None,
                )
                self._download_artifact(
                    store,
                    keys.pubtator_annotations,
                    pubtator_path,
                    expected_sha256=str(pubtator_file.get("sha256") or "") or None,
                )

                output_path = temp_root / ENTITY_OUTPUT_FILENAME
                output_chunk_count, counts = merge_branch_artifacts(
                    cell_path,
                    pubtator_path,
                    output_path,
                )
                output_sha = sha256_path(output_path)

                cell_stats = _stats_dict(cell_summary.get("stats"))
                pubtator_stats = _stats_dict(pubtator_summary.get("stats"))
                expected_chunks = int(
                    cell_stats.get("chunk_count")
                    or pubtator_stats.get("chunk_count")
                    or 0
                )
                if expected_chunks and output_chunk_count != expected_chunks:
                    raise RuntimeError(
                        "The final Stage 2 merge did not preserve every Stage 1 chunk."
                    )

                cell_count = int(counts["cell"])
                merge_elapsed = round(time.monotonic() - merge_started, 2)
                elapsed = round(
                    max(
                        _elapsed_since(job.get("started_at")),
                        float(cell_stats.get("cell_branch_elapsed_seconds") or 0.0),
                        float(
                            pubtator_stats.get("pubtator_branch_elapsed_seconds")
                            or pubtator_stats.get("pubtator_elapsed_seconds")
                            or 0.0
                        ),
                    ),
                    2,
                )
                stats = {
                    **cell_stats,
                    **pubtator_stats,
                    "output_schema": ANNOTATION_OUTPUT_SCHEMA,
                    "output_chunk_count": output_chunk_count,
                    "paper_count": int(
                        cell_stats.get("paper_count")
                        or pubtator_stats.get("paper_count")
                        or 0
                    ),
                    "chunk_count": output_chunk_count,
                    "mention_count": cell_count,
                    "cell_count": cell_count,
                    "gene_count": int(counts["gene"]),
                    "hormone_count": int(counts["hormone"]),
                    "entity_count": int(counts["total"]),
                    "normalized_count": int(
                        cell_stats.get("normalized_occurrences")
                        or cell_stats.get("normalized_count")
                        or 0
                    ),
                    "unresolved_count": int(
                        cell_stats.get("unresolved_occurrences")
                        or cell_stats.get("unresolved_count")
                        or 0
                    ),
                    "normalization_rate": round(
                        100.0
                        * int(
                            cell_stats.get("normalized_occurrences")
                            or cell_stats.get("normalized_count")
                            or 0
                        )
                        / max(1, cell_count),
                        2,
                    ),
                    "cell_branch_status": "completed",
                    "pubtator_branch_status": "completed",
                    "merge_runtime": "Railway CPU",
                    "merge_elapsed_seconds": merge_elapsed,
                    "elapsed_seconds": elapsed,
                    "execution_layout": (
                        "modal-cellexlink-plus-railway-pubtator3"
                    ),
                    "modal_gpu_released_before_merge": (
                        str(job.get("executor") or "") == "modal"
                    ),
                    "model_signature": str(job.get("model_signature") or ""),
                    "output_sha256": output_sha,
                    "output_bytes": output_path.stat().st_size,
                }

                output_ref, _ = store.put_file(
                    output_path,
                    key=keys.final_annotations,
                    content_type="application/gzip",
                    content_encoding=None,
                    sha256=output_sha,
                )

                message = (
                    f"Finished: {stats['cell_count']:,} cell-type, "
                    f"{stats['gene_count']:,} human gene, and "
                    f"{stats['hormone_count']:,} hormone annotations."
                )
                models = {}
                for branch_summary in (cell_summary, pubtator_summary):
                    branch_models = branch_summary.get("models")
                    if isinstance(branch_models, Mapping):
                        models.update(dict(branch_models))
                completed_at = utc_now()
                summary = {
                    "status": "completed",
                    "message": message,
                    "job_id": job_id,
                    "pipeline_version": ANNOTATION_PIPELINE_VERSION,
                    "output_schema": ANNOTATION_OUTPUT_SCHEMA,
                    "model_signature": job.get("model_signature"),
                    "source": {
                        "artifact_key": job.get("source_artifact_key"),
                        "sha256": job.get("source_artifact_sha256"),
                    },
                    "models": models,
                    "execution": {
                        "cell_branch": (
                            "Modal T4"
                            if str(job.get("executor") or "") == "modal"
                            else "local CPU"
                        ),
                        "pubtator3_branch": "Railway CPU",
                        "branches_ran_concurrently": True,
                        "final_merge": "Railway CPU",
                    },
                    "stats": stats,
                    "files": {
                        "annotations": {
                            **output_ref.to_dict(),
                            "content_encoding": None,
                        }
                    },
                    "completed_at": completed_at,
                }
                summary_path = temp_root / "summary.json"
                _write_json_atomic(summary_path, summary)
                summary_sha = sha256_path(summary_path)
                store.put_file(
                    summary_path,
                    key=keys.final_summary,
                    content_type="application/json",
                    sha256=summary_sha,
                )

            if store.head(keys.final_annotations) is None or store.head(
                keys.final_summary
            ) is None:
                raise RuntimeError("The final Stage 2 artifacts were not published.")

            current = get_annotation_job(job_id)
            if current is None or current.get("status") not in {
                "queued",
                "processing",
            }:
                return
            self._complete_from_final_summary(current, summary)

            # The final pair is now durable and deterministic; branch artifacts
            # are temporary coordination objects and can be removed.
            for key in (
                keys.cell_annotations,
                keys.cell_summary,
                keys.pubtator_annotations,
                keys.pubtator_summary,
            ):
                try:
                    store.delete(key)
                except Exception:
                    logger.warning(
                        "Could not delete temporary Stage 2 branch artifact %s",
                        key,
                        exc_info=True,
                    )
        except Exception as exc:
            logger.exception("Railway final merge %s failed", job_id)
            self._fail(
                job_id,
                message="Railway could not merge the CellExLink and PubTator3 results.",
                error=str(exc),
                stats={"merge_status": "failed"},
            )
            raise

    def shutdown(self) -> None:
        """Stop accepting new temporary Stage 2 coordination work."""

        self._pool.shutdown(wait=False, cancel_futures=True)


railway_annotation_executor = RailwayAnnotationExecutor()

__all__ = ["RailwayAnnotationExecutor", "railway_annotation_executor"]
