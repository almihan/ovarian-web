"""Simple four-stage orchestration for shared defaults and isolated user deltas."""

from __future__ import annotations

import logging
import secrets
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping

from backend.config import settings
from backend.worker_state import (
    create_annotation_job,
    create_relation_job,
    get_annotation_job,
    get_relation_job,
    clear_worker_state,
    update_annotation_job,
    update_relation_job,
    utc_now,
)
from backend.default_cache import (
    build_custom_stage1,
    cached_json_pair,
    get_or_build_default_stage1,
    materialize_artifact,
    run_scoped_signature,
)
from backend.pipeline.annotation_contract import (
    ANNOTATION_PIPELINE_VERSION,
    annotation_artifact_keys,
    annotation_model_signature,
    callback_token_hash,
    callback_token_matches,
)
from backend.pipeline.network_builder import build_interaction_network
from backend.pipeline.relation_contract import (
    relation_artifact_keys,
    relation_model_signature,
)
from backend.pipeline.retrieval import RetrievalError, build_effective_inputs
from backend.runtime import run_registry
from backend.services.local_executor import (
    local_executor,
    local_ml_dependencies_available,
)
from backend.services.modal_executor import modal_executor
from backend.services.railway_annotation_executor import railway_annotation_executor
from backend.services.relation_executor import relation_executor
from backend.storage.artifacts import ArtifactRef, get_artifact_store, prefixed_key
from backend.storage.bundles import build_deterministic_gzip_bundle

logger = logging.getLogger(__name__)


def _int(stats: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = stats.get(name)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _float(stats: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value = stats.get(name)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _chunk_count(stats: Mapping[str, Any]) -> int:
    return _int(
        stats,
        "chunk_count",
        "chunks_written",
        "output_chunk_count",
        "chunks_processed",
    )


def _stage_progress(
    run_id: str,
    stage_name: str,
    *,
    progress: int,
    stage: str,
    message: str,
    stats: Mapping[str, Any] | None = None,
) -> None:
    if not run_registry.exists(run_id):
        return
    current = run_registry.public(run_id)["stages"][stage_name]
    if current.get("status") in {"completed", "failed"}:
        return
    run_registry.update_stage(
        run_id,
        stage_name,
        status="processing",
        stage=stage,
        progress=max(int(current.get("progress") or 0), min(99, int(progress))),
        message=message,
        stats=dict(stats or current.get("stats") or {}),
    )


def _scaled(value: int, start: int, end: int) -> int:
    safe = max(0, min(100, int(value)))
    return start + round((end - start) * safe / 100)


def _annotation_counts(stats: Mapping[str, Any]) -> dict[str, int]:
    return {
        "paper_count": _int(stats, "paper_count", "papers_processed"),
        "chunk_count": _chunk_count(stats),
        "mention_count": _int(
            stats,
            "cell_count",
            "mention_count",
            "mention_occurrences",
            "mentions_detected",
        ),
        "normalized_count": _int(
            stats,
            "normalized_count",
            "normalized_occurrences",
        ),
        "unresolved_count": _int(
            stats,
            "unresolved_count",
            "unresolved_occurrences",
        ),
    }


def _cell_branch_ready(job: Mapping[str, Any]) -> bool:
    keys = annotation_artifact_keys(
        source_sha256=str(job.get("source_artifact_sha256") or ""),
        model_signature=str(job.get("model_signature") or ""),
    )
    store = get_artifact_store()
    return bool(
        store.head(keys.cell_annotations) is not None
        and store.head(keys.cell_summary) is not None
    )


def _apply_cell_result(
    job: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    current = get_annotation_job(str(job["id"])) or dict(job)
    if current.get("status") in {"completed", "failed"}:
        return current

    incoming_stats = result.get("stats")
    incoming_stats = incoming_stats if isinstance(incoming_stats, Mapping) else {}
    stats = dict(current.get("stats") or {})
    stats.update(dict(incoming_stats))
    status = str(result.get("status") or "cell_completed").casefold()
    if status == "failed":
        stats["cell_branch_status"] = "failed"
        return update_annotation_job(
            str(current["id"]),
            status="failed",
            stage="failed",
            progress=100,
            message=str(result.get("message") or "CellExLink extraction failed."),
            stats=stats,
            error=str(result.get("error") or "CellExLink extraction failed."),
            completed_at=utc_now(),
            last_remote_check_at=utc_now(),
            **_annotation_counts(stats),
        ) or current

    if not _cell_branch_ready(current):
        raise RuntimeError(
            "CellExLink completed without publishing its branch artifacts."
        )
    stats["cell_branch_status"] = "completed"
    keys = annotation_artifact_keys(
        source_sha256=str(current.get("source_artifact_sha256") or ""),
        model_signature=str(current.get("model_signature") or ""),
    )
    store = get_artifact_store()
    pubtator_ready = bool(
        store.head(keys.pubtator_annotations) is not None
        and store.head(keys.pubtator_summary) is not None
    )
    updated = update_annotation_job(
        str(current["id"]),
        status="processing",
        stage="merging_entities" if pubtator_ready else "waiting_for_pubtator3",
        progress=max(int(current.get("progress") or 0), 86 if pubtator_ready else 80),
        message=(
            "Both entity branches are ready; Railway is preparing the final merge."
            if pubtator_ready
            else "CellExLink is complete; Railway is waiting for PubTator3."
        ),
        stats=stats,
        last_remote_check_at=utc_now(),
        error=None,
        **_annotation_counts(stats),
    ) or current
    railway_annotation_executor.schedule_finalize(str(current["id"]))
    return updated


class PipelineOrchestrator:
    """Bounded stage queues with no persistent user job history."""

    def __init__(self) -> None:
        self._stage1_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="public-stage1"
        )
        self._stage2_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="public-stage2"
        )
        self._stage3_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="public-stage3"
        )
        self._stage4_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="public-stage4"
        )
        self._guard = threading.RLock()
        self._running: set[tuple[str, str]] = set()

    def initialize(self) -> None:
        settings.ensure_directories()
        clear_worker_state()
        run_registry.clear()
        for path in (
            settings.data_dir / "runs",
            settings.data_dir / "work",
            settings.local_annotation_jobs_dir,
            settings.relation_jobs_dir,
        ):
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
        try:
            get_artifact_store().delete_prefix(prefixed_key("runs"))
        except Exception as exc:
            logger.warning("Could not remove old temporary run artifacts: %s", exc)

    def shutdown(self) -> None:
        for pool in (
            self._stage1_pool,
            self._stage2_pool,
            self._stage3_pool,
            self._stage4_pool,
        ):
            pool.shutdown(wait=False, cancel_futures=True)
        relation_executor.shutdown()
        shutdown = getattr(railway_annotation_executor, "shutdown", None)
        if callable(shutdown):
            shutdown()
        try:
            get_artifact_store().delete_prefix(prefixed_key("runs"))
        except Exception as exc:
            logger.warning("Could not remove temporary run artifacts: %s", exc)
        shutil.rmtree(settings.data_dir / "runs", ignore_errors=True)
        shutil.rmtree(settings.data_dir / "work", ignore_errors=True)
        run_registry.clear()
        clear_worker_state()

    def _cleanup_expired_runs(self) -> None:
        for record in run_registry.pop_expired(settings.run_retention_seconds):
            run_id = str(record.get("id") or "")
            if not run_id:
                continue
            shutil.rmtree(settings.data_dir / "runs" / run_id, ignore_errors=True)
            try:
                get_artifact_store().delete_prefix(
                    prefixed_key(f"runs/{run_id}")
                )
            except Exception as exc:
                logger.warning(
                    "Could not remove expired temporary artifacts for run %s: %s",
                    run_id,
                    exc,
                )

    def create_run(self, query: str) -> dict[str, Any]:
        self._cleanup_expired_runs()
        cleaned = " ".join(str(query or "").split())
        effective = build_effective_inputs("keywords", cleaned)
        has_custom = bool(
            effective.user_keyword_count
            or effective.user_exclusion_count
            or effective.user_pmid_count
            or effective.user_pmcid_count
        )
        run = run_registry.create(query=cleaned, has_custom_input=has_custom)
        run_id = str(run["id"])
        run_registry.start_stage(
            run_id,
            "retrieval",
            "Stage 1 is queued. Shared defaults will be reused when available.",
        )
        self._submit(run_id, "retrieval", self._run_stage1)
        return run_registry.public(run_id)

    def start_stage(
        self,
        run_id: str,
        stage_name: str,
        *,
        callback_base_url: str,
    ) -> dict[str, Any]:
        if stage_name not in {"annotation", "relation", "network"}:
            raise ValueError("Unknown pipeline stage.")
        run = run_registry.get(run_id)
        if run is None:
            raise KeyError(run_id)
        order = ["retrieval", "annotation", "relation", "network"]
        previous = order[order.index(stage_name) - 1]
        if run["stages"][previous].get("status") != "completed":
            raise RuntimeError(f"{previous.title()} must complete first.")
        current = run["stages"][stage_name]
        if current.get("status") in {"queued", "processing"}:
            return run_registry.public(run_id)
        if current.get("status") == "completed":
            run_registry.reset_downstream_stages(run_id, stage_name)
        run_registry.start_stage(
            run_id,
            stage_name,
            f"{current.get('label') or stage_name.title()} is queued.",
        )
        target = {
            "annotation": lambda value: self._run_stage2(
                value, callback_base_url=callback_base_url
            ),
            "relation": self._run_stage3,
            "network": self._run_stage4,
        }[stage_name]
        self._submit(run_id, stage_name, target)
        return run_registry.public(run_id)

    def _submit(
        self,
        run_id: str,
        stage_name: str,
        target: Callable[[str], None],
    ) -> None:
        token = (run_id, stage_name)
        with self._guard:
            if token in self._running:
                return
            self._running.add(token)
        pool = {
            "retrieval": self._stage1_pool,
            "annotation": self._stage2_pool,
            "relation": self._stage3_pool,
            "network": self._stage4_pool,
        }[stage_name]

        def runner() -> None:
            try:
                target(run_id)
            except Exception as exc:
                logger.error(
                    "%s failed for run %s: %s",
                    stage_name,
                    run_id,
                    exc,
                )
                run_registry.fail_stage(run_id, stage_name, str(exc))
            finally:
                with self._guard:
                    self._running.discard(token)

        pool.submit(runner)

    def _run_stage1(self, run_id: str) -> None:
        started = time.monotonic()
        run = run_registry.get(run_id)
        if run is None:
            return

        def default_progress(
            stage: str,
            progress: int,
            message: str,
            stats: dict[str, Any],
        ) -> None:
            _stage_progress(
                run_id,
                "retrieval",
                progress=_scaled(progress, 2, 62),
                stage=f"default_{stage}",
                message=f"Shared default corpus: {message}",
                stats=stats,
            )

        _stage_progress(
            run_id,
            "retrieval",
            progress=2,
            stage="loading_default_cache",
            message="Loading the shared default Stage 1 artifact.",
        )
        default_result = get_or_build_default_stage1(default_progress)
        run_registry.set_private(run_id, "stage1_default", default_result)
        _stage_progress(
            run_id,
            "retrieval",
            progress=64,
            stage="default_ready",
            message=(
                "Reused the shared default Stage 1 artifact."
                if default_result.get("reused")
                else "Built and saved the shared default Stage 1 artifact."
            ),
            stats=dict(default_result.get("stats") or {}),
        )

        custom_result: dict[str, Any] | None = None
        if bool(run.get("has_custom_input")):
            def custom_progress(
                stage: str,
                progress: int,
                message: str,
                stats: dict[str, Any],
            ) -> None:
                _stage_progress(
                    run_id,
                    "retrieval",
                    progress=_scaled(progress, 65, 98),
                    stage=f"custom_{stage}",
                    message=f"This run's added papers: {message}",
                    stats=stats,
                )

            custom_result = build_custom_stage1(
                run_id=run_id,
                query=str(run.get("query") or ""),
                baseline=default_result,
                progress=custom_progress,
            )
        run_registry.set_private(run_id, "stage1_custom", custom_result)

        default_stats = dict(default_result.get("stats") or {})
        custom_stats = dict((custom_result or {}).get("stats") or {})
        paper_count = _int(default_stats, "paper_count") + _int(
            custom_stats, "paper_count"
        )
        abstract_count = _int(default_stats, "abstract_count") + _int(
            custom_stats, "abstract_count"
        )
        fulltext_count = _int(
            default_stats,
            "fulltext_available",
            "fulltexts_downloaded",
        ) + _int(custom_stats, "fulltext_available", "fulltexts_downloaded")
        stats = {
            "paper_count": paper_count,
            "abstract_count": abstract_count,
            "fulltext_count": fulltext_count,
            "fulltexts_downloaded": fulltext_count,
            "papers_without_pmcid": _int(
                default_stats, "papers_without_pmcid", "without_pmcid"
            )
            + _int(custom_stats, "papers_without_pmcid", "without_pmcid"),
            "chunk_count": _chunk_count(default_stats) + _chunk_count(custom_stats),
            "default_paper_count": _int(default_stats, "paper_count"),
            "custom_paper_count": _int(custom_stats, "paper_count"),
            "default_reused": bool(default_result.get("reused")),
            "custom_recomputed": bool(run.get("has_custom_input")),
            "custom_query_changed_results": custom_result is not None,
        }
        elapsed = round(time.monotonic() - started, 2)
        stats["elapsed_seconds"] = elapsed
        run_registry.complete_stage(
            run_id,
            "retrieval",
            message=(
                f"Ready: {paper_count:,} papers ({stats['custom_paper_count']:,} "
                "added only for this run)."
            ),
            stats=stats,
            elapsed_seconds=elapsed,
            download_url=f"/api/runs/{run_id}/download/stage1",
        )

    def _annotation_backend(self, callback_base_url: str) -> tuple[str, str]:
        backend = settings.cell_annotation_backend
        if backend == "local":
            if settings.artifact_backend != "local":
                raise RuntimeError("Local Stage 2 requires ARTIFACT_BACKEND=local.")
            if not local_ml_dependencies_available():
                raise RuntimeError(
                    "Local CellExLink dependencies are missing. Install requirements-local.txt."
                )
            return backend, callback_base_url.rstrip("/")
        if backend == "modal":
            if settings.artifact_backend != "s3":
                raise RuntimeError(
                    "Modal Stage 2 requires the Railway S3-compatible artifact bucket."
                )
            if not settings.modal_configured:
                raise RuntimeError("Modal credentials or the deployed function are missing.")
            if not settings.public_base_url:
                raise RuntimeError("PUBLIC_BASE_URL is required for Modal callbacks.")
            return backend, settings.public_base_url.rstrip("/")
        raise RuntimeError("Stage 2 entity extraction is disabled.")

    def _annotation_artifact(
        self,
        *,
        run_id: str,
        scope: str,
        source: Mapping[str, Any],
        model_signature: str,
        callback_base_url: str,
        progress_start: int,
        progress_end: int,
    ) -> dict[str, Any]:
        source_ref = ArtifactRef.from_dict(source["artifact"])
        keys = annotation_artifact_keys(
            source_sha256=source_ref.sha256,
            model_signature=model_signature,
        )
        cached = cached_json_pair(
            output_key=keys.final_annotations,
            summary_key=keys.final_summary,
            expected_model_signature=model_signature,
            expected_source_sha256=source_ref.sha256,
        )
        if cached is not None:
            cached["worker_job_id"] = "shared-default-stage2"
            cached["reused"] = scope == "default"
            return cached

        backend, callback_root = self._annotation_backend(callback_base_url)
        worker_job_id = f"a2-{uuid.uuid4().hex[:20]}"
        callback_token = secrets.token_urlsafe(32)
        create_annotation_job(
            job_id=worker_job_id,
            source_job_id=f"{run_id}-{scope}-stage1",
            executor=backend,
            model_signature=model_signature,
            source_artifact_key=source_ref.key,
            source_artifact_sha256=source_ref.sha256,
            output_artifact_key=keys.final_annotations,
            summary_artifact_key=keys.final_summary,
            callback_token_hash=callback_token_hash(callback_token),
        )
        store = get_artifact_store()
        source_stats = dict(source.get("stats") or {})
        update_annotation_job(
            worker_job_id,
            status="processing",
            stage="running_parallel_branches",
            progress=2,
            message=(
                "CellExLink and PubTator3 are starting for the shared default corpus."
                if scope == "default"
                else "CellExLink and PubTator3 are recomputing this run's added papers."
            ),
            stats={
                "cell_branch_status": "starting",
                "pubtator_branch_status": "starting",
                "cache_scope": "shared_default" if scope == "default" else "run_only",
            },
            paper_count=_int(source_stats, "paper_count"),
            chunk_count=_chunk_count(source_stats),
            started_at=utc_now(),
            last_remote_check_at=utc_now(),
        )

        if not _cell_branch_ready(get_annotation_job(worker_job_id) or {}):
            payload = {
                "job_id": worker_job_id,
                "pipeline_version": ANNOTATION_PIPELINE_VERSION,
                "model_signature": model_signature,
                "input": {
                    "url": store.presign_get(
                        source_ref.key,
                        expires_seconds=settings.artifact_presigned_ttl_seconds,
                    ),
                    "key": source_ref.key,
                    "sha256": source_ref.sha256,
                    "size_bytes": source_ref.size_bytes,
                },
                "output": {
                    "cell_annotations_url": store.presign_put(
                        keys.cell_annotations,
                        content_type="application/gzip",
                    ),
                    "cell_annotations_key": keys.cell_annotations,
                    "cell_summary_url": store.presign_put(
                        keys.cell_summary,
                        content_type="application/json",
                    ),
                    "cell_summary_key": keys.cell_summary,
                },
                "callback": {
                    "url": (
                        f"{callback_root}/api/internal/annotations/"
                        f"{worker_job_id}/callback"
                    ),
                    "token": callback_token,
                },
                "models": {
                    "ner": settings.cell_ner_model,
                    "ner_revision": settings.cell_ner_revision,
                    "nen": settings.cell_nen_model,
                    "nen_revision": settings.cell_nen_revision,
                },
                "options": {
                    "disable_abbreviations": settings.cell_disable_abbreviations,
                    "cpu_threads": settings.cell_cpu_threads,
                    "ner_text_batch_size": settings.cell_ner_text_batch_size,
                    "ner_window_batch_size": settings.cell_ner_window_batch_size,
                    "nen_batch_size": settings.cell_nen_batch_size,
                    "nen_request_batch_size": settings.cell_nen_request_batch_size,
                },
                "source_stats": {
                    "paper_count": _int(source_stats, "paper_count"),
                    "chunk_count": _chunk_count(source_stats),
                },
            }
            remote_call_id = (
                local_executor.submit(payload)
                if backend == "local"
                else modal_executor.submit(payload)
            )
            update_annotation_job(
                worker_job_id,
                remote_call_id=remote_call_id,
                stats={
                    "cell_branch_status": "submitted",
                    "pubtator_branch_status": "submitted",
                    "cache_scope": (
                        "shared_default" if scope == "default" else "run_only"
                    ),
                },
            )
        else:
            cell_summary = store.read_json(keys.cell_summary)
            _apply_cell_result(
                get_annotation_job(worker_job_id) or {},
                {
                    "status": "cell_completed",
                    "stats": dict(cell_summary.get("stats") or {}),
                },
            )

        railway_annotation_executor.submit_pubtator(worker_job_id)
        deadline = time.monotonic() + settings.cell_job_timeout_seconds
        last_remote_poll = 0.0
        while time.monotonic() < deadline:
            job = get_annotation_job(worker_job_id)
            if job is None:
                raise RuntimeError("The temporary Stage 2 worker record disappeared.")
            local_progress = int(job.get("progress") or 0)
            _stage_progress(
                run_id,
                "annotation",
                progress=_scaled(local_progress, progress_start, progress_end),
                stage=f"{scope}_{job.get('stage') or 'processing'}",
                message=(
                    f"Shared default entities: {job.get('message') or ''}"
                    if scope == "default"
                    else f"Added-paper entities: {job.get('message') or ''}"
                ),
                stats=dict(job.get("stats") or {}),
            )
            if job.get("status") == "completed":
                output_ref = store.head(keys.final_annotations)
                summary_ref = store.head(keys.final_summary)
                if output_ref is None or summary_ref is None:
                    raise RuntimeError("Stage 2 completed without its final artifacts.")
                summary = store.read_json(keys.final_summary)
                result = {
                    "artifact": output_ref.to_dict(),
                    "summary_artifact": summary_ref.to_dict(),
                    "summary": summary,
                    "stats": dict(summary.get("stats") or {}),
                    "worker_job_id": worker_job_id,
                    "reused": False,
                }
                if scope == "custom":
                    try:
                        store.delete(keys.final_summary)
                    except Exception as exc:
                        logger.warning(
                            "Could not remove temporary Stage 2 summary for run %s: %s",
                            run_id,
                            exc,
                        )
                    result.pop("summary_artifact", None)
                if backend == "local":
                    local_executor.cleanup(worker_job_id)
                return result
            if job.get("status") == "failed":
                if backend == "local":
                    local_executor.cleanup(worker_job_id)
                raise RuntimeError(
                    str(job.get("error") or job.get("message") or "Stage 2 failed.")
                )

            railway_annotation_executor.ensure_job(worker_job_id)
            remote_call_id = str(job.get("remote_call_id") or "")
            if remote_call_id and time.monotonic() - last_remote_poll >= 1.5:
                last_remote_poll = time.monotonic()
                poll = (
                    local_executor.poll(remote_call_id)
                    if backend == "local"
                    else modal_executor.poll(remote_call_id)
                )
                if poll.state == "completed":
                    _apply_cell_result(job, poll.result or {})
                elif poll.state in {"failed", "expired"}:
                    if _cell_branch_ready(job):
                        _apply_cell_result(
                            job,
                            {
                                "status": "cell_completed",
                                "message": "Recovered the published CellExLink branch.",
                            },
                        )
                    else:
                        update_annotation_job(
                            worker_job_id,
                            status="failed",
                            stage="failed",
                            progress=100,
                            message="The CellExLink branch failed.",
                            error=poll.error or "CellExLink executor failed.",
                            completed_at=utc_now(),
                        )
            time.sleep(1.0)
        if backend == "local":
            local_executor.cleanup(worker_job_id)
        raise TimeoutError("Stage 2 exceeded CELL_JOB_TIMEOUT_SECONDS.")

    def _run_stage2(self, run_id: str, *, callback_base_url: str) -> None:
        started = time.monotonic()
        default_source = run_registry.get_private(run_id, "stage1_default")
        custom_source = run_registry.get_private(run_id, "stage1_custom")
        base_signature = annotation_model_signature()
        has_custom = custom_source is not None
        default_result = self._annotation_artifact(
            run_id=run_id,
            scope="default",
            source=default_source,
            model_signature=base_signature,
            callback_base_url=callback_base_url,
            progress_start=2,
            progress_end=48 if has_custom else 98,
        )
        custom_result = None
        if has_custom:
            custom_result = self._annotation_artifact(
                run_id=run_id,
                scope="custom",
                source=custom_source,
                model_signature=run_scoped_signature(run_id, base_signature),
                callback_base_url=callback_base_url,
                progress_start=50,
                progress_end=98,
            )
        run_registry.set_private(run_id, "stage2_default", default_result)
        run_registry.set_private(run_id, "stage2_custom", custom_result)

        default_stats = dict(default_result.get("stats") or {})
        custom_stats = dict((custom_result or {}).get("stats") or {})
        stats = {
            "paper_count": _int(default_stats, "paper_count")
            + _int(custom_stats, "paper_count"),
            "chunk_count": _chunk_count(default_stats) + _chunk_count(custom_stats),
            "cell_count": _int(default_stats, "cell_count", "mention_count")
            + _int(custom_stats, "cell_count", "mention_count"),
            "gene_count": _int(default_stats, "gene_count")
            + _int(custom_stats, "gene_count"),
            "hormone_count": _int(default_stats, "hormone_count")
            + _int(custom_stats, "hormone_count"),
            "normalized_count": _int(
                default_stats, "normalized_count", "normalized_occurrences"
            )
            + _int(custom_stats, "normalized_count", "normalized_occurrences"),
            "default_reused": bool(default_result.get("reused")),
            "custom_recomputed": custom_result is not None,
        }
        elapsed = round(time.monotonic() - started, 2)
        stats["elapsed_seconds"] = elapsed
        run_registry.complete_stage(
            run_id,
            "annotation",
            message=(
                f"Ready: {stats['cell_count']:,} cell types, "
                f"{stats['gene_count']:,} genes, and "
                f"{stats['hormone_count']:,} hormones."
            ),
            stats=stats,
            elapsed_seconds=elapsed,
            download_url=f"/api/runs/{run_id}/download/stage2",
        )

    def handle_annotation_callback(
        self,
        worker_job_id: str,
        *,
        token: str,
        payload: Mapping[str, Any],
    ) -> None:
        job = get_annotation_job(worker_job_id)
        if job is None:
            raise KeyError(worker_job_id)
        if not callback_token_matches(token, job.get("callback_token_hash")):
            raise PermissionError("Invalid callback token.")
        if job.get("status") in {"completed", "failed"}:
            return
        status = str(payload.get("status") or "processing").casefold()
        incoming_stats = payload.get("stats")
        incoming_stats = incoming_stats if isinstance(incoming_stats, Mapping) else {}
        stats = dict(job.get("stats") or {})
        stats.update(dict(incoming_stats))
        if payload.get("output_sha256"):
            stats["cell_output_sha256"] = str(payload["output_sha256"])
        if payload.get("summary_sha256"):
            stats["cell_summary_sha256"] = str(payload["summary_sha256"])
        if status in {"cell_completed", "completed"}:
            _apply_cell_result(
                job,
                {
                    "status": "cell_completed",
                    "message": payload.get("message"),
                    "stats": stats,
                },
            )
            railway_annotation_executor.ensure_job(worker_job_id)
            return
        if status == "failed":
            update_annotation_job(
                worker_job_id,
                status="failed",
                stage="failed",
                progress=100,
                message=str(payload.get("message") or "CellExLink extraction failed."),
                stats=stats,
                error=str(payload.get("error") or "CellExLink worker failed."),
                completed_at=utc_now(),
                last_remote_check_at=utc_now(),
                **_annotation_counts(stats),
            )
            return
        update_annotation_job(
            worker_job_id,
            status="processing",
            stage=str(payload.get("stage") or "cell_annotation"),
            progress=max(
                int(job.get("progress") or 0),
                max(0, min(100, int(payload.get("progress") or 0))),
            ),
            message=str(payload.get("message") or "CellExLink is running."),
            stats=stats,
            last_remote_check_at=utc_now(),
            **_annotation_counts(stats),
        )
        railway_annotation_executor.ensure_job(worker_job_id)

    def _relation_artifact(
        self,
        *,
        run_id: str,
        scope: str,
        chunks: Mapping[str, Any],
        annotations: Mapping[str, Any],
        model_signature: str,
        progress_start: int,
        progress_end: int,
    ) -> dict[str, Any]:
        chunks_ref = ArtifactRef.from_dict(chunks["artifact"])
        annotations_ref = ArtifactRef.from_dict(annotations["artifact"])
        keys = relation_artifact_keys(
            source_annotation_sha256=annotations_ref.sha256,
            source_chunks_sha256=chunks_ref.sha256,
            model_signature=model_signature,
        )
        cached = cached_json_pair(
            output_key=keys.relations,
            summary_key=keys.summary,
            expected_model_signature=model_signature,
        )
        if cached is not None:
            cached["worker_job_id"] = "shared-default-stage3"
            cached["reused"] = scope == "default"
            return cached
        if not settings.relation_configured:
            raise RuntimeError("OPENAI_API_KEY is required for Stage 3.")

        worker_job_id = f"r3-{uuid.uuid4().hex[:20]}"
        create_relation_job(
            job_id=worker_job_id,
            source_annotation_job_id=str(
                annotations.get("worker_job_id") or f"{run_id}-{scope}-stage2"
            ),
            model_signature=model_signature,
            source_chunks_artifact_key=chunks_ref.key,
            source_chunks_artifact_sha256=chunks_ref.sha256,
            source_annotation_artifact_key=annotations_ref.key,
            source_annotation_artifact_sha256=annotations_ref.sha256,
            output_artifact_key=keys.relations,
            summary_artifact_key=keys.summary,
        )
        annotation_stats = dict(annotations.get("stats") or {})
        chunk_stats = dict(chunks.get("stats") or {})
        update_relation_job(
            worker_job_id,
            status="processing",
            stage="preparing_relations",
            progress=1,
            message=(
                "Preparing shared default relation extraction."
                if scope == "default"
                else "Recomputing relations for this run's added papers."
            ),
            paper_count=_int(chunk_stats, "paper_count"),
            chunk_count=_chunk_count(annotation_stats) or _chunk_count(chunk_stats),
            stats={
                "cache_scope": "shared_default" if scope == "default" else "run_only"
            },
            started_at=utc_now(),
        )
        relation_executor.submit(worker_job_id)
        while True:
            job = get_relation_job(worker_job_id)
            if job is None:
                raise RuntimeError("The temporary Stage 3 worker record disappeared.")
            _stage_progress(
                run_id,
                "relation",
                progress=_scaled(
                    int(job.get("progress") or 0), progress_start, progress_end
                ),
                stage=f"{scope}_{job.get('stage') or 'processing'}",
                message=(
                    f"Shared default relations: {job.get('message') or ''}"
                    if scope == "default"
                    else f"Added-paper relations: {job.get('message') or ''}"
                ),
                stats=dict(job.get("stats") or {}),
            )
            if job.get("status") == "completed":
                store = get_artifact_store()
                output_ref = store.head(keys.relations)
                summary_ref = store.head(keys.summary)
                if output_ref is None or summary_ref is None:
                    raise RuntimeError("Stage 3 completed without its final artifacts.")
                summary = store.read_json(keys.summary)
                result = {
                    "artifact": output_ref.to_dict(),
                    "summary_artifact": summary_ref.to_dict(),
                    "summary": summary,
                    "stats": dict(summary.get("stats") or {}),
                    "worker_job_id": worker_job_id,
                    "reused": False,
                }
                if scope == "custom":
                    try:
                        store.delete(keys.summary)
                    except Exception as exc:
                        logger.warning(
                            "Could not remove temporary Stage 3 summary for run %s: %s",
                            run_id,
                            exc,
                        )
                    result.pop("summary_artifact", None)
                return result
            if job.get("status") == "failed":
                raise RuntimeError(str(job.get("error") or job.get("message") or "Stage 3 failed."))
            relation_executor.submit(worker_job_id)
            time.sleep(1.0)

    def _run_stage3(self, run_id: str) -> None:
        started = time.monotonic()
        default_chunks = run_registry.get_private(run_id, "stage1_default")
        custom_chunks = run_registry.get_private(run_id, "stage1_custom")
        default_annotations = run_registry.get_private(run_id, "stage2_default")
        custom_annotations = run_registry.get_private(run_id, "stage2_custom")
        base_signature = relation_model_signature()
        has_custom = custom_chunks is not None and custom_annotations is not None
        default_result = self._relation_artifact(
            run_id=run_id,
            scope="default",
            chunks=default_chunks,
            annotations=default_annotations,
            model_signature=base_signature,
            progress_start=2,
            progress_end=48 if has_custom else 98,
        )
        custom_result = None
        if has_custom:
            custom_result = self._relation_artifact(
                run_id=run_id,
                scope="custom",
                chunks=custom_chunks,
                annotations=custom_annotations,
                model_signature=run_scoped_signature(run_id, base_signature),
                progress_start=50,
                progress_end=98,
            )
        run_registry.set_private(run_id, "stage3_default", default_result)
        run_registry.set_private(run_id, "stage3_custom", custom_result)

        default_stats = dict(default_result.get("stats") or {})
        custom_stats = dict((custom_result or {}).get("stats") or {})
        input_tokens = _int(default_stats, "input_tokens") + _int(
            custom_stats, "input_tokens"
        )
        cached_tokens = _int(default_stats, "cached_input_tokens") + _int(
            custom_stats, "cached_input_tokens"
        )
        stats = {
            "paper_count": _int(default_stats, "paper_count")
            + _int(custom_stats, "paper_count"),
            "chunk_count": _chunk_count(default_stats) + _chunk_count(custom_stats),
            "relation_count": _int(default_stats, "relation_count")
            + _int(custom_stats, "relation_count"),
            "eligible_chunk_count": _int(default_stats, "eligible_chunk_count")
            + _int(custom_stats, "eligible_chunk_count"),
            "api_request_count": _int(default_stats, "api_request_count")
            + _int(custom_stats, "api_request_count"),
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "prompt_cache_rate": round(cached_tokens / max(1, input_tokens), 6),
            "default_reused": bool(default_result.get("reused")),
            "custom_recomputed": custom_result is not None,
        }
        elapsed = round(time.monotonic() - started, 2)
        stats["elapsed_seconds"] = elapsed
        run_registry.complete_stage(
            run_id,
            "relation",
            message=f"Ready: {stats['relation_count']:,} validated relations.",
            stats=stats,
            elapsed_seconds=elapsed,
            download_url=f"/api/runs/{run_id}/download/stage3",
        )

    def _run_stage4(self, run_id: str) -> None:
        started = time.monotonic()
        work_root = settings.data_dir / "runs" / run_id / "stage4"
        input_root = work_root / "inputs"
        shutil.rmtree(work_root, ignore_errors=True)
        input_root.mkdir(parents=True, exist_ok=True)

        scopes = ["default"]
        if run_registry.get_private(run_id, "stage1_custom") is not None:
            scopes.append("custom")
        stage_sources: dict[str, list[Path]] = {
            "chunks": [],
            "annotations": [],
            "relations": [],
        }
        for index, scope in enumerate(scopes):
            for private_name, output_name in (
                (f"stage1_{scope}", "chunks"),
                (f"stage2_{scope}", "annotations"),
                (f"stage3_{scope}", "relations"),
            ):
                source = run_registry.get_private(run_id, private_name)
                if source is None:
                    raise RuntimeError(f"Missing {private_name} artifact for Stage 4.")
                destination = input_root / f"{index:02d}-{output_name}-{scope}.jsonl.gz"
                materialize_artifact(source["artifact"], destination)
                stage_sources[output_name].append(destination)

        _stage_progress(
            run_id,
            "network",
            progress=8,
            stage="merging_inputs",
            message="Combining the aligned default and run-specific artifacts.",
        )
        combined_chunks = input_root / "chunks.jsonl.gz"
        combined_annotations = input_root / "annotations.jsonl.gz"
        combined_relations = input_root / "relations.jsonl.gz"
        build_deterministic_gzip_bundle(stage_sources["chunks"], combined_chunks)
        build_deterministic_gzip_bundle(
            stage_sources["annotations"], combined_annotations
        )
        build_deterministic_gzip_bundle(stage_sources["relations"], combined_relations)

        graph_path = work_root / "graph.sqlite"
        entity_index_path = work_root / "entity-index.jsonl.gz"
        total_chunks = _int(
            run_registry.public(run_id)["stages"]["relation"].get("stats") or {},
            "chunk_count",
        )

        def progress(row_count: int, message: str, stats: dict[str, Any]) -> None:
            percentage = 10 + round(84 * min(total_chunks, row_count) / max(1, total_chunks))
            _stage_progress(
                run_id,
                "network",
                progress=min(96, percentage),
                stage="building_network",
                message=message,
                stats=stats,
            )

        result = build_interaction_network(
            relation_path=combined_relations,
            chunks_path=combined_chunks,
            annotations_path=combined_annotations,
            graph_path=graph_path,
            entity_index_path=entity_index_path,
            progress=progress,
        )
        shutil.rmtree(input_root, ignore_errors=True)
        run_registry.set_private(run_id, "graph_path", str(result.graph_path))
        run_registry.set_private(
            run_id, "entity_index_path", str(result.entity_index_path)
        )
        stats = dict(result.stats)
        elapsed = round(time.monotonic() - started, 2)
        stats["elapsed_seconds"] = elapsed
        stats["persistent_artifact"] = False
        run_registry.complete_stage(
            run_id,
            "network",
            message=(
                f"Ready: {int(stats.get('node_count') or 0):,} nodes and "
                f"{int(stats.get('edge_count') or 0):,} edges."
            ),
            stats=stats,
            elapsed_seconds=elapsed,
            open_url=f"/network/{run_id}",
        )

    def artifacts_for_download(
        self, run_id: str, stage_name: str
    ) -> list[dict[str, Any]]:
        if stage_name not in {"stage1", "stage2", "stage3"}:
            raise ValueError("Unknown download stage.")
        mapping = {
            "stage1": "stage1",
            "stage2": "stage2",
            "stage3": "stage3",
        }
        prefix = mapping[stage_name]
        refs: list[dict[str, Any]] = []
        default = run_registry.get_private(run_id, f"{prefix}_default")
        custom = run_registry.get_private(run_id, f"{prefix}_custom")
        for value in (default, custom):
            if isinstance(value, Mapping) and isinstance(value.get("artifact"), Mapping):
                refs.append(dict(value["artifact"]))
        return refs


pipeline_orchestrator = PipelineOrchestrator()

__all__ = [
    "PipelineOrchestrator",
    "RetrievalError",
    "pipeline_orchestrator",
]
