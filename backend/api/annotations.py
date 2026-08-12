"""Stage 2 API for reusable cell, gene, and hormone extraction jobs."""

from __future__ import annotations

import json
import logging
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.database.database import (
    create_annotation_job,
    find_active_annotation,
    find_any_active_annotation,
    find_reusable_annotation,
    get_annotation_job,
    get_job,
    list_annotation_jobs,
    update_annotation_job,
    utc_now,
)
from backend.pipeline.annotation_contract import (
    ANNOTATION_PIPELINE_VERSION,
    annotation_artifact_keys,
    annotation_model_signature,
    callback_token_hash,
    callback_token_matches,
    source_artifact_from_summary,
)
from backend.services.modal_executor import modal_executor
from backend.services.railway_annotation_executor import railway_annotation_executor
from backend.services.local_executor import (
    local_executor,
    local_ml_dependencies_available,
)
from backend.storage.artifacts import get_artifact_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/annotations", tags=["entity annotations"])


class AnnotationCreate(BaseModel):
    source_job_id: str = Field(min_length=1, max_length=64)

    @field_validator("source_job_id")
    @classmethod
    def clean_source_job_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(
            not (character.isalnum() or character in "-_")
            for character in cleaned
        ):
            raise ValueError("source_job_id contains unsupported characters")
        return cleaned


class WorkerCallback(BaseModel):
    status: str = Field(default="processing", max_length=32)
    stage: str = Field(default="processing", max_length=64)
    progress: int = Field(default=0, ge=0, le=100)
    message: str = Field(default="", max_length=2000)
    stats: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(default=None, max_length=8000)
    output_sha256: str | None = Field(default=None, max_length=128)
    summary_sha256: str | None = Field(default=None, max_length=128)


def _source_payload(source_job_id: str) -> dict[str, Any] | None:
    source = get_job(source_job_id)
    if source is None:
        return None
    return {
        "id": source["id"],
        "status": source["status"],
        "input_type": source["input_type"],
        "query": source.get("query") or "",
        "paper_count": int(source.get("paper_count") or 0),
        "created_at": source.get("created_at"),
        "completed_at": source.get("completed_at"),
    }


def _public_job(job: Mapping[str, Any], *, reused: bool = False) -> dict[str, Any]:
    hidden = {"callback_token_hash", "result_path"}
    payload = {key: value for key, value in dict(job).items() if key not in hidden}
    payload["source_job"] = _source_payload(str(job["source_job_id"]))
    payload["reused"] = reused or bool(job.get("reused_from_job_id"))
    if job.get("status") == "completed":
        payload["downloads"] = {
            "annotations": f"/api/annotations/{job['id']}/download",
        }
    return payload


def _load_source_summary(source_job: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = source_job.get("result_path")
    if not raw_path:
        raise HTTPException(
            status_code=409,
            detail="The retrieval job did not record its chunk summary.",
        )
    summary_path = Path(str(raw_path)).expanduser().resolve()
    papers_root = settings.papers_dir.expanduser().resolve()
    if not summary_path.is_relative_to(papers_root) or not summary_path.is_file():
        raise HTTPException(
            status_code=409,
            detail="The retrieval summary is missing or outside the paper cache.",
        )
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500, detail="The retrieval summary is unreadable."
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=500, detail="The retrieval summary has an invalid format."
        )
    return payload


def _counts_from_stats(stats: Mapping[str, Any]) -> dict[str, int]:
    return {
        "paper_count": int(
            stats.get("paper_count")
            or stats.get("papers_total")
            or stats.get("papers_processed")
            or 0
        ),
        "chunk_count": int(
            stats.get("chunk_count") or stats.get("chunks_processed") or 0
        ),
        "mention_count": int(
            stats.get("mention_count")
            or stats.get("mention_occurrences")
            or stats.get("mentions_detected")
            or 0
        ),
        "normalized_count": int(
            stats.get("normalized_count")
            or stats.get("normalized_occurrences")
            or 0
        ),
        "unresolved_count": int(
            stats.get("unresolved_count")
            or stats.get("unresolved_occurrences")
            or 0
        ),
    }


def _elapsed_from_stats(stats: Mapping[str, Any], fallback: float = 0.0) -> float:
    try:
        return max(0.0, float(stats.get("elapsed_seconds") or fallback))
    except (TypeError, ValueError):
        return max(0.0, fallback)


def _verify_completed_artifacts(job: Mapping[str, Any]) -> None:
    store = get_artifact_store()
    output_key = str(job.get("output_artifact_key") or "")
    summary_key = str(job.get("summary_artifact_key") or "")
    if not output_key or store.head(output_key) is None:
        raise RuntimeError(
            "The annotation worker completed without publishing the annotation artifact."
        )
    if not summary_key or store.head(summary_key) is None:
        raise RuntimeError(
            "The annotation worker completed without publishing the summary artifact."
        )


def _artifact_keys_for_job(job: Mapping[str, Any]):
    source_sha = str(job.get("source_artifact_sha256") or "")
    model_signature = str(job.get("model_signature") or "")
    if not source_sha or not model_signature:
        raise RuntimeError("The annotation job is missing its artifact fingerprints.")
    return annotation_artifact_keys(
        source_sha256=source_sha,
        model_signature=model_signature,
    )


def _verify_cell_branch_artifacts(job: Mapping[str, Any]) -> None:
    keys = _artifact_keys_for_job(job)
    store = get_artifact_store()
    if store.head(keys.cell_annotations) is None:
        raise RuntimeError(
            "Modal completed without publishing the CellExLink branch artifact."
        )
    if store.head(keys.cell_summary) is None:
        raise RuntimeError(
            "Modal completed without publishing the CellExLink branch summary."
        )


def _apply_final_result(
    job: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    status = str(result.get("status") or "completed").casefold()
    if status == "failed":
        return update_annotation_job(
            str(job["id"]),
            status="failed",
            stage="failed",
            progress=100,
            message=str(result.get("message") or "Entity extraction failed."),
            error=str(result.get("error") or "Entity extraction failed."),
            completed_at=utc_now(),
            last_remote_check_at=utc_now(),
        ) or dict(job)

    stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
    merged_stats = dict(job.get("stats") or {})
    merged_stats.update(stats)
    counts = _counts_from_stats(merged_stats)
    candidate = dict(job)
    candidate.update(
        {
            "output_artifact_key": result.get("output_artifact_key")
            or job.get("output_artifact_key"),
            "summary_artifact_key": result.get("summary_artifact_key")
            or job.get("summary_artifact_key"),
        }
    )
    _verify_completed_artifacts(candidate)
    return update_annotation_job(
        str(job["id"]),
        status="completed",
        stage="completed",
        progress=100,
        message=str(
            result.get("message")
            or (
                f"Finished: {int(merged_stats.get('cell_count') or counts['mention_count']):,} "
                f"cell-type annotations ({counts['normalized_count']:,} normalized), "
                f"{int(merged_stats.get('gene_count') or 0):,} human gene annotations, and "
                f"{int(merged_stats.get('hormone_count') or 0):,} hormone annotations."
            )
        ),
        stats=merged_stats,
        elapsed_seconds=_elapsed_from_stats(merged_stats),
        completed_at=str(result.get("completed_at") or utc_now()),
        last_remote_check_at=utc_now(),
        error=None,
        **counts,
    ) or dict(job)


def _apply_cell_result(
    job: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Record Modal/local CellExLink completion without completing Stage 2."""

    current = get_annotation_job(str(job["id"])) or dict(job)
    if current.get("status") in {"completed", "failed"}:
        return current

    status = str(result.get("status") or "cell_completed").casefold()
    stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
    merged_stats = dict(current.get("stats") or {})
    merged_stats.update(stats)

    if status == "failed":
        merged_stats["cell_branch_status"] = "failed"
        return update_annotation_job(
            str(current["id"]),
            status="failed",
            stage="failed",
            progress=100,
            message=str(result.get("message") or "CellExLink extraction failed."),
            stats=merged_stats,
            error=str(result.get("error") or "CellExLink extraction failed."),
            elapsed_seconds=_elapsed_from_stats(
                merged_stats, current.get("elapsed_seconds") or 0
            ),
            completed_at=utc_now(),
            last_remote_check_at=utc_now(),
        ) or current

    _verify_cell_branch_artifacts(current)
    merged_stats["cell_branch_status"] = "completed"
    keys = _artifact_keys_for_job(current)
    store = get_artifact_store()
    pubtator_ready = (
        store.head(keys.pubtator_annotations) is not None
        and store.head(keys.pubtator_summary) is not None
    )
    counts = _counts_from_stats(merged_stats)
    updated = update_annotation_job(
        str(current["id"]),
        status="processing",
        stage="merging_entities" if pubtator_ready else "waiting_for_pubtator3",
        progress=max(int(current.get("progress") or 0), 86 if pubtator_ready else 80),
        message=(
            "Both annotation branches are ready; Railway is preparing the final merge."
            if pubtator_ready
            else "CellExLink is complete; Railway is waiting for PubTator3."
        ),
        stats=merged_stats,
        elapsed_seconds=_elapsed_from_stats(
            merged_stats, current.get("elapsed_seconds") or 0
        ),
        last_remote_check_at=utc_now(),
        error=None,
        **counts,
    ) or current
    railway_annotation_executor.schedule_finalize(str(current["id"]))
    return updated


def _published_final_result(job: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read the exact final artifact pair even if SQLite missed completion."""

    output_key = str(job.get("output_artifact_key") or "")
    summary_key = str(job.get("summary_artifact_key") or "")
    if not output_key or not summary_key:
        return None
    store = get_artifact_store()
    if store.head(output_key) is None or store.head(summary_key) is None:
        return None
    summary = store.read_json(summary_key)
    source = summary.get("source") if isinstance(summary.get("source"), Mapping) else {}
    expected_source = str(job.get("source_artifact_sha256") or "")
    actual_source = str(source.get("sha256") or "")
    if expected_source and actual_source != expected_source:
        raise RuntimeError("Published annotation source fingerprint does not match.")
    expected_signature = str(job.get("model_signature") or "")
    actual_signature = str(summary.get("model_signature") or "")
    if expected_signature and actual_signature != expected_signature:
        raise RuntimeError("Published annotation model signature does not match.")
    stats = summary.get("stats") if isinstance(summary.get("stats"), dict) else {}
    return {
        "status": "completed",
        "message": summary.get("message")
        or "A matching completed Stage 2 entity artifact was reused.",
        "stats": stats,
        "output_artifact_key": output_key,
        "summary_artifact_key": summary_key,
        "completed_at": summary.get("completed_at") or utc_now(),
    }


def _recover_published_job(job: Mapping[str, Any]) -> dict[str, Any] | None:
    result = _published_final_result(job)
    return _apply_final_result(job, result) if result is not None else None


def _seconds_since(value: object) -> float:
    if not value:
        return 1e9
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return 1e9
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def _reconcile_if_stale(job: dict[str, Any]) -> dict[str, Any]:
    """Recover final artifacts and resume both Stage 2 branches when needed."""

    if job.get("status") in {"completed", "failed"}:
        return job

    # The deterministic final pair is authoritative. Check it before relying on
    # callbacks or executor polling so a completed merge survives a restart.
    try:
        recovered = _recover_published_job(job)
        if recovered is not None:
            return recovered
    except Exception:
        logger.warning(
            "Published-artifact recovery failed for annotation %s",
            job["id"],
            exc_info=True,
        )

    # PubTator3 and the final merge are Railway responsibilities. Both methods
    # are idempotent, so this also resumes work after a web-service restart.
    try:
        railway_annotation_executor.ensure_job(str(job["id"]))
    except Exception:
        logger.warning(
            "Could not resume Railway Stage 2 work for annotation %s",
            job["id"],
            exc_info=True,
        )

    if (
        job.get("executor") not in {"modal", "local"}
        or not job.get("remote_call_id")
        or _seconds_since(job.get("last_remote_check_at") or job.get("updated_at"))
        < settings.modal_status_stale_seconds
    ):
        return get_annotation_job(str(job["id"])) or job

    executor_name = str(job.get("executor") or "modal")
    poll = (
        local_executor.poll(str(job["remote_call_id"]))
        if executor_name == "local"
        else modal_executor.poll(str(job["remote_call_id"]))
    )
    if poll.state in {"running", "unavailable"}:
        fields: dict[str, Any] = {"last_remote_check_at": utc_now()}
        if poll.state == "unavailable" and poll.error:
            fields["message"] = (
                "CellExLink and Railway PubTator3 are still active; the latest "
                "CellExLink executor status could not be refreshed yet."
            )
        return update_annotation_job(str(job["id"]), **fields) or job

    if poll.state == "completed":
        try:
            updated = _apply_cell_result(job, poll.result or {})
            railway_annotation_executor.ensure_job(str(job["id"]))
            return _recover_published_job(updated) or updated
        except Exception as exc:
            logger.exception("Could not verify CellExLink branch %s", job["id"])
            try:
                recovered = _recover_published_job(job)
                if recovered is not None:
                    return recovered
            except Exception:
                logger.warning(
                    "Fallback final-artifact recovery failed for annotation %s",
                    job["id"],
                    exc_info=True,
                )
            return update_annotation_job(
                str(job["id"]),
                status="failed",
                stage="failed",
                progress=100,
                message=(
                    "CellExLink finished, but its branch artifact could not be verified."
                ),
                error=str(exc),
                completed_at=utc_now(),
                last_remote_check_at=utc_now(),
            ) or job

    # A completion callback or Modal return value can be lost. If the stable
    # cell branch exists, continue on Railway instead of failing the whole job.
    try:
        _verify_cell_branch_artifacts(job)
        updated = _apply_cell_result(
            job,
            {
                "status": "cell_completed",
                "message": (
                    "Recovered the published CellExLink branch; Railway is "
                    "continuing PubTator3 and the final merge."
                ),
            },
        )
        railway_annotation_executor.ensure_job(str(job["id"]))
        return _recover_published_job(updated) or updated
    except Exception:
        logger.warning(
            "Cell-branch artifact recovery failed for annotation %s",
            job["id"],
            exc_info=True,
        )

    try:
        recovered = _recover_published_job(job)
        if recovered is not None:
            return recovered
    except Exception:
        logger.warning(
            "Terminal final-artifact recovery failed for annotation %s",
            job["id"],
            exc_info=True,
        )
    return update_annotation_job(
        str(job["id"]),
        status="failed",
        stage="failed",
        progress=100,
        message="The CellExLink branch could not be completed.",
        error=poll.error or "CellExLink executor failed.",
        completed_at=utc_now(),
        last_remote_check_at=utc_now(),
    ) or job


@router.get("/status")
def annotation_pipeline_status() -> dict[str, Any]:
    backend = settings.cell_annotation_backend
    local_dependencies = local_ml_dependencies_available() if backend == "local" else None
    configured = (
        settings.local_annotation_configured and bool(local_dependencies)
        if backend == "local"
        else settings.modal_configured and settings.artifact_backend == "s3"
        if backend == "modal"
        else False
    )
    cell_compute = (
        "local CPU"
        if backend == "local"
        else "Modal T4"
        if backend == "modal"
        else "disabled"
    )
    return {
        "status": "connected" if configured else "configuration_required",
        "executor": backend,
        "compute": cell_compute,
        "cell_compute": cell_compute,
        "pubtator3_compute": "Railway CPU",
        "final_merge_compute": "Railway CPU",
        "parallel_branches": True,
        "gpu": "T4" if backend == "modal" else None,
        "input": "one_compressed_retrieval_artifact",
        "output": "one_compressed_entity_annotation_artifact",
        "models_loaded_sequentially": True,
        "persistent_checkpoint_cache": True,
        "max_active_annotation_jobs": 1,
        "max_active_gpu_jobs": 1 if backend == "modal" else 0,
        "recognition_model": settings.cell_ner_model,
        "normalization_model": settings.cell_nen_model,
        "artifact_backend": settings.artifact_backend,
        "abbreviation_context": not settings.cell_disable_abbreviations,
        "pubtator3_gene_hormone_augmentation": True,
        "pubtator3_required": settings.pubtator3_required,
        "preferred_label_resolution": settings.pubtator3_resolve_preferred_labels,
        "local_ml_dependencies_installed": local_dependencies,
        "model_cache_directory": (
            str(settings.cell_model_cache_dir) if backend == "local" else None
        ),
    }


@router.post("", status_code=202)
def submit_annotation(payload: AnnotationCreate, request: Request) -> dict[str, Any]:
    source_job = get_job(payload.source_job_id)
    if source_job is None:
        raise HTTPException(status_code=404, detail="Retrieval job not found.")
    if source_job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="Entity extraction requires a completed retrieval.",
        )

    source_summary = _load_source_summary(source_job)
    try:
        source_artifact = source_artifact_from_summary(source_summary)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    store = get_artifact_store()
    if store.head(source_artifact.key) is None:
        raise HTTPException(
            status_code=409,
            detail="The published retrieval artifact is missing. Run Stage 1 again.",
        )

    model_signature = annotation_model_signature()
    artifact_keys = annotation_artifact_keys(
        source_sha256=source_artifact.sha256,
        model_signature=model_signature,
    )
    output_key = artifact_keys.final_annotations
    summary_key = artifact_keys.final_summary

    reusable = find_reusable_annotation(
        source_artifact_sha256=source_artifact.sha256,
        model_signature=model_signature,
    )
    if reusable is not None:
        try:
            recovered = _recover_published_job(reusable)
            if recovered is not None:
                return _public_job(recovered, reused=True)
        except Exception:
            logger.warning(
                "Stored annotation job %s could not be reused",
                reusable["id"],
                exc_info=True,
            )

    active = find_active_annotation(
        source_artifact_sha256=source_artifact.sha256,
        model_signature=model_signature,
    )
    if active is not None:
        active = _reconcile_if_stale(active)
        if active.get("status") in {"queued", "processing", "completed"}:
            return _public_job(active, reused=True)

    # The final object pair is the durable cache. It restores reuse even if
    # SQLite was recreated or an earlier completion callback was lost.
    job_id = uuid.uuid4().hex[:12]
    bucket_candidate = {
        "id": job_id,
        "source_job_id": payload.source_job_id,
        "executor": settings.cell_annotation_backend,
        "model_signature": model_signature,
        "source_artifact_key": source_artifact.key,
        "source_artifact_sha256": source_artifact.sha256,
        "output_artifact_key": output_key,
        "summary_artifact_key": summary_key,
    }
    try:
        published_result = _published_final_result(bucket_candidate)
    except Exception:
        logger.warning("Existing bucket artifacts were not reusable", exc_info=True)
        published_result = None
    if published_result is not None:
        cached_job = create_annotation_job(
            job_id=job_id,
            source_job_id=payload.source_job_id,
            executor=settings.cell_annotation_backend,
            model_signature=model_signature,
            source_artifact_key=source_artifact.key,
            source_artifact_sha256=source_artifact.sha256,
            output_artifact_key=output_key,
            summary_artifact_key=summary_key,
            reused_from_job_id="bucket-artifact",
        )
        return _public_job(
            _apply_final_result(cached_job, published_result),
            reused=True,
        )

    # Strictly allow one active Stage 2 job. This caps local memory and avoids
    # accidental queues of billable Modal GPU work.
    other_active = find_any_active_annotation()
    if other_active is not None:
        other_active = _reconcile_if_stale(other_active)
        if other_active.get("status") in {"queued", "processing"}:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Another entity-extraction job is already running "
                    f"({other_active['id']}). Wait for it to finish before "
                    "starting a different annotation job."
                ),
            )

    backend = settings.cell_annotation_backend
    if backend == "disabled":
        raise HTTPException(status_code=503, detail="Entity extraction is disabled.")
    if backend == "local":
        if settings.artifact_backend != "local":
            raise HTTPException(
                status_code=503,
                detail="Local annotation requires ARTIFACT_BACKEND=local.",
            )
        if not local_ml_dependencies_available():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Local CellExLink dependencies are missing. Install PyTorch "
                    "and requirements-local.txt, then restart Uvicorn."
                ),
            )
        callback_base_url = str(request.base_url).rstrip("/")
    elif backend == "modal":
        if settings.artifact_backend != "s3":
            raise HTTPException(
                status_code=503,
                detail=(
                    "Modal requires the Railway S3-compatible object bucket. "
                    "Set ARTIFACT_BACKEND=s3 and attach a Railway Bucket."
                ),
            )
        if not settings.modal_configured:
            raise HTTPException(
                status_code=503,
                detail="Modal credentials or the deployed Modal function are not configured.",
            )
        if not settings.public_base_url:
            raise HTTPException(
                status_code=503,
                detail="PUBLIC_BASE_URL is required so Modal can report progress.",
            )
        callback_base_url = settings.public_base_url
    else:
        raise HTTPException(status_code=503, detail="Unknown annotation backend.")

    callback_token = secrets.token_urlsafe(32)
    job = create_annotation_job(
        job_id=job_id,
        source_job_id=payload.source_job_id,
        executor=backend,
        model_signature=model_signature,
        source_artifact_key=source_artifact.key,
        source_artifact_sha256=source_artifact.sha256,
        output_artifact_key=output_key,
        summary_artifact_key=summary_key,
        callback_token_hash=callback_token_hash(callback_token),
    )

    cell_payload = {
        "job_id": job_id,
        "pipeline_version": ANNOTATION_PIPELINE_VERSION,
        "model_signature": model_signature,
        "input": {
            "url": store.presign_get(
                source_artifact.key,
                expires_seconds=settings.artifact_presigned_ttl_seconds,
            ),
            "key": source_artifact.key,
            "sha256": source_artifact.sha256,
            "size_bytes": source_artifact.size_bytes,
        },
        "output": {
            "cell_annotations_url": store.presign_put(
                artifact_keys.cell_annotations,
                content_type="application/gzip",
                content_encoding=None,
            ),
            "cell_annotations_key": artifact_keys.cell_annotations,
            "cell_summary_url": store.presign_put(
                artifact_keys.cell_summary,
                content_type="application/json",
            ),
            "cell_summary_key": artifact_keys.cell_summary,
        },
        "callback": {
            "url": f"{callback_base_url}/api/annotations/{job_id}/callback",
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
            "paper_count": int(source_job.get("paper_count") or 0),
            "chunk_count": int(
                (source_summary.get("stats") or {}).get("chunk_count") or 0
            ),
        },
    }

    try:
        remote_call_id = (
            local_executor.submit(cell_payload)
            if backend == "local"
            else modal_executor.submit(cell_payload)
        )
    except Exception as exc:
        logger.exception("Could not submit CellExLink %s to %s", job_id, backend)
        update_annotation_job(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            message=f"The CellExLink branch could not be submitted to {backend}.",
            error=str(exc),
            completed_at=utc_now(),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Could not submit the {backend} CellExLink job: {exc}",
        ) from exc

    initial_stats = {
        "cell_branch_status": "submitted",
        "pubtator_branch_status": "submitted",
        "cell_branch_runtime": "local CPU" if backend == "local" else "Modal T4",
        "pubtator_branch_runtime": "Railway CPU",
        "execution_layout": "modal-cellexlink-plus-railway-pubtator3",
    }
    job = update_annotation_job(
        job_id,
        remote_call_id=remote_call_id,
        status="processing",
        stage="running_parallel_branches",
        progress=2,
        message=(
            "CellExLink is starting locally while PubTator3 starts on Railway CPU."
            if backend == "local"
            else "CellExLink is starting on Modal T4 while PubTator3 starts on Railway CPU."
        ),
        stats=initial_stats,
        started_at=utc_now(),
        last_remote_check_at=utc_now(),
    ) or job

    try:
        railway_annotation_executor.submit_pubtator(job_id)
    except Exception as exc:
        logger.exception("Could not start Railway PubTator3 branch %s", job_id)
        update_annotation_job(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            message="The Railway PubTator3 branch could not be started.",
            stats={**initial_stats, "pubtator_branch_status": "failed"},
            error=str(exc),
            completed_at=utc_now(),
            last_remote_check_at=utc_now(),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Could not start Railway PubTator3: {exc}",
        ) from exc

    return _public_job(get_annotation_job(job_id) or job)


@router.post("/{job_id}/callback")
def annotation_callback(
    job_id: str,
    payload: WorkerCallback,
    x_annotation_token: str | None = Header(default=None, alias="X-Annotation-Token"),
) -> dict[str, str]:
    job = get_annotation_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Annotation job not found.")
    if not callback_token_matches(
        x_annotation_token or "", job.get("callback_token_hash")
    ):
        raise HTTPException(status_code=401, detail="Invalid callback token.")

    # A delayed Modal callback must never overwrite a final Railway completion
    # or a terminal PubTator3 failure.
    if job.get("status") in {"completed", "failed"}:
        return {"status": "accepted"}

    # Merge only the incoming CellExLink fields into a freshly loaded record.
    # This prevents a callback from overwriting PubTator3 progress written by a
    # Railway thread at nearly the same time.
    incoming_stats = dict(payload.stats)
    if payload.output_sha256:
        incoming_stats["cell_output_sha256"] = payload.output_sha256
    if payload.summary_sha256:
        incoming_stats["cell_summary_sha256"] = payload.summary_sha256
    normalized_status = payload.status.casefold()

    if normalized_status in {"cell_completed", "completed"}:
        try:
            _apply_cell_result(
                job,
                {
                    "status": "cell_completed",
                    "message": payload.message,
                    "stats": incoming_stats,
                },
            )
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        railway_annotation_executor.ensure_job(job_id)
    elif normalized_status == "failed":
        current = get_annotation_job(job_id) or job
        stats = dict(current.get("stats") or {})
        stats.update(incoming_stats)
        stats["cell_branch_status"] = "failed"
        counts = _counts_from_stats(stats)
        update_annotation_job(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            message=payload.message or "CellExLink extraction failed.",
            stats=stats,
            elapsed_seconds=_elapsed_from_stats(
                stats, current.get("elapsed_seconds") or 0
            ),
            completed_at=utc_now(),
            last_remote_check_at=utc_now(),
            error=payload.error or "CellExLink worker failed.",
            **counts,
        )
    else:
        current = get_annotation_job(job_id) or job
        stats = dict(current.get("stats") or {})
        stats.update(incoming_stats)
        counts = _counts_from_stats(stats)
        update_annotation_job(
            job_id,
            status="processing",
            stage=payload.stage,
            progress=max(int(current.get("progress") or 0), payload.progress),
            message=payload.message,
            stats=stats,
            elapsed_seconds=_elapsed_from_stats(
                stats, current.get("elapsed_seconds") or 0
            ),
            last_remote_check_at=utc_now(),
            **counts,
        )
        railway_annotation_executor.ensure_job(job_id)
    return {"status": "accepted"}


@router.get("")
def recent_annotation_jobs(
    limit: int = Query(default=8, ge=1, le=50),
) -> dict[str, Any]:
    jobs = [_reconcile_if_stale(job) for job in list_annotation_jobs(limit)]
    return {"jobs": [_public_job(job) for job in jobs]}


@router.get("/{job_id}")
def annotation_job_status(job_id: str) -> dict[str, Any]:
    job = get_annotation_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Annotation job not found.")
    return _public_job(_reconcile_if_stale(job))


@router.get("/{job_id}/summary")
def annotation_summary(job_id: str) -> dict[str, Any]:
    job = get_annotation_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Annotation job not found.")
    job = _reconcile_if_stale(job)
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="The annotation summary is available after Stage 2 completes.",
        )
    summary_key = str(job.get("summary_artifact_key") or "")
    if not summary_key:
        raise HTTPException(status_code=404, detail="Summary artifact was not recorded.")
    try:
        summary = get_artifact_store().read_json(summary_key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    summary["job"] = _public_job(job)
    summary["downloads"] = {
        "annotations": f"/api/annotations/{job_id}/download"
    }
    return summary


@router.get("/{job_id}/download")
def download_annotations(job_id: str):
    job = get_annotation_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Annotation job not found.")
    job = _reconcile_if_stale(job)
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="Entity annotations are available after Stage 2 completes.",
        )
    key = str(job.get("output_artifact_key") or "")
    store = get_artifact_store()
    if not key or store.head(key) is None:
        raise HTTPException(status_code=404, detail="Annotation artifact was not found.")
    filename = f"{job_id}-entity-annotations.jsonl.gz"
    local_path = store.local_path(key)
    if local_path is not None:
        return FileResponse(
            local_path,
            media_type="application/gzip",
            filename=filename,
        )
    return RedirectResponse(
        store.presign_get(key, download_name=filename),
        status_code=307,
    )
