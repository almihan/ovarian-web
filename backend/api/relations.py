"""Stage 3 API for cached, resumable biological relation extraction."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Mapping

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.database.database import (
    create_relation_job,
    find_active_relation,
    find_any_active_relation,
    find_reusable_relation,
    get_annotation_job,
    get_job,
    get_relation_job,
    list_relation_jobs,
    update_relation_job,
    utc_now,
)
from backend.pipeline.relation_contract import (
    relation_artifact_keys,
    relation_model_signature,
)
from backend.services.relation_executor import relation_executor
from backend.storage.artifacts import get_artifact_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/relations", tags=["relations"])


class RelationCreate(BaseModel):
    source_annotation_job_id: str = Field(min_length=1, max_length=64)

    @field_validator("source_annotation_job_id")
    @classmethod
    def clean_source_annotation_job_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(
            not (character.isalnum() or character in "-_")
            for character in cleaned
        ):
            raise ValueError("source_annotation_job_id contains unsupported characters")
        return cleaned


def _source_payload(annotation_job_id: str) -> dict[str, Any] | None:
    annotation = get_annotation_job(annotation_job_id)
    if annotation is None:
        return None
    retrieval = get_job(str(annotation.get("source_job_id") or ""))
    return {
        "id": annotation["id"],
        "status": annotation["status"],
        "source_job_id": annotation.get("source_job_id"),
        "chunk_count": int(annotation.get("chunk_count") or 0),
        "mention_count": int(annotation.get("mention_count") or 0),
        "created_at": annotation.get("created_at"),
        "completed_at": annotation.get("completed_at"),
        "retrieval": (
            {
                "id": retrieval["id"],
                "input_type": retrieval.get("input_type"),
                "query": retrieval.get("query") or "",
                "paper_count": int(retrieval.get("paper_count") or 0),
            }
            if retrieval is not None
            else None
        ),
    }


def _public_job(job: Mapping[str, Any], *, reused: bool = False) -> dict[str, Any]:
    payload = dict(job)
    # Retained database columns from older deployments are intentionally hidden.
    payload.pop("remote_batch_id", None)
    payload.pop("remote_input_file_id", None)
    payload.pop("batch_count", None)
    payload["source_annotation_job"] = _source_payload(
        str(job["source_annotation_job_id"])
    )
    payload["reused"] = reused or bool(job.get("reused_from_job_id"))
    if job.get("status") == "completed":
        payload["downloads"] = {
            "relations": f"/api/relations/{job['id']}/download"
        }
    return payload


def _artifacts_exist(job: Mapping[str, Any]) -> bool:
    store = get_artifact_store()
    return bool(
        job.get("output_artifact_key")
        and job.get("summary_artifact_key")
        and store.head(str(job["output_artifact_key"])) is not None
        and store.head(str(job["summary_artifact_key"])) is not None
    )


def _recover_completed(job: Mapping[str, Any]) -> dict[str, Any] | None:
    if not _artifacts_exist(job):
        return None
    summary = get_artifact_store().read_json(str(job["summary_artifact_key"]))
    stats = summary.get("stats")
    stats = dict(stats) if isinstance(stats, Mapping) else {}
    return update_relation_job(
        str(job["id"]),
        status="completed",
        stage="completed",
        progress=100,
        message=str(
            summary.get("message")
            or f"Finished: {int(stats.get('relation_count') or 0):,} validated relations."
        ),
        paper_count=int(stats.get("paper_count") or job.get("paper_count") or 0),
        chunk_count=int(stats.get("chunk_count") or job.get("chunk_count") or 0),
        eligible_chunk_count=int(
            stats.get("eligible_chunk_count")
            or job.get("eligible_chunk_count")
            or 0
        ),
        processed_chunk_count=int(
            stats.get("processed_chunk_count")
            or stats.get("chunk_count")
            or job.get("processed_chunk_count")
            or 0
        ),
        relation_count=int(
            stats.get("relation_count") or job.get("relation_count") or 0
        ),
        cell_context_count=int(
            stats.get("cell_context_count")
            or job.get("cell_context_count")
            or 0
        ),
        api_request_count=int(
            stats.get("api_request_count") or job.get("api_request_count") or 0
        ),
        stats=stats,
        elapsed_seconds=float(stats.get("elapsed_seconds") or 0.0),
        completed_at=str(summary.get("completed_at") or utc_now()),
        error=None,
    )


@router.get("/status")
def relation_pipeline_status() -> dict[str, Any]:
    return {
        "status": "connected" if settings.relation_configured else "disconnected",
        "executor": "openai_responses_async",
        "compute": f"OpenAI Responses API · {settings.relation_model}",
        "model": settings.relation_model,
        "window_size": settings.relation_window_size,
        "concurrency": settings.relation_concurrency,
        "request_timeout_seconds": settings.relation_request_timeout_seconds,
        "max_request_retries": settings.relation_max_request_retries,
        "prompt_caching": "best-effort repeated-prefix caching",
        "prompt_cache_shards": settings.relation_prompt_cache_shards,
        "hormone_gene_cell_context_required": (
            settings.relation_require_hormone_gene_cell_context
        ),
        "message": (
            "Stage 3 is ready."
            if settings.relation_configured
            else "Set OPENAI_API_KEY to enable Stage 3."
        ),
    }


@router.post("", status_code=202)
def submit_relation(payload: RelationCreate) -> dict[str, Any]:
    if not settings.relation_configured:
        raise HTTPException(
            status_code=503,
            detail="Relation extraction is disabled until OPENAI_API_KEY is set.",
        )

    annotation = get_annotation_job(payload.source_annotation_job_id)
    if annotation is None:
        raise HTTPException(status_code=404, detail="Entity-extraction job not found.")
    if annotation.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="Stage 2 must complete before relation extraction starts.",
        )

    store = get_artifact_store()
    chunks_key = str(annotation.get("source_artifact_key") or "")
    chunks_sha = str(annotation.get("source_artifact_sha256") or "")
    annotations_key = str(annotation.get("output_artifact_key") or "")
    chunks_ref = store.head(chunks_key) if chunks_key else None
    annotations_ref = store.head(annotations_key) if annotations_key else None
    if chunks_ref is None or not chunks_sha:
        raise HTTPException(
            status_code=409,
            detail="The Stage 1 chunk artifact is missing. Run retrieval again.",
        )
    if chunks_ref.sha256 != chunks_sha:
        raise HTTPException(
            status_code=409,
            detail="The Stage 1 artifact fingerprint no longer matches Stage 2.",
        )
    if annotations_ref is None:
        raise HTTPException(
            status_code=409,
            detail="The Stage 2 annotation artifact is missing. Run Stage 2 again.",
        )

    model_signature = relation_model_signature()
    keys = relation_artifact_keys(
        source_annotation_sha256=annotations_ref.sha256,
        source_chunks_sha256=chunks_sha,
        model_signature=model_signature,
    )

    reusable = find_reusable_relation(
        source_annotation_artifact_sha256=annotations_ref.sha256,
        source_chunks_artifact_sha256=chunks_sha,
        model_signature=model_signature,
    )
    if reusable is not None:
        recovered = _recover_completed(reusable)
        if recovered is not None:
            return _public_job(recovered, reused=True)

    active = find_active_relation(
        source_annotation_artifact_sha256=annotations_ref.sha256,
        source_chunks_artifact_sha256=chunks_sha,
        model_signature=model_signature,
    )
    if active is not None:
        relation_executor.submit(str(active["id"]))
        return _public_job(active, reused=True)

    other_active = find_any_active_relation()
    if other_active is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Another relation-extraction job is already active "
                f"({other_active['id']}). This one-job policy prevents an "
                "accidental queue of billable OpenAI requests."
            ),
        )

    job_id = uuid.uuid4().hex[:12]
    job = create_relation_job(
        job_id=job_id,
        source_annotation_job_id=payload.source_annotation_job_id,
        model_signature=model_signature,
        source_chunks_artifact_key=chunks_key,
        source_chunks_artifact_sha256=chunks_sha,
        source_annotation_artifact_key=annotations_key,
        source_annotation_artifact_sha256=annotations_ref.sha256,
        output_artifact_key=keys.relations,
        summary_artifact_key=keys.summary,
    )
    job = update_relation_job(
        job_id,
        paper_count=int(annotation.get("paper_count") or 0),
        chunk_count=int(annotation.get("chunk_count") or 0),
        stats={
            "model": settings.relation_model,
            "execution_mode": "online_async_responses",
            "window_size": settings.relation_window_size,
            "concurrency": settings.relation_concurrency,
            "request_timeout_seconds": settings.relation_request_timeout_seconds,
            "max_request_retries": settings.relation_max_request_retries,
            "prompt_cache_shards": settings.relation_prompt_cache_shards,
            "cell_context_required": (
                settings.relation_require_hormone_gene_cell_context
            ),
        },
    ) or job

    # Deterministic object keys allow recovery even if SQLite history was reset.
    recovered = _recover_completed(job)
    if recovered is not None:
        return _public_job(recovered, reused=True)

    try:
        relation_executor.submit(job_id)
    except Exception as exc:
        logger.exception("Could not start relation job %s", job_id)
        update_relation_job(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            message="The relation worker could not be started.",
            error=str(exc),
            completed_at=utc_now(),
        )
        raise HTTPException(
            status_code=502,
            detail=f"Could not start relation extraction: {exc}",
        ) from exc
    return _public_job(get_relation_job(job_id) or job)


@router.get("")
def recent_relation_jobs(
    limit: int = Query(default=8, ge=1, le=50),
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for job in list_relation_jobs(limit):
        if (
            settings.relation_configured
            and job.get("status") in {"queued", "processing"}
        ):
            relation_executor.submit(str(job["id"]))
        jobs.append(_public_job(job))
    return {"jobs": jobs}


@router.get("/{job_id}")
def relation_job_status(job_id: str) -> dict[str, Any]:
    job = get_relation_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Relation job not found.")
    if (
        settings.relation_configured
        and job.get("status") in {"queued", "processing"}
    ):
        relation_executor.submit(job_id)
        job = get_relation_job(job_id) or job
    return _public_job(job)


@router.get("/{job_id}/summary")
def relation_summary(job_id: str) -> dict[str, Any]:
    job = get_relation_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Relation job not found.")
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="The relation summary is available after Stage 3 completes.",
        )
    key = str(job.get("summary_artifact_key") or "")
    if not key:
        raise HTTPException(status_code=404, detail="Summary artifact was not recorded.")
    try:
        summary = get_artifact_store().read_json(key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    summary["job"] = _public_job(job)
    summary["downloads"] = {"relations": f"/api/relations/{job_id}/download"}
    return summary


@router.get("/{job_id}/download")
def download_relations(job_id: str):
    job = get_relation_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Relation job not found.")
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="Relations are available after Stage 3 completes.",
        )
    key = str(job.get("output_artifact_key") or "")
    store = get_artifact_store()
    if not key or store.head(key) is None:
        raise HTTPException(status_code=404, detail="Relation artifact was not found.")
    filename = f"{job_id}-relations.jsonl.gz"
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
