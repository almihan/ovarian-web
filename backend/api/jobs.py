"""Paper-retrieval job endpoints."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.config import settings
from backend.database.database import (
    create_job,
    get_job,
    list_jobs,
    update_job,
    utc_now,
)
from backend.pipeline.locks import PIPELINE_LOCK
from backend.pipeline.retrieval import (
    RetrievalError,
    build_effective_inputs,
    run_paper_retrieval,
)
from backend.storage.bundles import publish_retrieval_bundle

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# Retrieval alone writes the shared per-paper cache. GPU annotation runs on Modal
# and therefore does not occupy the Railway retrieval lock.
_RETRIEVAL_LOCK = PIPELINE_LOCK


class JobCreate(BaseModel):
    input_type: Literal["keywords", "pmid", "pmcid"] = "keywords"
    query: str = Field(default="", max_length=4000)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_optional_additions(self) -> "JobCreate":
        try:
            build_effective_inputs(self.input_type, self.query)
        except RetrievalError as exc:
            raise ValueError(str(exc)) from exc
        return self


class JobCreated(BaseModel):
    id: str
    status: str
    stage: str
    progress: int
    message: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


def process_retrieval_job(job_id: str, input_type: str, query: str) -> None:
    """Run one retrieval and continuously persist browser-facing progress."""

    acquired = _RETRIEVAL_LOCK.acquire(blocking=False)
    if not acquired:
        update_job(
            job_id,
            status="queued",
            stage="queued",
            progress=1,
            message=(
                "Another retrieval is using the shared paper cache. "
                "This job will start automatically when it finishes."
            ),
        )
        _RETRIEVAL_LOCK.acquire()

    started_monotonic = time.monotonic()
    started_at = utc_now()

    try:
        update_job(
            job_id,
            status="processing",
            stage="preparing",
            progress=2,
            message="Preparing the built-in corpus and your additions...",
            started_at=started_at,
            completed_at=None,
            error=None,
        )

        def report_progress(
            stage: str,
            progress: int,
            message: str,
            stats: dict[str, Any],
        ) -> None:
            live_stats = dict(stats)
            elapsed = round(time.monotonic() - started_monotonic, 2)
            live_stats["elapsed_seconds"] = elapsed
            update_job(
                job_id,
                # The pipeline writes summary.json before emitting its final event,
                # but the database result_path is recorded only after the function
                # returns. Keep the public status processing until that final update.
                status="processing",
                stage=stage,
                progress=progress,
                message=message,
                paper_count=int(live_stats.get("paper_count") or 0),
                stats=live_stats,
                elapsed_seconds=elapsed,
            )

        result = run_paper_retrieval(
            job_id=job_id,
            input_type=input_type,  # type: ignore[arg-type]
            user_input=query,
            papers_root=settings.papers_dir,
            ncbi_email=settings.ncbi_email,
            ncbi_tool=settings.ncbi_tool,
            ncbi_api_key=settings.ncbi_api_key,
            keyword_limit=settings.retrieval_keyword_limit,
            batch_size=settings.retrieval_batch_size,
            request_timeout=settings.retrieval_request_timeout,
            progress_callback=report_progress,
        )

        report_progress(
            "publishing",
            99,
            "Publishing one compressed reusable chunk artifact to object storage...",
            dict(result.stats),
        )
        artifact, artifact_reused, artifact_records = publish_retrieval_bundle(
            summary_path=result.summary_path,
            chunk_paths=result.chunk_paths,
        )

        elapsed = round(time.monotonic() - started_monotonic, 2)
        final_stats = dict(result.stats)
        final_stats.update(
            {
                "elapsed_seconds": elapsed,
                "artifact_key": artifact.key,
                "artifact_sha256": artifact.sha256,
                "artifact_bytes": artifact.size_bytes,
                "artifact_records": artifact_records,
                "artifact_reused": artifact_reused,
            }
        )
        completion_message = (
            f"Finished: {final_stats.get('paper_count', 0)} papers. "
            f"{final_stats.get('fulltexts_downloaded', 0)} full texts downloaded; "
            f"{final_stats.get('papers_without_pmcid', 0)} papers without a PMCID."
        )
        pending_retry = int(final_stats.get("fulltext_pending_retry") or 0)
        if pending_retry:
            completion_message += (
                f" {pending_retry} temporary full-text failures remain eligible "
                "for a later retry."
            )
        update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message=completion_message,
            paper_count=int(final_stats.get("paper_count") or 0),
            stats=final_stats,
            elapsed_seconds=elapsed,
            result_path=str(result.summary_path),
            completed_at=utc_now(),
            error=None,
        )
    except Exception as exc:  # pragma: no cover - defensive background handling
        logger.exception("Paper retrieval job %s failed", job_id)
        elapsed = round(time.monotonic() - started_monotonic, 2)
        update_job(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            message="The paper retrieval could not be completed.",
            elapsed_seconds=elapsed,
            completed_at=utc_now(),
            error=str(exc),
        )
    finally:
        _RETRIEVAL_LOCK.release()


@router.post("", response_model=JobCreated, status_code=202)
def submit_job(
    payload: JobCreate,
    background_tasks: BackgroundTasks,
) -> JobCreated:
    job_id = uuid.uuid4().hex[:12]
    job = create_job(
        job_id=job_id,
        input_type=payload.input_type,
        query=payload.query,
        relation_extraction=False,
    )
    background_tasks.add_task(
        process_retrieval_job,
        job_id,
        payload.input_type,
        payload.query,
    )

    return JobCreated(
        id=job["id"],
        status=job["status"],
        stage=job["stage"],
        progress=job["progress"],
        message=job.get("message"),
        stats=job.get("stats") or {},
    )


@router.get("")
def recent_jobs(limit: int = Query(default=8, ge=1, le=50)) -> dict[str, Any]:
    return {"jobs": list_jobs(limit)}


@router.get("/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
