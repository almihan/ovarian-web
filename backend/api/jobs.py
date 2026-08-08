"""Analysis job endpoints and the first demo processing workflow."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from pydantic import BaseModel, Field, SecretStr, field_validator

from backend.config import settings
from backend.database.database import (
    create_job,
    get_job,
    list_jobs,
    result_file_for,
    update_job,
)
from backend.pipeline.graph_builder import build_demo_network

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobCreate(BaseModel):
    input_type: Literal["keywords", "pmid", "pmcid"] = "keywords"
    query: str = Field(min_length=1, max_length=4000)
    relation_extraction: bool = True
    api_key: SecretStr | None = None

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Enter keywords, PMID values, or PMCID values.")
        return cleaned


class JobCreated(BaseModel):
    id: str
    status: str
    progress: int
    message: str | None = None


def _write_network(job_id: str, payload: dict) -> Path:
    path = result_file_for(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def process_demo_job(
    job_id: str,
    query: str,
    relation_extraction: bool,
    user_api_key: str | None,
) -> None:
    """Make the first UI functional before the real pipeline is connected.

    Replace each timed stage with calls into ``backend.pipeline``. The API key
    remains only in this function's memory and is never written to SQLite.
    """

    active_api_key = user_api_key or settings.openai_api_key
    try:
        stages = [
            ("retrieving", 12, "Searching and downloading related papers...", 0.9),
            ("retrieving", 28, "Parsing titles, abstracts, and full text...", 0.8),
            ("entities", 48, "Extracting cell types, genes, and chemicals...", 1.2),
            ("entities", 64, "Normalizing entities and combining mentions...", 0.8),
        ]
        if relation_extraction:
            relation_message = (
                "Extracting relations with the configured OpenAI API key..."
                if active_api_key
                else "Previewing relation extraction with demonstration data..."
            )
            stages.append(("relations", 80, relation_message, 1.1))
        stages.extend(
            [
                ("network", 92, "Constructing the interaction network...", 0.8),
                ("network", 98, "Preparing the interactive visualization...", 0.5),
            ]
        )

        update_job(
            job_id,
            status="processing",
            stage="starting",
            progress=4,
            message="Starting analysis...",
        )

        for stage, progress, message, delay in stages:
            update_job(
                job_id,
                status="processing",
                stage=stage,
                progress=progress,
                message=message,
            )
            time.sleep(delay)

        network = build_demo_network(query)
        result_path = _write_network(job_id, network)
        summary = network["summary"]
        update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message="Your interactive network is ready.",
            paper_count=summary["papers"],
            entity_count=summary["entities"],
            relation_count=summary["relations"],
            result_path=str(result_path),
            error=None,
        )
    except Exception as exc:  # pragma: no cover - defensive background handling
        logger.exception("Job %s failed", job_id)
        update_job(
            job_id,
            status="failed",
            stage="failed",
            message="The analysis could not be completed.",
            error=str(exc),
        )
    finally:
        # Do not persist or log user-provided credentials.
        active_api_key = None
        user_api_key = None


@router.post("", response_model=JobCreated, status_code=202)
def submit_job(
    payload: JobCreate,
    background_tasks: BackgroundTasks,
    request: Request,
) -> JobCreated:
    job_id = uuid.uuid4().hex[:12]
    job = create_job(
        job_id=job_id,
        input_type=payload.input_type,
        query=payload.query,
        relation_extraction=payload.relation_extraction,
    )

    # SecretStr keeps the value redacted in model representations and errors.
    api_key = payload.api_key.get_secret_value() if payload.api_key else None
    background_tasks.add_task(
        process_demo_job,
        job_id,
        payload.query,
        payload.relation_extraction,
        api_key,
    )

    return JobCreated(
        id=job["id"],
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
    )


@router.get("")
def recent_jobs(limit: int = Query(default=8, ge=1, le=50)) -> dict:
    return {"jobs": list_jobs(limit)}


@router.get("/{job_id}")
def job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
