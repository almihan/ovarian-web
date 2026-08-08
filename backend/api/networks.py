"""Network-result endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.database.database import get_job

router = APIRouter(prefix="/api/networks", tags=["networks"])


@router.get("/{job_id}")
def get_network(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "completed" or not job.get("result_path"):
        raise HTTPException(status_code=409, detail="Network is not ready.")

    path = Path(job["result_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Network result file is missing.")
    return json.loads(path.read_text(encoding="utf-8"))
