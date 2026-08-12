"""Paper-retrieval status, summaries, and generated-file downloads."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse

from backend.config import settings
from backend.database.database import get_job
from backend.pipeline.retrieval import defaults_payload
from backend.storage.artifacts import ArtifactRef, get_artifact_store

router = APIRouter(prefix="/api/papers", tags=["papers"])


@router.get("/status")
def paper_pipeline_status() -> dict[str, Any]:
    return {
        "status": "connected",
        "message": (
            "PubMed retrieval, resilient multi-source PMC full-text downloads, "
            "compressed storage, and paper-ID-based chunk reuse are connected."
        ),
        "shared_cache": True,
        "incremental_chunks": True,
        "transient_fulltext_failures_are_retryable": True,
    }


@router.get("/defaults")
def paper_pipeline_defaults() -> dict[str, Any]:
    return defaults_payload(settings.retrieval_keyword_limit)


def _load_summary(job_id: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="The retrieval summary is available after the job completes.",
        )

    raw_path = job.get("result_path")
    if not raw_path:
        raise HTTPException(status_code=404, detail="Summary file was not recorded.")

    summary_path = Path(str(raw_path)).expanduser().resolve()
    papers_root = settings.papers_dir.expanduser().resolve()
    if not summary_path.is_relative_to(papers_root):
        raise HTTPException(status_code=400, detail="Invalid summary path.")
    if not summary_path.is_file():
        raise HTTPException(status_code=404, detail="Summary file was not found.")

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Summary file is unreadable.") from exc
    if not isinstance(summary, dict):
        raise HTTPException(status_code=500, detail="Summary file has an invalid format.")
    return job, summary, summary_path


@router.get("/jobs/{job_id}")
def paper_job_summary(job_id: str) -> dict[str, Any]:
    job, summary, _ = _load_summary(job_id)
    payload = dict(summary)
    payload["job"] = job
    payload["downloads"] = {
        "chunks": f"/api/papers/jobs/{job_id}/download/chunks",
    }
    return payload


def _published_artifact(summary: dict[str, Any]) -> ArtifactRef | None:
    files = summary.get("files") or {}
    if not isinstance(files, dict):
        return None
    raw = files.get("artifact")
    if not isinstance(raw, dict):
        return None
    try:
        return ArtifactRef.from_dict(raw)
    except (KeyError, TypeError, ValueError):
        raise HTTPException(
            status_code=500, detail="The published chunk artifact metadata is invalid."
        )


def _legacy_job_chunks_file(
    job_id: str,
    summary: dict[str, Any],
    summary_path: Path,
) -> Path:
    files = summary.get("files") or {}
    relative_value = files.get("chunks")
    if not isinstance(relative_value, str) or not relative_value.strip():
        raise HTTPException(status_code=404, detail="Chunks file was not recorded.")

    papers_root = settings.papers_dir.expanduser().resolve()
    output_path = (papers_root / relative_value).resolve()
    expected_job_dir = (papers_root / "jobs" / job_id).resolve()
    if not output_path.is_relative_to(expected_job_dir):
        raise HTTPException(status_code=400, detail="Invalid output file path.")
    if output_path == summary_path:
        raise HTTPException(status_code=400, detail="Invalid output file selection.")
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="Chunks file was not found.")
    return output_path


def _shared_chunk_parts(
    summary: dict[str, Any],
) -> tuple[Path, ...] | None:
    files = summary.get("files") or {}
    if "chunk_parts" not in files:
        return None

    raw_parts = files.get("chunk_parts")
    if not isinstance(raw_parts, list):
        raise HTTPException(status_code=500, detail="Chunk manifest has an invalid format.")

    papers_root = settings.papers_dir.expanduser().resolve()
    chunks_root = (papers_root / "chunks_by_paper").resolve()
    parts: list[Path] = []
    for raw_part in raw_parts:
        if not isinstance(raw_part, str) or not raw_part.strip():
            raise HTTPException(status_code=500, detail="Chunk manifest has an invalid entry.")
        part = (papers_root / raw_part).resolve()
        if not part.is_relative_to(chunks_root):
            raise HTTPException(status_code=400, detail="Invalid shared chunk path.")
        if not part.is_file():
            raise HTTPException(status_code=404, detail="A cached chunk part was not found.")
        parts.append(part)
    return tuple(parts)


def _iter_chunk_parts(paths: tuple[Path, ...]) -> Iterator[bytes]:
    for path in paths:
        last_byte = b""
        if path.suffix.casefold() == ".gz":
            handle_context = gzip.open(path, "rb")
        else:
            handle_context = path.open("rb")
        with handle_context as handle:
            while block := handle.read(1024 * 1024):
                last_byte = block[-1:]
                yield block
        if last_byte and last_byte != b"\n":
            yield b"\n"


@router.get("/jobs/{job_id}/download/chunks")
def download_chunks(job_id: str):
    _job, summary, summary_path = _load_summary(job_id)
    artifact = _published_artifact(summary)
    if artifact is not None:
        store = get_artifact_store()
        current = store.head(artifact.key)
        if current is None:
            raise HTTPException(status_code=404, detail="Published chunks were not found.")
        filename = f"{job_id}-chunks.jsonl.gz"
        local_path = store.local_path(artifact.key)
        if local_path is not None:
            return FileResponse(
                local_path,
                media_type="application/gzip",
                filename=filename,
            )
        return RedirectResponse(
            store.presign_get(artifact.key, download_name=filename),
            status_code=307,
        )

    # Backward compatibility for retrievals completed before object publishing.
    shared_parts = _shared_chunk_parts(summary)
    filename = f"{job_id}-chunks.jsonl"
    if shared_parts is not None:
        return StreamingResponse(
            _iter_chunk_parts(shared_parts),
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Chunk-Parts": str(len(shared_parts)),
            },
        )

    path = _legacy_job_chunks_file(job_id, summary, summary_path)
    return FileResponse(
        path,
        media_type="application/x-ndjson",
        filename=filename,
    )
