"""Small process-local records for the active Stage 2 and Stage 3 workers.

These records are coordination state, not user history. They are held only in
memory and are cleared whenever the FastAPI process starts or stops. Reusable
shared-default artifacts live in the artifact store instead.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Iterable

_LOCK = RLock()
_ANNOTATION_JOBS: dict[str, dict[str, Any]] = {}
_RELATION_JOBS: dict[str, dict[str, Any]] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _copy(value: dict[str, Any] | None) -> dict[str, Any] | None:
    return deepcopy(value) if value is not None else None


def _sorted(values: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 100))
    ordered = sorted(
        values,
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    )
    return [deepcopy(item) for item in ordered[:safe_limit]]


def _create(
    store: dict[str, dict[str, Any]],
    job_id: str,
    payload: dict[str, Any],
    *,
    message: str,
) -> dict[str, Any]:
    now = utc_now()
    record = {
        "id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "message": message,
        "stats": {},
        "elapsed_seconds": 0.0,
        "started_at": None,
        "completed_at": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        **payload,
    }
    with _LOCK:
        store[job_id] = record
        return deepcopy(record)


def _update(
    store: dict[str, dict[str, Any]],
    job_id: str,
    fields: dict[str, Any],
) -> dict[str, Any] | None:
    with _LOCK:
        record = store.get(job_id)
        if record is None:
            return None
        for key, value in fields.items():
            record[key] = deepcopy(value)
        record["updated_at"] = utc_now()
        return deepcopy(record)


def clear_worker_state() -> None:
    """Clear all process-local worker records; no database file is created."""

    with _LOCK:
        _ANNOTATION_JOBS.clear()
        _RELATION_JOBS.clear()


def create_annotation_job(
    *,
    job_id: str,
    source_job_id: str,
    executor: str = "modal",
    model_signature: str | None = None,
    source_artifact_key: str | None = None,
    source_artifact_sha256: str | None = None,
    output_artifact_key: str | None = None,
    summary_artifact_key: str | None = None,
    callback_token_hash: str | None = None,
    reused_from_job_id: str | None = None,
) -> dict[str, Any]:
    return _create(
        _ANNOTATION_JOBS,
        job_id,
        {
            "source_job_id": source_job_id,
            "executor": executor,
            "model_signature": model_signature,
            "source_artifact_key": source_artifact_key,
            "source_artifact_sha256": source_artifact_sha256,
            "output_artifact_key": output_artifact_key,
            "summary_artifact_key": summary_artifact_key,
            "remote_call_id": None,
            "callback_token_hash": callback_token_hash,
            "reused_from_job_id": reused_from_job_id,
            "last_remote_check_at": None,
            "paper_count": 0,
            "chunk_count": 0,
            "mention_count": 0,
            "normalized_count": 0,
            "unresolved_count": 0,
            "result_path": None,
        },
        message=(
            "Reusing a completed Stage 2 entity result."
            if reused_from_job_id
            else "Entity extraction has been queued."
        ),
    )


def update_annotation_job(job_id: str, **fields: Any) -> dict[str, Any] | None:
    return _update(_ANNOTATION_JOBS, job_id, fields)


def get_annotation_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        return _copy(_ANNOTATION_JOBS.get(job_id))


def list_annotation_jobs(limit: int = 100) -> list[dict[str, Any]]:
    with _LOCK:
        return _sorted(_ANNOTATION_JOBS.values(), limit)


def create_relation_job(
    *,
    job_id: str,
    source_annotation_job_id: str,
    model_signature: str,
    source_chunks_artifact_key: str,
    source_chunks_artifact_sha256: str,
    source_annotation_artifact_key: str,
    source_annotation_artifact_sha256: str,
    output_artifact_key: str,
    summary_artifact_key: str,
    reused_from_job_id: str | None = None,
) -> dict[str, Any]:
    return _create(
        _RELATION_JOBS,
        job_id,
        {
            "source_annotation_job_id": source_annotation_job_id,
            "model_signature": model_signature,
            "source_chunks_artifact_key": source_chunks_artifact_key,
            "source_chunks_artifact_sha256": source_chunks_artifact_sha256,
            "source_annotation_artifact_key": source_annotation_artifact_key,
            "source_annotation_artifact_sha256": source_annotation_artifact_sha256,
            "output_artifact_key": output_artifact_key,
            "summary_artifact_key": summary_artifact_key,
            "remote_batch_id": None,
            "remote_input_file_id": None,
            "reused_from_job_id": reused_from_job_id,
            "paper_count": 0,
            "chunk_count": 0,
            "eligible_chunk_count": 0,
            "processed_chunk_count": 0,
            "relation_count": 0,
            "cell_context_count": 0,
            "api_request_count": 0,
            "batch_count": 0,
        },
        message=(
            "Reusing a completed Stage 3 relation result."
            if reused_from_job_id
            else "Relation extraction has been queued."
        ),
    )


def update_relation_job(job_id: str, **fields: Any) -> dict[str, Any] | None:
    return _update(_RELATION_JOBS, job_id, fields)


def get_relation_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        return _copy(_RELATION_JOBS.get(job_id))


def list_relation_jobs(limit: int = 100) -> list[dict[str, Any]]:
    with _LOCK:
        return _sorted(_RELATION_JOBS.values(), limit)


__all__ = [
    "create_annotation_job",
    "create_relation_job",
    "get_annotation_job",
    "get_relation_job",
    "clear_worker_state",
    "list_annotation_jobs",
    "list_relation_jobs",
    "update_annotation_job",
    "update_relation_job",
    "utc_now",
]
