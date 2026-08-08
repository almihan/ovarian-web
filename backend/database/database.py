"""Small SQLite persistence layer for analysis jobs."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        settings.database_path,
        timeout=30,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    return connection


def init_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                input_type TEXT NOT NULL,
                query TEXT NOT NULL,
                relation_extraction INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT,
                paper_count INTEGER NOT NULL DEFAULT 0,
                entity_count INTEGER NOT NULL DEFAULT 0,
                relation_count INTEGER NOT NULL DEFAULT 0,
                result_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)"
        )
        connection.commit()


def create_job(
    *,
    job_id: str,
    input_type: str,
    query: str,
    relation_extraction: bool,
) -> dict[str, Any]:
    now = utc_now()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                id, input_type, query, relation_extraction, status, stage,
                progress, message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                input_type,
                query,
                int(relation_extraction),
                "queued",
                "queued",
                0,
                "Analysis has been queued.",
                now,
                now,
            ),
        )
        connection.commit()
    return get_job(job_id) or {}


def update_job(job_id: str, **fields: Any) -> dict[str, Any] | None:
    if not fields:
        return get_job(job_id)

    allowed_fields = {
        "status",
        "stage",
        "progress",
        "message",
        "paper_count",
        "entity_count",
        "relation_count",
        "result_path",
        "error",
    }
    invalid = set(fields) - allowed_fields
    if invalid:
        raise ValueError(f"Unsupported job fields: {sorted(invalid)}")

    fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{name} = ?" for name in fields)
    values = list(fields.values()) + [job_id]

    with _connect() as connection:
        connection.execute(
            f"UPDATE jobs SET {assignments} WHERE id = ?",  # noqa: S608
            values,
        )
        connection.commit()
    return get_job(job_id)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["relation_extraction"] = bool(result["relation_extraction"])
    return result


def list_jobs(limit: int = 8) -> list[dict[str, Any]]:
    safe_limit = max(1, min(limit, 50))
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (safe_limit,)
        ).fetchall()
    jobs = [dict(row) for row in rows]
    for job in jobs:
        job["relation_extraction"] = bool(job["relation_extraction"])
    return jobs


def result_file_for(job_id: str) -> Path:
    return settings.results_dir / job_id / "network.json"
