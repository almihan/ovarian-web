"""SQLite persistence for Railway-controlled pipeline stages."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import settings


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    connection.execute("PRAGMA busy_timeout=30000;")
    return connection


def _deserialize_stats(result: dict[str, Any]) -> dict[str, Any]:
    raw_stats = result.pop("stats_json", "{}") or "{}"
    try:
        stats = json.loads(raw_stats)
    except (TypeError, json.JSONDecodeError):
        stats = {}
    result["stats"] = stats if isinstance(stats, dict) else {}
    result["elapsed_seconds"] = float(result.get("elapsed_seconds") or 0.0)
    return result


def _deserialize_job(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    result["relation_extraction"] = bool(result.get("relation_extraction"))
    return _deserialize_stats(result)


def _deserialize_annotation_job(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return _deserialize_stats(dict(row))


def _deserialize_relation_job(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return _deserialize_stats(dict(row))


def _deserialize_network_job(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return _deserialize_stats(dict(row))


def _add_missing_columns(
    connection: sqlite3.Connection,
    table: str,
    definitions: dict[str, str],
) -> None:
    existing = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, definition in definitions.items():
        if name not in existing:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {name} {definition}"  # noqa: S608
            )


def init_database() -> None:
    """Create tables and perform additive migrations in place."""

    with closing(_connect()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                input_type TEXT NOT NULL,
                query TEXT NOT NULL DEFAULT '',
                relation_extraction INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT,
                paper_count INTEGER NOT NULL DEFAULT 0,
                entity_count INTEGER NOT NULL DEFAULT 0,
                relation_count INTEGER NOT NULL DEFAULT 0,
                stats_json TEXT NOT NULL DEFAULT '{}',
                elapsed_seconds REAL NOT NULL DEFAULT 0,
                started_at TEXT,
                completed_at TEXT,
                result_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _add_missing_columns(
            connection,
            "jobs",
            {
                "stats_json": "TEXT NOT NULL DEFAULT '{}'",
                "elapsed_seconds": "REAL NOT NULL DEFAULT 0",
                "started_at": "TEXT",
                "completed_at": "TEXT",
                "result_path": "TEXT",
                "error": "TEXT",
            },
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS annotation_jobs (
                id TEXT PRIMARY KEY,
                source_job_id TEXT NOT NULL,
                executor TEXT NOT NULL DEFAULT 'modal',
                model_signature TEXT,
                source_artifact_key TEXT,
                source_artifact_sha256 TEXT,
                output_artifact_key TEXT,
                summary_artifact_key TEXT,
                remote_call_id TEXT,
                callback_token_hash TEXT,
                reused_from_job_id TEXT,
                last_remote_check_at TEXT,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT,
                paper_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                mention_count INTEGER NOT NULL DEFAULT 0,
                normalized_count INTEGER NOT NULL DEFAULT 0,
                unresolved_count INTEGER NOT NULL DEFAULT 0,
                stats_json TEXT NOT NULL DEFAULT '{}',
                elapsed_seconds REAL NOT NULL DEFAULT 0,
                started_at TEXT,
                completed_at TEXT,
                result_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (source_job_id) REFERENCES jobs(id)
            )
            """
        )
        _add_missing_columns(
            connection,
            "annotation_jobs",
            {
                "executor": "TEXT NOT NULL DEFAULT 'local'",
                "model_signature": "TEXT",
                "source_artifact_key": "TEXT",
                "source_artifact_sha256": "TEXT",
                "output_artifact_key": "TEXT",
                "summary_artifact_key": "TEXT",
                "remote_call_id": "TEXT",
                "callback_token_hash": "TEXT",
                "reused_from_job_id": "TEXT",
                "last_remote_check_at": "TEXT",
                "paper_count": "INTEGER NOT NULL DEFAULT 0",
                "chunk_count": "INTEGER NOT NULL DEFAULT 0",
                "mention_count": "INTEGER NOT NULL DEFAULT 0",
                "normalized_count": "INTEGER NOT NULL DEFAULT 0",
                "unresolved_count": "INTEGER NOT NULL DEFAULT 0",
                "stats_json": "TEXT NOT NULL DEFAULT '{}'",
                "elapsed_seconds": "REAL NOT NULL DEFAULT 0",
                "started_at": "TEXT",
                "completed_at": "TEXT",
                "result_path": "TEXT",
                "error": "TEXT",
            },
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS relation_jobs (
                id TEXT PRIMARY KEY,
                source_annotation_job_id TEXT NOT NULL,
                model_signature TEXT NOT NULL,
                source_chunks_artifact_key TEXT NOT NULL,
                source_chunks_artifact_sha256 TEXT NOT NULL,
                source_annotation_artifact_key TEXT NOT NULL,
                source_annotation_artifact_sha256 TEXT NOT NULL,
                output_artifact_key TEXT NOT NULL,
                summary_artifact_key TEXT NOT NULL,
                remote_batch_id TEXT,
                remote_input_file_id TEXT,
                reused_from_job_id TEXT,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT,
                paper_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                eligible_chunk_count INTEGER NOT NULL DEFAULT 0,
                processed_chunk_count INTEGER NOT NULL DEFAULT 0,
                relation_count INTEGER NOT NULL DEFAULT 0,
                cell_context_count INTEGER NOT NULL DEFAULT 0,
                api_request_count INTEGER NOT NULL DEFAULT 0,
                batch_count INTEGER NOT NULL DEFAULT 0,
                stats_json TEXT NOT NULL DEFAULT '{}',
                elapsed_seconds REAL NOT NULL DEFAULT 0,
                started_at TEXT,
                completed_at TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (source_annotation_job_id) REFERENCES annotation_jobs(id)
            )
            """
        )
        _add_missing_columns(
            connection,
            "relation_jobs",
            {
                "model_signature": "TEXT NOT NULL DEFAULT ''",
                "source_chunks_artifact_key": "TEXT NOT NULL DEFAULT ''",
                "source_chunks_artifact_sha256": "TEXT NOT NULL DEFAULT ''",
                "source_annotation_artifact_key": "TEXT NOT NULL DEFAULT ''",
                "source_annotation_artifact_sha256": "TEXT NOT NULL DEFAULT ''",
                "output_artifact_key": "TEXT NOT NULL DEFAULT ''",
                "summary_artifact_key": "TEXT NOT NULL DEFAULT ''",
                "remote_batch_id": "TEXT",
                "remote_input_file_id": "TEXT",
                "reused_from_job_id": "TEXT",
                "paper_count": "INTEGER NOT NULL DEFAULT 0",
                "chunk_count": "INTEGER NOT NULL DEFAULT 0",
                "eligible_chunk_count": "INTEGER NOT NULL DEFAULT 0",
                "processed_chunk_count": "INTEGER NOT NULL DEFAULT 0",
                "relation_count": "INTEGER NOT NULL DEFAULT 0",
                "cell_context_count": "INTEGER NOT NULL DEFAULT 0",
                "api_request_count": "INTEGER NOT NULL DEFAULT 0",
                "batch_count": "INTEGER NOT NULL DEFAULT 0",
                "stats_json": "TEXT NOT NULL DEFAULT '{}'",
                "elapsed_seconds": "REAL NOT NULL DEFAULT 0",
                "started_at": "TEXT",
                "completed_at": "TEXT",
                "error": "TEXT",
            },
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS network_jobs (
                id TEXT PRIMARY KEY,
                source_relation_job_id TEXT NOT NULL,
                network_signature TEXT NOT NULL,
                source_relation_artifact_key TEXT NOT NULL,
                source_relation_artifact_sha256 TEXT NOT NULL,
                source_chunks_artifact_key TEXT NOT NULL,
                source_chunks_artifact_sha256 TEXT NOT NULL,
                source_annotation_artifact_key TEXT NOT NULL,
                source_annotation_artifact_sha256 TEXT NOT NULL,
                graph_artifact_key TEXT NOT NULL,
                entity_index_artifact_key TEXT NOT NULL,
                summary_artifact_key TEXT NOT NULL,
                reused_from_job_id TEXT,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT,
                paper_count INTEGER NOT NULL DEFAULT 0,
                node_count INTEGER NOT NULL DEFAULT 0,
                edge_count INTEGER NOT NULL DEFAULT 0,
                evidence_count INTEGER NOT NULL DEFAULT 0,
                stats_json TEXT NOT NULL DEFAULT '{}',
                elapsed_seconds REAL NOT NULL DEFAULT 0,
                started_at TEXT,
                completed_at TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (source_relation_job_id) REFERENCES relation_jobs(id)
            )
            """
        )
        _add_missing_columns(
            connection,
            "network_jobs",
            {
                "network_signature": "TEXT NOT NULL DEFAULT ''",
                "source_relation_artifact_key": "TEXT NOT NULL DEFAULT ''",
                "source_relation_artifact_sha256": "TEXT NOT NULL DEFAULT ''",
                "source_chunks_artifact_key": "TEXT NOT NULL DEFAULT ''",
                "source_chunks_artifact_sha256": "TEXT NOT NULL DEFAULT ''",
                "source_annotation_artifact_key": "TEXT NOT NULL DEFAULT ''",
                "source_annotation_artifact_sha256": "TEXT NOT NULL DEFAULT ''",
                "graph_artifact_key": "TEXT NOT NULL DEFAULT ''",
                "entity_index_artifact_key": "TEXT NOT NULL DEFAULT ''",
                "summary_artifact_key": "TEXT NOT NULL DEFAULT ''",
                "reused_from_job_id": "TEXT",
                "paper_count": "INTEGER NOT NULL DEFAULT 0",
                "node_count": "INTEGER NOT NULL DEFAULT 0",
                "edge_count": "INTEGER NOT NULL DEFAULT 0",
                "evidence_count": "INTEGER NOT NULL DEFAULT 0",
                "stats_json": "TEXT NOT NULL DEFAULT '{}'",
                "elapsed_seconds": "REAL NOT NULL DEFAULT 0",
                "started_at": "TEXT",
                "completed_at": "TEXT",
                "error": "TEXT",
            },
        )

        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotation_jobs_created_at "
            "ON annotation_jobs(created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotation_jobs_source "
            "ON annotation_jobs(source_job_id, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotation_jobs_cache "
            "ON annotation_jobs(source_artifact_sha256, model_signature, status)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotation_jobs_remote "
            "ON annotation_jobs(remote_call_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_jobs_created_at "
            "ON relation_jobs(created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_jobs_source "
            "ON relation_jobs(source_annotation_job_id, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_jobs_cache "
            "ON relation_jobs(source_annotation_artifact_sha256, "
            "source_chunks_artifact_sha256, model_signature, status)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_jobs_remote "
            "ON relation_jobs(remote_batch_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_network_jobs_created_at "
            "ON network_jobs(created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_network_jobs_source "
            "ON network_jobs(source_relation_job_id, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_network_jobs_cache_v2 "
            "ON network_jobs(source_relation_artifact_sha256, "
            "source_chunks_artifact_sha256, source_annotation_artifact_sha256, "
            "network_signature, status)"
        )

        # Retrieval is an in-process Railway task and cannot survive a restart.
        now = utc_now()
        connection.execute(
            """
            UPDATE jobs
            SET status = 'failed',
                stage = 'failed',
                message = 'This retrieval was interrupted by an application restart.',
                error = 'Application restart interrupted the in-process retrieval job.',
                completed_at = COALESCE(completed_at, ?),
                updated_at = ?
            WHERE status IN ('queued', 'processing')
            """,
            (now, now),
        )
        # Legacy local annotation workers are also non-resumable. Modal calls are
        # left active because their call IDs can be reconciled after Railway restarts.
        connection.execute(
            """
            UPDATE annotation_jobs
            SET status = 'failed',
                stage = 'failed',
                message = 'This local annotation was interrupted by an application restart.',
                error = 'Application restart interrupted the local annotation worker.',
                completed_at = COALESCE(completed_at, ?),
                updated_at = ?
            WHERE status IN ('queued', 'processing')
              AND COALESCE(executor, 'local') != 'modal'
            """,
            (now, now),
        )
        connection.execute(
            """
            UPDATE annotation_jobs
            SET status = 'failed',
                stage = 'failed',
                message = 'Modal submission was interrupted before a remote call was created.',
                error = 'Application restart occurred before Modal returned a call ID.',
                completed_at = COALESCE(completed_at, ?),
                updated_at = ?
            WHERE status IN ('queued', 'processing')
              AND executor = 'modal'
              AND remote_call_id IS NULL
            """,
            (now, now),
        )
        connection.commit()


def create_job(
    *,
    job_id: str,
    input_type: str,
    query: str,
    relation_extraction: bool = False,
) -> dict[str, Any]:
    now = utc_now()
    with closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                id, input_type, query, relation_extraction, status, stage,
                progress, message, stats_json, elapsed_seconds,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                input_type,
                query,
                int(relation_extraction),
                "queued",
                "queued",
                0,
                "Paper retrieval has been queued.",
                "{}",
                0.0,
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
        "stats",
        "elapsed_seconds",
        "started_at",
        "completed_at",
        "result_path",
        "error",
    }
    invalid = set(fields) - allowed_fields
    if invalid:
        raise ValueError(f"Unsupported job fields: {sorted(invalid)}")
    if "stats" in fields:
        fields["stats_json"] = json.dumps(
            fields.pop("stats") or {}, ensure_ascii=False
        )
    fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{name} = ?" for name in fields)
    values = list(fields.values()) + [job_id]
    with closing(_connect()) as connection:
        connection.execute(
            f"UPDATE jobs SET {assignments} WHERE id = ?",  # noqa: S608
            values,
        )
        connection.commit()
    return get_job(job_id)


def get_job(job_id: str) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return _deserialize_job(row)


def list_jobs(limit: int = 8) -> list[dict[str, Any]]:
    safe_limit = max(1, min(limit, 50))
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (safe_limit,)
        ).fetchall()
    return [job for row in rows if (job := _deserialize_job(row)) is not None]


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
    now = utc_now()
    with closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO annotation_jobs (
                id, source_job_id, executor, model_signature,
                source_artifact_key, source_artifact_sha256,
                output_artifact_key, summary_artifact_key,
                callback_token_hash, reused_from_job_id,
                status, stage, progress, message,
                stats_json, elapsed_seconds, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                source_job_id,
                executor,
                model_signature,
                source_artifact_key,
                source_artifact_sha256,
                output_artifact_key,
                summary_artifact_key,
                callback_token_hash,
                reused_from_job_id,
                "queued",
                "queued",
                0,
                (
                    "Reusing a completed Stage 2 entity result."
                    if reused_from_job_id
                    else "Entity extraction has been queued."
                ),
                "{}",
                0.0,
                now,
                now,
            ),
        )
        connection.commit()
    return get_annotation_job(job_id) or {}


def update_annotation_job(job_id: str, **fields: Any) -> dict[str, Any] | None:
    if not fields:
        return get_annotation_job(job_id)
    allowed_fields = {
        "executor",
        "model_signature",
        "source_artifact_key",
        "source_artifact_sha256",
        "output_artifact_key",
        "summary_artifact_key",
        "remote_call_id",
        "callback_token_hash",
        "reused_from_job_id",
        "last_remote_check_at",
        "status",
        "stage",
        "progress",
        "message",
        "paper_count",
        "chunk_count",
        "mention_count",
        "normalized_count",
        "unresolved_count",
        "stats",
        "elapsed_seconds",
        "started_at",
        "completed_at",
        "result_path",
        "error",
    }
    invalid = set(fields) - allowed_fields
    if invalid:
        raise ValueError(f"Unsupported annotation-job fields: {sorted(invalid)}")
    if "stats" in fields:
        fields["stats_json"] = json.dumps(
            fields.pop("stats") or {}, ensure_ascii=False
        )
    fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{name} = ?" for name in fields)
    values = list(fields.values()) + [job_id]
    with closing(_connect()) as connection:
        connection.execute(
            f"UPDATE annotation_jobs SET {assignments} WHERE id = ?",  # noqa: S608
            values,
        )
        connection.commit()
    return get_annotation_job(job_id)


def get_annotation_job(job_id: str) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT * FROM annotation_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return _deserialize_annotation_job(row)


def list_annotation_jobs(limit: int = 8) -> list[dict[str, Any]]:
    safe_limit = max(1, min(limit, 50))
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT * FROM annotation_jobs ORDER BY created_at DESC LIMIT ?",
            (safe_limit,),
        ).fetchall()
    return [
        job
        for row in rows
        if (job := _deserialize_annotation_job(row)) is not None
    ]


def latest_annotation_for_source(source_job_id: str) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT * FROM annotation_jobs
            WHERE source_job_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (source_job_id,),
        ).fetchone()
    return _deserialize_annotation_job(row)


def find_reusable_annotation(
    *, source_artifact_sha256: str, model_signature: str
) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT * FROM annotation_jobs
            WHERE source_artifact_sha256 = ?
              AND model_signature = ?
              AND status = 'completed'
              AND output_artifact_key IS NOT NULL
            ORDER BY completed_at DESC, created_at DESC
            LIMIT 1
            """,
            (source_artifact_sha256, model_signature),
        ).fetchone()
    return _deserialize_annotation_job(row)


def find_active_annotation(
    *, source_artifact_sha256: str, model_signature: str
) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT * FROM annotation_jobs
            WHERE source_artifact_sha256 = ?
              AND model_signature = ?
              AND status IN ('queued', 'processing')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (source_artifact_sha256, model_signature),
        ).fetchone()
    return _deserialize_annotation_job(row)


def find_any_active_annotation() -> dict[str, Any] | None:
    """Return the single active GPU job used by the cost-control policy."""

    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT * FROM annotation_jobs
            WHERE status IN ('queued', 'processing')
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()
    return _deserialize_annotation_job(row)


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
    now = utc_now()
    with closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO relation_jobs (
                id, source_annotation_job_id, model_signature,
                source_chunks_artifact_key, source_chunks_artifact_sha256,
                source_annotation_artifact_key, source_annotation_artifact_sha256,
                output_artifact_key, summary_artifact_key, reused_from_job_id,
                status, stage, progress, message, stats_json, elapsed_seconds,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                source_annotation_job_id,
                model_signature,
                source_chunks_artifact_key,
                source_chunks_artifact_sha256,
                source_annotation_artifact_key,
                source_annotation_artifact_sha256,
                output_artifact_key,
                summary_artifact_key,
                reused_from_job_id,
                "queued",
                "queued",
                0,
                (
                    "Reusing a completed Stage 3 relation result."
                    if reused_from_job_id
                    else "Relation extraction has been queued."
                ),
                "{}",
                0.0,
                now,
                now,
            ),
        )
        connection.commit()
    return get_relation_job(job_id) or {}


def update_relation_job(job_id: str, **fields: Any) -> dict[str, Any] | None:
    if not fields:
        return get_relation_job(job_id)
    allowed_fields = {
        "model_signature",
        "source_chunks_artifact_key",
        "source_chunks_artifact_sha256",
        "source_annotation_artifact_key",
        "source_annotation_artifact_sha256",
        "output_artifact_key",
        "summary_artifact_key",
        "remote_batch_id",
        "remote_input_file_id",
        "reused_from_job_id",
        "status",
        "stage",
        "progress",
        "message",
        "paper_count",
        "chunk_count",
        "eligible_chunk_count",
        "processed_chunk_count",
        "relation_count",
        "cell_context_count",
        "api_request_count",
        "batch_count",
        "stats",
        "elapsed_seconds",
        "started_at",
        "completed_at",
        "error",
    }
    invalid = set(fields) - allowed_fields
    if invalid:
        raise ValueError(f"Unsupported relation-job fields: {sorted(invalid)}")
    if "stats" in fields:
        fields["stats_json"] = json.dumps(
            fields.pop("stats") or {}, ensure_ascii=False
        )
    fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{name} = ?" for name in fields)
    values = list(fields.values()) + [job_id]
    with closing(_connect()) as connection:
        connection.execute(
            f"UPDATE relation_jobs SET {assignments} WHERE id = ?",  # noqa: S608
            values,
        )
        connection.commit()
    return get_relation_job(job_id)


def get_relation_job(job_id: str) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT * FROM relation_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return _deserialize_relation_job(row)


def list_relation_jobs(limit: int = 8) -> list[dict[str, Any]]:
    safe_limit = max(1, min(limit, 50))
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT * FROM relation_jobs ORDER BY created_at DESC LIMIT ?",
            (safe_limit,),
        ).fetchall()
    return [
        job
        for row in rows
        if (job := _deserialize_relation_job(row)) is not None
    ]


def find_reusable_relation(
    *,
    source_annotation_artifact_sha256: str,
    source_chunks_artifact_sha256: str,
    model_signature: str,
) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT * FROM relation_jobs
            WHERE source_annotation_artifact_sha256 = ?
              AND source_chunks_artifact_sha256 = ?
              AND model_signature = ?
              AND status = 'completed'
              AND output_artifact_key IS NOT NULL
            ORDER BY completed_at DESC, created_at DESC
            LIMIT 1
            """,
            (
                source_annotation_artifact_sha256,
                source_chunks_artifact_sha256,
                model_signature,
            ),
        ).fetchone()
    return _deserialize_relation_job(row)


def find_active_relation(
    *,
    source_annotation_artifact_sha256: str,
    source_chunks_artifact_sha256: str,
    model_signature: str,
) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT * FROM relation_jobs
            WHERE source_annotation_artifact_sha256 = ?
              AND source_chunks_artifact_sha256 = ?
              AND model_signature = ?
              AND status IN ('queued', 'processing')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (
                source_annotation_artifact_sha256,
                source_chunks_artifact_sha256,
                model_signature,
            ),
        ).fetchone()
    return _deserialize_relation_job(row)


def find_any_active_relation() -> dict[str, Any] | None:
    """Return the one active Stage 3 job allowed by the cost policy."""

    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT * FROM relation_jobs
            WHERE status IN ('queued', 'processing')
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()
    return _deserialize_relation_job(row)



def create_network_job(
    *,
    job_id: str,
    source_relation_job_id: str,
    network_signature: str,
    source_relation_artifact_key: str,
    source_relation_artifact_sha256: str,
    source_chunks_artifact_key: str,
    source_chunks_artifact_sha256: str,
    source_annotation_artifact_key: str,
    source_annotation_artifact_sha256: str,
    graph_artifact_key: str,
    entity_index_artifact_key: str,
    summary_artifact_key: str,
    reused_from_job_id: str | None = None,
) -> dict[str, Any]:
    now = utc_now()
    with closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO network_jobs (
                id, source_relation_job_id, network_signature,
                source_relation_artifact_key, source_relation_artifact_sha256,
                source_chunks_artifact_key, source_chunks_artifact_sha256,
                source_annotation_artifact_key, source_annotation_artifact_sha256,
                graph_artifact_key, entity_index_artifact_key, summary_artifact_key,
                reused_from_job_id, status, stage, progress, message,
                stats_json, elapsed_seconds, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id, source_relation_job_id, network_signature,
                source_relation_artifact_key, source_relation_artifact_sha256,
                source_chunks_artifact_key, source_chunks_artifact_sha256,
                source_annotation_artifact_key, source_annotation_artifact_sha256,
                graph_artifact_key, entity_index_artifact_key, summary_artifact_key,
                reused_from_job_id, "queued", "queued", 0,
                (
                    "Reusing a completed Stage 4 network."
                    if reused_from_job_id
                    else "Interaction-network construction has been queued."
                ),
                "{}", 0.0, now, now,
            ),
        )
        connection.commit()
    return get_network_job(job_id) or {}


def update_network_job(job_id: str, **fields: Any) -> dict[str, Any] | None:
    if not fields:
        return get_network_job(job_id)
    allowed_fields = {
        "network_signature",
        "source_relation_artifact_key",
        "source_relation_artifact_sha256",
        "source_chunks_artifact_key",
        "source_chunks_artifact_sha256",
        "source_annotation_artifact_key",
        "source_annotation_artifact_sha256",
        "graph_artifact_key",
        "entity_index_artifact_key",
        "summary_artifact_key",
        "reused_from_job_id",
        "status",
        "stage",
        "progress",
        "message",
        "paper_count",
        "node_count",
        "edge_count",
        "evidence_count",
        "stats",
        "elapsed_seconds",
        "started_at",
        "completed_at",
        "error",
    }
    invalid = set(fields) - allowed_fields
    if invalid:
        raise ValueError(f"Unsupported network-job fields: {sorted(invalid)}")
    if "stats" in fields:
        fields["stats_json"] = json.dumps(
            fields.pop("stats") or {}, ensure_ascii=False
        )
    fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{name} = ?" for name in fields)
    values = list(fields.values()) + [job_id]
    with closing(_connect()) as connection:
        connection.execute(
            f"UPDATE network_jobs SET {assignments} WHERE id = ?",  # noqa: S608
            values,
        )
        connection.commit()
    return get_network_job(job_id)


def get_network_job(job_id: str) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT * FROM network_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return _deserialize_network_job(row)


def list_network_jobs(limit: int = 8) -> list[dict[str, Any]]:
    safe_limit = max(1, min(limit, 50))
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT * FROM network_jobs ORDER BY created_at DESC LIMIT ?",
            (safe_limit,),
        ).fetchall()
    return [
        job
        for row in rows
        if (job := _deserialize_network_job(row)) is not None
    ]


def find_reusable_network(
    *,
    source_relation_artifact_sha256: str,
    source_chunks_artifact_sha256: str,
    source_annotation_artifact_sha256: str,
    network_signature: str,
) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT * FROM network_jobs
            WHERE source_relation_artifact_sha256 = ?
              AND source_chunks_artifact_sha256 = ?
              AND source_annotation_artifact_sha256 = ?
              AND network_signature = ?
              AND status = 'completed'
              AND graph_artifact_key IS NOT NULL
            ORDER BY completed_at DESC, created_at DESC
            LIMIT 1
            """,
            (
                source_relation_artifact_sha256,
                source_chunks_artifact_sha256,
                source_annotation_artifact_sha256,
                network_signature,
            ),
        ).fetchone()
    return _deserialize_network_job(row)


def find_active_network(
    *,
    source_relation_artifact_sha256: str,
    source_chunks_artifact_sha256: str,
    source_annotation_artifact_sha256: str,
    network_signature: str,
) -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT * FROM network_jobs
            WHERE source_relation_artifact_sha256 = ?
              AND source_chunks_artifact_sha256 = ?
              AND source_annotation_artifact_sha256 = ?
              AND network_signature = ?
              AND status IN ('queued', 'processing')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (
                source_relation_artifact_sha256,
                source_chunks_artifact_sha256,
                source_annotation_artifact_sha256,
                network_signature,
            ),
        ).fetchone()
    return _deserialize_network_job(row)


def find_any_active_network() -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT * FROM network_jobs
            WHERE status IN ('queued', 'processing')
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()
    return _deserialize_network_job(row)

def result_file_for(job_id: str) -> Path:
    """Return the legacy JSON result path retained for backward compatibility."""

    return settings.results_dir / job_id / "network.json"
