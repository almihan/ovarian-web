"""Bounded Railway/local executor for Stage 4 network construction."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests

from backend.config import settings
from backend.database.database import (
    find_any_active_network,
    get_network_job,
    update_network_job,
    utc_now,
)
from backend.pipeline.entity_artifacts import sha256_path
from backend.pipeline.network_builder import build_interaction_network
from backend.storage.artifacts import ArtifactStore, get_artifact_store

logger = logging.getLogger(__name__)
_ONE_MIB = 1024 * 1024


class NetworkExecutorStopping(RuntimeError):
    pass


def _elapsed(started_at: Any) -> float:
    if not started_at:
        return 0.0
    try:
        started = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
    except (TypeError, ValueError):
        return 0.0


class NetworkExecutor:
    """Run one CPU/SQLite Stage 4 build at a time."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage4")
        self._guard = threading.Lock()
        self._running: set[str] = set()
        self._stop = threading.Event()

    def shutdown(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _download_artifact(
        store: ArtifactStore,
        key: str,
        destination: Path,
        *,
        expected_sha256: str,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        local = store.local_path(key)
        if local is not None:
            shutil.copyfile(local, destination)
        else:
            url = store.presign_get(
                key, expires_seconds=settings.artifact_presigned_ttl_seconds
            )
            with requests.get(url, stream=True, timeout=(20, 900)) as response:
                response.raise_for_status()
                with destination.open("wb") as output:
                    for block in response.iter_content(chunk_size=_ONE_MIB):
                        if block:
                            output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
        if sha256_path(destination) != expected_sha256:
            raise ValueError(f"Artifact {key} failed its SHA-256 check.")

    @staticmethod
    def _merge_stats(job: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
        current = job.get("stats")
        merged = dict(current) if isinstance(current, Mapping) else {}
        merged.update(dict(extra))
        return merged

    def _update(
        self,
        job_id: str,
        *,
        stage: str,
        progress: int,
        message: str,
        stats: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        job = get_network_job(job_id)
        if job is None or job.get("status") not in {"queued", "processing"}:
            return
        update_network_job(
            job_id,
            status="processing",
            stage=stage,
            progress=max(int(job.get("progress") or 0), min(99, int(progress))),
            message=message,
            stats=self._merge_stats(job, stats or {}),
            elapsed_seconds=max(
                float(job.get("elapsed_seconds") or 0.0),
                _elapsed(job.get("started_at")),
            ),
            **fields,
        )

    def submit(self, job_id: str) -> bool:
        if self._stop.is_set():
            return False
        job = get_network_job(job_id)
        if job is None or job.get("status") not in {"queued", "processing"}:
            return False
        with self._guard:
            if job_id in self._running:
                return False
            self._running.add(job_id)
        self._pool.submit(self._run_guarded, job_id)
        return True

    def resume_active_jobs(self) -> None:
        active = find_any_active_network()
        if active is not None:
            self.submit(str(active["id"]))

    def _run_guarded(self, job_id: str) -> None:
        try:
            self._run(job_id)
        except NetworkExecutorStopping:
            logger.info("Stage 4 job %s paused for application shutdown", job_id)
        except Exception as exc:
            logger.exception("Stage 4 job %s failed", job_id)
            job = get_network_job(job_id)
            if job is not None and job.get("status") != "completed":
                update_network_job(
                    job_id,
                    status="failed",
                    stage="failed",
                    progress=100,
                    message="Interaction-network construction failed.",
                    elapsed_seconds=max(
                        float(job.get("elapsed_seconds") or 0.0),
                        _elapsed(job.get("started_at")),
                    ),
                    completed_at=utc_now(),
                    error=f"{type(exc).__name__}: {exc}",
                )
        finally:
            with self._guard:
                self._running.discard(job_id)

    def _run(self, job_id: str) -> None:
        if self._stop.is_set():
            raise NetworkExecutorStopping()
        job = get_network_job(job_id)
        if job is None or job.get("status") not in {"queued", "processing"}:
            return
        started_at = str(job.get("started_at") or utc_now())
        update_network_job(
            job_id,
            status="processing",
            stage="preparing_network_sources",
            progress=max(1, int(job.get("progress") or 0)),
            message="Preparing Stage 3 relations, source passages, and Stage 2 entity spans.",
            started_at=started_at,
            error=None,
        )

        work_dir = settings.network_jobs_dir / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        relation_path = work_dir / "relations.jsonl.gz"
        chunks_path = work_dir / "chunks.jsonl.gz"
        annotations_path = work_dir / "entity-annotations.jsonl.gz"
        graph_path = work_dir / "interaction-network.sqlite"
        index_path = work_dir / "entity-relation-index.jsonl.gz"
        summary_path = work_dir / "summary.json"
        for path in (
            relation_path,
            chunks_path,
            annotations_path,
            graph_path,
            index_path,
            summary_path,
        ):
            path.unlink(missing_ok=True)

        store = get_artifact_store()
        self._download_artifact(
            store,
            str(job["source_relation_artifact_key"]),
            relation_path,
            expected_sha256=str(job["source_relation_artifact_sha256"]),
        )
        self._download_artifact(
            store,
            str(job["source_chunks_artifact_key"]),
            chunks_path,
            expected_sha256=str(job["source_chunks_artifact_sha256"]),
        )
        self._download_artifact(
            store,
            str(job["source_annotation_artifact_key"]),
            annotations_path,
            expected_sha256=str(job["source_annotation_artifact_sha256"]),
        )
        self._update(
            job_id,
            stage="building_entity_index",
            progress=5,
            message="Building global cell, gene/protein, and hormone identities.",
        )

        total_chunks = max(
            1,
            int(job.get("stats", {}).get("source_chunk_count") or 0),
        )
        last_update = 0.0

        def progress(rows: int, message: str, live_stats: Mapping[str, Any]) -> None:
            nonlocal last_update
            if self._stop.is_set():
                raise NetworkExecutorStopping()
            now = time.monotonic()
            if now - last_update < 1.0 and rows < total_chunks:
                return
            last_update = now
            percent = 5 + round(72 * min(rows, total_chunks) / total_chunks)
            self._update(
                job_id,
                stage=(
                    "writing_entity_index"
                    if message.startswith("Writing")
                    else "building_interaction_network"
                ),
                progress=percent,
                message=message,
                stats=live_stats,
            )

        result = build_interaction_network(
            relation_path=relation_path,
            chunks_path=chunks_path,
            annotations_path=annotations_path,
            graph_path=graph_path,
            entity_index_path=index_path,
            progress=progress,
        )
        if self._stop.is_set():
            raise NetworkExecutorStopping()

        self._update(
            job_id,
            stage="publishing_network",
            progress=88,
            message="Publishing the SQLite graph and compressed entity index.",
            stats=result.stats,
        )
        graph_ref, graph_reused = store.put_file(
            graph_path,
            key=str(job["graph_artifact_key"]),
            content_type="application/vnd.sqlite3",
        )
        index_ref, index_reused = store.put_file(
            index_path,
            key=str(job["entity_index_artifact_key"]),
            content_type="application/x-ndjson",
            content_encoding="gzip",
        )
        elapsed = max(_elapsed(started_at), 0.001)
        final_stats = {
            **result.stats,
            "elapsed_seconds": elapsed,
            "graph_size_bytes": graph_ref.size_bytes,
            "entity_index_size_bytes": index_ref.size_bytes,
            "graph_artifact_sha256": graph_ref.sha256,
            "entity_index_artifact_sha256": index_ref.sha256,
            "graph_artifact_reused": graph_reused,
            "entity_index_artifact_reused": index_reused,
        }
        summary = {
            "schema": "ovarian-network-stage4-summary-v1",
            "job_id": job_id,
            "source_relation_job_id": job["source_relation_job_id"],
            "completed_at": utc_now(),
            "message": (
                f"Finished: {int(final_stats.get('node_count') or 0):,} nodes and "
                f"{int(final_stats.get('edge_count') or 0):,} directed edges are ready."
            ),
            "stats": final_stats,
            "artifacts": {
                "graph": graph_ref.to_dict(),
                "entity_index": index_ref.to_dict(),
            },
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary_ref, _ = store.put_file(
            summary_path,
            key=str(job["summary_artifact_key"]),
            content_type="application/json",
        )
        final_stats["summary_size_bytes"] = summary_ref.size_bytes

        update_network_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message=summary["message"],
            paper_count=int(final_stats.get("paper_count") or 0),
            node_count=int(final_stats.get("node_count") or 0),
            edge_count=int(final_stats.get("edge_count") or 0),
            evidence_count=int(final_stats.get("evidence_count") or 0),
            stats=final_stats,
            elapsed_seconds=elapsed,
            completed_at=summary["completed_at"],
            error=None,
        )

        # Keep only the durable artifacts. S3 deployments do not need a second
        # local copy; local deployments already have the canonical artifact.
        for path in (
            relation_path,
            chunks_path,
            annotations_path,
            graph_path,
            index_path,
            summary_path,
        ):
            path.unlink(missing_ok=True)
        try:
            work_dir.rmdir()
        except OSError:
            pass


network_executor = NetworkExecutor()

__all__ = ["NetworkExecutor", "network_executor"]
