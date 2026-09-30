"""Cancellation stops local queued and in-flight work without API requests."""

import asyncio
import threading

from backend.services import relation_executor as module


def test_cancel_stops_active_task_and_preserves_other_jobs(monkeypatch):
    executor = module.RelationExecutor()
    started = threading.Event()
    stopped = threading.Event()
    other_finished = threading.Event()
    failures = []

    async def fake_run(job_id):
        if job_id == "other":
            other_finished.set()
            return
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            stopped.set()

    monkeypatch.setattr(module, "get_relation_job", lambda job_id: {"status": "queued"})
    monkeypatch.setattr(executor, "_run_async", fake_run)
    monkeypatch.setattr(executor, "_fail", lambda job_id, error: failures.append(job_id))
    try:
        assert executor.submit("update")
        assert started.wait(2)
        executor.cancel("update")
        assert stopped.wait(2)
        assert executor.submit("update") is False
        assert executor.submit("other")
        assert other_finished.wait(2)
        assert failures == ["update"]
    finally:
        executor.shutdown()
        executor._pool.shutdown(wait=True)
