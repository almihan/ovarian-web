"""Offline checks of private updates, deduplication, bounds, and crash recovery."""

import json
import re
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import config
from backend.api import updates as api
from backend.pipeline import precomputed_corpora
from backend.services import corpus_store as store
from backend.services import corpus_updates as updates


NN, CA = list(store.CORPUS_FILES)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    for corpus_id, pmid in ((NN, "101"), (CA, "202")):
        (seeds / store.CORPUS_FILES[corpus_id]).write_text(json.dumps({"pmid": pmid, "chunks": []}) + "\n")
    selected = replace(config.settings, data_dir=tmp_path / "data",
                       precomputed_corpora_dir=tmp_path / "persistent",
                       monthly_updates_enabled=True, monthly_update_max_new_papers=3,
                       monthly_update_token="private-token", public_base_url="https://example.test")
    for module in (store, updates, api, precomputed_corpora):
        monkeypatch.setattr(module, "settings", selected)
    monkeypatch.setattr(store, "SEED_CORPUS_ROOT", seeds)
    store.ensure_corpus_store()
    return selected


def run_update(service, *, dry_run=False):
    service.initialize()
    service.start(dry_run=dry_run)
    service._thread.join(timeout=5)
    assert not service._thread.is_alive()
    return service.status()


def process_stub(calls):
    def process(pmid, **kwargs):
        kwargs["check_active"]()
        calls.append(pmid)
        # Zero relations still represent a successfully computed paper.
        return {"rows": [{"pmid": pmid, "chunks": []}], "stats": {"input_tokens": 7}}
    return process


def test_only_union_new_papers_are_processed_once_and_both_topics_grow(configured):
    calls = []
    discover = Mock(return_value={NN: ["101", "202", "301"], CA: ["202", "301", "302"]})
    service = updates.MonthlyCorpusUpdater(discover=discover, process=process_stub(calls))
    status = run_update(service)
    assert status["status"] == "completed"
    assert sorted(calls) == ["301", "302"]
    assert store.get_known_pmids() == {"101", "202", "301", "302"}
    # A paper already in the other corpus is copied without inference.
    assert status["copied_known_memberships"] == 1
    for corpus_id in (NN, CA):
        rows = list(store._rows(store.store_root() / store.CORPUS_FILES[corpus_id]))
        assert "301" in {store.prediction_pmid(row) for row in rows}
    again = service.start(dry_run=False)
    assert again["already_completed_this_month"]
    assert discover.call_count == 1
    assert sorted(calls) == ["301", "302"]


def test_dry_run_does_discovery_without_processing_or_publishing(configured):
    process = Mock(side_effect=AssertionError("Preview launched computation"))
    service = updates.MonthlyCorpusUpdater(discover=lambda **kwargs: {NN: ["303"], CA: ["101"]}, process=process)
    status = run_update(service, dry_run=True)
    assert status["status"] == "dry_run_completed"
    assert status["preview_pmids"] == ["303"]
    assert status["pending_count"] == 0
    assert store.get_known_pmids() == {"101", "202"}
    process.assert_not_called()
    assert not status["last_completed_month"]


def test_cap_is_total_union_and_alternates_topics_with_persistent_backlog(configured):
    calls = []
    discover = lambda **kwargs: {NN: ["301", "302", "303", "304"], CA: ["401", "402", "403", "404"]}
    service = updates.MonthlyCorpusUpdater(discover=discover, process=process_stub(calls))
    status = run_update(service)
    assert len(calls) == 3
    assert calls == ["301", "401", "302"]
    assert status["pending_count"] == 5
    state = json.loads(service.state_path.read_text())
    assert len(state["months"][datetime.now(timezone.utc).strftime("%Y-%m")]["attempted_pmids"]) == 3
    # A fresh controller cannot reset the monthly limit.
    restarted = updates.MonthlyCorpusUpdater(discover=discover, process=process_stub(calls))
    restarted.initialize()
    assert restarted.start(dry_run=False)["already_completed_this_month"]
    assert len(calls) == 3


def test_interrupted_month_keeps_attempt_reservations(configured, monkeypatch):
    selected = replace(configured, monthly_update_max_new_papers=1)
    monkeypatch.setattr(updates, "settings", selected)
    calls = []
    service = updates.MonthlyCorpusUpdater(discover=lambda **kwargs: {NN: ["303", "304"], CA: []}, process=process_stub(calls))
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    service._save({"status": "running", "pending": {"303": [NN]}, "months": {
        month: {"attempted_pmids": ["303"], "succeeded_pmids": [], "failed_pmids": {}}}, "last_completed_month": None})
    service.initialize()
    assert service.status()["status"] == "interrupted"
    status = run_update(service)
    assert calls == []
    assert status["pending_count"] == 2
    assert store.get_known_pmids() == {"101", "202"}


def test_completed_checkpoint_republished_without_processing(configured):
    process = Mock(side_effect=AssertionError("Recovery recomputed a completed paper"))
    service = updates.MonthlyCorpusUpdater(discover=lambda **kwargs: {NN: ["303"], CA: ["303"]}, process=process)
    service.initialize()
    updates._atomic_json(service.checkpoint_root / "303.json", {
        "pmid": "303", "corpus_ids": [NN, CA], "rows": [{"pmid": "303", "chunks": []}]})
    status = run_update(service)
    assert status["status"] == "completed"
    assert "303" in store.get_known_pmids()
    assert not list(service.checkpoint_root.glob("*.json"))
    process.assert_not_called()


def test_failure_remains_pending_and_cannot_retry_paid_work_in_same_month(configured):
    process = Mock(side_effect=RuntimeError("provider failed"))
    service = updates.MonthlyCorpusUpdater(discover=lambda **kwargs: {NN: ["303"], CA: []}, process=process)
    status = run_update(service)
    assert status["status"] == "completed"
    assert status["pending_count"] == 1
    assert "303" not in store.get_known_pmids()
    assert service.start(dry_run=False)["already_completed_this_month"]
    process.assert_called_once()


def test_batch_deadline_prevents_launching_more_papers(configured):
    calls = []
    service = updates.MonthlyCorpusUpdater(discover=lambda **kwargs: {NN: ["303", "304"], CA: []})
    def process(pmid, **kwargs):
        calls.append(pmid)
        service._deadline = time.monotonic() - 1
        return {"rows": [{"pmid": pmid, "chunks": []}]}
    service._process = process
    status = run_update(service)
    assert status["status"] == "failed"
    assert calls == ["303"]
    assert status["pending_count"] == 1
    assert store.get_known_pmids() == {"101", "202", "303"}


def test_disabled_initialization_leaves_completed_checkpoints_inert(configured, monkeypatch):
    monkeypatch.setattr(updates, "settings", replace(configured, monthly_updates_enabled=False))
    service = updates.MonthlyCorpusUpdater(discover=Mock(), process=Mock())
    service.checkpoint_root.mkdir(parents=True)
    updates._atomic_json(service.checkpoint_root / "303.json", {
        "pmid": "303", "corpus_ids": [NN], "rows": [{"pmid": "303", "chunks": []}]})
    service.initialize()
    assert "303" not in store.get_known_pmids()
    with pytest.raises(InterruptedError):
        service.start(dry_run=False)
    assert (service.checkpoint_root / "303.json").exists()


def test_disabled_updates_and_unconfigured_or_wrong_token_cannot_start(configured, monkeypatch):
    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    service = Mock()
    monkeypatch.setattr(api, "corpus_update_service", service)
    endpoint = "/api/internal/corpus-updates"
    assert client.post(endpoint, json={}).status_code == 401
    assert client.post(endpoint, json={}, headers={"X-Corpus-Update-Token": "wrong"}).status_code == 401
    monkeypatch.setattr(api, "settings", replace(configured, monthly_updates_enabled=False))
    assert client.post(endpoint, json={}, headers={"X-Corpus-Update-Token": "private-token"}).status_code == 409
    monkeypatch.setattr(api, "settings", replace(configured, monthly_update_token=""))
    assert client.get(endpoint, headers={"X-Corpus-Update-Token": "private-token"}).status_code == 503
    service.start.assert_not_called()


def test_api_defaults_to_preview_and_rejects_user_supplied_queries(configured, monkeypatch):
    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    service = Mock()
    service.start.return_value = {"status": "running"}
    monkeypatch.setattr(api, "corpus_update_service", service)
    headers = {"X-Corpus-Update-Token": "private-token"}
    assert client.post("/api/internal/corpus-updates", json={}, headers=headers).status_code == 202
    service.start.assert_called_once_with(dry_run=True, callback_base_url="https://example.test")
    assert client.post("/api/internal/corpus-updates", json={"query": "arbitrary"}, headers=headers).status_code == 422


def test_duplicate_trigger_cannot_launch_a_second_worker(configured):
    entered, release = threading.Event(), threading.Event()
    def discover(**kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return {NN: [], CA: []}
    service = updates.MonthlyCorpusUpdater(discover=discover, process=Mock())
    service.start(dry_run=False)
    assert entered.wait(timeout=5)
    try:
        other = updates.MonthlyCorpusUpdater(discover=Mock(side_effect=AssertionError("duplicate")))
        assert other.start(dry_run=False)["already_running"]
    finally:
        release.set()
        service._thread.join(timeout=5)
    assert service.status()["status"] == "completed"


def test_discovery_partitions_more_than_10000_identifiers_and_pages_without_loss(configured, monkeypatch):
    ids = list(range(1, 10002))
    seen = []
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(updates, "build_retry_session", lambda *args: session)
    monkeypatch.setattr(updates, "CORPUS_DISCOVERY_QUERIES", {NN: "human ovarian inflammation"})
    def respond(*args, **kwargs):
        data = kwargs["data"]
        low, high = map(int, re.search(r"(\d+):(\d+)\[UID\]", data["term"]).groups())
        values = [value for value in ids if low <= value <= high]
        start = int(data["retstart"])
        seen.append((len(values), start))
        response = Mock()
        response.json.return_value = {"esearchresult": {"count": str(len(values)), "idlist": [str(value) for value in values[start:start + 1000]]}}
        return response
    monkeypatch.setattr(updates, "perform_request_with_retries", respond)
    result = updates.discover_corpus_pmids()
    assert result[NN] == [str(value) for value in ids]
    assert any(count > 10000 for count, _ in seen)
    assert all(start < 10000 for _, start in seen)


def test_truncated_discovery_fails_before_processing(configured, monkeypatch):
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(updates, "build_retry_session", lambda *args: session)
    response = Mock()
    response.json.return_value = {"esearchresult": {"count": "2", "idlist": ["1"]}}
    monkeypatch.setattr(updates, "perform_request_with_retries", lambda *args, **kwargs: response)
    with pytest.raises(updates.RetrievalError, match="truncated"):
        updates.discover_corpus_pmids()
