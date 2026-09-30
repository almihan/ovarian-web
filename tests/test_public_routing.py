"""Public saved PMID runs must never dispatch retrieval, GPU, or API work."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend import config, orchestrator, runtime
from backend.api import runs
from backend.pipeline import precomputed_corpora
from backend.services import corpus_store


def prediction(pmid):
    return {
        "pmid": pmid,
        "canonical_id": f"pmid:{pmid}",
        "title": f"Paper {pmid}",
        "chunks": [{
            "chunk_id": 1, "base": f"paper-{pmid}",
            "section_type": "ABSTRACT", "text_source": "abstract",
            "text": "IL21 activates T cells.",
            "annotations": [
                {"obj": "gene", "mention": "IL21", "start": 0, "end": 4,
                 "concept_id": "HGNC:6005", "preferred_label": "IL21"},
                {"obj": "cell", "mention": "T cells", "start": 15, "end": 22,
                 "concept_id": "CL:0000084", "preferred_label": "T cell"},
            ],
            "relations": [{"subject": "IL21 (HGNC:6005)",
                           "predicate": "activation", "object": "T cell (CL:0000084)",
                           "conditions": "", "evidence": "IL21 activates T cells."}],
        }],
    }


@pytest.fixture
def public_app(tmp_path, monkeypatch):
    data = tmp_path / "data"
    corpus_dir = data / "precomputed_corpora"
    corpus_dir.mkdir(parents=True)
    for corpus_id, pmid in zip(corpus_store.CORPUS_FILES, ("101", "202")):
        (corpus_dir / corpus_store.CORPUS_FILES[corpus_id]).write_text(
            json.dumps(prediction(pmid)) + "\n", encoding="utf-8"
        )
    selected = replace(config.settings, data_dir=data,
                       precomputed_corpora_dir=corpus_dir, public_precomputed_only=True,
                       monthly_updates_enabled=True)
    for module in (config, orchestrator, runtime, runs, corpus_store, precomputed_corpora):
        monkeypatch.setattr(module, "settings", selected)
    runtime.run_registry.clear()
    monkeypatch.setattr(orchestrator, "_PRECOMPUTED_STAGE_MIN_SECONDS", 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("A public saved-results run attempted fresh computation")

    pipeline = orchestrator.PipelineOrchestrator()
    # Run one stage at a time so tests can inspect each transition deterministically.
    monkeypatch.setattr(pipeline, "_submit", lambda run_id, stage, target: target(run_id))
    monkeypatch.setattr(orchestrator, "build_isolated_pmid_stage1", forbidden)
    monkeypatch.setattr(orchestrator, "get_or_build_default_stage1", forbidden)
    monkeypatch.setattr(pipeline, "_annotation_artifact", forbidden)
    monkeypatch.setattr(pipeline, "_relation_artifact", forbidden)
    monkeypatch.setattr(runs, "pipeline_orchestrator", pipeline)
    app = FastAPI()
    app.include_router(runs.router)
    yield TestClient(app), pipeline, selected
    runtime.run_registry.clear()
    for pool in (pipeline._stage1_pool, pipeline._stage2_pool,
                 pipeline._stage3_pool, pipeline._stage4_pool, pipeline._update_pool):
        pool.shutdown(wait=False, cancel_futures=True)


def test_public_pmids_use_both_corpora_and_report_missing(public_app):
    client, pipeline, _ = public_app
    response = client.post("/api/runs", json={"query": "101, 202, 999"})
    assert response.status_code == 202
    run = response.json()
    assert run["input_mode"] == "precomputed"
    assert run["stages"]["retrieval"]["stats"]["found_pmids"] == ["101", "202"]
    assert run["stages"]["retrieval"]["stats"]["missing_pmids"] == ["999"]
    assert "999" in run["stages"]["retrieval"]["message"]
    for number in (2, 3, 4):
        response = client.post(f"/api/runs/{run['id']}/stages/{number}")
        assert response.status_code == 202
    final = response.json()
    assert final["stages"]["network"]["status"] == "completed"
    assert final["stages"]["network"]["stats"]["edge_count"] == 1
    downloaded = client.get(f"/api/runs/{run['id']}/download/stage3")
    assert downloaded.status_code == 200
    rows = [json.loads(line) for line in downloaded.text.splitlines() if line]
    assert {row["pmid"] for row in rows} == {"101", "202"}


def test_public_unknown_and_invalid_pmids_are_rejected(public_app):
    client, _, _ = public_app
    for query in ("999", "ovary", "PMC1234", "101, nonsense"):
        response = client.post("/api/runs", json={"query": query})
        assert response.status_code == 422
    assert "saved paper list" in client.post("/api/runs", json={"query": "999"}).json()["detail"]


def test_dropdown_saved_corpus_still_works(public_app):
    client, _, _ = public_app
    response = client.post("/api/runs", json={"corpus_id": "cancer_associated_inflammatory"})
    assert response.status_code == 202
    assert response.json()["stages"]["retrieval"]["stats"]["paper_count"] == 1


def test_local_arbitrary_pmids_and_trusted_update_bypass(public_app, monkeypatch):
    _, pipeline, selected = public_app
    # Suppress workers: verify dispatch and privileges without paid requests.
    monkeypatch.setattr(pipeline, "_submit", lambda *args: None)
    monkeypatch.setattr(orchestrator, "settings", replace(selected, public_precomputed_only=False))
    local = pipeline.create_run("999")
    assert local["input_mode"] == "pmid_only"
    assert not runtime.run_registry.get_private(local["id"], "trusted_update")
    monkeypatch.setattr(orchestrator, "settings", selected)
    with pytest.raises(PermissionError):
        pipeline._run_stage1(local["id"])
    with pytest.raises(PermissionError):
        pipeline._run_stage2(local["id"], callback_base_url="")
    with pytest.raises(PermissionError):
        pipeline._run_stage3(local["id"])
    trusted = pipeline.create_update_run(["999"])
    assert trusted["input_mode"] == "pmid_only"
    assert runtime.run_registry.get_private(trusted["id"], "trusted_update")
    pipeline._assert_run_compute_allowed(trusted["id"])
    with pytest.raises(HTTPException) as error:
        runs._run(trusted["id"])
    assert error.value.status_code == 404


def test_updater_run_is_retained_until_checkpoint(public_app):
    _, pipeline, _ = public_app
    # No worker is dispatched; the retained marker protects a completed stage.
    pipeline._submit = lambda *args: None
    run = pipeline.create_update_run(["999"])
    runtime.run_registry.update_stage(run["id"], "retrieval", status="completed")
    with runtime.run_registry._lock:
        runtime.run_registry._runs[run["id"]]["updated_at"] = "2020-01-01T00:00:00+00:00"
    assert runtime.run_registry.pop_expired(3600) == []
    runtime.run_registry.set_private(run["id"], "retain_for_update", False)
    with runtime.run_registry._lock:
        runtime.run_registry._runs[run["id"]]["updated_at"] = "2020-01-01T00:00:00+00:00"
    assert len(runtime.run_registry.pop_expired(3600)) == 1


def test_expired_or_unknown_run_returns_404(public_app):
    client, _, _ = public_app
    response = client.get("/api/runs/" + "0" * 32)
    assert response.status_code == 404


def test_canceled_update_cancels_workers_and_cannot_start_next_stage(public_app, monkeypatch):
    _, pipeline, selected = public_app
    monkeypatch.setattr(pipeline, "_submit", lambda *args: None)
    run = pipeline.create_update_run(["999"])
    run_id = run["id"]
    runtime.run_registry.set_private(run_id, "active_annotation_job_id", "annotation-1")
    runtime.run_registry.set_private(run_id, "active_modal_call_id", "modal-call-1")
    runtime.run_registry.set_private(run_id, "active_relation_job_id", "relation-1")
    calls = []
    monkeypatch.setattr(orchestrator, "update_annotation_job", lambda job_id, **fields: calls.append(("annotation", job_id, fields["status"])))
    monkeypatch.setattr(orchestrator.local_executor, "cleanup", lambda job_id, **kwargs: calls.append(("local", job_id, kwargs["terminate"])))
    monkeypatch.setattr(orchestrator.modal_executor, "cancel", lambda call_id: calls.append(("modal", call_id)))
    monkeypatch.setattr(orchestrator.relation_executor, "cancel", lambda job_id: calls.append(("relation", job_id)))
    pipeline.cancel_update_run(run_id)
    assert ("annotation", "annotation-1", "failed") in calls
    assert ("local", "annotation-1", True) in calls
    assert ("modal", "modal-call-1") in calls
    assert ("relation", "relation-1") in calls
    for stage in (pipeline._run_stage1, pipeline._run_stage3):
        with pytest.raises(PermissionError, match="canceled or disabled"):
            stage(run_id)
    with pytest.raises(PermissionError):
        pipeline._run_stage2(run_id, callback_base_url="")
    runtime.run_registry.complete_stage(run_id, "retrieval", message="done", stats={}, elapsed_seconds=0)
    launches = []
    monkeypatch.setattr(pipeline, "_submit", lambda *args: launches.append(args))
    pipeline._queue_automatic_next_stage(run_id, "retrieval")
    assert launches == []

    # Disabling monthly updates also stops a trusted run before paid work starts.
    enabled_run = pipeline.create_update_run(["998"])
    monkeypatch.setattr(orchestrator, "settings", replace(selected, monthly_updates_enabled=False))
    with pytest.raises(PermissionError):
        pipeline._run_stage2(enabled_run["id"], callback_base_url="")
    assert runtime.run_registry.get_private(enabled_run["id"], "cancel_requested")


def test_monthly_runs_have_a_separate_queue(public_app, monkeypatch):
    _, pipeline, _ = public_app
    monkeypatch.setattr(pipeline, "_submit", lambda *args: None)
    update = pipeline.create_update_run(["999"])
    public = pipeline.create_run("101")
    dispatched = []
    monkeypatch.setattr(pipeline._update_pool, "submit", lambda target: dispatched.append("update"))
    monkeypatch.setattr(pipeline._stage2_pool, "submit", lambda target: dispatched.append("public"))
    submit = orchestrator.PipelineOrchestrator._submit.__get__(pipeline)
    submit(update["id"], "annotation", lambda run_id: None)
    submit(public["id"], "annotation", lambda run_id: None)
    assert dispatched == ["update", "public"]


@pytest.mark.parametrize("trusted", [True, False])
def test_completed_updater_artifact_summaries_are_retained(public_app, monkeypatch, trusted):
    _, pipeline, selected = public_app
    monkeypatch.setattr(pipeline, "_submit", lambda *args: None)
    monkeypatch.setattr(orchestrator, "settings", replace(
        selected, public_precomputed_only=False, openai_api_key="offline-test-key"
    ))
    run = pipeline.create_update_run(["999"]) if trusted else pipeline.create_run("999")
    ref = orchestrator.ArtifactRef(key="source.jsonl.gz", size_bytes=1, sha256="abc")
    deleted = []

    class FinishedArtifactStore:
        def head(self, key):
            return orchestrator.ArtifactRef(key=key, size_bytes=1, sha256="result")

        def read_json(self, key):
            return {"stats": {"paper_count": 1}}

        def delete(self, key):
            deleted.append(key)

    monkeypatch.setattr(orchestrator, "get_artifact_store", lambda: FinishedArtifactStore())
    monkeypatch.setattr(orchestrator, "cached_json_pair", lambda **kwargs: None)
    monkeypatch.setattr(pipeline, "_annotation_backend", lambda url: ("local", ""))
    monkeypatch.setattr(orchestrator, "create_annotation_job", lambda **kwargs: None)
    monkeypatch.setattr(orchestrator, "update_annotation_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "get_annotation_job", lambda job_id: {"id": job_id, "status": "completed"})
    monkeypatch.setattr(orchestrator, "_cell_branch_ready", lambda job: True)
    monkeypatch.setattr(orchestrator, "_apply_cell_result", lambda *args: None)
    monkeypatch.setattr(orchestrator.annotation_coordinator, "submit_pubtator", lambda job_id: None)
    monkeypatch.setattr(orchestrator.local_executor, "cleanup", lambda *args, **kwargs: None)
    annotation = orchestrator.PipelineOrchestrator._annotation_artifact(
        pipeline, run_id=run["id"], scope="custom", source={"artifact": ref.to_dict()},
        model_signature="test", callback_base_url="", progress_start=1, progress_end=99,
    )
    monkeypatch.setattr(orchestrator, "create_relation_job", lambda **kwargs: None)
    monkeypatch.setattr(orchestrator, "update_relation_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "get_relation_job", lambda job_id: {"id": job_id, "status": "completed"})
    monkeypatch.setattr(orchestrator.relation_executor, "submit", lambda job_id: None)
    monkeypatch.setattr(pipeline, "_ensure_final_annotation_artifact", lambda **kwargs: (ref.to_dict(), {}))
    relations = orchestrator.PipelineOrchestrator._relation_artifact(
        pipeline, run_id=run["id"], scope="custom", chunks={"artifact": ref.to_dict()},
        annotations={"artifact": ref.to_dict()}, model_signature="test",
        progress_start=1, progress_end=99,
    )
    assert ("summary_artifact" in annotation) is trusted
    assert ("summary_artifact" in relations) is trusted
    assert len(deleted) == (0 if trusted else 2)


def test_cancel_during_modal_submission_stops_returned_call(public_app, monkeypatch):
    _, pipeline, _ = public_app
    monkeypatch.setattr(pipeline, "_submit", lambda *args: None)
    run = pipeline.create_update_run(["999"])
    ref = orchestrator.ArtifactRef(key="source.jsonl.gz", size_bytes=1, sha256="abc")
    monkeypatch.setattr(orchestrator, "cached_json_pair", lambda **kwargs: None)
    monkeypatch.setattr(pipeline, "_annotation_backend", lambda url: ("modal", "https://example.invalid"))
    monkeypatch.setattr(orchestrator, "create_annotation_job", lambda **kwargs: None)
    monkeypatch.setattr(orchestrator, "update_annotation_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "get_annotation_job", lambda job_id: {"id": job_id, "status": "processing"})
    monkeypatch.setattr(orchestrator, "_cell_branch_ready", lambda job: False)
    monkeypatch.setattr(orchestrator, "get_artifact_store", lambda: SimpleNamespace(
        presign_get=lambda *args, **kwargs: "https://example.invalid/source",
        presign_put=lambda *args, **kwargs: "https://example.invalid/output",
    ))
    canceled = []

    def submitted_then_canceled(payload):
        # The updater times out before Modal returns the new function call ID.
        runtime.run_registry.set_private(run["id"], "cancel_requested", True)
        return "new-modal-call"

    monkeypatch.setattr(orchestrator.modal_executor, "submit", submitted_then_canceled)
    monkeypatch.setattr(orchestrator.modal_executor, "cancel", lambda call_id: canceled.append(call_id))
    monkeypatch.setattr(orchestrator.local_executor, "cleanup", lambda *args, **kwargs: None)

    def no_followup_worker(job_id):
        raise AssertionError("Canceled update launched another worker")

    monkeypatch.setattr(orchestrator.annotation_coordinator, "submit_pubtator", no_followup_worker)
    with pytest.raises(PermissionError, match="canceled or disabled"):
        orchestrator.PipelineOrchestrator._annotation_artifact(
            pipeline, run_id=run["id"], scope="custom", source={"artifact": ref.to_dict()},
            model_signature="test", callback_base_url="", progress_start=1, progress_end=99,
        )
    assert canceled == ["new-modal-call"]
