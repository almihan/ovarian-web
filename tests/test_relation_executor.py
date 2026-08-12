from __future__ import annotations

import asyncio
import gzip
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import backend.config as config_module
import backend.database.database as database
import backend.services.relation_executor as executor_module
import backend.storage.artifacts as artifacts_module
from backend.pipeline.entity_artifacts import iter_jsonl, sha256_path
from backend.pipeline.relation_extraction import PreparedChunk


def _write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            fileobj=raw,
            mode="wb",
            compresslevel=6,
            mtime=0,
        ) as output:
            for row in rows:
                output.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )


def _prepared(custom_id: str) -> PreparedChunk:
    return PreparedChunk(
        custom_id=custom_id,
        identity={"base": "paper", "chunk_id": int(custom_id.rsplit("-", 1)[-1])},
        tagged_text=(
            "[H1]Estradiol[/H1] increased [G1]FSHR[/G1] expression in "
            "[C1]granulosa cells[/C1]."
        ),
        entities={
            "H1": {"id": "H1", "obj": "hormone"},
            "G1": {"id": "G1", "obj": "gene"},
            "C1": {"id": "C1", "obj": "cell"},
        },
        eligible=True,
        valid_annotation_count=3,
        dropped_overlap_count=0,
    )


def test_gateway_calls_online_responses_and_validates_locally(monkeypatch) -> None:
    test_settings = replace(
        config_module.settings,
        openai_api_key="test-key",
        relation_model="gpt-5.4-nano",
        relation_max_output_tokens=1200,
        relation_reasoning_effort="none",
        relation_prompt_cache_key="ovarian-relations-v4",
        relation_enable_biosynthesis=False,
        relation_require_hormone_gene_cell_context=False,
    )
    monkeypatch.setattr(executor_module, "settings", test_settings)

    class FakeResponse:
        status = "completed"
        output_text = json.dumps(
            {
                "triples": [
                    {
                        "subject": "H1",
                        "predicate": "upregulation",
                        "object": "G1",
                        "cell_context": ["C1"],
                    },
                    {
                        "subject": "G1",
                        "predicate": "activation",
                        "object": "G2",
                        "cell_context": [],
                    },
                ]
            }
        )

        @staticmethod
        def model_dump() -> dict[str, Any]:
            return {
                "status": "completed",
                "usage": {
                    "input_tokens": 1000,
                    "input_tokens_details": {"cached_tokens": 800},
                    "output_tokens": 50,
                },
                "output": [],
            }

    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> FakeResponse:
            self.calls.append(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self) -> None:
            self.responses = FakeResponses()

    gateway = object.__new__(executor_module.OpenAIResponsesGateway)
    gateway.client = FakeClient()
    item = _prepared("r-0000000000")
    triples, usage = asyncio.run(gateway.extract(item, cache_shards=1))

    assert len(gateway.client.responses.calls) == 1
    request = gateway.client.responses.calls[0]
    assert request["model"] == "gpt-5.4-nano"
    assert request["store"] is False
    assert request["input"].endswith(item.tagged_text)
    assert request["text"]["format"]["strict"] is True
    assert request["prompt_cache_key"].startswith("ovarian-relations-v4:")
    assert triples == [
        {
            "subject": "H1",
            "predicate": "upregulation",
            "object": "G1",
            "cell_context": ["C1"],
        }
    ]
    assert usage == {
        "input_tokens": 1000,
        "cached_input_tokens": 800,
        "output_tokens": 50,
    }


def test_pending_window_bounds_concurrency_and_retries(monkeypatch, tmp_path: Path) -> None:
    test_settings = replace(
        config_module.settings,
        relation_concurrency=3,
        relation_max_request_retries=1,
        relation_retry_base_seconds=1,
        relation_progress_update_every=1,
        relation_window_size=500,
    )
    monkeypatch.setattr(executor_module, "settings", test_settings)

    job: dict[str, Any] = {
        "id": "online-window",
        "status": "processing",
        "stage": "preparing_online_requests",
        "progress": 5,
        "message": "",
        "stats": {},
        "started_at": database.utc_now(),
        "elapsed_seconds": 0.0,
    }

    def fake_get(job_id: str) -> dict[str, Any] | None:
        return dict(job) if job_id == job["id"] else None

    def fake_update(job_id: str, **fields: Any) -> dict[str, Any] | None:
        assert job_id == job["id"]
        job.update(fields)
        return dict(job)

    async def no_retry_delay(_stop: Any, _seconds: float) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(executor_module, "get_relation_job", fake_get)
    monkeypatch.setattr(executor_module, "update_relation_job", fake_update)
    monkeypatch.setattr(executor_module, "_sleep_with_stop", no_retry_delay)

    class RetryableRateLimit(RuntimeError):
        status_code = 429

    class FakeGateway:
        active = 0
        maximum_active = 0
        calls: dict[str, int] = {}

        async def extract(
            self,
            item: PreparedChunk,
            *,
            cache_shards: int,
        ) -> tuple[list[dict[str, Any]], dict[str, int]]:
            assert cache_shards == 1
            self.__class__.calls[item.custom_id] = (
                self.__class__.calls.get(item.custom_id, 0) + 1
            )
            self.__class__.active += 1
            self.__class__.maximum_active = max(
                self.__class__.maximum_active,
                self.__class__.active,
            )
            try:
                await asyncio.sleep(0.01)
                if (
                    item.custom_id == "r-0000000000"
                    and self.__class__.calls[item.custom_id] == 1
                ):
                    raise RetryableRateLimit("retry this request")
                return [], {
                    "input_tokens": 10,
                    "cached_input_tokens": 5,
                    "output_tokens": 2,
                }
            finally:
                self.__class__.active -= 1

    items = [_prepared(f"r-{index:010d}") for index in range(6)]
    state_path = tmp_path / "pending.state.json"
    state_path.write_text(
        json.dumps(
            {
                "execution": executor_module._ONLINE_STATE_VERSION,
                "window_start": 0,
                "cache_shards": 1,
                "checkpoint_started": False,
                "output_appended": False,
            }
        ),
        encoding="utf-8",
    )
    events_path = tmp_path / "pending.events.jsonl"

    executor = executor_module.RelationExecutor()
    try:
        relations, usage, attempts, retries = asyncio.run(
            executor._run_pending_window(
                job_id="online-window",
                gateway=FakeGateway(),
                items=items,
                state_path=state_path,
                events_path=events_path,
                processed_before=0,
                total_chunks=len(items),
                durable_counters={
                    "api_request_count": 0,
                    "retry_count": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                },
            )
        )
    finally:
        executor._pool.shutdown(wait=True)

    assert set(relations) == {item.custom_id for item in items}
    assert all(value == [] for value in relations.values())
    assert FakeGateway.maximum_active == 3
    assert attempts == 7
    assert retries == 1
    assert FakeGateway.calls["r-0000000000"] == 2
    assert usage == {
        "input_tokens": 60,
        "cached_input_tokens": 30,
        "output_tokens": 12,
    }
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert sum(event["type"] == "attempt_started" for event in events) == 7
    assert sum(event["type"] == "attempt_succeeded" for event in events) == 6
    assert sum(event["type"] == "attempt_failed" for event in events) == 1


def test_executor_resumes_successful_online_response_without_duplicate_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    test_settings = replace(
        config_module.settings,
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "database.sqlite",
        papers_dir=tmp_path / "data" / "papers",
        results_dir=tmp_path / "data" / "results",
        artifact_local_dir=tmp_path / "data" / "artifacts",
        cell_model_cache_dir=tmp_path / "data" / "model_cache",
        local_annotation_jobs_dir=tmp_path / "data" / "local_annotation_jobs",
        relation_jobs_dir=tmp_path / "data" / "relation_jobs",
        artifact_backend="local",
        artifact_prefix="test",
        openai_api_key="test-key",
        relation_model="gpt-5.4-nano",
        relation_window_size=500,
        relation_concurrency=2,
        relation_request_timeout_seconds=180,
        relation_max_request_retries=0,
        relation_retry_base_seconds=1,
        relation_progress_update_every=1,
        relation_prompt_cache_key="ovarian-relations-v4",
        relation_prompt_cache_shards=32,
        relation_enable_biosynthesis=False,
        relation_require_hormone_gene_cell_context=False,
    )
    test_settings.ensure_directories()
    monkeypatch.setattr(config_module, "settings", test_settings)
    monkeypatch.setattr(database, "settings", test_settings)
    monkeypatch.setattr(artifacts_module, "settings", test_settings)
    monkeypatch.setattr(executor_module, "settings", test_settings)

    database.init_database()
    store = artifacts_module.LocalArtifactStore(test_settings.artifact_local_dir)
    monkeypatch.setattr(executor_module, "get_artifact_store", lambda: store)

    text_1 = "Estradiol increased FSHR expression in granulosa cells."
    text_2 = "Granulosa cells were counted."
    identity_1 = {
        "base": "paper-1",
        "doc_key": "PMID1",
        "canonical_id": "pmid:1",
        "pmid": "1",
        "pmcid": None,
        "journal": "Test Journal",
        "pub_year": 2026,
        "section_type": "ABSTRACT",
        "chunk_id": 0,
    }
    identity_2 = {**identity_1, "chunk_id": 1}
    source_rows = [
        {**identity_1, "chunk": text_1},
        {**identity_2, "chunk": text_2},
    ]
    annotation_rows = [
        {
            **identity_1,
            "annotations": [
                {
                    "obj": "hormone",
                    "start": text_1.index("Estradiol"),
                    "end": text_1.index("Estradiol") + len("Estradiol"),
                    "mention": "Estradiol",
                    "concept_id": "MESH:D004958",
                    "hormone_id": "D004958",
                    "preferred_label": "Estradiol",
                },
                {
                    "obj": "gene",
                    "start": text_1.index("FSHR"),
                    "end": text_1.index("FSHR") + len("FSHR"),
                    "mention": "FSHR",
                    "concept_id": "NCBIGene:2492",
                    "gene_id": "2492",
                    "tax_id": "9606",
                    "tax_name": "Homo sapiens",
                    "preferred_label": "FSHR",
                },
                {
                    "obj": "cell",
                    "start": text_1.index("granulosa cells"),
                    "end": text_1.index("granulosa cells")
                    + len("granulosa cells"),
                    "mention": "granulosa cells",
                    "concept_id": "CL:0000501",
                    "preferred_label": "granulosa cell",
                },
            ],
        },
        {
            **identity_2,
            "annotations": [
                {
                    "obj": "cell",
                    "start": 0,
                    "end": len("Granulosa cells"),
                    "mention": "Granulosa cells",
                    "concept_id": "CL:0000501",
                    "preferred_label": "granulosa cell",
                }
            ],
        },
    ]

    source_path = tmp_path / "chunks.jsonl.gz"
    annotation_path = tmp_path / "entity_annotations.jsonl.gz"
    _write_gzip_jsonl(source_path, source_rows)
    _write_gzip_jsonl(annotation_path, annotation_rows)
    source_ref, _ = store.put_file(
        source_path,
        key="test/retrieval/chunks.jsonl.gz",
        content_type="application/gzip",
        sha256=sha256_path(source_path),
    )
    annotation_ref, _ = store.put_file(
        annotation_path,
        key="test/annotations/entity_annotations.jsonl.gz",
        content_type="application/gzip",
        sha256=sha256_path(annotation_path),
    )

    database.create_job(job_id="retrieval", input_type="pmid", query="1")
    database.update_job(
        "retrieval",
        status="completed",
        stage="completed",
        progress=100,
        paper_count=1,
    )
    database.create_annotation_job(
        job_id="annotations",
        source_job_id="retrieval",
        executor="local",
        model_signature="entity-signature",
        source_artifact_key=source_ref.key,
        source_artifact_sha256=source_ref.sha256,
        output_artifact_key=annotation_ref.key,
        summary_artifact_key="test/annotations/summary.json",
    )
    database.update_annotation_job(
        "annotations",
        status="completed",
        stage="completed",
        progress=100,
        paper_count=1,
        chunk_count=2,
        mention_count=4,
    )
    database.create_relation_job(
        job_id="relations",
        source_annotation_job_id="annotations",
        model_signature="relation-signature",
        source_chunks_artifact_key=source_ref.key,
        source_chunks_artifact_sha256=source_ref.sha256,
        source_annotation_artifact_key=annotation_ref.key,
        source_annotation_artifact_sha256=annotation_ref.sha256,
        output_artifact_key="test/relations/relations.jsonl.gz",
        summary_artifact_key="test/relations/summary.json",
    )
    database.update_relation_job("relations", paper_count=1, chunk_count=2)

    class FakeOnlineGateway:
        calls = 0
        closes = 0

        async def extract(
            self,
            item: PreparedChunk,
            *,
            cache_shards: int,
        ) -> tuple[list[dict[str, Any]], dict[str, int]]:
            self.__class__.calls += 1
            assert cache_shards == 1
            assert "[H1]Estradiol[/H1]" in item.tagged_text
            return [
                {
                    "subject": "H1",
                    "predicate": "upregulation",
                    "object": "G1",
                    "cell_context": ["C1"],
                }
            ], {
                "input_tokens": 1000,
                "cached_input_tokens": 800,
                "output_tokens": 50,
            }

        async def close(self) -> None:
            self.__class__.closes += 1

    monkeypatch.setattr(
        executor_module,
        "OpenAIResponsesGateway",
        FakeOnlineGateway,
    )
    executor = executor_module.RelationExecutor()
    try:
        actual_update_relation_job = executor_module.update_relation_job
        interrupted = False

        def interrupt_after_append(job_id: str, **fields: Any):
            nonlocal interrupted
            if not interrupted and fields.get("stage") == "checkpointed_relations":
                interrupted = True
                raise KeyboardInterrupt("simulated process interruption")
            return actual_update_relation_job(job_id, **fields)

        monkeypatch.setattr(
            executor_module,
            "update_relation_job",
            interrupt_after_append,
        )
        with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
            executor._run("relations")

        interrupted_job = database.get_relation_job("relations")
        assert interrupted_job is not None
        assert interrupted_job["status"] == "processing"
        assert interrupted_job["processed_chunk_count"] == 0
        assert FakeOnlineGateway.calls == 1
        work_root = test_settings.relation_jobs_dir / "relations"
        assert (work_root / "pending.events.jsonl").is_file()
        assert (work_root / "relations.partial.jsonl.gz").is_file()

        monkeypatch.setattr(
            executor_module,
            "update_relation_job",
            actual_update_relation_job,
        )
        executor._run("relations")
    finally:
        executor._pool.shutdown(wait=True)

    completed = database.get_relation_job("relations")
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["processed_chunk_count"] == 2
    assert completed["eligible_chunk_count"] == 1
    assert completed["api_request_count"] == 1
    assert completed["relation_count"] == 1
    assert completed["cell_context_count"] == 1
    assert completed["stats"]["execution_mode"] == "online_async_responses"
    assert completed["stats"]["window_count"] == 1
    assert completed["stats"]["retry_count"] == 0
    assert completed["stats"]["input_tokens"] == 1000
    assert completed["stats"]["cached_input_tokens"] == 800
    assert completed["stats"]["prompt_cache_rate"] == 0.8
    assert FakeOnlineGateway.calls == 1

    output_path = store.local_path("test/relations/relations.jsonl.gz")
    assert output_path is not None
    rows = list(iter_jsonl(output_path))
    assert len(rows) == 2
    assert "chunk" not in rows[0]
    assert rows[0]["relations"] == [
        {
            "subject": "H1",
            "predicate": "upregulation",
            "object": "G1",
            "cell_context": ["C1"],
        }
    ]
    assert {entity["id"] for entity in rows[0]["entities"]} == {
        "C1",
        "G1",
        "H1",
    }
    assert rows[1]["relations"] == []
    assert rows[1]["entities"] == []
    assert not (test_settings.relation_jobs_dir / "relations").exists()


def test_legacy_batch_job_is_not_resubmitted_online(monkeypatch) -> None:
    test_settings = replace(
        config_module.settings,
        openai_api_key="test-key",
        relation_model="gpt-5.4-nano",
    )
    monkeypatch.setattr(executor_module, "settings", test_settings)
    legacy_job = {
        "id": "legacy-batch",
        "status": "processing",
        "stage": "openai_batch_in_progress",
        "stats": {"batch_api": True},
        "remote_batch_id": "batch-old",
        "remote_input_file_id": "file-old",
        "batch_count": 1,
    }
    monkeypatch.setattr(
        executor_module,
        "get_relation_job",
        lambda job_id: dict(legacy_job) if job_id == "legacy-batch" else None,
    )

    class MustNotInitialize:
        def __init__(self) -> None:
            raise AssertionError("A legacy Batch job must not create online requests")

    monkeypatch.setattr(
        executor_module,
        "OpenAIResponsesGateway",
        MustNotInitialize,
    )
    executor = executor_module.RelationExecutor()
    try:
        with pytest.raises(RuntimeError, match="retired OpenAI Batch executor"):
            asyncio.run(executor._run_async("legacy-batch"))
    finally:
        executor._pool.shutdown(wait=True)
