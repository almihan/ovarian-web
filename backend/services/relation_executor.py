"""Resumable online Responses API executor for Stage 3 relation extraction.

The FastAPI process keeps one bounded source window in memory (500 chunks by
 default) and runs only a small configurable number of OpenAI Responses API
calls concurrently. Each request result is appended to a durable local journal
before it is counted, allowing local or Railway restarts to resume unfinished
chunks without an offline completion window.
"""

from __future__ import annotations

import asyncio
import gzip
import inspect
import json
import logging
import os
import random
import shutil
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests

from backend.config import settings
from backend.database.database import (
    get_relation_job,
    list_relation_jobs,
    update_relation_job,
    utc_now,
)
from backend.pipeline.entity_artifacts import iter_jsonl, sha256_path
from backend.pipeline.relation_extraction import (
    RELATION_OUTPUT_SCHEMA,
    RELATION_PIPELINE_VERSION,
    PreparedChunk,
    compact_json,
    effective_prompt_cache_shards,
    extract_response_text,
    is_hormone_gene_relation,
    output_row,
    prepare_chunk,
    prompt_cache_key_for_request,
    request_body,
    sanitize_triples,
)
from backend.storage.artifacts import ArtifactStore, get_artifact_store

logger = logging.getLogger(__name__)
_ONE_MIB = 1024 * 1024
_ONLINE_STATE_VERSION = "openai-responses-online-v1"
_USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens")
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class RelationExecutorStopping(Exception):
    """Internal signal used to pause resumable work during app shutdown."""


class RelationResponseError(RuntimeError):
    """A completed response that could not be accepted locally."""

    def __init__(
        self,
        message: str,
        *,
        usage: Mapping[str, int] | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.usage = _zero_usage()
        if usage:
            _add_usage(self.usage, usage)
        self.retryable = retryable


def _zero_usage() -> dict[str, int]:
    return {key: 0 for key in _USAGE_KEYS}


def _add_usage(target: dict[str, int], source: Mapping[str, Any] | None) -> None:
    if not source:
        return
    for key in _USAGE_KEYS:
        target[key] = int(target.get(key) or 0) + int(source.get(key) or 0)


def _elapsed_since(value: object) -> float:
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def _model_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump()
        return dict(payload) if isinstance(payload, Mapping) else {}
    return {}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _write_prepared(path: Path, items: Sequence[PreparedChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                fileobj=raw,
                mode="wb",
                compresslevel=6,
                mtime=0,
            ) as output:
                for item in items:
                    row = {
                        "custom_id": item.custom_id,
                        "identity": item.identity,
                        "tagged_text": item.tagged_text,
                        "entities": item.entities,
                        "eligible": item.eligible,
                        "valid_annotation_count": item.valid_annotation_count,
                        "dropped_overlap_count": item.dropped_overlap_count,
                    }
                    output.write(compact_json(row).encode("utf-8") + b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_prepared(path: Path) -> list[PreparedChunk]:
    items: list[PreparedChunk] = []
    for row in iter_jsonl(path):
        identity = row.get("identity")
        entities = row.get("entities")
        if not isinstance(identity, dict) or not isinstance(entities, dict):
            raise ValueError("The pending relation window is invalid.")
        items.append(
            PreparedChunk(
                custom_id=str(row.get("custom_id") or ""),
                identity=identity,
                tagged_text=str(row.get("tagged_text") or ""),
                entities={str(key): dict(value) for key, value in entities.items()},
                eligible=bool(row.get("eligible")),
                valid_annotation_count=int(row.get("valid_annotation_count") or 0),
                dropped_overlap_count=int(row.get("dropped_overlap_count") or 0),
            )
        )
    return items


def _append_gzip_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Append one deterministic gzip member and fsync it before checkpointing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("ab") as raw:
        with gzip.GzipFile(
            filename="",
            fileobj=raw,
            mode="wb",
            compresslevel=6,
            mtime=0,
        ) as output:
            for row in rows:
                output.write(compact_json(row).encode("utf-8") + b"\n")
                count += 1
        raw.flush()
        os.fsync(raw.fileno())
    return count


def _truncate_and_fsync(path: Path, size: int) -> None:
    if size < 0:
        raise ValueError("A checkpoint byte boundary cannot be negative.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.truncate(size)
        handle.flush()
        os.fsync(handle.fileno())


def _identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        "" if row.get(field) is None else str(row.get(field))
        for field in (
            "base",
            "doc_key",
            "canonical_id",
            "pmid",
            "pmcid",
            "section_type",
            "chunk_id",
        )
    )


def _usage_from_response(body: Mapping[str, Any]) -> dict[str, int]:
    usage = body.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    details = usage.get("input_tokens_details")
    details = details if isinstance(details, Mapping) else {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cached_input_tokens": int(details.get("cached_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }


def _response_refusal(body: Mapping[str, Any]) -> str | None:
    output = body.get("output")
    if not isinstance(output, list):
        return None
    for item in output:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "refusal":
                continue
            refusal = part.get("refusal")
            if isinstance(refusal, str) and refusal.strip():
                return refusal.strip()
    return None


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RelationResponseError):
        return exc.retryable
    status = _status_code(exc)
    if status in _RETRYABLE_STATUS_CODES:
        return True
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    return any(
        marker in name or marker in message
        for marker in (
            "timeout",
            "connection",
            "temporar",
            "rate limit",
            "ratelimit",
            "overloaded",
        )
    )


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw_ms = headers.get("retry-after-ms")
    if raw_ms is not None:
        try:
            return max(0.0, min(120.0, float(raw_ms) / 1000.0))
        except (TypeError, ValueError):
            pass
    raw = headers.get("retry-after")
    if raw is not None:
        try:
            return max(0.0, min(120.0, float(raw)))
        except (TypeError, ValueError):
            return None
    return None


async def _sleep_with_stop(stop: threading.Event, seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        if stop.is_set():
            raise RelationExecutorStopping(
                "Application shutdown paused relation extraction."
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(0.5, remaining))


class OpenAIResponsesGateway:
    """One reusable asynchronous OpenAI client for all requests in a job."""

    def __init__(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - deployment guard
            raise RuntimeError(
                "The openai package is missing. Install requirements.txt."
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": settings.openai_api_key,
            "timeout": float(settings.relation_request_timeout_seconds),
            # Application-level retries are explicit, journaled, and counted.
            "max_retries": 0,
        }
        if settings.openai_organization:
            kwargs["organization"] = settings.openai_organization
        if settings.openai_project:
            kwargs["project"] = settings.openai_project
        self.client = AsyncOpenAI(**kwargs)

    async def close(self) -> None:
        close = getattr(self.client, "close", None) or getattr(
            self.client, "aclose", None
        )
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    async def extract(
        self,
        item: PreparedChunk,
        *,
        cache_shards: int,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        body = request_body(
            tagged_text=item.tagged_text,
            model=settings.relation_model,
            max_output_tokens=settings.relation_max_output_tokens,
            reasoning_effort=settings.relation_reasoning_effort,
            cache_key=prompt_cache_key_for_request(
                settings.relation_prompt_cache_key,
                custom_id=item.custom_id,
                shard_count=cache_shards,
            ),
            enable_biosynthesis=settings.relation_enable_biosynthesis,
        )
        response = await self.client.responses.create(**body)
        payload = _model_dict(response)
        usage = _usage_from_response(payload)
        status = str(
            getattr(response, "status", None) or payload.get("status") or ""
        ).casefold()
        if status == "incomplete":
            raise RelationResponseError(
                f"OpenAI returned an incomplete response: {payload.get('incomplete_details')}",
                usage=usage,
                retryable=True,
            )
        refusal = _response_refusal(payload)
        if refusal:
            raise RelationResponseError(
                f"OpenAI refused the relation request: {refusal}",
                usage=usage,
                retryable=False,
            )
        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text:
            text = extract_response_text(payload)
        if not text:
            raise RelationResponseError(
                "OpenAI returned no structured response text.",
                usage=usage,
                retryable=True,
            )
        try:
            parsed = json.loads(text)
            triples = sanitize_triples(
                parsed,
                entities=item.entities,
                enable_biosynthesis=settings.relation_enable_biosynthesis,
                require_hormone_gene_cell_context=(
                    settings.relation_require_hormone_gene_cell_context
                ),
            )
        except Exception as exc:
            raise RelationResponseError(
                f"The structured relation response failed local validation: {exc}",
                usage=usage,
                retryable=True,
            ) from exc
        return triples, usage


@dataclass(slots=True)
class _JournalState:
    attempts_by_id: dict[str, int] = field(default_factory=dict)
    relations_by_id: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=_zero_usage)
    attempt_count: int = 0
    last_errors: dict[str, str] = field(default_factory=dict)

    @property
    def retry_count(self) -> int:
        return sum(max(0, value - 1) for value in self.attempts_by_id.values())


def _load_event_journal(path: Path, *, valid_ids: set[str]) -> _JournalState:
    state = _JournalState()
    if not path.is_file():
        return state

    file_size = path.stat().st_size
    safe_end = 0
    truncate_to: int | None = None
    with path.open("rb") as handle:
        while True:
            line_start = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            line_end = handle.tell()
            if not raw.endswith(b"\n"):
                truncate_to = safe_end
                break
            try:
                event = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if line_end == file_size:
                    truncate_to = safe_end
                    break
                raise ValueError(
                    f"The online request journal is invalid near byte {line_start}."
                ) from exc
            if not isinstance(event, Mapping):
                raise ValueError("The online request journal contains a non-object row.")
            custom_id = str(event.get("custom_id") or "")
            if custom_id not in valid_ids:
                raise ValueError(
                    f"The online request journal contains unexpected ID {custom_id!r}."
                )
            kind = str(event.get("type") or "")
            if kind == "attempt_started":
                attempt = max(1, int(event.get("attempt") or 1))
                state.attempts_by_id[custom_id] = max(
                    int(state.attempts_by_id.get(custom_id) or 0), attempt
                )
                state.attempt_count += 1
            elif kind in {"attempt_failed", "attempt_succeeded"}:
                usage = event.get("usage")
                if isinstance(usage, Mapping):
                    _add_usage(state.usage, usage)
                if kind == "attempt_succeeded":
                    triples = event.get("triples")
                    if not isinstance(triples, list):
                        raise ValueError(
                            "A successful online request journal row has no triples."
                        )
                    state.relations_by_id[custom_id] = [
                        dict(value) for value in triples if isinstance(value, Mapping)
                    ]
                    state.last_errors.pop(custom_id, None)
                else:
                    state.last_errors[custom_id] = str(event.get("error") or "")
            else:
                raise ValueError(f"Unsupported online request journal event {kind!r}.")
            safe_end = line_end

    if truncate_to is not None:
        _truncate_and_fsync(path, truncate_to)
    return state


class _EventJournal:
    """Append-only, fsynced request journal for one local source window."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")

    def append(self, event: Mapping[str, Any]) -> None:
        self._handle.write(compact_json(event) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


class RelationExecutor:
    """One bounded, resumable Stage 3 worker per web process."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="railway-stage3",
        )
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
                key,
                expires_seconds=settings.artifact_presigned_ttl_seconds,
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
    ) -> dict[str, Any] | None:
        job = get_relation_job(job_id)
        if job is None or job.get("status") not in {"queued", "processing"}:
            return job
        return update_relation_job(
            job_id,
            status="processing",
            stage=stage,
            progress=max(int(job.get("progress") or 0), min(99, int(progress))),
            message=message,
            stats=self._merge_stats(job, stats or {}),
            elapsed_seconds=max(
                float(job.get("elapsed_seconds") or 0.0),
                _elapsed_since(job.get("started_at")),
            ),
            **fields,
        )

    def _fail(self, job_id: str, exc: Exception) -> None:
        job = get_relation_job(job_id)
        if job is None or job.get("status") == "completed":
            return
        update_relation_job(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            message="Relation extraction could not be completed.",
            elapsed_seconds=max(
                float(job.get("elapsed_seconds") or 0.0),
                _elapsed_since(job.get("started_at")),
            ),
            completed_at=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
        )

    def submit(self, job_id: str) -> bool:
        if self._stop.is_set():
            return False
        job = get_relation_job(job_id)
        if job is None or job.get("status") not in {"queued", "processing"}:
            return False
        with self._guard:
            if job_id in self._running:
                return False
            self._running.add(job_id)
        try:
            future = self._pool.submit(self._run, job_id)
        except Exception:
            with self._guard:
                self._running.discard(job_id)
            raise
        future.add_done_callback(
            lambda completed, current=job_id: self._done(current, completed)
        )
        return True

    def _done(self, job_id: str, future: Future[None]) -> None:
        with self._guard:
            self._running.discard(job_id)
        try:
            future.result()
        except Exception:
            logger.exception("Stage 3 relation job %s exited with an error", job_id)

    def resume_active_jobs(self) -> None:
        if not settings.relation_configured:
            return
        for job in list_relation_jobs(50):
            if job.get("status") in {"queued", "processing"}:
                self.submit(str(job["id"]))

    @staticmethod
    def _advance_and_verify(
        source_iter: Iterable[dict[str, Any]],
        annotation_iter: Iterable[dict[str, Any]],
        count: int,
        expected: Sequence[PreparedChunk] | None = None,
    ) -> None:
        source_iterator = iter(source_iter)
        annotation_iterator = iter(annotation_iter)
        for index in range(count):
            source = next(source_iterator, None)
            annotations = next(annotation_iterator, None)
            if source is None or annotations is None:
                raise ValueError("Stage 1 and Stage 2 artifacts ended during resume.")
            if _identity(source) != _identity(annotations):
                raise ValueError("Stage 1 and Stage 2 artifacts are not aligned.")
            if expected is not None and _identity(annotations) != _identity(
                expected[index].identity
            ):
                raise ValueError(
                    "The pending relation window no longer matches its source."
                )

    async def _run_pending_window(
        self,
        *,
        job_id: str,
        gateway: OpenAIResponsesGateway,
        items: Sequence[PreparedChunk],
        state_path: Path,
        events_path: Path,
        processed_before: int,
        total_chunks: int,
        durable_counters: Mapping[str, int],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], int, int]:
        state = _read_json_object(state_path)
        if state.get("execution") != _ONLINE_STATE_VERSION:
            raise RuntimeError(
                "This unfinished Stage 3 checkpoint uses an unsupported executor. "
                "Start a new relation job."
            )
        if int(state.get("window_start") or 0) != processed_before:
            raise ValueError("The online request checkpoint has the wrong window start.")

        cache_shards = max(1, int(state.get("cache_shards") or 1))
        eligible_items = [item for item in items if item.eligible]
        item_lookup = {item.custom_id: item for item in eligible_items}
        journal_state = _load_event_journal(
            events_path,
            valid_ids=set(item_lookup),
        )
        unresolved = [
            item
            for item in eligible_items
            if item.custom_id not in journal_state.relations_by_id
        ]
        max_attempts = settings.relation_max_request_retries + 1
        exhausted = [
            item.custom_id
            for item in unresolved
            if int(journal_state.attempts_by_id.get(item.custom_id) or 0)
            >= max_attempts
        ]
        if exhausted:
            raise RuntimeError(
                f"{len(exhausted)} online requests exhausted their configured attempts "
                "before this resume."
            )

        skipped_in_window = len(items) - len(eligible_items)
        state_lock = asyncio.Lock()
        journal = _EventJournal(events_path)
        permanent_failures: dict[str, str] = {}
        in_flight = 0
        retrying = 0
        last_report_time = 0.0
        last_report_completed = -1

        def live_usage() -> dict[str, int]:
            return {
                key: int(durable_counters.get(key) or 0)
                + int(journal_state.usage.get(key) or 0)
                for key in _USAGE_KEYS
            }

        def report(*, force: bool = False) -> None:
            nonlocal last_report_time, last_report_completed
            now = time.monotonic()
            completed = len(journal_state.relations_by_id)
            completed_delta = completed - last_report_completed
            if not force:
                enough_items = (
                    completed_delta >= settings.relation_progress_update_every
                )
                enough_time = now - last_report_time >= 2.0
                if not enough_items and not enough_time:
                    return
            last_report_time = now
            last_report_completed = completed
            remaining = max(
                0,
                len(eligible_items) - completed - len(permanent_failures),
            )
            equivalent = processed_before + skipped_in_window + completed
            progress = 5 + round(
                89 * min(total_chunks, equivalent) / max(1, total_chunks)
            )
            usage = live_usage()
            input_tokens = usage["input_tokens"]
            cache_rate = (
                usage["cached_input_tokens"] / input_tokens
                if input_tokens
                else 0.0
            )
            attempts = int(durable_counters.get("api_request_count") or 0) + int(
                journal_state.attempt_count
            )
            retries = int(durable_counters.get("retry_count") or 0) + int(
                journal_state.retry_count
            )
            self._update(
                job_id,
                stage="openai_responses_in_progress",
                progress=progress,
                message=(
                    f"OpenAI Responses: {completed:,}/{len(eligible_items):,} "
                    f"eligible chunks completed in the current {len(items):,}-chunk "
                    f"window ({min(settings.relation_concurrency, max(1, len(eligible_items))):,} concurrent)."
                ),
                stats={
                    "execution_mode": "online_async_responses",
                    "window_size": settings.relation_window_size,
                    "concurrency": settings.relation_concurrency,
                    "current_window_chunk_count": len(items),
                    "current_window_eligible_chunk_count": len(eligible_items),
                    "current_window_skipped_chunk_count": skipped_in_window,
                    "current_window_completed_requests": completed,
                    "current_window_remaining_requests": remaining,
                    "current_window_in_flight_requests": in_flight,
                    "current_window_retrying_requests": retrying,
                    "live_api_request_count": attempts,
                    "live_retry_count": retries,
                    "live_input_tokens": usage["input_tokens"],
                    "live_cached_input_tokens": usage["cached_input_tokens"],
                    "live_output_tokens": usage["output_tokens"],
                    "prompt_cache_rate": round(cache_rate, 6),
                    "prompt_cache_shards": cache_shards,
                },
            )

        async def append_event(event: Mapping[str, Any]) -> None:
            async with state_lock:
                journal.append(event)

        async def worker(queue: "asyncio.Queue[PreparedChunk | None]") -> None:
            nonlocal in_flight, retrying
            while True:
                item = await queue.get()
                if item is None:
                    return
                custom_id = item.custom_id
                while custom_id not in journal_state.relations_by_id:
                    if self._stop.is_set():
                        raise RelationExecutorStopping(
                            "Application shutdown paused relation extraction."
                        )
                    prior_attempts = int(
                        journal_state.attempts_by_id.get(custom_id) or 0
                    )
                    if prior_attempts >= max_attempts:
                        permanent_failures[custom_id] = str(
                            journal_state.last_errors.get(custom_id)
                            or "Maximum request attempts were exhausted."
                        )
                        break
                    attempt = prior_attempts + 1
                    await append_event(
                        {
                            "type": "attempt_started",
                            "custom_id": custom_id,
                            "attempt": attempt,
                            "created_at": utc_now(),
                        }
                    )
                    journal_state.attempts_by_id[custom_id] = attempt
                    journal_state.attempt_count += 1
                    in_flight += 1
                    report()
                    try:
                        triples, usage = await gateway.extract(
                            item,
                            cache_shards=cache_shards,
                        )
                    except asyncio.CancelledError:
                        in_flight = max(0, in_flight - 1)
                        raise
                    except Exception as exc:
                        in_flight = max(0, in_flight - 1)
                        usage = (
                            exc.usage
                            if isinstance(exc, RelationResponseError)
                            else _zero_usage()
                        )
                        retryable = _is_retryable(exc)
                        error = f"{type(exc).__name__}: {exc}"
                        await append_event(
                            {
                                "type": "attempt_failed",
                                "custom_id": custom_id,
                                "attempt": attempt,
                                "retryable": retryable,
                                "error": error,
                                "usage": usage,
                                "created_at": utc_now(),
                            }
                        )
                        _add_usage(journal_state.usage, usage)
                        journal_state.last_errors[custom_id] = error
                        if not retryable or attempt >= max_attempts:
                            permanent_failures[custom_id] = error
                            report(force=True)
                            break
                        retrying += 1
                        report(force=True)
                        delay = _retry_after(exc)
                        if delay is None:
                            base = float(settings.relation_retry_base_seconds)
                            delay = min(60.0, base * (2 ** max(0, attempt - 1)))
                            delay += random.random() * base
                        try:
                            await _sleep_with_stop(self._stop, delay)
                        finally:
                            retrying = max(0, retrying - 1)
                        continue
                    else:
                        in_flight = max(0, in_flight - 1)
                        await append_event(
                            {
                                "type": "attempt_succeeded",
                                "custom_id": custom_id,
                                "attempt": attempt,
                                "triples": triples,
                                "usage": usage,
                                "created_at": utc_now(),
                            }
                        )
                        journal_state.relations_by_id[custom_id] = triples
                        journal_state.last_errors.pop(custom_id, None)
                        _add_usage(journal_state.usage, usage)
                        report()
                        break

        try:
            report(force=True)
            if unresolved:
                queue: asyncio.Queue[PreparedChunk | None] = asyncio.Queue()
                for item in unresolved:
                    queue.put_nowait(item)
                worker_count = min(settings.relation_concurrency, len(unresolved))
                for _ in range(worker_count):
                    queue.put_nowait(None)
                workers = [
                    asyncio.create_task(worker(queue)) for _ in range(worker_count)
                ]
                try:
                    await asyncio.gather(*workers)
                except BaseException:
                    for task in workers:
                        task.cancel()
                    await asyncio.gather(*workers, return_exceptions=True)
                    raise
            report(force=True)
        finally:
            journal.close()

        if permanent_failures:
            example_id, example_error = next(iter(permanent_failures.items()))
            raise RuntimeError(
                f"{len(permanent_failures)} online relation requests failed after "
                f"up to {max_attempts} attempts. Example {example_id}: {example_error}"
            )
        missing = set(item_lookup) - set(journal_state.relations_by_id)
        if missing:
            raise RuntimeError(
                f"{len(missing)} eligible online requests ended without a result."
            )
        return (
            journal_state.relations_by_id,
            journal_state.usage,
            journal_state.attempt_count,
            journal_state.retry_count,
        )

    @staticmethod
    def _stats_snapshot(
        counters: Mapping[str, Any],
        *,
        by_predicate: Mapping[str, Any],
        by_direction: Mapping[str, Any],
    ) -> dict[str, Any]:
        input_tokens = int(counters.get("input_tokens") or 0)
        cached_tokens = int(counters.get("cached_input_tokens") or 0)
        return {
            **dict(counters),
            "relations_by_predicate": dict(by_predicate),
            "relations_by_direction": dict(by_direction),
            "prompt_cache_rate": round(
                cached_tokens / input_tokens if input_tokens else 0.0,
                6,
            ),
            "model": settings.relation_model,
            "execution_mode": "online_async_responses",
            "window_size": settings.relation_window_size,
            "concurrency": settings.relation_concurrency,
            "request_timeout_seconds": settings.relation_request_timeout_seconds,
            "max_request_retries": settings.relation_max_request_retries,
            "prompt_cache_shards": settings.relation_prompt_cache_shards,
            "biosynthesis_enabled": settings.relation_enable_biosynthesis,
            "cell_context_required": (
                settings.relation_require_hormone_gene_cell_context
            ),
        }

    def _run(self, job_id: str) -> None:
        try:
            asyncio.run(self._run_async(job_id))
        except RelationExecutorStopping:
            logger.info(
                "Relation extraction job %s paused for application shutdown",
                job_id,
            )
        except Exception as exc:
            logger.exception("Relation extraction job %s failed", job_id)
            self._fail(job_id, exc)
            shutil.rmtree(settings.relation_jobs_dir / job_id, ignore_errors=True)

    async def _run_async(self, job_id: str) -> None:
        job = get_relation_job(job_id)
        if job is None or job.get("status") not in {"queued", "processing"}:
            return
        if not settings.relation_configured:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        legacy_stats = job.get("stats")
        legacy_stats = legacy_stats if isinstance(legacy_stats, Mapping) else {}
        if (
            job.get("remote_batch_id")
            or job.get("remote_input_file_id")
            or int(job.get("batch_count") or 0) > 0
            or bool(legacy_stats.get("batch_api"))
        ):
            raise RuntimeError(
                "This unfinished job was created by the retired OpenAI Batch "
                "executor and will not be resubmitted online. Start a new Stage 3 job."
            )

        work_root = settings.relation_jobs_dir / job_id
        work_root.mkdir(parents=True, exist_ok=True)
        partial_path = work_root / "relations.partial.jsonl.gz"
        pending_path = work_root / "pending.jsonl.gz"
        pending_state_path = work_root / "pending.state.json"
        events_path = work_root / "pending.events.jsonl"

        if not job.get("started_at"):
            job = update_relation_job(
                job_id,
                status="processing",
                stage="preparing_sources",
                progress=2,
                message="Preparing aligned Stage 1 text and Stage 2 entities...",
                started_at=utc_now(),
                completed_at=None,
                error=None,
            ) or job

        processed = int(job.get("processed_chunk_count") or 0)
        if processed == 0 and not pending_path.exists():
            partial_path.unlink(missing_ok=True)
            pending_state_path.unlink(missing_ok=True)
            events_path.unlink(missing_ok=True)

        gateway: OpenAIResponsesGateway | None = None
        try:
            store = get_artifact_store()
            with tempfile.TemporaryDirectory(
                prefix=f"ovarian-relations-{job_id}-"
            ) as temp:
                temp_root = Path(temp)
                chunks_path = temp_root / "chunks.jsonl.gz"
                annotations_path = temp_root / "entity_annotations.jsonl.gz"
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

                total_chunks = int(job.get("chunk_count") or 0)
                if total_chunks <= 0:
                    total_chunks = sum(1 for _ in iter_jsonl(annotations_path))
                    job = update_relation_job(job_id, chunk_count=total_chunks) or job
                if total_chunks == 0:
                    _append_gzip_rows(partial_path, [])

                source_iterator = iter_jsonl(chunks_path)
                annotation_iterator = iter_jsonl(annotations_path)
                self._advance_and_verify(
                    source_iterator,
                    annotation_iterator,
                    processed,
                )

                gateway = OpenAIResponsesGateway()
                stats = dict(job.get("stats") or {})
                counters: dict[str, int] = {
                    "eligible_chunk_count": int(
                        job.get("eligible_chunk_count") or 0
                    ),
                    "skipped_chunk_count": int(
                        stats.get("skipped_chunk_count") or 0
                    ),
                    "dropped_overlap_count": int(
                        stats.get("dropped_overlap_count") or 0
                    ),
                    "relation_count": int(job.get("relation_count") or 0),
                    "cell_context_count": int(
                        job.get("cell_context_count") or 0
                    ),
                    "hormone_gene_relation_count": int(
                        stats.get("hormone_gene_relation_count") or 0
                    ),
                    "hormone_gene_without_context_count": int(
                        stats.get("hormone_gene_without_context_count") or 0
                    ),
                    "api_request_count": int(job.get("api_request_count") or 0),
                    "retry_count": int(stats.get("retry_count") or 0),
                    "window_count": int(stats.get("window_count") or 0),
                    "input_tokens": int(stats.get("input_tokens") or 0),
                    "cached_input_tokens": int(
                        stats.get("cached_input_tokens") or 0
                    ),
                    "output_tokens": int(stats.get("output_tokens") or 0),
                }
                by_predicate = dict(stats.get("relations_by_predicate") or {})
                by_direction = dict(stats.get("relations_by_direction") or {})

                pending_items: list[PreparedChunk] | None = None
                if pending_path.is_file():
                    if not pending_state_path.is_file():
                        raise ValueError(
                            "A pending relation window exists without checkpoint state."
                        )
                    candidate_items = _read_prepared(pending_path)
                    pending_state = _read_json_object(pending_state_path)
                    if pending_state.get("execution") != _ONLINE_STATE_VERSION:
                        raise RuntimeError(
                            "This active relation job uses an unsupported checkpoint "
                            "format. Start a new Stage 3 job."
                        )
                    window_start = int(pending_state.get("window_start") or 0)
                    window_end = window_start + len(candidate_items)
                    if window_start == processed:
                        pending_items = candidate_items
                        self._advance_and_verify(
                            source_iterator,
                            annotation_iterator,
                            len(pending_items),
                            pending_items,
                        )
                    elif (
                        window_end == processed
                        and bool(pending_state.get("output_appended"))
                    ):
                        pending_path.unlink(missing_ok=True)
                        pending_state_path.unlink(missing_ok=True)
                        events_path.unlink(missing_ok=True)
                    else:
                        raise ValueError(
                            "The pending relation checkpoint does not match SQLite progress."
                        )

                row_index = processed
                while processed < total_chunks:
                    if self._stop.is_set():
                        raise RelationExecutorStopping(
                            "Application shutdown paused relation extraction."
                        )
                    if pending_items is None:
                        items: list[PreparedChunk] = []
                        for _ in range(settings.relation_window_size):
                            source_row = next(source_iterator, None)
                            annotation_row = next(annotation_iterator, None)
                            if source_row is None and annotation_row is None:
                                break
                            if source_row is None or annotation_row is None:
                                raise ValueError(
                                    "Stage 1 and Stage 2 artifacts contain different row counts."
                                )
                            if _identity(source_row) != _identity(annotation_row):
                                raise ValueError(
                                    f"Stage 1 and Stage 2 are not aligned at row {row_index + 1}."
                                )
                            items.append(
                                prepare_chunk(
                                    row_index=row_index,
                                    source_row=source_row,
                                    annotation_row=annotation_row,
                                    enable_biosynthesis=(
                                        settings.relation_enable_biosynthesis
                                    ),
                                )
                            )
                            row_index += 1
                        if not items:
                            break
                        _write_prepared(pending_path, items)
                        events_path.unlink(missing_ok=True)
                        _write_json_atomic(
                            pending_state_path,
                            {
                                "execution": _ONLINE_STATE_VERSION,
                                "window_start": processed,
                                "cache_shards": effective_prompt_cache_shards(
                                    sum(item.eligible for item in items),
                                    maximum_shards=(
                                        settings.relation_prompt_cache_shards
                                    ),
                                ),
                                "checkpoint_started": False,
                                "output_appended": False,
                            },
                        )
                    else:
                        items = pending_items
                        pending_items = None

                    eligible = sum(item.eligible for item in items)
                    skipped_in_window = len(items) - eligible
                    dropped_in_window = sum(
                        item.dropped_overlap_count for item in items
                    )
                    self._update(
                        job_id,
                        stage="preparing_online_requests",
                        progress=5 + round(
                            89 * processed / max(1, total_chunks)
                        ),
                        message=(
                            f"Prepared {len(items):,} chunks; {eligible:,} require "
                            f"online relation requests with up to "
                            f"{settings.relation_concurrency:,} running concurrently."
                        ),
                        stats={
                            "execution_mode": "online_async_responses",
                            "window_size": settings.relation_window_size,
                            "concurrency": settings.relation_concurrency,
                            "current_window_eligible_chunk_count": eligible,
                            "current_window_skipped_chunk_count": skipped_in_window,
                        },
                    )

                    relations_by_id: dict[str, list[dict[str, Any]]] = {}
                    window_usage = _zero_usage()
                    attempt_count = 0
                    retry_count = 0
                    if eligible:
                        (
                            relations_by_id,
                            window_usage,
                            attempt_count,
                            retry_count,
                        ) = await self._run_pending_window(
                            job_id=job_id,
                            gateway=gateway,
                            items=items,
                            state_path=pending_state_path,
                            events_path=events_path,
                            processed_before=processed,
                            total_chunks=total_chunks,
                            durable_counters=counters,
                        )

                    rows: list[dict[str, Any]] = []
                    window_relation_count = 0
                    window_cell_context_count = 0
                    window_hg_count = 0
                    window_hg_without_context = 0
                    window_by_predicate: dict[str, int] = {}
                    window_by_direction: dict[str, int] = {}
                    for item in items:
                        triples = relations_by_id.get(item.custom_id, [])
                        rows.append(output_row(item, triples))
                        window_relation_count += len(triples)
                        for triple in triples:
                            predicate = str(triple["predicate"])
                            direction = (
                                f"{triple['subject'][0]}->{triple['object'][0]}"
                            )
                            window_by_predicate[predicate] = (
                                window_by_predicate.get(predicate, 0) + 1
                            )
                            window_by_direction[direction] = (
                                window_by_direction.get(direction, 0) + 1
                            )
                            if is_hormone_gene_relation(
                                str(triple["subject"]),
                                str(triple["object"]),
                            ):
                                window_hg_count += 1
                                if triple.get("cell_context"):
                                    window_cell_context_count += 1
                                else:
                                    window_hg_without_context += 1

                    checkpoint_state = _read_json_object(pending_state_path)
                    if int(checkpoint_state.get("window_start") or 0) != processed:
                        raise ValueError(
                            "The relation output checkpoint has the wrong window start."
                        )
                    if bool(checkpoint_state.get("output_appended")):
                        expected_after = int(
                            checkpoint_state.get("partial_size_after") or -1
                        )
                        actual_after = (
                            partial_path.stat().st_size
                            if partial_path.exists()
                            else 0
                        )
                        if expected_after < 0 or actual_after != expected_after:
                            raise ValueError(
                                "The durable relation output size does not match its checkpoint."
                            )
                    else:
                        before_value = checkpoint_state.get("partial_size_before")
                        if before_value is None:
                            before_size = (
                                partial_path.stat().st_size
                                if partial_path.exists()
                                else 0
                            )
                            checkpoint_state.update(
                                {
                                    "checkpoint_started": True,
                                    "partial_size_before": before_size,
                                }
                            )
                            _write_json_atomic(
                                pending_state_path,
                                checkpoint_state,
                            )
                        else:
                            before_size = int(before_value)
                            actual_size = (
                                partial_path.stat().st_size
                                if partial_path.exists()
                                else 0
                            )
                            if actual_size != before_size:
                                _truncate_and_fsync(partial_path, before_size)
                        appended = _append_gzip_rows(partial_path, rows)
                        if appended != len(items):
                            raise RuntimeError(
                                "Could not checkpoint every relation row."
                            )
                        checkpoint_state.update(
                            {
                                "checkpoint_started": True,
                                "output_appended": True,
                                "partial_size_after": partial_path.stat().st_size,
                            }
                        )
                        _write_json_atomic(
                            pending_state_path,
                            checkpoint_state,
                        )

                    # Advance counters only after the source window is durable.
                    processed += len(items)
                    counters["eligible_chunk_count"] += eligible
                    counters["skipped_chunk_count"] += skipped_in_window
                    counters["dropped_overlap_count"] += dropped_in_window
                    counters["relation_count"] += window_relation_count
                    counters["cell_context_count"] += window_cell_context_count
                    counters["hormone_gene_relation_count"] += window_hg_count
                    counters["hormone_gene_without_context_count"] += (
                        window_hg_without_context
                    )
                    counters["api_request_count"] += attempt_count
                    counters["retry_count"] += retry_count
                    counters["window_count"] += 1
                    _add_usage(counters, window_usage)
                    for key, value in window_by_predicate.items():
                        by_predicate[key] = int(by_predicate.get(key) or 0) + value
                    for key, value in window_by_direction.items():
                        by_direction[key] = int(by_direction.get(key) or 0) + value

                    stats_snapshot = self._stats_snapshot(
                        counters,
                        by_predicate=by_predicate,
                        by_direction=by_direction,
                    )
                    update_relation_job(
                        job_id,
                        status="processing",
                        stage="checkpointed_relations",
                        progress=5 + round(
                            89 * processed / max(1, total_chunks)
                        ),
                        message=(
                            f"Processed {processed:,}/{total_chunks:,} chunks and "
                            f"validated {counters['relation_count']:,} relations."
                        ),
                        processed_chunk_count=processed,
                        eligible_chunk_count=counters["eligible_chunk_count"],
                        relation_count=counters["relation_count"],
                        cell_context_count=counters["cell_context_count"],
                        api_request_count=counters["api_request_count"],
                        stats=stats_snapshot,
                        elapsed_seconds=_elapsed_since(
                            (get_relation_job(job_id) or {}).get("started_at")
                        ),
                    )
                    pending_path.unlink(missing_ok=True)
                    pending_state_path.unlink(missing_ok=True)
                    events_path.unlink(missing_ok=True)
                    row_index = processed

                if next(source_iterator, None) is not None or next(
                    annotation_iterator, None
                ) is not None:
                    raise ValueError(
                        "Stage 1/Stage 2 row counts exceed the recorded chunk count."
                    )
                if processed != total_chunks:
                    raise ValueError(
                        f"Processed {processed} rows but expected {total_chunks}."
                    )

                self._update(
                    job_id,
                    stage="publishing_relations",
                    progress=97,
                    message="Publishing one compressed reusable relation artifact...",
                )
                output_sha = sha256_path(partial_path)
                output_ref, output_reused = store.put_file(
                    partial_path,
                    key=str(job["output_artifact_key"]),
                    content_type="application/gzip",
                    content_encoding=None,
                    sha256=output_sha,
                )
                current = get_relation_job(job_id) or job
                final_stats = self._stats_snapshot(
                    counters,
                    by_predicate=by_predicate,
                    by_direction=by_direction,
                )
                final_stats.update(
                    {
                        "paper_count": int(current.get("paper_count") or 0),
                        "chunk_count": total_chunks,
                        "processed_chunk_count": processed,
                        "output_sha256": output_ref.sha256,
                        "output_bytes": output_ref.size_bytes,
                        "output_reused": output_reused,
                        "elapsed_seconds": round(
                            _elapsed_since(current.get("started_at")),
                            2,
                        ),
                    }
                )
                summary = {
                    "status": "completed",
                    "message": (
                        f"Finished: {counters['relation_count']:,} validated relations "
                        f"from {processed:,} chunks."
                    ),
                    "job_id": job_id,
                    "pipeline_version": RELATION_PIPELINE_VERSION,
                    "output_schema": RELATION_OUTPUT_SCHEMA,
                    "model_signature": current.get("model_signature"),
                    "model": settings.relation_model,
                    "source": {
                        "annotation_job_id": current.get(
                            "source_annotation_job_id"
                        ),
                        "chunks": {
                            "key": current.get("source_chunks_artifact_key"),
                            "sha256": current.get(
                                "source_chunks_artifact_sha256"
                            ),
                        },
                        "annotations": {
                            "key": current.get(
                                "source_annotation_artifact_key"
                            ),
                            "sha256": current.get(
                                "source_annotation_artifact_sha256"
                            ),
                        },
                    },
                    "configuration": {
                        "execution_mode": "online_async_responses",
                        "window_size": settings.relation_window_size,
                        "concurrency": settings.relation_concurrency,
                        "request_timeout_seconds": (
                            settings.relation_request_timeout_seconds
                        ),
                        "max_request_retries": (
                            settings.relation_max_request_retries
                        ),
                        "prompt_cache_key": settings.relation_prompt_cache_key,
                        "prompt_cache_shards": (
                            settings.relation_prompt_cache_shards
                        ),
                        "reasoning_effort": settings.relation_reasoning_effort,
                        "max_output_tokens": settings.relation_max_output_tokens,
                        "biosynthesis_enabled": (
                            settings.relation_enable_biosynthesis
                        ),
                        "hormone_gene_cell_context_required": (
                            settings.relation_require_hormone_gene_cell_context
                        ),
                    },
                    "stats": final_stats,
                    "files": {"relations": output_ref.to_dict()},
                    "completed_at": utc_now(),
                }
                summary_path = work_root / "summary.json"
                with summary_path.open("w", encoding="utf-8") as handle:
                    json.dump(summary, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                summary_ref, _ = store.put_file(
                    summary_path,
                    key=str(job["summary_artifact_key"]),
                    content_type="application/json",
                    sha256=sha256_path(summary_path),
                )
                summary["files"]["summary"] = summary_ref.to_dict()

                elapsed = round(_elapsed_since(current.get("started_at")), 2)
                update_relation_job(
                    job_id,
                    status="completed",
                    stage="completed",
                    progress=100,
                    message=summary["message"],
                    processed_chunk_count=processed,
                    eligible_chunk_count=counters["eligible_chunk_count"],
                    relation_count=counters["relation_count"],
                    cell_context_count=counters["cell_context_count"],
                    api_request_count=counters["api_request_count"],
                    stats={**final_stats, "elapsed_seconds": elapsed},
                    elapsed_seconds=elapsed,
                    completed_at=utc_now(),
                    error=None,
                )

            shutil.rmtree(work_root, ignore_errors=True)
        finally:
            if gateway is not None:
                await gateway.close()


relation_executor = RelationExecutor()

__all__ = [
    "OpenAIResponsesGateway",
    "RelationExecutor",
    "RelationExecutorStopping",
    "RelationResponseError",
    "relation_executor",
]
