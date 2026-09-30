"""Private, bounded monthly discovery and incremental corpus publication.

Discovery downloads identifiers only. The union of both saved corpora is checked
before metadata, GPU inference, or relation extraction is requested. Pending IDs,
monthly attempt reservations, and completed predictions live outside temporary
user runs so restarts and duplicate scheduler deliveries do not reset the budget.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    import fcntl
except ImportError:  # Windows local installations can import the app as well.
    fcntl = None
    import msvcrt

from backend.config import settings
from backend.pipeline.precomputed_corpora import CORPUS_DEFINITIONS, iter_prediction_rows
from backend.pipeline.retrieval import (
    DEFAULT_PUBMED_ANIMAL_EXCLUSION_QUERY,
    DEFAULT_PUBMED_CANCER_EXCLUSION_QUERY,
    DEFAULT_PUBMED_HUMAN_QUERY,
    DEFAULT_PUBMED_IMMUNE_INFLAMMATION_QUERY,
    DEFAULT_PUBMED_OVARIAN_CONTEXT_QUERY,
    DEFAULT_PUBMED_QUERY,
    PUBMED_ESEARCH,
    RequestPacer,
    RetrievalError,
    build_retry_session,
    ncbi_params,
    normalize_pmid,
    perform_request_with_retries,
    sanitize_query,
)

logger = logging.getLogger(__name__)

# The cancer corpus uses the same ovarian, immune, and human selection blocks.
# The non-neoplastic cancer exclusion becomes an inclusion for this corpus.
CORPUS_DISCOVERY_QUERIES = {
    "non_neoplastic_inflammatory": DEFAULT_PUBMED_QUERY,
    "cancer_associated_inflammatory": (
        f"({DEFAULT_PUBMED_OVARIAN_CONTEXT_QUERY}) "
        f"AND ({DEFAULT_PUBMED_CANCER_EXCLUSION_QUERY}) "
        f"AND ({DEFAULT_PUBMED_IMMUNE_INFLAMMATION_QUERY}) "
        f"AND ({DEFAULT_PUBMED_HUMAN_QUERY}) "
        f"AND NOT ({DEFAULT_PUBMED_ANIMAL_EXCLUSION_QUERY})"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def discover_corpus_pmids(*, check_active: Callable[[], None] | None = None) -> dict[str, list[str]]:
    """Search both topic queries from the beginning, with no paid processing.

PubMed ESearch cannot page beyond 10,000 matching records. Split oversized
searches into disjoint numeric UID ranges before paging, rather than silently
discarding the remainder. No creation/publication-date watermark is advanced;
older papers that become newly indexed or acquire matching metadata are found.
"""
    result: dict[str, list[str]] = {}
    pacer = RequestPacer(0.11 if settings.ncbi_api_key else 0.34)
    with build_retry_session(f"{settings.ncbi_tool}/monthly-corpus-update") as session:
        for corpus_id, topic_query in CORPUS_DISCOVERY_QUERIES.items():
            ids: set[str] = set()

            def page(query: str, start: int = 0) -> tuple[int, list[str]]:
                if check_active:
                    check_active()
                response = perform_request_with_retries(
                    session,
                    "POST",
                    PUBMED_ESEARCH,
                    context="Monthly PubMed identifier discovery",
                    timeout=settings.retrieval_request_timeout,
                    pacer=pacer,
                    data={
                        "db": "pubmed", "term": sanitize_query(query),
                        "retmode": "json", "retmax": "1000", "retstart": str(start),
                        "sort": "pub_date",
                        **ncbi_params(settings.ncbi_email, settings.ncbi_tool, settings.ncbi_api_key),
                    },
                )
                try:
                    payload = response.json()["esearchresult"]
                    if payload.get("errorlist") or payload.get("ERROR"):
                        raise ValueError("PubMed rejected the monthly search query.")
                    count = int(payload["count"])
                    values = [normalize_pmid(value) for value in payload.get("idlist", [])]
                except (KeyError, TypeError, ValueError) as exc:
                    raise RetrievalError("PubMed returned an invalid monthly search response.") from exc
                return count, [value for value in values if value]

            def collect(low: int, high: int) -> int:
                query = f"({topic_query}) AND {low}:{high}[UID]"
                count, first = page(query)
                if count > 10_000:
                    if low >= high:
                        raise RetrievalError("Could not partition all PubMed identifiers safely.")
                    middle = (low + high) // 2
                    combined_count = collect(low, middle) + collect(middle + 1, high)
                    if combined_count != count:
                        raise RetrievalError("PubMed changed during identifier partitioning; retry the update.")
                    return combined_count
                branch = set(first)
                for start in range(1000, count, 1000):
                    _count, values = page(query, start)
                    if _count != count:
                        raise RetrievalError("PubMed changed during identifier pagination; retry the update.")
                    branch.update(values)
                if len(branch) != count or any(not low <= int(pmid) <= high for pmid in branch):
                    # Do not publish a completed month after a truncated search.
                    raise RetrievalError("PubMed changed or truncated the search; retry the update.")
                ids.update(branch)
                return count

            collect(1, 999_999_999)
            result[corpus_id] = sorted(ids, key=int)
    return result


def process_new_pmid(pmid: str, *, callback_base_url: str, check_active: Callable[[], None]) -> dict[str, Any]:
    """Use the existing trusted pipeline for one genuinely new PMID."""
    from backend.orchestrator import pipeline_orchestrator
    from backend.runtime import run_registry

    check_active()
    run = pipeline_orchestrator.create_update_run(
        [pmid],
        text_mode=getattr(settings, "monthly_update_text_mode", "fulltext"),
        callback_base_url=callback_base_url,
    )
    run_id = str(run["id"])
    timeout = max(60, int(getattr(settings, "monthly_update_timeout_seconds", 43200)))
    interval = max(0.1, float(getattr(settings, "monthly_update_poll_seconds", 5)))
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            check_active()
            current = run_registry.public(run_id)
            for stage_name in ("retrieval", "annotation", "relation"):
                stage = current["stages"][stage_name]
                if stage.get("status") == "failed":
                    raise RuntimeError(str(stage.get("error") or stage.get("message") or f"{stage_name} failed"))
            relation = current["stages"]["relation"]
            if relation.get("status") == "completed":
                paths = run_registry.get_private(run_id, "stage3_prediction_paths", [])
                rows = [row for path in paths for row in iter_prediction_rows(Path(path))]
                if not rows or any(normalize_pmid(row.get("pmid")) != pmid for row in rows):
                    raise RuntimeError("The pipeline did not publish an aligned final prediction for the new PMID.")
                return {"rows": rows, "stats": dict(relation.get("stats") or {}), "run_id": run_id}
            # Runs normally advance automatically. This also supports a controller
            # configured for explicit stage advancement without running network build.
            for previous, next_name in (("retrieval", "annotation"), ("annotation", "relation")):
                if current["stages"][previous].get("status") == "completed" and current["stages"][next_name].get("status") in {"ready", "locked"}:
                    pipeline_orchestrator.start_stage(run_id, next_name, callback_base_url=callback_base_url)
                    break
            time.sleep(interval)
        raise TimeoutError(f"Monthly processing timed out for PMID {pmid}; it remains pending.")
    except BaseException:
        cancel = getattr(pipeline_orchestrator, "cancel_update_run", None)
        if callable(cancel):
            try:
                cancel(run_id)
            except Exception:
                logger.exception("Could not cancel the interrupted trusted update run %s", run_id)
        if run_registry.exists(run_id):
            run_registry.set_private(run_id, "retain_for_update", False)
        raise


def _release_update_run(result: Mapping[str, Any]) -> None:
    run_id = str(result.get("run_id") or "")
    if run_id:
        from backend.runtime import run_registry
        if run_registry.exists(run_id):
            run_registry.set_private(run_id, "retain_for_update", False)


def _balanced_queue(pending: dict[str, list[str]], previous_pending: Mapping[str, Any]) -> list[str]:
    """Alternate topic queues while prioritizing their previously deferred IDs."""
    queues = {corpus_id: deque() for corpus_id in CORPUS_DEFINITIONS}
    previous = [pmid for pmid in previous_pending if pmid in pending]
    order = previous + sorted(set(pending) - set(previous), key=int)
    for pmid in order:
        memberships = pending[pmid]
        # A paper in both queries still consumes exactly one paid slot.
        owner = min(memberships, key=lambda corpus_id: len(queues[corpus_id]))
        queues[owner].append(pmid)
    result: list[str] = []
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                result.append(queue.popleft())
    return result


class MonthlyCorpusUpdater:
    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        discover: Callable[..., dict[str, list[str]]] | None = None,
        process: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.root = (data_dir or settings.data_dir) / "corpus_updates"
        self.state_path = self.root / "state.json"
        self.checkpoint_root = self.root / "completed"
        self._discover = discover or discover_corpus_pmids
        self._process = process or process_new_pmid
        self._guard = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._deadline: float | None = None

    def _state(self) -> dict[str, Any]:
        if self.state_path.exists():
            with self.state_path.open(encoding="utf-8") as handle:
                return json.load(handle)
        return {"version": 1, "status": "idle", "pending": {}, "months": {}, "last_completed_month": None}

    def _save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        _atomic_json(self.state_path, state)

    def _lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        handle = (self.root / "update.lock").open("a+b")
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            handle.close()
            return None
        return handle

    def initialize(self) -> None:
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        handle = self._lock()
        if handle is None:
            return
        try:
            state = self._state()
            if state.get("status") == "running":
                state["status"] = "interrupted"
                state["error"] = "The controller restarted. Pending IDs and monthly attempt reservations were retained."
                self._save(state)
            # Completed predictions are republished by the next authorized
            # enabled update. Initialization never starts paid processing.
        finally:
            handle.close()

    def shutdown(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)

    def _check_active(self) -> None:
        if self._stop.is_set() or not getattr(settings, "monthly_updates_enabled", False):
            raise InterruptedError("Monthly updates are disabled or the controller is shutting down.")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TimeoutError("The monthly update time limit was reached. Remaining papers stay pending.")

    def status(self) -> dict[str, Any]:
        state = self._state()
        state["enabled"] = bool(getattr(settings, "monthly_updates_enabled", False))
        state["pending_count"] = len(state.pop("pending", {}))
        state["max_new_papers_per_month"] = int(getattr(settings, "monthly_update_max_new_papers", 300))
        return state

    def start(self, *, dry_run: bool = True, callback_base_url: str = "") -> dict[str, Any]:
        with self._guard:
            if not getattr(settings, "monthly_updates_enabled", False):
                raise InterruptedError("Monthly corpus updates are disabled.")
            handle = self._lock()
            if handle is None:
                result = self.status()
                result["already_running"] = True
                return result
            state = self._state()
            month = datetime.now(timezone.utc).strftime("%Y-%m")
            if not dry_run and state.get("last_completed_month") == month:
                handle.close()
                result = self.status()
                result["already_completed_this_month"] = True
                return result
            self._stop.clear()
            state.update(status="running", started_at=utc_now(), dry_run=dry_run, active_pmid=None, error=None)
            self._save(state)
            self._deadline = time.monotonic() + max(60, int(getattr(settings, "monthly_update_timeout_seconds", 43200)))
            self._thread = threading.Thread(
                target=self._execute,
                kwargs={"state": state, "month": month, "dry_run": dry_run,
                        "callback_base_url": callback_base_url or settings.public_base_url,
                        "lock_handle": handle},
                name="monthly-corpus-update", daemon=True,
            )
            self._thread.start()
            return self.status()

    def _publish_checkpoint(self, path: Path, state: dict[str, Any]) -> None:
        from backend.services.corpus_store import add_prediction_rows

        with path.open(encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        pmid = str(checkpoint["pmid"])
        memberships = set(checkpoint["corpus_ids"]) | set(state["pending"].get(pmid, []))
        for corpus_id in sorted(memberships):
            if corpus_id not in CORPUS_DEFINITIONS:
                raise ValueError("Invalid corpus in the update checkpoint.")
            add_prediction_rows(corpus_id, checkpoint["rows"])
        state["pending"].pop(pmid, None)
        completed_month = checkpoint.get("month")
        if completed_month:
            month_state = state["months"].setdefault(completed_month, {
                "attempted_pmids": [], "succeeded_pmids": [], "failed_pmids": {},
            })
            if pmid not in month_state["succeeded_pmids"]:
                month_state["succeeded_pmids"].append(pmid)
                for name in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"):
                    month_state[name] = int(month_state.get(name, 0)) + int(checkpoint.get("stats", {}).get(name, 0))
            month_state["failed_pmids"].pop(pmid, None)
        self._save(state)
        path.unlink(missing_ok=True)

    def _copy_known_memberships(self, memberships: dict[str, list[str]], known: set[str]) -> int:
        from backend.services.corpus_store import add_prediction_rows, get_corpus_pmids, prediction_pmid, select_pmid_predictions

        missing_by_corpus = {
            corpus_id: {pmid for pmid, targets in memberships.items() if pmid in known and corpus_id in targets}
                       - get_corpus_pmids(corpus_id)
            for corpus_id in CORPUS_DEFINITIONS
        }
        selected = sorted(set().union(*missing_by_corpus.values()), key=int)
        if not selected:
            return 0
        destination = self.root / "known-memberships.jsonl"
        select_pmid_predictions(selected, destination)
        copied = 0
        rows_by_corpus: dict[str, list[dict[str, Any]]] = {corpus_id: [] for corpus_id in CORPUS_DEFINITIONS}
        try:
            for row in iter_prediction_rows(destination):
                self._check_active()
                pmid = prediction_pmid(row)
                for corpus_id, missing in missing_by_corpus.items():
                    if pmid in missing:
                        rows_by_corpus[corpus_id].append(row)
            for corpus_id, rows in rows_by_corpus.items():
                self._check_active()
                if rows:
                    copied += int(add_prediction_rows(corpus_id, rows).get("added", 0))
        finally:
            destination.unlink(missing_ok=True)
        return copied

    def _execute(self, *, state: dict[str, Any], month: str, dry_run: bool, callback_base_url: str, lock_handle: Any) -> None:
        from backend.services.corpus_store import get_known_pmids

        try:
            self._check_active()
            state.setdefault("pending", {})
            state.setdefault("months", {})
            if not dry_run:
                for path in sorted(self.checkpoint_root.glob("*.json")):
                    self._check_active()
                    self._publish_checkpoint(path, state)
            discoveries = self._discover(check_active=self._check_active)
            self._check_active()
            memberships: dict[str, list[str]] = {}
            for corpus_id, ids in discoveries.items():
                if corpus_id not in CORPUS_DEFINITIONS:
                    raise ValueError("Unknown corpus in monthly discovery.")
                for value in ids:
                    pmid = normalize_pmid(value)
                    if not pmid:
                        raise ValueError("Monthly discovery returned an invalid PMID.")
                    memberships.setdefault(pmid, []).append(corpus_id)
            known = get_known_pmids()
            newly_discovered = {pmid: ids for pmid, ids in memberships.items() if pmid not in known}
            pending = dict(state["pending"])
            for pmid, ids in newly_discovered.items():
                pending[pmid] = sorted(set(pending.get(pmid, [])) | set(ids))
            pending = {pmid: ids for pmid, ids in pending.items() if pmid not in known}
            month_state = state["months"].setdefault(month, {"attempted_pmids": [], "succeeded_pmids": [], "failed_pmids": {}})
            attempted = set(month_state["attempted_pmids"])
            cap = max(0, int(getattr(settings, "monthly_update_max_new_papers", 300)))
            remaining = max(0, cap - len(attempted))
            # Previously deferred work comes first, then new IDs in PMID order.
            queue = _balanced_queue(pending, state["pending"])
            selected = [pmid for pmid in queue if pmid not in attempted][:remaining]
            state["discovery"] = {
                "corpus_counts": {corpus: len(ids) for corpus, ids in discoveries.items()},
                "retrieved_unique_papers": len(memberships),
                "already_known_papers": len(set(memberships) & known),
                "newly_discovered_papers": len(newly_discovered),
                "pending_papers": len(pending), "selected_papers": len(selected),
                "remaining_monthly_attempts": remaining,
            }
            if dry_run:
                state.update(status="dry_run_completed", preview_pmids=selected, completed_at=utc_now())
                self._save(state)
                return
            state["pending"] = pending
            self._save(state)
            state["copied_known_memberships"] = self._copy_known_memberships(memberships, known)
            for pmid in selected:
                self._check_active()
                # Reserve before submitting any pipeline work. Failed attempts
                # consume a monthly slot and are retried in a later month.
                month_state["attempted_pmids"].append(pmid)
                state["active_pmid"] = pmid
                self._save(state)
                result: dict[str, Any] = {}
                try:
                    result = self._process(pmid, callback_base_url=callback_base_url, check_active=self._check_active)
                    rows = result.get("rows")
                    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) or normalize_pmid(row.get("pmid")) != pmid for row in rows):
                        raise ValueError("New paper predictions must retain the requested PMID.")
                    checkpoint = self.checkpoint_root / f"{pmid}.json"
                    _atomic_json(checkpoint, {"pmid": pmid, "month": month, "corpus_ids": pending[pmid], "rows": rows,
                                              "stats": result.get("stats", {}), "completed_at": utc_now()})
                    self._publish_checkpoint(checkpoint, state)
                    self._save(state)
                except InterruptedError:
                    raise
                except Exception as exc:
                    logger.exception("Monthly processing failed for PMID %s", pmid)
                    month_state["failed_pmids"][pmid] = str(exc)
                    self._save(state)
                finally:
                    _release_update_run(result)
            state.update(status="completed", active_pmid=None, completed_at=utc_now(), last_completed_month=month)
            self._save(state)
        except InterruptedError as exc:
            state.update(status="stopped", active_pmid=None, error=str(exc))
            self._save(state)
        except Exception as exc:
            logger.exception("Monthly corpus update failed")
            state.update(status="failed", active_pmid=None, error=str(exc))
            self._save(state)
        finally:
            lock_handle.close()


corpus_update_service = MonthlyCorpusUpdater()
