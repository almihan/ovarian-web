"""Four-stage orchestration for shared defaults or isolated PMID runs."""

from __future__ import annotations

import logging
import secrets
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping

from backend.config import settings
from backend.worker_state import (
    create_annotation_job,
    create_relation_job,
    get_annotation_job,
    get_relation_job,
    clear_worker_state,
    update_annotation_job,
    update_relation_job,
    utc_now,
)
from backend.default_cache import (
    build_isolated_pmid_stage1,
    cached_json_pair,
    get_or_build_default_stage1,
    materialize_artifact,
    run_scoped_signature,
)
from backend.pipeline.annotation_contract import (
    ANNOTATION_PIPELINE_VERSION,
    annotation_artifact_keys,
    annotation_model_signature,
    callback_token_hash,
    callback_token_matches,
)
from backend.pipeline.final_annotations import build_final_annotation_artifact
from backend.pipeline.network_builder import (
    build_interaction_network_from_final_annotations,
)
from backend.pipeline.precomputed_corpora import (
    DEFAULT_CORPUS_ID,
    CorpusNotFoundError,
    corpus_definition,
    corpus_path,
    corpus_summary,
    summarize_entity_annotations,
    summarize_prediction_file,
)
from backend.pipeline.relation_contract import (
    final_annotation_artifact_key,
    relation_artifact_keys,
    relation_model_signature,
)
from backend.pipeline.retrieval import (
    RetrievalError,
    TextMode,
    build_explicit_pmid_inputs,
    normalize_text_mode,
)
from backend.runtime import run_registry
from backend.services.local_executor import (
    local_executor,
    missing_local_ml_dependencies,
)
from backend.services.modal_executor import modal_executor
from backend.services.annotation_coordinator import annotation_coordinator
from backend.services.corpus_store import get_known_pmids, select_pmid_predictions
from backend.services.relation_executor import relation_executor
from backend.storage.artifacts import (
    ArtifactRef,
    get_artifact_store,
    prefixed_key,
    sha256_file,
)

logger = logging.getLogger(__name__)

_PRECOMPUTED_STAGE_MIN_SECONDS = 3.0


def _hold_precomputed_stage(started: float) -> None:
    remaining = _PRECOMPUTED_STAGE_MIN_SECONDS - (time.monotonic() - started)
    if remaining > 0:
        time.sleep(remaining)


def _int(stats: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = stats.get(name)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _float(stats: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value = stats.get(name)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _chunk_count(stats: Mapping[str, Any]) -> int:
    return _int(
        stats,
        "chunk_count",
        "chunks_written",
        "output_chunk_count",
        "chunks_processed",
    )


def _stage_progress(
    run_id: str,
    stage_name: str,
    *,
    progress: int,
    stage: str,
    message: str,
    stats: Mapping[str, Any] | None = None,
) -> None:
    if not run_registry.exists(run_id):
        return
    current = run_registry.public(run_id)["stages"][stage_name]
    if current.get("status") in {"completed", "failed"}:
        return
    run_registry.update_stage(
        run_id,
        stage_name,
        status="processing",
        stage=stage,
        progress=max(int(current.get("progress") or 0), min(99, int(progress))),
        message=message,
        stats=dict(stats or current.get("stats") or {}),
    )


def _scaled(value: int, start: int, end: int) -> int:
    safe = max(0, min(100, int(value)))
    return start + round((end - start) * safe / 100)


def _scope_progress_bounds(index: int, total: int) -> tuple[int, int]:
    """Allocate the 2..98 progress range across one or two artifact scopes."""

    safe_total = max(1, int(total))
    safe_index = max(0, min(safe_total - 1, int(index)))
    start = 2 + round(96 * safe_index / safe_total)
    end = 2 + round(96 * (safe_index + 1) / safe_total)
    return start, end


def _annotation_counts(stats: Mapping[str, Any]) -> dict[str, int]:
    return {
        "paper_count": _int(stats, "paper_count", "papers_processed"),
        "chunk_count": _chunk_count(stats),
        "mention_count": _int(
            stats,
            "cell_count",
            "mention_count",
            "mention_occurrences",
            "mentions_detected",
        ),
        "normalized_count": _int(
            stats,
            "normalized_count",
            "normalized_occurrences",
        ),
        "unresolved_count": _int(
            stats,
            "unresolved_count",
            "unresolved_occurrences",
        ),
    }


def _cell_branch_ready(job: Mapping[str, Any]) -> bool:
    keys = annotation_artifact_keys(
        source_sha256=str(job.get("source_artifact_sha256") or ""),
        model_signature=str(job.get("model_signature") or ""),
    )
    store = get_artifact_store()
    return bool(
        store.head(keys.cell_annotations) is not None
        and store.head(keys.cell_summary) is not None
    )


def _apply_cell_result(
    job: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    current = get_annotation_job(str(job["id"])) or dict(job)
    if current.get("status") in {"completed", "failed"}:
        return current

    incoming_stats = result.get("stats")
    incoming_stats = incoming_stats if isinstance(incoming_stats, Mapping) else {}
    stats = dict(current.get("stats") or {})
    stats.update(dict(incoming_stats))
    status = str(result.get("status") or "cell_completed").casefold()
    if status == "failed":
        stats["cell_branch_status"] = "failed"
        return update_annotation_job(
            str(current["id"]),
            status="failed",
            stage="failed",
            progress=100,
            message=str(result.get("message") or "CellExLink extraction failed."),
            stats=stats,
            error=str(result.get("error") or "CellExLink extraction failed."),
            completed_at=utc_now(),
            last_remote_check_at=utc_now(),
            **_annotation_counts(stats),
        ) or current

    if not _cell_branch_ready(current):
        raise RuntimeError(
            "CellExLink completed without publishing its branch artifacts."
        )
    stats["cell_branch_status"] = "completed"
    keys = annotation_artifact_keys(
        source_sha256=str(current.get("source_artifact_sha256") or ""),
        model_signature=str(current.get("model_signature") or ""),
    )
    store = get_artifact_store()
    pubtator_ready = bool(
        store.head(keys.pubtator_annotations) is not None
        and store.head(keys.pubtator_summary) is not None
    )
    updated = update_annotation_job(
        str(current["id"]),
        status="processing",
        stage="merging_entities" if pubtator_ready else "waiting_for_pubtator3",
        progress=max(int(current.get("progress") or 0), 86 if pubtator_ready else 80),
        message=(
            "Both entity branches are ready; the Stage 2 coordinator is "
            "preparing the final merge."
            if pubtator_ready
            else "CellExLink is complete; waiting for PubTator3."
        ),
        stats=stats,
        last_remote_check_at=utc_now(),
        error=None,
        **_annotation_counts(stats),
    ) or current
    annotation_coordinator.schedule_finalize(str(current["id"]))
    return updated


def _apply_cell_progress(
    job: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply progress reported by a local file or a remote callback."""

    current = get_annotation_job(str(job["id"])) or dict(job)
    if current.get("status") in {"completed", "failed"}:
        return current

    status = str(payload.get("status") or "processing").casefold()
    incoming_stats = payload.get("stats")
    incoming_stats = incoming_stats if isinstance(incoming_stats, Mapping) else {}
    stats = dict(current.get("stats") or {})
    stats.update(dict(incoming_stats))
    if payload.get("output_sha256"):
        stats["cell_output_sha256"] = str(payload["output_sha256"])
    if payload.get("summary_sha256"):
        stats["cell_summary_sha256"] = str(payload["summary_sha256"])

    if status in {"cell_completed", "completed"}:
        return _apply_cell_result(
            current,
            {
                "status": "cell_completed",
                "message": payload.get("message"),
                "stats": stats,
            },
        )

    if status == "failed":
        return update_annotation_job(
            str(current["id"]),
            status="failed",
            stage="failed",
            progress=100,
            message=str(payload.get("message") or "CellExLink extraction failed."),
            stats=stats,
            error=str(payload.get("error") or "CellExLink worker failed."),
            completed_at=utc_now(),
            last_remote_check_at=utc_now(),
            **_annotation_counts(stats),
        ) or current

    return update_annotation_job(
        str(current["id"]),
        status="processing",
        stage=str(payload.get("stage") or "cell_annotation"),
        progress=max(
            int(current.get("progress") or 0),
            max(0, min(100, int(payload.get("progress") or 0))),
        ),
        message=str(payload.get("message") or "Local CellExLink is running."),
        stats=stats,
        last_remote_check_at=utc_now(),
        **_annotation_counts(stats),
    ) or current


class PipelineOrchestrator:
    """Bounded stage queues with no persistent user job history."""

    def __init__(self) -> None:
        self._stage1_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="public-stage1"
        )
        self._stage2_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="public-stage2"
        )
        self._stage3_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="public-stage3"
        )
        self._stage4_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="public-stage4"
        )
        # Monthly GPU/API work must not occupy the visitor saved-results queues.
        self._update_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="monthly-update"
        )
        self._guard = threading.RLock()
        self._running: set[tuple[str, str]] = set()

    def initialize(self) -> None:
        settings.ensure_directories()
        clear_worker_state()
        run_registry.clear()
        for path in (
            settings.data_dir / "runs",
            settings.data_dir / "work",
            settings.local_annotation_jobs_dir,
            settings.relation_jobs_dir,
        ):
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
        if settings.artifact_backend == "local":
            try:
                get_artifact_store().delete_prefix(prefixed_key("runs"))
            except Exception as exc:
                logger.warning("Could not remove old temporary run artifacts: %s", exc)
        else:
            # Serving /health must not wait for a bucket connection or retries.
            # Remote runs are removed by per-run expiry or bucket lifecycle rules.
            logger.info(
                "Skipping remote temporary-run cleanup during startup; "
                "use per-run expiry cleanup or bucket lifecycle expiration."
            )

    def shutdown(self) -> None:
        for run_id in {run_id for run_id, _stage in tuple(self._running)}:
            try:
                if run_registry.get_private(run_id, "trusted_update", False):
                    self.cancel_update_run(run_id)
            except KeyError:
                pass
        for pool in (
            self._stage1_pool,
            self._stage2_pool,
            self._stage3_pool,
            self._stage4_pool,
            self._update_pool,
        ):
            pool.shutdown(wait=False, cancel_futures=True)
        relation_executor.shutdown()
        local_executor.shutdown()
        shutdown = getattr(annotation_coordinator, "shutdown", None)
        if callable(shutdown):
            shutdown()
        if settings.artifact_backend == "local":
            try:
                get_artifact_store().delete_prefix(prefixed_key("runs"))
            except Exception as exc:
                logger.warning("Could not remove temporary run artifacts: %s", exc)
        else:
            logger.info(
                "Skipping remote temporary-run cleanup during shutdown; "
                "use per-run expiry cleanup or bucket lifecycle expiration."
            )
        shutil.rmtree(settings.data_dir / "runs", ignore_errors=True)
        shutil.rmtree(settings.data_dir / "work", ignore_errors=True)
        run_registry.clear()
        clear_worker_state()

    def _cleanup_expired_runs(self) -> None:
        for record in run_registry.pop_expired(settings.run_retention_seconds):
            run_id = str(record.get("id") or "")
            if not run_id:
                continue
            shutil.rmtree(settings.data_dir / "runs" / run_id, ignore_errors=True)
            try:
                get_artifact_store().delete_prefix(
                    prefixed_key(f"runs/{run_id}")
                )
            except Exception as exc:
                logger.warning(
                    "Could not remove expired temporary artifacts for run %s: %s",
                    run_id,
                    exc,
                )

    def create_run(
        self,
        query: str,
        *,
        text_mode: TextMode = "fulltext",
        corpus_id: str = DEFAULT_CORPUS_ID,
        callback_base_url: str = "",
    ) -> dict[str, Any]:
        """Create a visitor run using the server's public/local policy."""
        return self._create_run(
            query,
            text_mode=text_mode,
            corpus_id=corpus_id,
            callback_base_url=callback_base_url,
            trusted_update=False,
        )

    def create_update_run(
        self,
        pmids: list[str],
        *,
        text_mode: TextMode = "fulltext",
        callback_base_url: str = "",
    ) -> dict[str, Any]:
        """Private Python entry point for the scheduled corpus updater.

        No public request field or endpoint can select this trusted path.
        """
        if not pmids:
            raise ValueError("An update run requires at least one new PMID.")
        return self._create_run(
            ", ".join(str(pmid) for pmid in pmids),
            text_mode=text_mode,
            corpus_id=DEFAULT_CORPUS_ID,
            callback_base_url=callback_base_url,
            trusted_update=True,
        )

    def _create_run(
        self,
        query: str,
        *,
        text_mode: TextMode,
        corpus_id: str,
        callback_base_url: str,
        trusted_update: bool,
    ) -> dict[str, Any]:
        self._cleanup_expired_runs()
        cleaned = " ".join(str(query or "").split())
        selected_text_mode = normalize_text_mode(text_mode)
        pmid_only = bool(cleaned)
        saved_pmid_selection = bool(
            pmid_only and settings.public_precomputed_only and not trusted_update
        )
        requested_pmids: list[str] = []
        if pmid_only:
            requested_pmids = list(build_explicit_pmid_inputs("pmid", cleaned).pmids)
            selected_corpus = corpus_definition(corpus_id)
            has_custom = True
            if saved_pmid_selection:
                known_pmids = get_known_pmids()
                if not any(pmid in known_pmids for pmid in requested_pmids):
                    raise RetrievalError(
                        "These PMIDs are not available in the saved paper list: "
                        + ", ".join(requested_pmids)
                        + ". This website loads existing results only. "
                        "Run the application locally to analyze other papers."
                    )
                selected_corpus = {
                    **selected_corpus,
                    "label": "Saved papers selected by PMID",
                }
                input_mode = "precomputed"
            else:
                # Local and trusted monthly runs compute their own isolated data.
                input_mode = "pmid_only"
        else:
            selected_corpus = corpus_definition(corpus_id)
            selected_path = corpus_path(selected_corpus["id"])
            if not selected_path.is_file():
                raise CorpusNotFoundError(
                    f"Place {selected_corpus['filename']} in {selected_path.parent} "
                    "before starting."
                )
            has_custom = False
            input_mode = "precomputed"

        run = run_registry.create(
            query=cleaned,
            has_custom_input=has_custom,
            input_mode=input_mode,
            text_mode=selected_text_mode,
            corpus_id=selected_corpus["id"],
            corpus_label=selected_corpus["label"],
        )
        run_id = str(run["id"])
        run_registry.set_private(run_id, "trusted_update", bool(trusted_update))
        run_registry.set_private(run_id, "retain_for_update", bool(trusted_update))
        if saved_pmid_selection:
            subset_path = (
                settings.data_dir / "runs" / run_id / "precomputed-pmid-results.jsonl"
            )
            selected = select_pmid_predictions(requested_pmids, subset_path)
            if not selected["found_pmids"]:
                raise RetrievalError("None of the entered PMIDs has saved results.")
            summary = {
                **summarize_prediction_file(subset_path),
                **selected,
                "label": selected_corpus["label"],
            }
            run_registry.set_private(run_id, "precomputed_pmid_selection", True)
            run_registry.set_private(run_id, "precomputed_prediction_path", str(subset_path))
            run_registry.set_private(run_id, "precomputed_summary", summary)
        run_registry.set_private(
            run_id,
            "callback_base_url",
            str(callback_base_url or settings.public_base_url or "").rstrip("/"),
        )
        if saved_pmid_selection:
            message = "Stage 1 is queued. Loading saved results for the entered PMIDs."
        elif pmid_only:
            message = (
                "Stage 1 is queued. Only the entered PMIDs will be downloaded "
                "and processed in this isolated run."
            )
        else:
            message = (
                f"Stage 1 is queued. Loading the saved {selected_corpus['label']} "
                "corpus without changing its source file."
            )
        run_registry.start_stage(
            run_id,
            "retrieval",
            message,
        )
        self._submit(run_id, "retrieval", self._run_stage1)
        return run_registry.public(run_id)

    def _assert_run_compute_allowed(self, run_id: str) -> None:
        """Fail closed if a public run could enter a paid processing branch."""
        run = run_registry.get(run_id)
        if run is None:
            return
        trusted = bool(run_registry.get_private(run_id, "trusted_update", False))
        if trusted and (
            run_registry.get_private(run_id, "cancel_requested", False)
            or not settings.monthly_updates_enabled
        ):
            self.cancel_update_run(run_id)
            raise PermissionError("Monthly update computation is canceled or disabled.")
        if (
            settings.public_precomputed_only
            and str(run.get("input_mode") or "") != "precomputed"
            and not trusted
        ):
            raise PermissionError(
                "Public runs may load saved paper results only. "
                "Fresh paper analysis requires a local installation."
            )

    def cancel_update_run(self, run_id: str) -> None:
        """Request cancellation of a trusted update and stop known workers.

        This private Python operation is best effort: an in-flight external
        request may finish, but canceled runs cannot launch another paid stage.
        """
        if not run_registry.get_private(run_id, "trusted_update", False):
            raise PermissionError("Only an internal monthly update can be canceled here.")
        run_registry.set_private(run_id, "cancel_requested", True)
        annotation_job_id = str(run_registry.get_private(run_id, "active_annotation_job_id", "") or "")
        modal_call_id = str(run_registry.get_private(run_id, "active_modal_call_id", "") or "")
        relation_job_id = str(run_registry.get_private(run_id, "active_relation_job_id", "") or "")
        if annotation_job_id:
            update_annotation_job(
                annotation_job_id, status="failed", stage="failed", progress=100,
                error="Monthly update canceled.", message="Monthly update canceled.",
                completed_at=utc_now(),
            )
            try:
                local_executor.cleanup(annotation_job_id, terminate=True)
            except Exception as exc:
                logger.warning("Could not stop local update worker %s: %s", annotation_job_id, exc)
        if modal_call_id:
            try:
                modal_executor.cancel(modal_call_id)
            except Exception as exc:
                logger.warning("Could not cancel Modal update call %s: %s", modal_call_id, exc)
        if relation_job_id:
            try:
                relation_executor.cancel(relation_job_id)
            except Exception as exc:
                logger.warning("Could not cancel relation update job %s: %s", relation_job_id, exc)

    def start_stage(
        self,
        run_id: str,
        stage_name: str,
        *,
        callback_base_url: str,
    ) -> dict[str, Any]:
        if stage_name not in {"annotation", "relation", "network"}:
            raise ValueError("Unknown pipeline stage.")
        run = run_registry.get(run_id)
        if run is None:
            raise KeyError(run_id)
        self._assert_run_compute_allowed(run_id)
        order = ["retrieval", "annotation", "relation", "network"]
        previous = order[order.index(stage_name) - 1]
        if run["stages"][previous].get("status") != "completed":
            raise RuntimeError(f"{previous.title()} must complete first.")
        current = run["stages"][stage_name]
        if current.get("status") in {"queued", "processing"}:
            return run_registry.public(run_id)
        if current.get("status") == "completed":
            run_registry.reset_downstream_stages(run_id, stage_name)
        run_registry.start_stage(
            run_id,
            stage_name,
            f"{current.get('label') or stage_name.title()} is queued.",
        )
        target = {
            "annotation": lambda value: self._run_stage2(
                value, callback_base_url=callback_base_url
            ),
            "relation": self._run_stage3,
            "network": self._run_stage4,
        }[stage_name]
        self._submit(run_id, stage_name, target)
        return run_registry.public(run_id)

    def _queue_automatic_next_stage(self, run_id: str, stage_name: str) -> None:
        try:
            self._assert_run_compute_allowed(run_id)
        except PermissionError:
            return
        next_name = {"retrieval": "annotation", "annotation": "relation"}.get(
            stage_name
        )
        if next_name is None or not run_registry.exists(run_id):
            return
        run = run_registry.public(run_id)
        if run["stages"][stage_name].get("status") != "completed":
            return
        if run["stages"][next_name].get("status") not in {"ready", "failed"}:
            return
        run_registry.start_stage(
            run_id,
            next_name,
            (
                "Stage 2 started automatically after retrieval."
                if next_name == "annotation"
                else "Stage 3 started automatically after entity extraction."
            ),
        )
        callback_base_url = str(
            run_registry.get_private(run_id, "callback_base_url", "") or ""
        )
        target: Callable[[str], None]
        if next_name == "annotation":
            target = lambda value: self._run_stage2(
                value, callback_base_url=callback_base_url
            )
        else:
            target = self._run_stage3
        self._submit(run_id, next_name, target)

    def _submit(
        self,
        run_id: str,
        stage_name: str,
        target: Callable[[str], None],
    ) -> None:
        self._assert_run_compute_allowed(run_id)
        token = (run_id, stage_name)
        with self._guard:
            if token in self._running:
                return
            self._running.add(token)
        pool = self._update_pool if run_registry.get_private(run_id, "trusted_update", False) else {
            "retrieval": self._stage1_pool,
            "annotation": self._stage2_pool,
            "relation": self._stage3_pool,
            "network": self._stage4_pool,
        }[stage_name]

        def runner() -> None:
            try:
                self._assert_run_compute_allowed(run_id)
                target(run_id)
                self._queue_automatic_next_stage(run_id, stage_name)
            except Exception as exc:
                logger.error(
                    "%s failed for run %s: %s",
                    stage_name,
                    run_id,
                    exc,
                )
                run_registry.fail_stage(run_id, stage_name, str(exc))
            finally:
                with self._guard:
                    self._running.discard(token)

        pool.submit(runner)

    def _run_stage1(self, run_id: str) -> None:
        self._assert_run_compute_allowed(run_id)
        started = time.monotonic()
        run = run_registry.get(run_id)
        if run is None:
            return
        text_mode = normalize_text_mode(run.get("text_mode") or "fulltext")
        text_mode_label = (
            "abstracts only"
            if text_mode == "abstract"
            else "full text when available"
        )

        if str(run.get("input_mode") or "") == "precomputed":
            selected_corpus_id = str(run.get("corpus_id") or DEFAULT_CORPUS_ID)
            _stage_progress(
                run_id,
                "retrieval",
                progress=15,
                stage="loading_saved_corpus",
                message="Reading the selected permanent prediction file.",
            )
            subset = bool(run_registry.get_private(run_id, "precomputed_pmid_selection", False))
            summary = run_registry.get_private(run_id, "precomputed_summary") or corpus_summary(selected_corpus_id)
            raw_path = run_registry.get_private(run_id, "precomputed_prediction_path")
            selected_path = Path(str(raw_path)).resolve() if raw_path else corpus_path(selected_corpus_id)
            run_registry.set_private(run_id, "precomputed_prediction_path", str(selected_path))
            run_registry.set_private(run_id, "precomputed_summary", summary)
            run_registry.set_private(run_id, "stage1_default", None)
            run_registry.set_private(run_id, "stage1_custom", None)
            run_registry.set_private(run_id, "stage3_prediction_paths", [str(selected_path)])
            stats = {
                "paper_count": _int(summary, "paper_count"),
                "abstract_count": _int(summary, "abstract_count"),
                "fulltext_count": _int(summary, "fulltext_count"),
                "fulltexts_downloaded": _int(summary, "fulltext_count"),
                "relation_count": _int(summary, "unique_paper_relation_count"),
                "unique_paper_relation_count": _int(
                    summary, "unique_paper_relation_count"
                ),
                "metadata_only_count": _int(summary, "metadata_only_count"),
                "chunk_count": _chunk_count(summary),
                "corpus_id": selected_corpus_id,
                "corpus_label": str(summary.get("label") or run.get("corpus_label") or ""),
                "input_mode": "precomputed",
                "precomputed": True,
                "read_only_source": True,
                "defaults_included": False,
                "custom_recomputed": False,
                "saved_pmid_selection": subset,
                "requested_pmids": list(summary.get("requested_pmids") or []),
                "found_pmids": list(summary.get("found_pmids") or []),
                "missing_pmids": list(summary.get("missing_pmids") or []),
            }
            _stage_progress(
                run_id,
                "retrieval",
                progress=78,
                stage="summarizing_saved_corpus",
                message="Summarizing papers, available text, and saved relations.",
                stats=stats,
            )
            _hold_precomputed_stage(started)
            elapsed = round(time.monotonic() - started, 2)
            stats["elapsed_seconds"] = elapsed
            missing_notice = (
                " Unavailable PMIDs skipped: " + ", ".join(stats["missing_pmids"]) + "."
                if stats["missing_pmids"] else ""
            )
            run_registry.complete_stage(
                run_id,
                "retrieval",
                message=(
                    f"Ready: {stats['paper_count']:,} saved papers loaded from "
                    f"{stats['corpus_label']}." + missing_notice
                ),
                stats=stats,
                elapsed_seconds=elapsed,
            )
            return

        if str(run.get("input_mode") or "") == "pmid_only":
            def pmid_progress(
                stage: str,
                progress: int,
                message: str,
                stats: dict[str, Any],
            ) -> None:
                _stage_progress(
                    run_id,
                    "retrieval",
                    progress=_scaled(progress, 2, 98),
                    stage=f"pmid_{stage}",
                    message=f"Entered PMIDs: {message}",
                    stats=stats,
                )

            _stage_progress(
                run_id,
                "retrieval",
                progress=2,
                stage="preparing_pmids",
                message=(
                    "Preparing an isolated retrieval. The built-in query and "
                    "default corpus will not be used."
                ),
            )
            pmid_result = build_isolated_pmid_stage1(
                run_id=run_id,
                query=str(run.get("query") or ""),
                progress=pmid_progress,
                text_mode=text_mode,
            )
            run_registry.set_private(run_id, "stage1_default", None)
            run_registry.set_private(run_id, "stage1_custom", pmid_result)

            source_stats = dict(pmid_result.get("stats") or {})
            paper_count = _int(source_stats, "paper_count")
            abstract_count = _int(source_stats, "abstract_count")
            fulltext_count = _int(
                source_stats,
                "fulltext_available",
                "fulltexts_downloaded",
            )
            stats = {
                "paper_count": paper_count,
                "abstract_count": abstract_count,
                "fulltext_count": fulltext_count,
                "fulltexts_downloaded": fulltext_count,
                "papers_without_pmcid": _int(
                    source_stats, "papers_without_pmcid", "without_pmcid"
                ),
                "chunk_count": _chunk_count(source_stats),
                "default_paper_count": 0,
                "custom_paper_count": paper_count,
                "requested_pmid_count": _int(
                    source_stats, "explicit_pmid_selected", "user_pmid_count"
                ),
                "default_reused": False,
                "defaults_included": False,
                "input_mode": "pmid_only",
                "text_mode": text_mode,
                "abstract_only": text_mode == "abstract",
                "custom_recomputed": True,
                "custom_query_changed_results": True,
            }
            elapsed = round(time.monotonic() - started, 2)
            stats["elapsed_seconds"] = elapsed
            run_registry.complete_stage(
                run_id,
                "retrieval",
                message=(
                    f"Ready: {paper_count:,} papers retrieved only from the "
                    f"entered PMID list using {text_mode_label}."
                ),
                stats=stats,
                elapsed_seconds=elapsed,
            )
            return

        def default_progress(
            stage: str,
            progress: int,
            message: str,
            stats: dict[str, Any],
        ) -> None:
            _stage_progress(
                run_id,
                "retrieval",
                progress=_scaled(progress, 2, 98),
                stage=f"default_{stage}",
                message=f"Shared default corpus: {message}",
                stats=stats,
            )

        _stage_progress(
            run_id,
            "retrieval",
            progress=2,
            stage="loading_default_cache",
            message="Loading the shared default Stage 1 artifact.",
        )
        default_result = get_or_build_default_stage1(
            default_progress,
            text_mode=text_mode,
        )
        run_registry.set_private(run_id, "stage1_default", default_result)
        _stage_progress(
            run_id,
            "retrieval",
            progress=98,
            stage="default_ready",
            message=(
                "Reused the shared default Stage 1 artifact."
                if default_result.get("reused")
                else "Built and saved the shared default Stage 1 artifact."
            ),
            stats=dict(default_result.get("stats") or {}),
        )

        run_registry.set_private(run_id, "stage1_custom", None)
        default_stats = dict(default_result.get("stats") or {})
        paper_count = _int(default_stats, "paper_count")
        abstract_count = _int(default_stats, "abstract_count")
        fulltext_count = _int(
            default_stats,
            "fulltext_available",
            "fulltexts_downloaded",
        )
        stats = {
            "paper_count": paper_count,
            "abstract_count": abstract_count,
            "fulltext_count": fulltext_count,
            "fulltexts_downloaded": fulltext_count,
            "papers_without_pmcid": _int(
                default_stats, "papers_without_pmcid", "without_pmcid"
            ),
            "chunk_count": _chunk_count(default_stats),
            "default_paper_count": _int(default_stats, "paper_count"),
            "custom_paper_count": 0,
            "default_reused": bool((default_result or {}).get("reused")),
            "defaults_included": True,
            "input_mode": "default",
            "text_mode": text_mode,
            "abstract_only": text_mode == "abstract",
            "custom_recomputed": False,
            "custom_query_changed_results": False,
        }
        elapsed = round(time.monotonic() - started, 2)
        stats["elapsed_seconds"] = elapsed
        run_registry.complete_stage(
            run_id,
            "retrieval",
            message=(
                f"Ready: {paper_count:,} papers from the default corpus using "
                f"{text_mode_label}."
            ),
            stats=stats,
            elapsed_seconds=elapsed,
        )

    def _annotation_backend(self, callback_base_url: str) -> tuple[str, str]:
        backend = settings.cell_annotation_backend
        if backend == "local":
            if settings.artifact_backend != "local":
                raise RuntimeError("Local Stage 2 requires ARTIFACT_BACKEND=local.")
            missing_dependencies = missing_local_ml_dependencies(
                require_ab3p=not settings.cell_disable_abbreviations
            )
            if missing_dependencies:
                raise RuntimeError(
                    "Local CellExLink dependencies are missing: "
                    f"{', '.join(missing_dependencies)}. Install PyTorch and "
                    "requirements-local.txt in the same Python environment used "
                    "to run Uvicorn."
                )
            return backend, callback_base_url.rstrip("/")
        if backend == "modal":
            if settings.artifact_backend != "s3":
                raise RuntimeError(
                    "Modal Stage 2 requires an S3-compatible artifact backend."
                )
            if not settings.modal_configured:
                raise RuntimeError("Modal credentials or the deployed function are missing.")
            if not settings.public_base_url:
                raise RuntimeError("PUBLIC_BASE_URL is required for Modal callbacks.")
            return backend, settings.public_base_url.rstrip("/")
        raise RuntimeError("Stage 2 entity extraction is disabled.")

    def _annotation_artifact(
        self,
        *,
        run_id: str,
        scope: str,
        source: Mapping[str, Any],
        model_signature: str,
        callback_base_url: str,
        progress_start: int,
        progress_end: int,
    ) -> dict[str, Any]:
        self._assert_run_compute_allowed(run_id)
        source_ref = ArtifactRef.from_dict(source["artifact"])
        keys = annotation_artifact_keys(
            source_sha256=source_ref.sha256,
            model_signature=model_signature,
        )
        cached = cached_json_pair(
            output_key=keys.final_annotations,
            summary_key=keys.final_summary,
            expected_model_signature=model_signature,
            expected_source_sha256=source_ref.sha256,
        )
        if cached is not None:
            cached["worker_job_id"] = (
                "shared-default-stage2"
                if scope == "default"
                else f"{run_id}-run-stage2-cache"
            )
            cached["reused"] = scope == "default" or run_registry.get_private(run_id, "trusted_update", False)
            return cached

        backend, callback_root = self._annotation_backend(callback_base_url)
        worker_job_id = f"a2-{uuid.uuid4().hex[:20]}"
        callback_token = secrets.token_urlsafe(32)
        create_annotation_job(
            job_id=worker_job_id,
            source_job_id=f"{run_id}-{scope}-stage1",
            executor=backend,
            model_signature=model_signature,
            source_artifact_key=source_ref.key,
            source_artifact_sha256=source_ref.sha256,
            output_artifact_key=keys.final_annotations,
            summary_artifact_key=keys.final_summary,
            callback_token_hash=callback_token_hash(callback_token),
        )
        run_registry.set_private(run_id, "active_annotation_job_id", worker_job_id)
        self._assert_run_compute_allowed(run_id)
        store = get_artifact_store()
        source_stats = dict(source.get("stats") or {})
        update_annotation_job(
            worker_job_id,
            status="processing",
            stage="running_parallel_branches",
            progress=2,
            message=(
                "CellExLink and PubTator3 are starting for the shared default corpus."
                if scope == "default"
                else "CellExLink and PubTator3 are starting for this run's selected papers."
            ),
            stats={
                "cell_branch_status": "starting",
                "pubtator_branch_status": "starting",
                "cache_scope": "shared_default" if scope == "default" else "run_only",
            },
            paper_count=_int(source_stats, "paper_count"),
            chunk_count=_chunk_count(source_stats),
            started_at=utc_now(),
            last_remote_check_at=utc_now(),
        )

        if not _cell_branch_ready(get_annotation_job(worker_job_id) or {}):
            payload = {
                "job_id": worker_job_id,
                "pipeline_version": ANNOTATION_PIPELINE_VERSION,
                "model_signature": model_signature,
                "input": {
                    "url": store.presign_get(
                        source_ref.key,
                        expires_seconds=settings.artifact_presigned_ttl_seconds,
                    ),
                    "key": source_ref.key,
                    "sha256": source_ref.sha256,
                    "size_bytes": source_ref.size_bytes,
                },
                "output": {
                    "cell_annotations_url": store.presign_put(
                        keys.cell_annotations,
                        content_type="application/gzip",
                    ),
                    "cell_annotations_key": keys.cell_annotations,
                    "cell_summary_url": store.presign_put(
                        keys.cell_summary,
                        content_type="application/json",
                    ),
                    "cell_summary_key": keys.cell_summary,
                },
                "callback": {
                    "url": (
                        f"{callback_root}/api/internal/annotations/"
                        f"{worker_job_id}/callback"
                    ),
                    "token": callback_token,
                },
                "models": {
                    "ner": settings.cell_ner_model,
                    "ner_revision": settings.cell_ner_revision,
                    "nen": settings.cell_nen_model,
                    "nen_revision": settings.cell_nen_revision,
                },
                "options": {
                    "device": settings.cell_local_device,
                    "disable_abbreviations": settings.cell_disable_abbreviations,
                    "normalization_method_log": (
                        settings.cell_normalization_method_log
                    ),
                    "cpu_threads": settings.cell_cpu_threads,
                    "ner_text_batch_size": settings.cell_ner_text_batch_size,
                    "ner_window_batch_size": settings.cell_ner_window_batch_size,
                    "nen_batch_size": settings.cell_nen_batch_size,
                    "nen_request_batch_size": settings.cell_nen_request_batch_size,
                },
                "source_stats": {
                    "paper_count": _int(source_stats, "paper_count"),
                    "chunk_count": _chunk_count(source_stats),
                },
            }
            self._assert_run_compute_allowed(run_id)
            remote_call_id = (
                local_executor.submit(payload)
                if backend == "local"
                else modal_executor.submit(payload)
            )
            if backend == "modal":
                run_registry.set_private(run_id, "active_modal_call_id", str(remote_call_id))
            # Cancellation can arrive while Modal/local submit is in flight.
            self._assert_run_compute_allowed(run_id)
            update_annotation_job(
                worker_job_id,
                remote_call_id=remote_call_id,
                stats={
                    "cell_branch_status": "submitted",
                    "pubtator_branch_status": "submitted",
                    "cache_scope": (
                        "shared_default" if scope == "default" else "run_only"
                    ),
                },
            )
        else:
            cell_summary = store.read_json(keys.cell_summary)
            _apply_cell_result(
                get_annotation_job(worker_job_id) or {},
                {
                    "status": "cell_completed",
                    "stats": dict(cell_summary.get("stats") or {}),
                },
            )

        self._assert_run_compute_allowed(run_id)
        annotation_coordinator.submit_pubtator(worker_job_id)
        deadline = time.monotonic() + settings.cell_job_timeout_seconds
        last_remote_poll = 0.0
        while time.monotonic() < deadline:
            self._assert_run_compute_allowed(run_id)
            job = get_annotation_job(worker_job_id)
            if job is None:
                raise RuntimeError("The temporary Stage 2 worker record disappeared.")
            local_progress = int(job.get("progress") or 0)
            _stage_progress(
                run_id,
                "annotation",
                progress=_scaled(local_progress, progress_start, progress_end),
                stage=f"{scope}_{job.get('stage') or 'processing'}",
                message=(
                    f"Shared default entities: {job.get('message') or ''}"
                    if scope == "default"
                    else f"Run-specific entities: {job.get('message') or ''}"
                ),
                stats=dict(job.get("stats") or {}),
            )
            if job.get("status") == "completed":
                output_ref = store.head(keys.final_annotations)
                summary_ref = store.head(keys.final_summary)
                if output_ref is None or summary_ref is None:
                    raise RuntimeError("Stage 2 completed without its final artifacts.")
                summary = store.read_json(keys.final_summary)
                result = {
                    "artifact": output_ref.to_dict(),
                    "summary_artifact": summary_ref.to_dict(),
                    "summary": summary,
                    "stats": dict(summary.get("stats") or {}),
                    "worker_job_id": worker_job_id,
                    "reused": False,
                }
                if scope == "custom" and not run_registry.get_private(run_id, "trusted_update", False):
                    try:
                        store.delete(keys.final_summary)
                    except Exception as exc:
                        logger.warning(
                            "Could not remove temporary Stage 2 summary for run %s: %s",
                            run_id,
                            exc,
                        )
                    result.pop("summary_artifact", None)
                if backend == "local":
                    local_executor.cleanup(worker_job_id)
                return result
            if job.get("status") == "failed":
                if backend == "local":
                    local_executor.cleanup(worker_job_id, terminate=True)
                raise RuntimeError(
                    str(job.get("error") or job.get("message") or "Stage 2 failed.")
                )

            annotation_coordinator.ensure_job(worker_job_id)
            remote_call_id = str(job.get("remote_call_id") or "")
            if remote_call_id and time.monotonic() - last_remote_poll >= 1.5:
                last_remote_poll = time.monotonic()
                poll = (
                    local_executor.poll(remote_call_id)
                    if backend == "local"
                    else modal_executor.poll(remote_call_id)
                )
                if (
                    backend == "local"
                    and poll.state == "running"
                    and poll.progress
                ):
                    _apply_cell_progress(job, poll.progress)
                if poll.state == "completed":
                    _apply_cell_result(job, poll.result or {})
                elif poll.state in {"failed", "expired"}:
                    if _cell_branch_ready(job):
                        _apply_cell_result(
                            job,
                            {
                                "status": "cell_completed",
                                "message": "Recovered the published CellExLink branch.",
                            },
                        )
                    else:
                        update_annotation_job(
                            worker_job_id,
                            status="failed",
                            stage="failed",
                            progress=100,
                            message="The CellExLink branch failed.",
                            error=poll.error or "CellExLink executor failed.",
                            completed_at=utc_now(),
                        )
            time.sleep(1.0)
        if backend == "local":
            local_executor.cleanup(worker_job_id, terminate=True)
        raise TimeoutError("Stage 2 exceeded CELL_JOB_TIMEOUT_SECONDS.")

    def _run_stage2(self, run_id: str, *, callback_base_url: str) -> None:
        self._assert_run_compute_allowed(run_id)
        started = time.monotonic()
        run = run_registry.get(run_id)
        if run is None:
            return
        if str(run.get("input_mode") or "") == "precomputed":
            _stage_progress(
                run_id,
                "annotation",
                progress=22,
                stage="loading_saved_entities",
                message="Reading normalized entities from the saved prediction file.",
            )
            summary = run_registry.get_private(run_id, "precomputed_summary") or corpus_summary(
                str(run.get("corpus_id") or DEFAULT_CORPUS_ID)
            )
            stats = {
                "paper_count": _int(summary, "paper_count"),
                "chunk_count": _chunk_count(summary),
                "cell_count": _int(summary, "unique_cell_count"),
                "gene_count": _int(summary, "unique_gene_count"),
                "hormone_count": _int(summary, "unique_hormone_count"),
                "unique_cell_count": _int(summary, "unique_cell_count"),
                "unique_gene_count": _int(summary, "unique_gene_count"),
                "unique_hormone_count": _int(summary, "unique_hormone_count"),
                "unique_entity_count": _int(summary, "unique_entity_count"),
                "annotation_occurrence_count": _int(
                    summary, "annotation_occurrence_count"
                ),
                "precomputed": True,
            }
            _stage_progress(
                run_id,
                "annotation",
                progress=82,
                stage="summarizing_saved_entities",
                message="Counting globally unique normalized cells, genes/proteins, and hormones.",
                stats=stats,
            )
            _hold_precomputed_stage(started)
            elapsed = round(time.monotonic() - started, 2)
            stats["elapsed_seconds"] = elapsed
            run_registry.complete_stage(
                run_id,
                "annotation",
                message=(
                    f"Ready: {stats['unique_cell_count']:,} unique cell types, "
                    f"{stats['unique_gene_count']:,} unique genes/proteins, and "
                    f"{stats['unique_hormone_count']:,} unique hormones."
                ),
                stats=stats,
                elapsed_seconds=elapsed,
            )
            return

        sources = {
            scope: run_registry.get_private(run_id, f"stage1_{scope}")
            for scope in ("default", "custom")
        }
        scopes = [scope for scope, source in sources.items() if source is not None]
        if not scopes:
            raise RuntimeError("Stage 1 did not publish a source artifact.")

        base_signature = annotation_model_signature()
        results: dict[str, dict[str, Any]] = {}
        for index, scope in enumerate(scopes):
            progress_start, progress_end = _scope_progress_bounds(index, len(scopes))
            model_signature = (
                base_signature
                if scope == "default" or run_registry.get_private(run_id, "trusted_update", False)
                else run_scoped_signature(run_id, base_signature)
            )
            results[scope] = self._annotation_artifact(
                run_id=run_id,
                scope=scope,
                source=sources[scope],
                model_signature=model_signature,
                callback_base_url=callback_base_url,
                progress_start=progress_start,
                progress_end=progress_end,
            )

        default_result = results.get("default")
        custom_result = results.get("custom")
        run_registry.set_private(run_id, "stage2_default", default_result)
        run_registry.set_private(run_id, "stage2_custom", custom_result)

        unique_totals = {
            "unique_cell_count": 0,
            "unique_gene_count": 0,
            "unique_hormone_count": 0,
            "unique_entity_count": 0,
            "annotation_occurrence_count": 0,
        }
        summary_root = settings.data_dir / "runs" / run_id / "stage2-summary"
        summary_root.mkdir(parents=True, exist_ok=True)
        for scope, result in results.items():
            artifact = result.get("artifact") if isinstance(result, Mapping) else None
            if not isinstance(artifact, Mapping):
                continue
            local_path = summary_root / f"{scope}-annotations.jsonl.gz"
            materialize_artifact(artifact, local_path)
            unique_stats = summarize_entity_annotations(local_path)
            result["unique_stats"] = unique_stats
            for key in unique_totals:
                unique_totals[key] += _int(unique_stats, key)
        shutil.rmtree(summary_root, ignore_errors=True)

        default_stats = dict((default_result or {}).get("stats") or {})
        custom_stats = dict((custom_result or {}).get("stats") or {})
        stats = {
            "paper_count": _int(default_stats, "paper_count")
            + _int(custom_stats, "paper_count"),
            "chunk_count": _chunk_count(default_stats) + _chunk_count(custom_stats),
            "cell_count": unique_totals["unique_cell_count"],
            "gene_count": unique_totals["unique_gene_count"],
            "hormone_count": unique_totals["unique_hormone_count"],
            **unique_totals,
            "mention_count": _int(default_stats, "cell_count", "mention_count")
            + _int(custom_stats, "cell_count", "mention_count"),
            "normalized_count": _int(
                default_stats, "normalized_count", "normalized_occurrences"
            )
            + _int(custom_stats, "normalized_count", "normalized_occurrences"),
            "default_reused": bool((default_result or {}).get("reused")),
            "custom_recomputed": custom_result is not None,
        }
        elapsed = round(time.monotonic() - started, 2)
        stats["elapsed_seconds"] = elapsed
        run_registry.complete_stage(
            run_id,
            "annotation",
            message=(
                f"Ready: {stats['unique_cell_count']:,} unique cell types, "
                f"{stats['unique_gene_count']:,} unique genes/proteins, and "
                f"{stats['unique_hormone_count']:,} unique hormones."
            ),
            stats=stats,
            elapsed_seconds=elapsed,
        )

    def handle_annotation_callback(
        self,
        worker_job_id: str,
        *,
        token: str,
        payload: Mapping[str, Any],
    ) -> None:
        job = get_annotation_job(worker_job_id)
        if job is None:
            raise KeyError(worker_job_id)
        if not callback_token_matches(token, job.get("callback_token_hash")):
            raise PermissionError("Invalid callback token.")
        if job.get("status") in {"completed", "failed"}:
            return
        updated = _apply_cell_progress(job, payload)
        if updated.get("status") != "failed":
            annotation_coordinator.ensure_job(worker_job_id)

    def _ensure_final_annotation_artifact(
        self,
        *,
        run_id: str,
        scope: str,
        chunks_ref: ArtifactRef,
        annotations_ref: ArtifactRef,
        relations_ref: ArtifactRef,
        model_signature: str,
        progress_value: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create the Stage 3 download without changing Stage 4 inputs."""

        store = get_artifact_store()
        key = final_annotation_artifact_key(
            source_annotation_sha256=annotations_ref.sha256,
            source_chunks_sha256=chunks_ref.sha256,
            relation_sha256=relations_ref.sha256,
            model_signature=model_signature,
        )
        cached_ref = store.head(key)
        if cached_ref is not None:
            return cached_ref.to_dict(), {"reused": True}

        _stage_progress(
            run_id,
            "relation",
            progress=min(99, max(1, int(progress_value))),
            stage=f"{scope}_assembling_final_annotations",
            message=(
                "Assembling the final annotated paper download for the shared "
                "default corpus."
                if scope == "default"
                else "Assembling the final annotated paper download for this run."
            ),
        )
        work_root = (
            settings.data_dir
            / "work"
            / f"final-annotations-{run_id}-{scope}-{uuid.uuid4().hex}"
        )
        chunks_path = work_root / "chunks.jsonl.gz"
        annotations_path = work_root / "annotations.jsonl.gz"
        relations_path = work_root / "relations.jsonl.gz"
        output_path = work_root / "ovarian-final-annotations.jsonl.gz"
        work_root.mkdir(parents=True, exist_ok=True)
        try:
            materialize_artifact(chunks_ref, chunks_path, store=store)
            materialize_artifact(annotations_ref, annotations_path, store=store)
            materialize_artifact(relations_ref, relations_path, store=store)
            export_stats = build_final_annotation_artifact(
                chunks_path=chunks_path,
                annotations_path=annotations_path,
                relations_path=relations_path,
                output_path=output_path,
            )
            output_ref, reused = store.put_file(
                output_path,
                key=key,
                content_type="application/gzip",
                sha256=sha256_file(output_path),
            )
            export_stats["reused"] = bool(reused)
            return output_ref.to_dict(), export_stats
        finally:
            shutil.rmtree(work_root, ignore_errors=True)

    def _relation_artifact(
        self,
        *,
        run_id: str,
        scope: str,
        chunks: Mapping[str, Any],
        annotations: Mapping[str, Any],
        model_signature: str,
        progress_start: int,
        progress_end: int,
    ) -> dict[str, Any]:
        self._assert_run_compute_allowed(run_id)
        chunks_ref = ArtifactRef.from_dict(chunks["artifact"])
        annotations_ref = ArtifactRef.from_dict(annotations["artifact"])
        keys = relation_artifact_keys(
            source_annotation_sha256=annotations_ref.sha256,
            source_chunks_sha256=chunks_ref.sha256,
            model_signature=model_signature,
        )
        cached = cached_json_pair(
            output_key=keys.relations,
            summary_key=keys.summary,
            expected_model_signature=model_signature,
        )
        if cached is not None:
            cached["worker_job_id"] = (
                "shared-default-stage3"
                if scope == "default"
                else f"{run_id}-run-stage3-cache"
            )
            cached["reused"] = scope == "default" or run_registry.get_private(run_id, "trusted_update", False)
            relations_ref = ArtifactRef.from_dict(cached["artifact"])
            final_ref, final_stats = self._ensure_final_annotation_artifact(
                run_id=run_id,
                scope=scope,
                chunks_ref=chunks_ref,
                annotations_ref=annotations_ref,
                relations_ref=relations_ref,
                model_signature=model_signature,
                progress_value=max(progress_start, progress_end - 1),
            )
            cached["final_annotations_artifact"] = final_ref
            cached["final_annotations_stats"] = final_stats
            return cached
        if not settings.relation_configured:
            raise RuntimeError("OPENAI_API_KEY is required for Stage 3.")

        worker_job_id = f"r3-{uuid.uuid4().hex[:20]}"
        create_relation_job(
            job_id=worker_job_id,
            source_annotation_job_id=str(
                annotations.get("worker_job_id") or f"{run_id}-{scope}-stage2"
            ),
            model_signature=model_signature,
            source_chunks_artifact_key=chunks_ref.key,
            source_chunks_artifact_sha256=chunks_ref.sha256,
            source_annotation_artifact_key=annotations_ref.key,
            source_annotation_artifact_sha256=annotations_ref.sha256,
            output_artifact_key=keys.relations,
            summary_artifact_key=keys.summary,
        )
        run_registry.set_private(run_id, "active_relation_job_id", worker_job_id)
        self._assert_run_compute_allowed(run_id)
        annotation_stats = dict(annotations.get("stats") or {})
        chunk_stats = dict(chunks.get("stats") or {})
        update_relation_job(
            worker_job_id,
            status="processing",
            stage="preparing_relations",
            progress=1,
            message=(
                "Preparing shared default relation extraction."
                if scope == "default"
                else "Preparing relations for this run's selected papers."
            ),
            paper_count=_int(chunk_stats, "paper_count"),
            chunk_count=_chunk_count(annotation_stats) or _chunk_count(chunk_stats),
            stats={
                "cache_scope": "shared_default" if scope == "default" else "run_only"
            },
            started_at=utc_now(),
        )
        self._assert_run_compute_allowed(run_id)
        relation_executor.submit(worker_job_id)
        self._assert_run_compute_allowed(run_id)
        while True:
            self._assert_run_compute_allowed(run_id)
            job = get_relation_job(worker_job_id)
            if job is None:
                raise RuntimeError("The temporary Stage 3 worker record disappeared.")
            _stage_progress(
                run_id,
                "relation",
                progress=_scaled(
                    int(job.get("progress") or 0), progress_start, progress_end
                ),
                stage=f"{scope}_{job.get('stage') or 'processing'}",
                message=(
                    f"Shared default relations: {job.get('message') or ''}"
                    if scope == "default"
                    else f"Run-specific relations: {job.get('message') or ''}"
                ),
                stats=dict(job.get("stats") or {}),
            )
            if job.get("status") == "completed":
                store = get_artifact_store()
                output_ref = store.head(keys.relations)
                summary_ref = store.head(keys.summary)
                if output_ref is None or summary_ref is None:
                    raise RuntimeError("Stage 3 completed without its final artifacts.")
                summary = store.read_json(keys.summary)
                final_ref, final_stats = self._ensure_final_annotation_artifact(
                    run_id=run_id,
                    scope=scope,
                    chunks_ref=chunks_ref,
                    annotations_ref=annotations_ref,
                    relations_ref=output_ref,
                    model_signature=model_signature,
                    progress_value=max(progress_start, progress_end - 1),
                )
                result = {
                    "artifact": output_ref.to_dict(),
                    "final_annotations_artifact": final_ref,
                    "final_annotations_stats": final_stats,
                    "summary_artifact": summary_ref.to_dict(),
                    "summary": summary,
                    "stats": dict(summary.get("stats") or {}),
                    "worker_job_id": worker_job_id,
                    "reused": False,
                }
                if scope == "custom" and not run_registry.get_private(run_id, "trusted_update", False):
                    try:
                        store.delete(keys.summary)
                    except Exception as exc:
                        logger.warning(
                            "Could not remove temporary Stage 3 summary for run %s: %s",
                            run_id,
                            exc,
                        )
                    result.pop("summary_artifact", None)
                return result
            if job.get("status") == "failed":
                raise RuntimeError(str(job.get("error") or job.get("message") or "Stage 3 failed."))
            self._assert_run_compute_allowed(run_id)
            relation_executor.submit(worker_job_id)
            time.sleep(1.0)

    def _run_stage3(self, run_id: str) -> None:
        self._assert_run_compute_allowed(run_id)
        started = time.monotonic()
        run = run_registry.get(run_id)
        if run is None:
            return

        if str(run.get("input_mode") or "") == "precomputed":
            _stage_progress(
                run_id,
                "relation",
                progress=24,
                stage="loading_saved_relations",
                message="Reading normalized directed relations from the saved prediction file.",
            )
            prediction_path = Path(
                str(
                    run_registry.get_private(run_id, "precomputed_prediction_path")
                    or corpus_path(str(run.get("corpus_id") or DEFAULT_CORPUS_ID))
                )
            ).expanduser().resolve()
            summary = run_registry.get_private(
                run_id, "precomputed_summary"
            ) or corpus_summary(str(run.get("corpus_id") or DEFAULT_CORPUS_ID))
            run_registry.set_private(
                run_id, "stage3_prediction_paths", [str(prediction_path)]
            )
            stats = {
                "paper_count": _int(summary, "paper_count"),
                "chunk_count": _chunk_count(summary),
                "relation_count": _int(summary, "unique_paper_relation_count"),
                "global_relation_count": _int(summary, "global_relation_count"),
                "relation_occurrence_count": _int(
                    summary, "relation_occurrence_count"
                ),
                "duplicate_paper_relation_count": _int(
                    summary, "duplicate_paper_relation_count"
                ),
                "papers_with_relations": _int(summary, "papers_with_relations"),
                "precomputed": True,
            }
            _stage_progress(
                run_id,
                "relation",
                progress=84,
                stage="summarizing_saved_relations",
                message="Deduplicating directed paper-supported relations for the network.",
                stats=stats,
            )
            _hold_precomputed_stage(started)
            elapsed = round(time.monotonic() - started, 2)
            stats["elapsed_seconds"] = elapsed
            run_registry.complete_stage(
                run_id,
                "relation",
                message=(
                    f"Ready: {stats['relation_count']:,} paper-supported relations "
                    f"across {stats['global_relation_count']:,} normalized network edges."
                ),
                stats=stats,
                elapsed_seconds=elapsed,
                download_url=f"/api/runs/{run_id}/download/stage3",
            )
            return

        chunks = {
            scope: run_registry.get_private(run_id, f"stage1_{scope}")
            for scope in ("default", "custom")
        }
        annotations = {
            scope: run_registry.get_private(run_id, f"stage2_{scope}")
            for scope in ("default", "custom")
        }
        scopes = [
            scope
            for scope in ("default", "custom")
            if chunks[scope] is not None and annotations[scope] is not None
        ]
        if not scopes:
            raise RuntimeError("Stage 2 did not publish aligned source artifacts.")

        base_signature = relation_model_signature()
        results: dict[str, dict[str, Any]] = {}
        for index, scope in enumerate(scopes):
            progress_start, progress_end = _scope_progress_bounds(index, len(scopes))
            model_signature = (
                base_signature
                if scope == "default" or run_registry.get_private(run_id, "trusted_update", False)
                else run_scoped_signature(run_id, base_signature)
            )
            results[scope] = self._relation_artifact(
                run_id=run_id,
                scope=scope,
                chunks=chunks[scope],
                annotations=annotations[scope],
                model_signature=model_signature,
                progress_start=progress_start,
                progress_end=progress_end,
            )

        default_result = results.get("default")
        custom_result = results.get("custom")
        run_registry.set_private(run_id, "stage3_default", default_result)
        run_registry.set_private(run_id, "stage3_custom", custom_result)

        prediction_root = settings.data_dir / "runs" / run_id / "stage3"
        shutil.rmtree(prediction_root, ignore_errors=True)
        prediction_root.mkdir(parents=True, exist_ok=True)
        prediction_paths: list[str] = []
        prediction_summaries: list[dict[str, Any]] = []
        for scope in scopes:
            result = results.get(scope) or {}
            artifact = result.get("final_annotations_artifact")
            if not isinstance(artifact, Mapping):
                raise RuntimeError(
                    f"Stage 3 did not publish the final {scope} prediction artifact."
                )
            destination = prediction_root / f"final-{scope}.jsonl.gz"
            materialize_artifact(artifact, destination)
            prediction_paths.append(str(destination.resolve()))
            prediction_summaries.append(summarize_prediction_file(destination))
        run_registry.set_private(run_id, "stage3_prediction_paths", prediction_paths)

        default_stats = dict((default_result or {}).get("stats") or {})
        custom_stats = dict((custom_result or {}).get("stats") or {})
        input_tokens = _int(default_stats, "input_tokens") + _int(
            custom_stats, "input_tokens"
        )
        cached_tokens = _int(default_stats, "cached_input_tokens") + _int(
            custom_stats, "cached_input_tokens"
        )
        stats = {
            "paper_count": sum(_int(value, "paper_count") for value in prediction_summaries),
            "chunk_count": sum(_chunk_count(value) for value in prediction_summaries),
            "relation_count": sum(
                _int(value, "unique_paper_relation_count")
                for value in prediction_summaries
            ),
            "global_relation_count": sum(
                _int(value, "global_relation_count") for value in prediction_summaries
            ),
            "relation_occurrence_count": sum(
                _int(value, "relation_occurrence_count")
                for value in prediction_summaries
            ),
            "duplicate_paper_relation_count": sum(
                _int(value, "duplicate_paper_relation_count")
                for value in prediction_summaries
            ),
            "papers_with_relations": sum(
                _int(value, "papers_with_relations") for value in prediction_summaries
            ),
            "eligible_chunk_count": _int(default_stats, "eligible_chunk_count")
            + _int(custom_stats, "eligible_chunk_count"),
            "api_request_count": _int(default_stats, "api_request_count")
            + _int(custom_stats, "api_request_count"),
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "prompt_cache_rate": round(cached_tokens / max(1, input_tokens), 6),
            "default_reused": bool((default_result or {}).get("reused")),
            "custom_recomputed": custom_result is not None,
        }
        elapsed = round(time.monotonic() - started, 2)
        stats["elapsed_seconds"] = elapsed
        run_registry.complete_stage(
            run_id,
            "relation",
            message=(
                f"Ready: {stats['relation_count']:,} paper-supported relations "
                f"across {stats['global_relation_count']:,} normalized network edges."
            ),
            stats=stats,
            elapsed_seconds=elapsed,
            download_url=f"/api/runs/{run_id}/download/stage3",
        )

    def _run_stage4(self, run_id: str) -> None:
        started = time.monotonic()
        work_root = settings.data_dir / "runs" / run_id / "stage4"
        shutil.rmtree(work_root, ignore_errors=True)
        work_root.mkdir(parents=True, exist_ok=True)

        raw_paths = run_registry.get_private(run_id, "stage3_prediction_paths", [])
        prediction_paths = [
            Path(str(value)).expanduser().resolve()
            for value in raw_paths
            if str(value or "").strip()
        ]
        if not prediction_paths:
            run = run_registry.get(run_id) or {}
            if str(run.get("input_mode") or "") == "precomputed":
                prediction_paths = [
                    Path(str(run_registry.get_private(run_id, "precomputed_prediction_path")))
                    if run_registry.get_private(run_id, "precomputed_prediction_path")
                    else corpus_path(str(run.get("corpus_id") or DEFAULT_CORPUS_ID))
                ]
            else:
                fallback_root = work_root / "prediction-inputs"
                fallback_root.mkdir(parents=True, exist_ok=True)
                for scope in ("default", "custom"):
                    source = run_registry.get_private(run_id, f"stage3_{scope}")
                    artifact = (
                        source.get("final_annotations_artifact")
                        if isinstance(source, Mapping)
                        else None
                    )
                    if not isinstance(artifact, Mapping):
                        continue
                    destination = fallback_root / f"final-{scope}.jsonl.gz"
                    materialize_artifact(artifact, destination)
                    prediction_paths.append(destination.resolve())
        if not prediction_paths or any(not path.is_file() for path in prediction_paths):
            raise RuntimeError("The final Stage 3 prediction file is unavailable.")

        _stage_progress(
            run_id,
            "network",
            progress=6,
            stage="preparing_network",
            message=(
                "Preparing relation-bearing nodes and deduplicated paper support "
                "from the final prediction file."
            ),
        )
        graph_path = work_root / "graph.sqlite"
        entity_index_path = work_root / "entity-index.jsonl.gz"
        total_chunks = _int(
            run_registry.public(run_id)["stages"]["relation"].get("stats") or {},
            "chunk_count",
        )

        def progress(row_count: int, message: str, stats: dict[str, Any]) -> None:
            percentage = 8 + round(
                86 * min(max(1, total_chunks), row_count) / max(1, total_chunks)
            )
            _stage_progress(
                run_id,
                "network",
                progress=min(96, percentage),
                stage="building_network",
                message=message,
                stats=stats,
            )

        result = build_interaction_network_from_final_annotations(
            prediction_paths=prediction_paths,
            graph_path=graph_path,
            entity_index_path=entity_index_path,
            progress=progress,
        )
        run_registry.set_private(run_id, "graph_path", str(result.graph_path))
        run_registry.set_private(
            run_id, "entity_index_path", str(result.entity_index_path)
        )
        stats = dict(result.stats)
        elapsed = round(time.monotonic() - started, 2)
        stats["elapsed_seconds"] = elapsed
        stats["persistent_artifact"] = False
        run_registry.complete_stage(
            run_id,
            "network",
            message=(
                f"Ready: {int(stats.get('node_count') or 0):,} relation-bearing "
                f"nodes and {int(stats.get('edge_count') or 0):,} edges."
            ),
            stats=stats,
            elapsed_seconds=elapsed,
            open_url=f"/network/{run_id}",
        )

    def artifacts_for_download(
        self, run_id: str, stage_name: str
    ) -> list[dict[str, Any]]:
        if stage_name != "stage3":
            raise ValueError("Only the final Stage 3 annotation export is downloadable.")
        refs: list[dict[str, Any]] = []
        default = run_registry.get_private(run_id, "stage3_default")
        custom = run_registry.get_private(run_id, "stage3_custom")
        for value in (default, custom):
            if not isinstance(value, Mapping):
                continue
            artifact = value.get("final_annotations_artifact")
            if isinstance(artifact, Mapping):
                refs.append(dict(artifact))
        return refs


pipeline_orchestrator = PipelineOrchestrator()

__all__ = [
    "PipelineOrchestrator",
    "RetrievalError",
    "pipeline_orchestrator",
]
