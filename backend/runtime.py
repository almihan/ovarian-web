"""Ephemeral per-page pipeline state.

A run exists only in this FastAPI process and is addressed by a random ID kept
in the current page's JavaScript memory. Nothing here is written to SQLite,
cookies, browser storage, or a user-history table.
"""

from __future__ import annotations

import secrets
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from backend.config import settings

_STAGE_ORDER = ("retrieval", "annotation", "relation", "network")
_STAGE_LABELS = {
    "retrieval": "Stage 1",
    "annotation": "Stage 2",
    "relation": "Stage 3",
    "network": "Stage 4",
}
_ACTIVE_STATUSES = {"queued", "processing"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _new_stage(name: str, *, first: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "label": _STAGE_LABELS[name],
        "status": "ready" if first else "locked",
        "stage": "ready" if first else "locked",
        "progress": 0,
        "message": (
            "Ready to retrieve the shared default corpus and this run's additions."
            if first
            else "Complete the preceding stage to continue."
        ),
        "stats": {},
        "error": None,
        "started_at": None,
        "completed_at": None,
        "elapsed_seconds": 0.0,
        "download_url": None,
        "open_url": None,
    }


class RunRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._runs: dict[str, dict[str, Any]] = {}

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()

    def create(self, *, query: str, has_custom_input: bool) -> dict[str, Any]:
        run_id = secrets.token_hex(16)
        now = utc_now()
        record = {
            "id": run_id,
            "query": query,
            "has_custom_input": bool(has_custom_input),
            "created_at": now,
            "updated_at": now,
            "stages": {
                name: _new_stage(name, first=(index == 0))
                for index, name in enumerate(_STAGE_ORDER)
            },
            "private": {},
        }
        with self._lock:
            self._runs[run_id] = record
        return self.public(run_id)

    def exists(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._runs

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._runs.get(run_id)
            return deepcopy(value) if value is not None else None

    def public(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            return {
                key: deepcopy(value)
                for key, value in record.items()
                if key != "private"
            }

    def update_stage(
        self,
        run_id: str,
        stage_name: str,
        **fields: Any,
    ) -> dict[str, Any]:
        if stage_name not in _STAGE_ORDER:
            raise KeyError(stage_name)
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            stage = record["stages"][stage_name]
            for key, value in fields.items():
                stage[key] = deepcopy(value)
            stage["progress"] = max(0, min(100, int(stage.get("progress") or 0)))
            record["updated_at"] = utc_now()
            return deepcopy(stage)

    def start_stage(self, run_id: str, stage_name: str, message: str) -> None:
        self.update_stage(
            run_id,
            stage_name,
            status="queued",
            stage="queued",
            progress=0,
            message=message,
            error=None,
            started_at=utc_now(),
            completed_at=None,
            elapsed_seconds=0.0,
        )

    def complete_stage(
        self,
        run_id: str,
        stage_name: str,
        *,
        message: str,
        stats: dict[str, Any],
        elapsed_seconds: float,
        download_url: str | None = None,
        open_url: str | None = None,
    ) -> None:
        self.update_stage(
            run_id,
            stage_name,
            status="completed",
            stage="completed",
            progress=100,
            message=message,
            stats=stats,
            error=None,
            elapsed_seconds=max(0.0, float(elapsed_seconds)),
            completed_at=utc_now(),
            download_url=download_url,
            open_url=open_url,
        )
        index = _STAGE_ORDER.index(stage_name)
        if index + 1 >= len(_STAGE_ORDER):
            return
        next_name = _STAGE_ORDER[index + 1]
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return
            next_stage = record["stages"][next_name]
            if next_stage.get("status") == "locked":
                next_stage.update(
                    {
                        "status": "ready",
                        "stage": "ready",
                        "progress": 0,
                        "message": f"{_STAGE_LABELS[next_name]} is ready.",
                    }
                )
                record["updated_at"] = utc_now()

    def fail_stage(self, run_id: str, stage_name: str, error: str) -> None:
        current = self.get(run_id)
        if current is None:
            return
        self.update_stage(
            run_id,
            stage_name,
            status="failed",
            stage="failed",
            progress=100,
            message="This stage could not be completed.",
            error=str(error),
            completed_at=utc_now(),
            started_at=current["stages"][stage_name].get("started_at"),
        )

    def set_private(self, run_id: str, key: str, value: Any) -> None:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            record["private"][key] = deepcopy(value)
            record["updated_at"] = utc_now()

    def reset_downstream_stages(self, run_id: str, stage_name: str) -> None:
        """Discard results that depend on a stage before that stage is rerun."""

        if stage_name not in _STAGE_ORDER:
            raise KeyError(stage_name)
        index = _STAGE_ORDER.index(stage_name)
        private_prefixes = tuple(
            f"stage{number}_" for number in range(index + 1, len(_STAGE_ORDER) + 1)
        )
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            for name in _STAGE_ORDER[index + 1 :]:
                record["stages"][name] = _new_stage(name)
            for key in tuple(record["private"]):
                if key.startswith(private_prefixes) or key in {"graph_path", "entity_index_path"}:
                    del record["private"][key]
            record["updated_at"] = utc_now()

    def get_private(self, run_id: str, key: str, default: Any = None) -> Any:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            return deepcopy(record["private"].get(key, default))

    def pop_expired(self, max_age_seconds: int) -> list[dict[str, Any]]:
        """Remove inactive runs older than the small temporary retention window."""

        cutoff = max(3600, int(max_age_seconds))
        now = datetime.now(timezone.utc)
        removed: list[dict[str, Any]] = []
        with self._lock:
            for run_id, record in tuple(self._runs.items()):
                stages = record.get("stages") or {}
                if any(
                    (stages.get(name) or {}).get("status") in _ACTIVE_STATUSES
                    for name in _STAGE_ORDER
                ):
                    continue
                try:
                    updated = datetime.fromisoformat(
                        str(record.get("updated_at") or "").replace("Z", "+00:00")
                    )
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=timezone.utc)
                except ValueError:
                    updated = now
                if (now - updated).total_seconds() < cutoff:
                    continue
                removed.append(deepcopy(record))
                del self._runs[run_id]
        return removed

    def graph_path(self, run_id: str) -> Path:
        raw = self.get_private(run_id, "graph_path")
        if not raw:
            raise FileNotFoundError("The temporary network is not available.")
        path = Path(str(raw)).expanduser().resolve()
        root = (settings.data_dir / "runs").expanduser().resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError("The temporary network is not available.")
        if self.public(run_id)["stages"]["network"].get("status") != "completed":
            raise RuntimeError("The network is not ready yet.")
        return path


run_registry = RunRegistry()

__all__ = ["RunRegistry", "run_registry", "utc_now"]
