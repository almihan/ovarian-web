"""Short-lived local CellExLink worker launcher.

The FastAPI process remains lightweight. Each local annotation runs in a
separate Python process, loads recognition and normalization sequentially, and
exits after publishing the compressed result. Child stdout and stderr inherit
the Uvicorn console so optional normalization-method diagnostics are visible.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from backend.config import PROJECT_ROOT, settings


@dataclass(slots=True, frozen=True)
class LocalPollResult:
    state: str
    result: dict[str, Any] | None = None
    progress: dict[str, Any] | None = None
    error: str | None = None


_LOCAL_ML_DEPENDENCIES = {
    "torch": "PyTorch",
    "transformers": "transformers",
    "huggingface_hub": "huggingface-hub",
    "numpy": "NumPy",
    "pyab3p": "pyab3p",
}


def missing_local_ml_dependencies(
    *,
    require_ab3p: bool,
) -> tuple[str, ...]:
    return tuple(
        display_name
        for module_name, display_name in _LOCAL_ML_DEPENDENCIES.items()
        if (require_ab3p or module_name != "pyab3p")
        and importlib.util.find_spec(module_name) is None
    )


def local_ml_dependencies_available(
    *,
    require_ab3p: bool | None = None,
) -> bool:
    if require_ab3p is None:
        require_ab3p = not settings.cell_disable_abbreviations
    return not missing_local_ml_dependencies(require_ab3p=bool(require_ab3p))


class LocalAnnotationExecutor:
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._processes: dict[str, subprocess.Popen[Any]] = {}

    def _job_dir(self, job_id: str) -> Path:
        clean = "".join(
            character
            for character in str(job_id)
            if character.isalnum() or character in "-_"
        )
        if not clean:
            raise ValueError("Local annotation job ID is empty.")
        root = settings.local_annotation_jobs_dir.expanduser().resolve()
        job_dir = (root / clean).resolve()
        if not job_dir.is_relative_to(root):
            raise ValueError("Unsafe local annotation job path.")
        return job_dir

    def submit(self, payload: Mapping[str, Any]) -> str:
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            raise ValueError("The local annotation payload has no job ID.")
        options = (
            payload.get("options")
            if isinstance(payload.get("options"), Mapping)
            else {}
        )
        missing_dependencies = missing_local_ml_dependencies(
            require_ab3p=not bool(options.get("disable_abbreviations", False))
        )
        if missing_dependencies:
            raise RuntimeError(
                "Local CellExLink dependencies are missing: "
                f"{', '.join(missing_dependencies)}. Install PyTorch and "
                "requirements-local.txt in the same Python environment used to "
                "run Uvicorn before starting Stage 2."
            )

        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        payload_path = job_dir / "payload.json"
        result_path = job_dir / "result.json"
        progress_path = job_dir / "progress.json"
        result_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

        worker_payload = dict(payload)
        callback = (
            dict(worker_payload.get("callback") or {})
            if isinstance(worker_payload.get("callback"), Mapping)
            else {}
        )
        # A local worker does not need to make an HTTP callback to the same
        # Uvicorn process. It writes its latest progress atomically instead,
        # and the orchestrator reads that file while polling the child process.
        callback.update(
            {
                "url": "",
                "token": "",
                "local_progress_path": str(progress_path),
            }
        )
        worker_payload["callback"] = callback
        worker_payload["local_control"] = {
            "result_path": str(result_path),
            "progress_path": str(progress_path),
            "model_cache_root": str(settings.cell_model_cache_dir),
        }
        payload_path.write_text(
            json.dumps(worker_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        environment = os.environ.copy()
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        environment.setdefault("TRANSFORMERS_VERBOSITY", "error")
        environment.setdefault("HF_HUB_VERBOSITY", "error")
        environment.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        environment.setdefault("TQDM_DISABLE", "1")
        environment.setdefault("PYTHONUNBUFFERED", "1")
        environment["CELL_LOCAL_DEVICE"] = settings.cell_local_device
        environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        local_no_proxy = "127.0.0.1,localhost"
        environment["NO_PROXY"] = ",".join(
            item
            for item in (environment.get("NO_PROXY", ""), local_no_proxy)
            if item
        )
        environment["no_proxy"] = environment["NO_PROXY"]
        if settings.cell_local_device == "cpu":
            # Keep a local CPU run deterministic even on a workstation that has
            # CUDA. Set CELL_LOCAL_DEVICE=auto later to allow local GPU use.
            environment["CUDA_VISIBLE_DEVICES"] = ""

        popen_kwargs: dict[str, Any] = {
            "cwd": str(PROJECT_ROOT),
            "env": environment,
            "stdin": subprocess.DEVNULL,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(
            [sys.executable, "-m", "backend.local_worker", str(payload_path)],
            **popen_kwargs,
        )
        with self._guard:
            self._processes[job_id] = process
        return f"{job_id}:{process.pid}"

    def cleanup(self, job_id: str, *, terminate: bool = False) -> None:
        """Remove one finished local worker's small control directory."""

        with self._guard:
            process = self._processes.pop(job_id, None)
        if process is not None:
            return_code = process.poll()
            if return_code is None and terminate:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            elif return_code is not None:
                process.wait(timeout=0)
        shutil.rmtree(self._job_dir(job_id), ignore_errors=True)

    def shutdown(self) -> None:
        """Stop any local CellExLink subprocesses owned by this app process."""

        with self._guard:
            job_ids = tuple(self._processes)
        for job_id in job_ids:
            self.cleanup(job_id, terminate=True)

    def poll(self, call_id: str) -> LocalPollResult:
        raw = str(call_id or "")
        job_id, separator, pid_text = raw.partition(":")
        if not separator or not job_id:
            return LocalPollResult(
                state="failed", error="Invalid local worker call ID."
            )

        result_path = self._job_dir(job_id) / "result.json"
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return LocalPollResult(
                    state="failed", error=f"Could not read local worker result: {exc}"
                )
            if not isinstance(payload, dict):
                return LocalPollResult(
                    state="failed", error="Local worker returned an invalid result."
                )
            state = str(payload.get("state") or "failed").casefold()
            if state == "completed":
                result = payload.get("result")
                if isinstance(result, dict):
                    return LocalPollResult(state="completed", result=dict(result))
                return LocalPollResult(
                    state="failed", error="Local worker completion result is missing."
                )
            return LocalPollResult(
                state="failed",
                error=str(payload.get("error") or "Local annotation failed."),
            )

        progress: dict[str, Any] | None = None
        progress_path = self._job_dir(job_id) / "progress.json"
        if progress_path.is_file():
            try:
                raw_progress = json.loads(progress_path.read_text(encoding="utf-8"))
                if isinstance(raw_progress, dict):
                    progress = dict(raw_progress)
            except (OSError, json.JSONDecodeError):
                # The next poll will retry. Progress is informational and must
                # not fail an otherwise healthy local CellExLink process.
                progress = None

        with self._guard:
            process = self._processes.get(job_id)
        if process is not None:
            return_code = process.poll()
            if return_code is None:
                return LocalPollResult(state="running", progress=progress)
            with self._guard:
                self._processes.pop(job_id, None)
            # The worker writes result.json atomically immediately before it
            # exits. Recheck after observing its exit to avoid a small race.
            if result_path.is_file():
                return self.poll(call_id)
            detail = ""
            if progress:
                detail = str(
                    progress.get("error")
                    or progress.get("message")
                    or ""
                ).strip()
            suffix = f" Last worker message: {detail}" if detail else ""
            return LocalPollResult(
                state="failed",
                error=(
                    "The local CellExLink process exited with code "
                    f"{return_code} without writing result.json.{suffix}"
                ),
            )

        try:
            pid = int(pid_text)
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError):
            return LocalPollResult(
                state="failed",
                error="The local annotation process exited without a result file.",
            )
        except PermissionError:
            # The process exists but belongs to another security context.
            return LocalPollResult(state="running", progress=progress)
        return LocalPollResult(state="running", progress=progress)


local_executor = LocalAnnotationExecutor()

__all__ = [
    "LocalAnnotationExecutor",
    "LocalPollResult",
    "local_executor",
    "local_ml_dependencies_available",
    "missing_local_ml_dependencies",
]
