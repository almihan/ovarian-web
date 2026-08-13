"""Short-lived local CellExLink worker launcher.

The FastAPI process remains lightweight.  Each local annotation runs in a
separate Python process, loads recognition and normalization sequentially, and
exits after publishing the compressed result.  This is intended for local CPU
validation; production continues to use the Modal executor.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from backend.config import PROJECT_ROOT, settings


@dataclass(slots=True, frozen=True)
class LocalPollResult:
    state: str
    result: dict[str, Any] | None = None
    error: str | None = None


def local_ml_dependencies_available() -> bool:
    required = ("torch", "transformers", "huggingface_hub", "numpy")
    return all(importlib.util.find_spec(name) is not None for name in required)


class LocalAnnotationExecutor:
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
        if not local_ml_dependencies_available():
            raise RuntimeError(
                "Local CellExLink dependencies are missing. Install PyTorch and "
                "requirements-local.txt before starting Stage 2."
            )

        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        payload_path = job_dir / "payload.json"
        result_path = job_dir / "result.json"
        result_path.unlink(missing_ok=True)

        worker_payload = dict(payload)
        worker_payload["local_control"] = {
            "result_path": str(result_path),
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **popen_kwargs,
        )
        return f"{job_id}:{process.pid}"

    def cleanup(self, job_id: str) -> None:
        """Remove one finished local worker's small control directory."""

        shutil.rmtree(self._job_dir(job_id), ignore_errors=True)

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
            return LocalPollResult(state="running")
        return LocalPollResult(state="running")


local_executor = LocalAnnotationExecutor()

__all__ = [
    "LocalAnnotationExecutor",
    "LocalPollResult",
    "local_executor",
    "local_ml_dependencies_available",
]
