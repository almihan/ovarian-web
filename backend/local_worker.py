"""Command-line entry point for one local CPU Stage 2 entity-extraction job."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from backend.modal_worker.pipeline import run_annotation_bundle


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        return 2

    payload_path = Path(args[0]).expanduser().resolve()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Local worker payload must be a JSON object.")
    control = (
        payload.get("local_control")
        if isinstance(payload.get("local_control"), dict)
        else {}
    )
    result_path = Path(str(control.get("result_path") or "")).expanduser().resolve()
    model_cache_root = Path(
        str(control.get("model_cache_root") or "data/model_cache")
    ).expanduser().resolve()

    try:
        result = run_annotation_bundle(
            payload,
            model_cache_root=model_cache_root,
            require_cuda=False,
        )
        _write_json_atomic(
            result_path,
            {"state": "completed", "result": result},
        )
        return 0
    except Exception as exc:
        _write_json_atomic(
            result_path,
            {"state": "failed", "error": str(exc)},
        )
        return 1
    finally:
        # The payload contains a short-lived callback token. It is unnecessary
        # after the worker exits and should not become persistent local state.
        payload_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
