"""Command-line entry point for one local Stage 2 cell-extraction job."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from backend.cellexlink_lite.normalization import ensure_ab3p_healthy
from backend.local_annotation_pipeline import run_local_annotation_bundle

logger = logging.getLogger(__name__)


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


def _configure_logging(payload: Mapping[str, Any]) -> None:
    options = payload.get("options") if isinstance(payload.get("options"), Mapping) else {}
    method_log = bool(options.get("normalization_method_log", False))
    # Keep unrelated libraries quiet. The per-document diagnostic logger is
    # promoted to INFO only when the explicit switch is enabled.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if method_log:
        for logger_name in (
            "backend.cellexlink_lite.normalization",
            "backend.pipeline.cell_annotation_worker",
        ):
            logging.getLogger(logger_name).setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        return 2

    payload_path = Path(args[0]).expanduser().resolve()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Local worker payload must be a JSON object.")
    _configure_logging(payload)

    control = (
        payload.get("local_control")
        if isinstance(payload.get("local_control"), dict)
        else {}
    )
    result_path = Path(str(control.get("result_path") or "")).expanduser().resolve()
    model_cache_root = Path(
        str(control.get("model_cache_root") or "data/model_cache")
    ).expanduser().resolve()

    options = (
        payload.get("options")
        if isinstance(payload.get("options"), Mapping)
        else {}
    )

    try:
        if not bool(options.get("disable_abbreviations", False)):
            ensure_ab3p_healthy()
        result = run_local_annotation_bundle(
            payload,
            model_cache_root=model_cache_root,
        )
        _write_json_atomic(
            result_path,
            {"state": "completed", "result": result},
        )
        return 0
    except Exception as exc:
        logger.exception("Local Stage 2 cell worker failed")
        _write_json_atomic(
            result_path,
            {"state": "failed", "error": str(exc)},
        )
        return 1
    finally:
        # The payload contains a short-lived callback token and is not retained.
        payload_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
