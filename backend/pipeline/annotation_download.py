"""Rewrite annotation JSONL files with the final overlap policy applied."""

from __future__ import annotations

import gzip
import io
import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, TextIO

from backend.pipeline.entity_overlap import sanitize_annotation_payload


def _open_input(path: Path) -> TextIO:
    with path.open("rb") as handle:
        magic = handle.read(2)
    if magic == b"\x1f\x8b" or path.suffix.casefold() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _write_json_line(handle: TextIO, payload: Any) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def sanitize_jsonl_files(
    sources: Iterable[Path],
    destination: Path,
) -> int:
    """Write sanitized rows from one or more plain/gzip JSONL files.

    The output is gzip-compressed when ``destination`` ends in ``.gz``.
    Returns the number of rows written.
    """

    source_paths = [Path(path).expanduser().resolve() for path in sources]
    if not source_paths:
        raise ValueError("At least one annotation source is required.")
    for source in source_paths:
        if not source.is_file():
            raise FileNotFoundError(source)

    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    row_count = 0
    try:
        with ExitStack() as stack:
            raw_output = stack.enter_context(temporary.open("wb"))
            if destination.suffix.casefold() == ".gz":
                gzip_output = stack.enter_context(
                    gzip.GzipFile(
                        filename="",
                        fileobj=raw_output,
                        mode="wb",
                        compresslevel=6,
                        mtime=0,
                    )
                )
                output = stack.enter_context(
                    io.TextIOWrapper(
                        gzip_output,
                        encoding="utf-8",
                        newline="\n",
                    )
                )
            else:
                output = stack.enter_context(
                    io.TextIOWrapper(
                        raw_output,
                        encoding="utf-8",
                        newline="\n",
                    )
                )

            for source in source_paths:
                with _open_input(source) as input_handle:
                    for line_number, line in enumerate(input_handle, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Invalid JSON in {source} at line {line_number}: {exc}"
                            ) from exc
                        _write_json_line(output, sanitize_annotation_payload(payload))
                        row_count += 1
            output.flush()
            if destination.suffix.casefold() != ".gz":
                os.fsync(raw_output.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return row_count


__all__ = ["sanitize_jsonl_files"]
