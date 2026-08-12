"""Publish deterministic, compressed stage artifacts without local duplication."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from backend.storage.artifacts import (
    ArtifactRef,
    ArtifactStore,
    get_artifact_store,
    prefixed_key,
    sha256_file,
)

_ONE_MIB = 1024 * 1024


def _open_binary(path: Path):
    return gzip.open(path, "rb") if path.suffix.casefold() == ".gz" else path.open("rb")


def build_deterministic_gzip_bundle(paths: Iterable[Path], destination: Path) -> int:
    """Combine NDJSON parts into one deterministic gzip file.

    The gzip timestamp and filename fields are fixed so an unchanged collection
    produces the same SHA-256 and therefore reuses the same object-store key.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    line_count = 0
    with destination.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_output,
            compresslevel=6,
            mtime=0,
        ) as compressed:
            for source in paths:
                source = source.expanduser().resolve()
                if not source.is_file():
                    raise FileNotFoundError(source)
                last_byte = b""
                with _open_binary(source) as handle:
                    while block := handle.read(_ONE_MIB):
                        compressed.write(block)
                        line_count += block.count(b"\n")
                        last_byte = block[-1:]
                if last_byte and last_byte != b"\n":
                    compressed.write(b"\n")
                    line_count += 1
        raw_output.flush()
        os.fsync(raw_output.fileno())
    return line_count


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


def publish_retrieval_bundle(
    *,
    summary_path: Path,
    chunk_paths: Iterable[Path],
    store: ArtifactStore | None = None,
) -> tuple[ArtifactRef, bool, int]:
    """Create and publish the one canonical Stage 1 input for Modal."""

    selected_store = store or get_artifact_store()
    paths = tuple(path.expanduser().resolve() for path in chunk_paths)
    temporary_dir = Path(tempfile.mkdtemp(prefix="ovarian-retrieval-bundle-"))
    temporary_path = temporary_dir / "chunks.jsonl.gz"
    try:
        line_count = build_deterministic_gzip_bundle(paths, temporary_path)
        digest = sha256_file(temporary_path)
        key = prefixed_key(
            f"retrieval/chunks/{digest[:2]}/{digest}.jsonl.gz"
        )
        artifact, reused = selected_store.put_file(
            temporary_path,
            key=key,
            content_type="application/gzip",
            content_encoding=None,
            sha256=digest,
        )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("Retrieval summary is not a JSON object.")
        files = summary.setdefault("files", {})
        if not isinstance(files, dict):
            files = {}
            summary["files"] = files
        files["artifact"] = artifact.to_dict()
        files["artifact"]["record_count"] = line_count
        files["artifact"]["part_count"] = len(paths)
        files["artifact"]["reused_existing"] = reused
        _write_json_atomic(summary_path, summary)
        return artifact, reused, line_count
    finally:
        temporary_path.unlink(missing_ok=True)
        temporary_dir.rmdir()


__all__ = ["build_deterministic_gzip_bundle", "publish_retrieval_bundle"]
