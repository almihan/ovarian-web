"""Durable precomputed results and the union of available PubMed identifiers.

Bundled JSONL files seed the store once. Monthly updates change the configured
store, never replace it from the bundle, and commit each corpus atomically. The
PMID index is derived data and is rebuilt after an interrupted update.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:
    import fcntl
except ImportError:  # Windows local installations.
    fcntl = None
    import msvcrt

from backend.config import PROJECT_ROOT, settings

# Keep these filenames aligned with precomputed_corpora.CORPUS_DEFINITIONS.
# This module intentionally does not import that pipeline module: reading saved
# papers must not initialize the model or its relation-extraction dependencies.
CORPUS_FILES = {
    "non_neoplastic_inflammatory": "neoplastic_ovarian_prediction.jsonl",
    "cancer_associated_inflammatory": "cancer_associated_ovarian_prediction.jsonl",
}
SEED_CORPUS_ROOT = PROJECT_ROOT / "data" / "precomputed_corpora"
INDEX_FILENAME = "known_pmids.json"
_INDEX_VERSION = 1
_LOCK = threading.RLock()
_PMID_RE = re.compile(r"^(?:pmid\s*:\s*)?([0-9]+)$", re.IGNORECASE)
_PUBMED_URL_RE = re.compile(
    r"^https?://(?:www\.)?pubmed\.ncbi\.nlm\.nih\.gov/([0-9]+)/?$",
    re.IGNORECASE,
)


def store_root() -> Path:
    configured = getattr(settings, "precomputed_corpora_dir", None)
    return Path(configured or settings.data_dir / "precomputed_corpora").expanduser().resolve()


def normalize_pmid(value: Any) -> str:
    """Accept a PMID, ``pmid:123`` identity, or canonical PubMed URL."""
    if value is None or isinstance(value, bool):
        return ""
    text = str(value).strip()
    match = _PMID_RE.fullmatch(text) or _PUBMED_URL_RE.fullmatch(text)
    if match is None:
        return ""
    # PubMed identifiers are positive integers. Canonicalization avoids treating
    # an accidentally zero-padded identifier as a second paper.
    normalized = str(int(match.group(1)))
    return normalized if normalized != "0" else ""


def prediction_pmid(row: Mapping[str, Any]) -> str:
    for field in ("pmid", "canonical_id", "doc_key", "id"):
        identifier = normalize_pmid(row.get(field))
        if identifier:
            return identifier
    return ""


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name} at line {number}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path.name} at line {number}.")
            yield row


def _sync_directory(root: Path) -> None:
    if os.name == "nt":
        # Windows does not expose directory fsync through os.open. The temporary
        # file itself has already been flushed before its atomic replacement.
        return
    fd = os.open(str(root), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def _locked() -> Iterator[Path]:
    root = store_root()
    root.mkdir(parents=True, exist_ok=True)
    with _LOCK, (root / ".corpus-store.lock").open("a+b") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        else:
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield root
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            else:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def _atomic_copy(source: Path, destination: Path) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as original:
            shutil.copyfileobj(original, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(destination: Path, payload: Any) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _bootstrap(root: Path) -> None:
    for filename in CORPUS_FILES.values():
        destination = root / filename
        if destination.is_file():
            continue
        source = SEED_CORPUS_ROOT / filename
        if not source.is_file():
            raise FileNotFoundError(f"Bundled precomputed corpus is missing: {source}")
        _atomic_copy(source, destination)


def _signatures(root: Path) -> dict[str, dict[str, Any]]:
    signatures = {}
    for corpus_id, filename in CORPUS_FILES.items():
        stat = (root / filename).stat()
        signatures[corpus_id] = {"filename": filename, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return signatures


def _index(root: Path, *, force: bool = False) -> dict[str, Any]:
    signatures = _signatures(root)
    index_path = root / INDEX_FILENAME
    if not force:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            if (
                isinstance(payload, dict)
                and payload.get("schema_version") == _INDEX_VERSION
                and payload.get("corpora") == signatures
                and isinstance(payload.get("pmids"), list)
                and all(isinstance(value, str) and normalize_pmid(value) == value for value in payload["pmids"])
            ):
                return payload
        except (OSError, ValueError, TypeError):
            pass
    pmids = set()
    for filename in CORPUS_FILES.values():
        for row in _rows(root / filename):
            pmid = prediction_pmid(row)
            if pmid:
                pmids.add(pmid)
    payload = {
        "schema_version": _INDEX_VERSION,
        "corpora": signatures,
        "pmids": sorted(pmids, key=int),
        "paper_count": len(pmids),
    }
    _atomic_json(index_path, payload)
    return payload


def ensure_corpus_store() -> Path:
    """Seed absent files, preserve existing results, and refresh the PMID index."""
    with _locked() as root:
        _bootstrap(root)
        _index(root)
        return root


def get_known_pmids() -> set[str]:
    """Return all available PMIDs from both precomputed resources."""
    with _locked() as root:
        _bootstrap(root)
        return set(_index(root)["pmids"])


def get_corpus_pmids(corpus_id: str) -> set[str]:
    """Return PMID membership for one topic without loading prediction payloads.

    Monthly retrieval uses this once per topic to batch only membership additions
    for papers already computed in the other corpus.
    """
    if corpus_id not in CORPUS_FILES:
        raise ValueError("Unknown precomputed corpus.")
    with _locked() as root:
        _bootstrap(root)
        pmids = {prediction_pmid(row) for row in _rows(root / CORPUS_FILES[corpus_id])}
        pmids.discard("")
        return pmids


def add_prediction_rows(corpus_id: str, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Append complete new paper predictions once, including across restarts.

    Existing predictions are authoritative. A repeated PMID is skipped instead
    of replacing previously reviewed data. Failed pipeline runs must not call
    this function with partial results.
    """
    if corpus_id not in CORPUS_FILES:
        raise ValueError("Unknown precomputed corpus.")
    incoming = []
    for row in rows:
        if not isinstance(row, Mapping) or not prediction_pmid(row):
            raise ValueError("Every new prediction row must have a valid PMID.")
        # Validate serializability before entering the commit, not halfway into it.
        value = dict(row)
        json.dumps(value, ensure_ascii=False, allow_nan=False)
        incoming.append(value)

    with _locked() as root:
        _bootstrap(root)
        destination = root / CORPUS_FILES[corpus_id]
        existing_pmids = {prediction_pmid(row) for row in _rows(destination)}
        existing_pmids.discard("")
        added_rows = []
        skipped = 0
        for row in incoming:
            pmid = prediction_pmid(row)
            if pmid in existing_pmids:
                skipped += 1
                continue
            existing_pmids.add(pmid)
            added_rows.append(row)

        if added_rows:
            fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=root)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, "wb") as output, destination.open("rb") as source:
                    shutil.copyfileobj(source, output)
                    # Bundled or user-provided files need not end with a newline.
                    if destination.stat().st_size:
                        source.seek(-1, os.SEEK_END)
                        if source.read(1) != b"\n":
                            output.write(b"\n")
                    for row in added_rows:
                        output.write((json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8"))
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, destination)
                _sync_directory(root)
            finally:
                temporary.unlink(missing_ok=True)
            # Source signatures invalidate cached pipeline summaries; removing
            # the sidecar also protects a same-byte-length replacement.
            destination.with_name(f"{destination.name}.summary.json").unlink(missing_ok=True)
        index = _index(root)
        return {
            "corpus_id": corpus_id,
            "added": len(added_rows),
            "skipped": skipped,
            "paper_count": len(existing_pmids),
            "known_pmid_count": index["paper_count"],
        }


def select_pmid_predictions(pmids: Iterable[Any], destination: Path) -> dict[str, Any]:
    """Write cached predictions for requested papers, without any inference.

    When a paper belongs to both topic corpora, its rows from the first corpus
    are used consistently. Within that corpus all distinct rows are preserved.
    Missing identifiers are reported to the caller instead of silently omitted.
    """
    requested = []
    seen = set()
    for value in pmids:
        pmid = normalize_pmid(value)
        if not pmid:
            raise ValueError(f"Invalid PubMed identifier: {value!r}")
        if pmid not in seen:
            seen.add(pmid)
            requested.append(pmid)
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with _locked() as root:
        _bootstrap(root)
        _index(root)
        if destination.parent == root and destination.name in {*CORPUS_FILES.values(), INDEX_FILENAME}:
            raise ValueError("A PMID selection cannot replace the precomputed corpus or its index.")
        selected = {}
        for filename in CORPUS_FILES.values():
            current = {}
            for row in _rows(root / filename):
                pmid = prediction_pmid(row)
                if pmid not in seen or pmid in selected:
                    continue
                current.setdefault(pmid, []).append(row)
            selected.update(current)

        fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                for pmid in requested:
                    row_signatures = set()
                    for row in selected.get(pmid, []):
                        signature = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        if signature in row_signatures:
                            continue
                        row_signatures.add(signature)
                        output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            _sync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)

    found = [pmid for pmid in requested if pmid in selected]
    return {
        "requested_pmids": requested,
        "found_pmids": found,
        "missing_pmids": [pmid for pmid in requested if pmid not in selected],
        "paper_count": len(found),
    }
