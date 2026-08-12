"""Streaming artifacts shared by Modal CellExLink and Railway PubTator3.

The Stage 1 bundle is split into bounded per-paper work units.  Each Stage 2
branch then publishes one deterministic, text-free row per original chunk.  The
Railway controller can merge the two branch artifacts in lockstep without
loading the corpus or either full result into memory.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from backend.pipeline.pubtator3_annotation_worker import (
    PUBTATOR3_ANNOTATIONS_FILENAME,
)
MENTIONS_FILENAME = "cell_mentions.jsonl.gz"
MENTIONS_META_FILENAME = "cell_mentions.meta.json"
CELL_ANNOTATIONS_FILENAME = "cell_annotations.jsonl.gz"
CELL_ANNOTATIONS_META_FILENAME = "cell_annotations.meta.json"

CELL_BRANCH_FILENAME = "cell_branch.jsonl.gz"
PUBTATOR_BRANCH_FILENAME = "pubtator3_branch.jsonl.gz"
ENTITY_OUTPUT_FILENAME = "entity_annotations.jsonl.gz"

CELL_BRANCH_SCHEMA = "chunk-cell-annotations-v1"
PUBTATOR_BRANCH_SCHEMA = "chunk-human-gene-metadata-hormone-annotations-v5"
ANNOTATION_OUTPUT_SCHEMA = "chunk-entity-annotations-v6-human-gene-metadata"

_ONE_MIB = 1024 * 1024
_ENTITY_ORDER = {"cell": 0, "gene": 1, "hormone": 2}
_SUPPORTED_ENTITY_TYPES = frozenset(_ENTITY_ORDER)


def _empty_counts() -> dict[str, int]:
    return {
        "cell": 0,
        "gene": 0,
        "hormone": 0,
        "total": 0,
    }


def _count_annotation(counts: dict[str, int], annotation: Mapping[str, Any]) -> None:
    entity_type = str(annotation.get("obj") or "")
    if entity_type not in _SUPPORTED_ENTITY_TYPES:
        raise ValueError(f"Unsupported entity type: {entity_type or 'missing'}")
    counts[entity_type] += 1
    counts["total"] += 1


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_ONE_MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: object) -> str:
    raw = str(value or "paper").strip() or "paper"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-._")[:64] or "paper"
    suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{suffix}"


def _write_row(handle: gzip.GzipFile, row: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    handle.write(b"\n")


def _open_jsonl(path: Path):
    if path.suffix.casefold() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_jsonl(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_no}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_no}."
                )
            yield row


def split_bundle(bundle: Path, root: Path) -> tuple[list[dict[str, Any]], int]:
    """Split the ordered Stage 1 bundle into ephemeral per-paper files.

    ``base`` identifies a chunk, while ``canonical_id``/``doc_key`` identify a
    paper.  Keeping all chunks from one paper together preserves document-level
    abbreviation context for CellExLink and lets PubTator3 be requested once per
    article.
    """

    entries: list[dict[str, Any]] = []
    current_identity: str | None = None
    current_path: Path | None = None
    current_raw = None
    current_gzip: gzip.GzipFile | None = None
    chunk_count = 0
    closed_identities: set[str] = set()

    def close_current() -> None:
        nonlocal current_raw, current_gzip, current_path, current_identity
        if current_gzip is not None:
            current_gzip.close()
        if current_raw is not None:
            current_raw.flush()
            os.fsync(current_raw.fileno())
            current_raw.close()
        if current_path is not None:
            stat = current_path.stat()
            parent = current_path.parent
            entries.append(
                {
                    "paper_identity": current_identity,
                    "chunk_path": str(current_path),
                    "source_fingerprint": f"{stat.st_size}:{stat.st_mtime_ns}",
                    "mentions_path": str(parent / MENTIONS_FILENAME),
                    "mentions_meta_path": str(parent / MENTIONS_META_FILENAME),
                    "annotations_path": str(parent / CELL_ANNOTATIONS_FILENAME),
                    "annotations_meta_path": str(
                        parent / CELL_ANNOTATIONS_META_FILENAME
                    ),
                    "pubtator_annotations_path": str(
                        parent / PUBTATOR3_ANNOTATIONS_FILENAME
                    ),
                }
            )
            if current_identity is not None:
                closed_identities.add(current_identity)
        current_raw = None
        current_gzip = None
        current_path = None

    with gzip.open(bundle, "rt", encoding="utf-8") as source:
        for line_no, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid chunk JSON at line {line_no}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Chunk line {line_no} is not a JSON object.")

            identity = str(
                row.get("canonical_id")
                or row.get("doc_key")
                or row.get("pmcid")
                or row.get("pmid")
                or row.get("base")
                or f"line-{line_no}"
            )
            if identity != current_identity:
                close_current()
                if identity in closed_identities:
                    raise ValueError(
                        "The Stage 1 bundle contains non-contiguous chunks for "
                        f"paper {identity!r}."
                    )
                current_identity = identity
                paper_dir = root / _safe_name(identity)
                paper_dir.mkdir(parents=True, exist_ok=True)
                current_path = paper_dir / "chunks.jsonl.gz"
                current_raw = current_path.open("wb")
                current_gzip = gzip.GzipFile(
                    filename="",
                    fileobj=current_raw,
                    mode="wb",
                    compresslevel=5,
                    mtime=0,
                )
            assert current_gzip is not None
            _write_row(current_gzip, row)
            chunk_count += 1

    close_current()
    return entries, chunk_count


def _identity_text(value: Any) -> str:
    return "" if value is None else str(value)


def _candidate_chunk_keys(record: Mapping[str, Any]) -> list[tuple[str, ...]]:
    chunk_id = _identity_text(record.get("chunk_id"))
    section_type = _identity_text(record.get("section_type"))
    if not chunk_id:
        return []

    keys: list[tuple[str, ...]] = []
    for field in ("base", "doc_key", "canonical_id", "pmid", "pmcid"):
        value = _identity_text(record.get(field))
        if not value:
            continue
        if section_type:
            keys.append((field, value, section_type, chunk_id))
        keys.append((field, value, chunk_id))
    if section_type:
        keys.append(("section_type", section_type, chunk_id))
    keys.append(("chunk_id", chunk_id))
    return keys


def _chunk_result_row(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "base": source.get("base"),
        "doc_key": source.get("doc_key"),
        "canonical_id": source.get("canonical_id"),
        "pmid": source.get("pmid"),
        "pmcid": source.get("pmcid"),
        "journal": source.get("journal") or "",
        "pub_year": source.get("pub_year") or "",
        "section_type": source.get("section_type"),
        "chunk_id": source.get("chunk_id"),
        "annotations": [],
    }


def _compact_cell_annotation(source: Mapping[str, Any]) -> dict[str, Any]:
    concept_id = source.get("concept_id")
    if concept_id is None:
        concept_id = source.get("cell_ontology_id")

    preferred_label = source.get("preferred_label")
    if preferred_label is None:
        preferred_label = source.get("cell_ontology_label")

    try:
        start = int(source.get("start"))
        end = int(source.get("end"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "A CellExLink annotation is missing valid integer start/end offsets."
        ) from exc

    return {
        "obj": "cell",
        "start": start,
        "end": end,
        "mention": str(source.get("mention") or ""),
        "concept_id": concept_id,
        "preferred_label": preferred_label,
        "normalization_source": "CellExLink",
    }


def _compact_pubtator_annotation(source: Mapping[str, Any]) -> dict[str, Any]:
    entity_type = str(source.get("entity_type") or source.get("obj") or "").casefold()
    if entity_type not in {"gene", "hormone"}:
        raise ValueError(
            f"Unsupported PubTator3 entity type: {entity_type or 'missing'}"
        )
    try:
        start = int(source.get("start"))
        end = int(source.get("end"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "A PubTator3 annotation is missing valid integer start/end offsets."
        ) from exc

    row: dict[str, Any] = {
        "obj": entity_type,
        "start": start,
        "end": end,
        "mention": str(source.get("mention") or ""),
        "concept_id": source.get("concept_id"),
        "preferred_label": source.get("preferred_label"),
        "normalization_source": "PubTator3",
    }
    if entity_type == "gene":
        row["gene_id"] = source.get("gene_id")
        for field in (
            "tax_id",
            "tax_name",
            "taxonomy_source",
            "gene_record_status",
        ):
            value = source.get(field)
            if value not in (None, ""):
                row[field] = value
    else:
        hormone_id = source.get("hormone_id") or source.get("chemical_id")
        row["hormone_id"] = hormone_id
        # Retained for interoperability with existing MeSH-aware consumers.
        row["chemical_id"] = source.get("chemical_id") or hormone_id
        row["source_entity_type"] = source.get("source_entity_type") or "Chemical"
    label_source = source.get("label_source")
    if label_source:
        row["label_source"] = label_source
    classification_source = source.get("hormone_classification_source")
    if classification_source:
        row["hormone_classification_source"] = classification_source
    return row


def _annotation_signature(annotation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        annotation.get("obj"),
        annotation.get("start"),
        annotation.get("end"),
        annotation.get("mention"),
        annotation.get("concept_id"),
        annotation.get("preferred_label"),
    )


def _sort_annotations(annotations: list[dict[str, Any]]) -> None:
    annotations.sort(
        key=lambda item: (
            int(item.get("start") or 0),
            int(item.get("end") or 0),
            _ENTITY_ORDER.get(str(item.get("obj") or ""), 99),
            str(item.get("mention") or "").casefold(),
            str(item.get("concept_id") or ""),
        )
    )


def _build_branch_artifact(
    entries: Iterable[Mapping[str, Any]],
    output: Path,
    *,
    source_field: str,
    compact: Callable[[Mapping[str, Any]], dict[str, Any]],
    source_name: str,
) -> tuple[int, dict[str, int]]:
    """Consolidate sparse per-paper sidecars into one row per Stage 1 chunk."""

    output_chunk_count = 0
    counts = _empty_counts()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            fileobj=raw_output,
            mode="wb",
            compresslevel=6,
            mtime=0,
        ) as destination:
            for entry in entries:
                chunk_path = Path(str(entry["chunk_path"]))
                source_path = Path(str(entry[source_field]))
                if not source_path.is_file():
                    raise FileNotFoundError(
                        f"The {source_name} sidecar is missing: {source_path}"
                    )

                chunk_rows = [_chunk_result_row(row) for row in iter_jsonl(chunk_path)]
                annotations_by_chunk: list[list[dict[str, Any]]] = [
                    [] for _ in chunk_rows
                ]
                seen_annotations: list[set[tuple[Any, ...]]] = [
                    set() for _ in chunk_rows
                ]

                key_to_index: dict[tuple[str, ...], int] = {}
                ambiguous_keys: set[tuple[str, ...]] = set()
                for chunk_index, chunk_row in enumerate(chunk_rows):
                    for key in _candidate_chunk_keys(chunk_row):
                        if key in ambiguous_keys:
                            continue
                        if key in key_to_index:
                            key_to_index.pop(key, None)
                            ambiguous_keys.add(key)
                        else:
                            key_to_index[key] = chunk_index

                unmatched = 0
                for raw_annotation in iter_jsonl(source_path):
                    chunk_index: int | None = None
                    for key in _candidate_chunk_keys(raw_annotation):
                        candidate = key_to_index.get(key)
                        if candidate is not None:
                            chunk_index = candidate
                            break
                    if chunk_index is None and len(chunk_rows) == 1:
                        chunk_index = 0
                    if chunk_index is None:
                        unmatched += 1
                        continue

                    annotation = compact(raw_annotation)
                    signature = _annotation_signature(annotation)
                    if signature in seen_annotations[chunk_index]:
                        continue
                    seen_annotations[chunk_index].add(signature)
                    annotations_by_chunk[chunk_index].append(annotation)
                    _count_annotation(counts, annotation)

                if unmatched:
                    raise ValueError(
                        f"Could not map {unmatched} {source_name} annotation(s) "
                        f"from {source_path} back to their Stage 1 chunks."
                    )

                for chunk_row, annotations in zip(chunk_rows, annotations_by_chunk):
                    _sort_annotations(annotations)
                    chunk_row["annotations"] = annotations
                    _write_row(destination, chunk_row)
                    output_chunk_count += 1

        raw_output.flush()
        os.fsync(raw_output.fileno())

    return output_chunk_count, counts


def build_cell_branch(
    entries: Iterable[Mapping[str, Any]], output: Path
) -> tuple[int, dict[str, int]]:
    return _build_branch_artifact(
        entries,
        output,
        source_field="annotations_path",
        compact=_compact_cell_annotation,
        source_name="CellExLink",
    )


def build_pubtator_branch(
    entries: Iterable[Mapping[str, Any]], output: Path
) -> tuple[int, dict[str, int]]:
    return _build_branch_artifact(
        entries,
        output,
        source_field="pubtator_annotations_path",
        compact=_compact_pubtator_annotation,
        source_name="PubTator3",
    )


def _chunk_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        _identity_text(row.get(field))
        for field in (
            "base",
            "doc_key",
            "canonical_id",
            "pmid",
            "pmcid",
            "section_type",
            "chunk_id",
        )
    )


def merge_branch_artifacts(
    cell_branch: Path,
    pubtator_branch: Path,
    output: Path,
) -> tuple[int, dict[str, int]]:
    """Merge aligned CellExLink and PubTator3 branch rows in constant memory."""

    counts = _empty_counts()
    output_chunk_count = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            fileobj=raw_output,
            mode="wb",
            compresslevel=6,
            mtime=0,
        ) as destination:
            for row_number, pair in enumerate(
                zip_longest(iter_jsonl(cell_branch), iter_jsonl(pubtator_branch)),
                start=1,
            ):
                cell_row, pubtator_row = pair
                if cell_row is None or pubtator_row is None:
                    raise ValueError(
                        "The CellExLink and PubTator3 branch artifacts contain "
                        "different numbers of chunks."
                    )
                if _chunk_identity(cell_row) != _chunk_identity(pubtator_row):
                    raise ValueError(
                        "The CellExLink and PubTator3 branch artifacts are not "
                        f"aligned at row {row_number}."
                    )

                merged = {
                    key: value
                    for key, value in cell_row.items()
                    if key not in {"annotations", "chunk"}
                }
                annotations: list[dict[str, Any]] = []
                seen: set[tuple[Any, ...]] = set()
                for branch_row in (cell_row, pubtator_row):
                    raw_annotations = branch_row.get("annotations") or []
                    if not isinstance(raw_annotations, list):
                        raise ValueError(
                            f"Branch row {row_number} has an invalid annotations field."
                        )
                    for raw_annotation in raw_annotations:
                        if not isinstance(raw_annotation, Mapping):
                            raise ValueError(
                                f"Branch row {row_number} contains a non-object annotation."
                            )
                        annotation = dict(raw_annotation)
                        entity_type = str(annotation.get("obj") or "")
                        if entity_type not in _SUPPORTED_ENTITY_TYPES:
                            raise ValueError(
                                f"Branch row {row_number} contains unsupported entity "
                                f"type {entity_type!r}."
                            )
                        signature = _annotation_signature(annotation)
                        if signature in seen:
                            continue
                        seen.add(signature)
                        annotations.append(annotation)
                        _count_annotation(counts, annotation)

                _sort_annotations(annotations)
                merged["annotations"] = annotations
                _write_row(destination, merged)
                output_chunk_count += 1

        raw_output.flush()
        os.fsync(raw_output.fileno())

    return output_chunk_count, counts


__all__ = [
    "ANNOTATION_OUTPUT_SCHEMA",
    "CELL_ANNOTATIONS_FILENAME",
    "CELL_ANNOTATIONS_META_FILENAME",
    "CELL_BRANCH_FILENAME",
    "CELL_BRANCH_SCHEMA",
    "ENTITY_OUTPUT_FILENAME",
    "MENTIONS_FILENAME",
    "MENTIONS_META_FILENAME",
    "PUBTATOR_BRANCH_FILENAME",
    "PUBTATOR_BRANCH_SCHEMA",
    "build_cell_branch",
    "build_pubtator_branch",
    "iter_jsonl",
    "merge_branch_artifacts",
    "sha256_path",
    "split_bundle",
]
