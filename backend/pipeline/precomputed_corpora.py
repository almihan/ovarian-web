"""Precomputed ovarian corpora and streaming summary helpers.

Bundled results seed the persistent corpus store. Authorized monthly updates
append completed predictions; public requests read saved results only.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

from backend.config import settings
from backend.pipeline.entity_overlap import sanitize_annotation_payload
from backend.pipeline.identifier_identity import canonical_identifier
from backend.pipeline.network_builder import global_entity_id
from backend.pipeline.relation_extraction import relation_allowed
from backend.services.corpus_store import ensure_corpus_store

CORPUS_DEFINITIONS: dict[str, dict[str, str]] = {
    "non_neoplastic_inflammatory": {
        "label": "Non-neoplastic inflammatory",
        "filename": "neoplastic_ovarian_prediction.jsonl",
        "description": (
            "Saved non-neoplastic ovarian inflammatory and immune-related results."
        ),
    },
    "cancer_associated_inflammatory": {
        "label": "Cancer–associated inflammatory",
        "filename": "cancer_associated_ovarian_prediction.jsonl",
        "description": (
            "Saved ovarian cancer-associated inflammatory and immune-related results."
        ),
    },
}
DEFAULT_CORPUS_ID = "non_neoplastic_inflammatory"

_SPACE_RE = re.compile(r"\s+")
_ENDPOINT_RE = re.compile(r"^(.*?)\s*\(([^()]+)\)\s*$")
_CACHE_LOCK = threading.RLock()
_SUMMARY_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
CORPUS_SUMMARY_VERSION = "four-predicate-relations"
_TYPE_PREFIX = {"cell": "C", "gene": "G", "hormone": "H"}


class CorpusNotFoundError(ValueError):
    """Raised when a selected permanent corpus is unknown or unavailable."""


def _text(value: Any) -> str:
    if value is None:
        return ""
    return _SPACE_RE.sub(" ", str(value)).strip()


def corpus_root() -> Path:
    return ensure_corpus_store()


def corpus_definition(corpus_id: str) -> dict[str, str]:
    key = _text(corpus_id).casefold().replace("-", "_")
    definition = CORPUS_DEFINITIONS.get(key)
    if definition is None:
        raise CorpusNotFoundError("Unknown saved corpus selection.")
    return {"id": key, **definition}


def corpus_path(corpus_id: str) -> Path:
    definition = corpus_definition(corpus_id)
    return (corpus_root() / definition["filename"]).resolve()


def _open_text(path: Path):
    if path.suffix.casefold() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_prediction_rows(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object in {path.name} at line {line_number}."
                )
            sanitized = sanitize_annotation_payload(row)
            if not isinstance(sanitized, dict):
                raise ValueError(
                    f"Expected a JSON object in {path.name} at line {line_number}."
                )
            yield sanitized


def _paper_id(row: Mapping[str, Any], line_number: int) -> str:
    return (
        _text(row.get("canonical_id"))
        or _text(row.get("doc_key"))
        or (f"pmid:{_text(row.get('pmid'))}" if _text(row.get("pmid")) else "")
        or (f"pmcid:{_text(row.get('pmcid'))}" if _text(row.get("pmcid")) else "")
        or f"row:{line_number}"
    )


def _canonical_endpoint(value: Any) -> tuple[str, str, str]:
    """Return ``(label, entity_type, canonical identifier)`` for a relation endpoint."""

    text = _text(value)
    match = _ENDPOINT_RE.fullmatch(text)
    if match is None:
        return "", "", ""
    label = _text(match.group(1))
    identifier = _text(match.group(2))
    prefix = identifier.split(":", 1)[0].casefold() if ":" in identifier else ""
    if prefix == "cl":
        entity_type, field = "cell", "concept_id"
    elif prefix == "hgnc":
        entity_type, field = "gene", "hgnc_id"
    elif prefix in {"ncbigene", "geneid", "gene"}:
        entity_type, field = "gene", "ncbi_gene_id"
    elif prefix == "chebi":
        entity_type, field = "hormone", "chebi_id"
    elif prefix == "mesh":
        entity_type, field = "hormone", "mesh_id"
    else:
        return "", "", ""
    _namespace, canonical = canonical_identifier(entity_type, field, identifier)
    return label, entity_type, canonical or identifier


def _iter_chunks(row: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    chunks = row.get("chunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            if isinstance(chunk, Mapping):
                yield chunk
        return
    # Stage 2 entity artifacts are one row per chunk rather than one row per paper.
    yield row


def summarize_entity_annotations(path: Path) -> dict[str, int]:
    """Count globally unique normalized entities in nested or flat JSONL data."""

    unique: dict[str, set[str]] = {"cell": set(), "gene": set(), "hormone": set()}
    annotation_occurrences = 0
    chunk_count = 0
    for row in iter_prediction_rows(path):
        for chunk in _iter_chunks(row):
            chunk_count += 1
            values = chunk.get("annotations")
            annotations = values if isinstance(values, list) else []
            for annotation in annotations:
                if not isinstance(annotation, Mapping):
                    continue
                annotation_occurrences += 1
                node_id, entity_type, _identity = global_entity_id(annotation)
                if node_id and entity_type in unique:
                    unique[entity_type].add(node_id)
    return {
        "chunk_count": chunk_count,
        "annotation_occurrence_count": annotation_occurrences,
        "unique_cell_count": len(unique["cell"]),
        "unique_gene_count": len(unique["gene"]),
        "unique_hormone_count": len(unique["hormone"]),
        "unique_entity_count": sum(len(values) for values in unique.values()),
    }


def summarize_prediction_file(path: Path) -> dict[str, Any]:
    """Stream a one-row-per-paper final prediction JSONL and summarize it.

    Relation support is deduplicated by ``(paper, subject ID, predicate, object ID)``.
    ``global_relation_count`` is the number of globally distinct normalized edges.
    """

    path = path.expanduser().resolve()
    if not path.is_file():
        raise CorpusNotFoundError(f"Saved corpus file not found: {path}")

    unique_entities: dict[str, set[str]] = {
        "cell": set(),
        "gene": set(),
        "hormone": set(),
    }
    global_relations: set[tuple[str, str, str]] = set()
    predicate_counts: Counter[str] = Counter()
    stats: dict[str, Any] = {
        "paper_count": 0,
        "abstract_count": 0,
        "fulltext_count": 0,
        "metadata_only_count": 0,
        "chunk_count": 0,
        "annotation_occurrence_count": 0,
        "relation_occurrence_count": 0,
        "unique_paper_relation_count": 0,
        "duplicate_paper_relation_count": 0,
        "papers_with_relations": 0,
        "unresolved_relation_endpoint_count": 0,
    }

    for line_number, row in enumerate(iter_prediction_rows(path), start=1):
        stats["paper_count"] += 1
        paper_id = _paper_id(row, line_number)
        paper_relations: set[tuple[str, str, str]] = set()
        has_abstract = False
        has_fulltext = False
        chunks = list(_iter_chunks(row))
        if not chunks:
            stats["metadata_only_count"] += 1

        for chunk in chunks:
            stats["chunk_count"] += 1
            section_type = _text(chunk.get("section_type")).upper()
            text_source = _text(chunk.get("text_source")).casefold()
            chunk_text = str(chunk.get("text") or chunk.get("chunk") or "")
            if section_type == "ABSTRACT" and chunk_text.strip():
                has_abstract = True
            if text_source == "fulltext":
                has_fulltext = True

            raw_annotations = chunk.get("annotations")
            annotations = raw_annotations if isinstance(raw_annotations, list) else []
            for annotation in annotations:
                if not isinstance(annotation, Mapping):
                    continue
                stats["annotation_occurrence_count"] += 1
                node_id, entity_type, _identity = global_entity_id(annotation)
                if node_id and entity_type in unique_entities:
                    unique_entities[entity_type].add(node_id)

            raw_relations = chunk.get("relations")
            relations = raw_relations if isinstance(raw_relations, list) else []
            for relation in relations:
                stats["relation_occurrence_count"] += 1
                if not isinstance(relation, Mapping):
                    stats["unresolved_relation_endpoint_count"] += 1
                    continue
                _subject_label, subject_type, subject_id = _canonical_endpoint(
                    relation.get("subject")
                )
                _object_label, object_type, object_id = _canonical_endpoint(
                    relation.get("object")
                )
                predicate = _text(relation.get("predicate")).casefold()
                subject_prefix = _TYPE_PREFIX.get(subject_type, "")
                object_prefix = _TYPE_PREFIX.get(object_type, "")
                if (
                    not subject_id
                    or not object_id
                    or not subject_prefix
                    or not object_prefix
                    or not relation_allowed(
                        f"{subject_prefix}1",
                        predicate,
                        f"{object_prefix}2",
                    )
                ):
                    stats["unresolved_relation_endpoint_count"] += 1
                    continue
                key = (subject_id, predicate, object_id)
                if key in paper_relations:
                    stats["duplicate_paper_relation_count"] += 1
                    continue
                paper_relations.add(key)
                global_relations.add(key)
                predicate_counts[predicate] += 1

        selected_source = _text(row.get("selected_text_source")).casefold()
        if selected_source == "fulltext":
            has_fulltext = True
        if has_abstract:
            stats["abstract_count"] += 1
        if has_fulltext:
            stats["fulltext_count"] += 1
        if not chunks or selected_source.startswith("metadata_only"):
            stats["metadata_only_count"] += int(bool(chunks))
        if paper_relations:
            stats["papers_with_relations"] += 1
            stats["unique_paper_relation_count"] += len(paper_relations)

    stats.update(
        {
            "unique_cell_count": len(unique_entities["cell"]),
            "unique_gene_count": len(unique_entities["gene"]),
            "unique_hormone_count": len(unique_entities["hormone"]),
            "unique_entity_count": sum(
                len(values) for values in unique_entities.values()
            ),
            "global_relation_count": len(global_relations),
            "predicate_counts": dict(sorted(predicate_counts.items())),
            "source_size": path.stat().st_size,
        }
    )
    return stats


def _sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.summary.json")


def _read_sidecar(path: Path) -> dict[str, Any] | None:
    sidecar = _sidecar_path(path)
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("summary_version") != CORPUS_SUMMARY_VERSION:
        return None
    if int(payload.get("source_size") or -1) != path.stat().st_size:
        return None
    return payload


def _write_sidecar(path: Path, summary: Mapping[str, Any]) -> None:
    sidecar = _sidecar_path(path)
    payload = {"summary_version": CORPUS_SUMMARY_VERSION, **dict(summary)}
    temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, sidecar)
    except OSError:
        temporary.unlink(missing_ok=True)


def corpus_summary(corpus_id: str) -> dict[str, Any]:
    definition = corpus_definition(corpus_id)
    path = corpus_path(corpus_id)
    if not path.is_file():
        raise CorpusNotFoundError(
            f"Place {definition['filename']} in {corpus_root()} before starting."
        )
    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns)
    with _CACHE_LOCK:
        cached = _SUMMARY_CACHE.get(definition["id"])
        if cached and cached[0] == signature:
            return dict(cached[1])

    summary = _read_sidecar(path) or summarize_prediction_file(path)
    summary.update(
        {
            "id": definition["id"],
            "label": definition["label"],
            "description": definition["description"],
            "filename": definition["filename"],
            "available": True,
            "read_only": True,
        }
    )
    _write_sidecar(path, summary)
    with _CACHE_LOCK:
        _SUMMARY_CACHE[definition["id"]] = (signature, dict(summary))
    return summary


def list_corpora() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for corpus_id, raw in CORPUS_DEFINITIONS.items():
        path = corpus_path(corpus_id)
        results.append(
            {
                "id": corpus_id,
                **raw,
                "available": path.is_file(),
                "read_only": True,
            }
        )
    return results


__all__ = [
    "CORPUS_DEFINITIONS",
    "DEFAULT_CORPUS_ID",
    "CorpusNotFoundError",
    "corpus_definition",
    "corpus_path",
    "corpus_root",
    "corpus_summary",
    "iter_prediction_rows",
    "list_corpora",
    "summarize_entity_annotations",
    "summarize_prediction_file",
]
