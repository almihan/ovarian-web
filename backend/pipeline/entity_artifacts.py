"""Bounded per-paper artifacts shared by the Stage 2 annotation branches.

The cell branch temporarily carries exact source text for final reconciliation.
The published final artifact remains text-free. Memory is bounded by one paper,
not the whole corpus; branch alignment and original chunk order are preserved.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from itertools import zip_longest, groupby
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from backend.pipeline.abbreviation_prepass import (
    ABBREVIATION_ANNOTATIONS_FILENAME,
    ABBREVIATION_CONTEXT_FILENAME,
)
from backend.pipeline.pubtator3_annotation_worker import (
    PUBTATOR3_ANNOTATIONS_FILENAME,
    PUBTATOR3_PIPELINE_VERSION,
)
from backend.pipeline.entity_overlap import (
    assert_no_cell_gene_overlaps,
    prefer_longest_spans,
    sanitize_annotation_payload,
)
MENTIONS_FILENAME = "cell_mentions.jsonl.gz"
MENTIONS_META_FILENAME = "cell_mentions.meta.json"
CELL_ANNOTATIONS_FILENAME = "cell_annotations.jsonl.gz"
CELL_ANNOTATIONS_META_FILENAME = "cell_annotations.meta.json"

CELL_BRANCH_FILENAME = "cell_branch.jsonl.gz"
PUBTATOR_BRANCH_FILENAME = "pubtator3_branch.jsonl.gz"
ENTITY_OUTPUT_FILENAME = "entity_annotations.jsonl.gz"

CELL_BRANCH_SCHEMA = "chunk-local-entities-with-private-source-v6-longest-hgnc"
PUBTATOR_BRANCH_SCHEMA = "chunk-hgnc-uniprot-mesh-hormone-v13-longest"
ANNOTATION_OUTPUT_SCHEMA = "chunk-entity-annotations-v15-same-paper-cell-recovery"

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


def iter_jsonl(path: Path, *, sanitize: bool = True) -> Iterator[dict[str, Any]]:
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
            # Raw merge inputs need source validation before cell-over-gene
            # precedence. Public/legacy reads keep the existing default policy.
            sanitized = sanitize_annotation_payload(row) if sanitize else row
            if not isinstance(sanitized, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_no}."
                )
            yield sanitized


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
                    "abbreviation_context_path": str(
                        parent / ABBREVIATION_CONTEXT_FILENAME
                    ),
                    "abbreviation_annotations_path": str(
                        parent / ABBREVIATION_ANNOTATIONS_FILENAME
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
    from backend.pipeline.entity_span_rules import source_span_exclusions
    masks = source_span_exclusions(str(source.get("chunk") or ""))
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
        **({"entity_span_exclusions": masks} if masks else {}),
    }


def _compact_cell_annotation(source: Mapping[str, Any]) -> dict[str, Any]:
    if str(source.get("entity_type") or source.get("obj") or "").casefold() == "gene":
        return _compact_pubtator_annotation(source)
    raw_type = str(source.get("entity_type") or source.get("obj") or "").casefold()
    if not raw_type and (source.get("cell_ontology_id") or source.get("cell_ontology_label")):
        raw_type = "cell"
    elif not raw_type and (source.get("hormone_id") or source.get("mesh_id")):
        raw_type = "hormone"
    if raw_type in {"cell", "cell_type", "cell type"}:
        entity_type = "cell"
        concept_id = source.get("concept_id") or source.get("cell_ontology_id")
        preferred_label = (
            source.get("preferred_label") or source.get("cell_ontology_label")
        )
        default_system = "CellExLink/Cell Ontology"
    elif raw_type in {"hormone", "chemical"}:
        entity_type = "hormone"
        concept_id = (
            source.get("concept_id")
            or source.get("hormone_id")
            or source.get("mesh_id")
        )
        preferred_label = source.get("preferred_label")
        default_system = "Ab3P/MeSH"
    else:
        raise ValueError(
            f"Unsupported local annotation entity type: {raw_type or 'missing'}"
        )

    try:
        start = int(source.get("start"))
        end = int(source.get("end"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "A local cell/hormone annotation is missing valid integer offsets."
        ) from exc

    row: dict[str, Any] = {
        "obj": entity_type,
        "entity_type": entity_type,
        "start": start,
        "end": end,
        "mention": str(source.get("mention") or ""),
        "concept_id": concept_id,
        "normalized_id": concept_id,
        "preferred_label": preferred_label,
        "matched_term": source.get("matched_term"),
        "term_kind": source.get("term_kind"),
        "resource_version": source.get("resource_version"),
        "normalization_system": source.get("normalization_system") or default_system,
        "normalization_status": (
            source.get("normalization_status")
            or ("normalized" if concept_id else "unresolved")
        ),
        "normalization_source": source.get("normalization_source") or "unresolved",
    }
    if entity_type == "cell":
        row["cell_ontology_id"] = concept_id
        row["cell_ontology_label"] = preferred_label
    else:
        row["hormone_id"] = concept_id
        row["mesh_id"] = concept_id
        row["source_entity_type"] = "Chemical"

    evidence_fields = (
        "recognition_source",
        "definition_id",
        "definition_detector",
        "matched_abbreviation_key",
        "matched_static_abbreviation_key",
        "expanded_long_form",
        "abbreviation_key_cosine",
        "ab3p_key_cosine",
        "ab3p_match_method",
        "ontology_raw_cosine",
        "matched_ontology_alias",
        "normalization_score",
        "supporting_sources",
        "resource_file",
        "resource_line",
        "ontology_resource_version",
        "coordination_shared_head",
        "coordination_arms",
        "coordination_unresolved_arms",
        "coordination_rule",
        "normalization_scope",
        "seed_evidence",
        "locked",
    )
    for field in evidence_fields:
        value = source.get(field)
        if value not in (None, "", [], ()):
            row[field] = value
    return {key: value for key, value in row.items() if value not in (None, "", [], ())}


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
        "entity_type": entity_type,
        "start": start,
        "end": end,
        "mention": str(source.get("mention") or ""),
        "concept_id": source.get("concept_id"),
        "normalized_id": source.get("normalized_id")
        or source.get("concept_id"),
        "preferred_label": source.get("preferred_label"),
        "normalization_source": source.get("normalization_source") or "PubTator3",
        "matched_term": source.get("matched_term") or source.get("mention"),
        "term_kind": source.get("term_kind") or "pubtator3_mention",
        "resource_version": (
            source.get("resource_version") or PUBTATOR3_PIPELINE_VERSION
        ),
    }

    shared_fields = (
        "canonical_id_type",
        "canonical_name",
        "normalization_status",
        "source_concept_id",
        "source_entity_type",
        "label_source",
        "identified_source",
        "recognition_source",
        "supporting_sources",
        "definition_id", "definition_detector", "expanded_long_form",
        "matched_abbreviation_key", "entity_role", "entity_granularity",
        "hgnc_group_id", "reference_tax_id", "taxonomy_status", "seed_evidence",
    )
    gene_fields = (
        "hgnc_id",
        "ncbi_gene_id",
        "uniprot_ids",
        "pubtator_original_gene_id",
        "identity_correction",
    )
    hormone_fields = (
        "hormone_id",
        "mesh_id",
        "pubtator_mesh_id",
        "chemical_id",
        "hormone_classification_source",
    )

    entity_fields = gene_fields if entity_type == "gene" else hormone_fields
    for field in shared_fields + entity_fields:
        value = source.get(field)
        if value not in (None, "", [], ()):
            row[field] = value

    if entity_type == "gene":
        concept_id = str(source.get("concept_id") or "").strip()
        hgnc_id = str(source.get("hgnc_id") or "").strip()
        if concept_id.upper().startswith("HGNC:"):
            hgnc_id = hgnc_id or concept_id
        if hgnc_id:
            row["hgnc_id"] = hgnc_id
            row["concept_id"] = hgnc_id
            row["normalized_id"] = hgnc_id
        row.setdefault("identified_source", "pubtator3")
    else:
        row.setdefault("hormone_id", source.get("concept_id"))
        row.setdefault("source_entity_type", "Chemical")

    return {key: value for key, value in row.items() if value not in (None, "", [], ())}


def _annotation_signature(annotation: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        annotation.get("obj"),
        annotation.get("start"),
        annotation.get("end"),
        annotation.get("mention"),
        annotation.get("concept_id"),
        annotation.get("preferred_label"),
    )


def _resolve_annotation_conflicts(
    raw_annotations: Iterable[Mapping[str, Any]],
    *, text: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve receptors first, then retain longest non-overlapping spans."""
    from backend.pipeline.entity_span_rules import prune_cell_fragments, sanitize_source_annotations
    if text is not None:
        from backend.pipeline.document_entity_recovery import default_cell_span_resources
        from backend.pipeline.receptor_annotations import reconcile_receptor_annotations
        resources = default_cell_span_resources()
        raw_annotations = reconcile_receptor_annotations(text, raw_annotations, matcher=resources.genes)
        raw_annotations = sanitize_source_annotations(text, raw_annotations,
            known_cell=lambda value: resources.resolve_cell(value).status == "resolved_target",
            known_gene=lambda value: bool(resources.genes and resources.genes.resolve(value)),
            cell_surface_allowed=resources.cell_surface_allowed)
    accepted = prefer_longest_spans(prune_cell_fragments(raw_annotations))
    _sort_annotations(accepted)
    return accepted


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

                source_chunks = list(iter_jsonl(chunk_path))
                chunk_rows = [_chunk_result_row(row) for row in source_chunks]
                if source_name == "CellExLink":
                    # Used only during final per-paper reconciliation. Never
                    # copied into the final Stage 2 published annotation row.
                    for source_chunk, chunk_row in zip(source_chunks, chunk_rows):
                        if isinstance(source_chunk.get("chunk"), str):
                            chunk_row["_source_text"] = source_chunk["chunk"]
                        if entry.get("paper_identity"):
                            chunk_row["_paper_scope"] = str(entry["paper_identity"])
                context_path = entry.get("abbreviation_context_path")
                if source_name == "CellExLink" and context_path and Path(str(context_path)).is_file():
                    from backend.cellexlink_lite.normalization import build_document_text
                    from backend.pipeline.abbreviation_prepass import load_document_context
                    from backend.pipeline.document_entity_recovery import document_abbreviation_constraints, load_local_hgnc
                    context = load_document_context(context_path)
                    document = build_document_text(source_chunks, document_key=context.document_key)
                    constraints = document_abbreviation_constraints(document, context, load_local_hgnc())
                    for chunk_row in chunk_rows:
                        masks = [mask for mask in constraints if str(mask["chunk_id"]) == str(chunk_row.get("chunk_id"))]
                        if masks:
                            chunk_row["document_abbreviation_constraints"] = masks
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

                    # Legacy taxonomy masks are not entity annotations.
                    if raw_annotation.get("entity_type") == "excluded_nonhuman_gene":
                        continue
                    annotation = compact(raw_annotation)
                    signature = _annotation_signature(annotation)
                    if signature in seen_annotations[chunk_index]:
                        continue
                    seen_annotations[chunk_index].add(signature)
                    annotations_by_chunk[chunk_index].append(annotation)

                if unmatched:
                    raise ValueError(
                        f"Could not map {unmatched} {source_name} annotation(s) "
                        f"from {source_path} back to their Stage 1 chunks."
                    )

                for source_chunk, chunk_row, annotations in zip(source_chunks, chunk_rows, annotations_by_chunk):
                    text = source_chunk.get("chunk")
                    annotations = _resolve_annotation_conflicts(
                        annotations, text=text if isinstance(text, str) else None)
                    for annotation in annotations:
                        _count_annotation(counts, annotation)
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
    """Merge aligned branches, then reconcile repetitions one paper at a time."""
    from backend.pipeline.document_repeat_recovery import paper_scope, reconcile_document_rows

    counts = _empty_counts()
    output_chunk_count = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    def aligned_rows() -> Iterator[dict[str, Any]]:
        for row_number, pair in enumerate(
            zip_longest(iter_jsonl(cell_branch, sanitize=False),
                         iter_jsonl(pubtator_branch, sanitize=False)), start=1
        ):
            cell_row, pubtator_row = pair
            if cell_row is None or pubtator_row is None:
                raise ValueError("The CellExLink and PubTator3 branch artifacts contain different numbers of chunks.")
            if _chunk_identity(cell_row) != _chunk_identity(pubtator_row):
                raise ValueError(f"The CellExLink and PubTator3 branch artifacts are not aligned at row {row_number}.")
            merged = {key: value for key, value in cell_row.items() if key != "annotations"}
            candidates = []
            for branch_row in (cell_row, pubtator_row):
                annotations = branch_row.get("annotations") or []
                if not isinstance(annotations, list):
                    raise ValueError(f"Branch row {row_number} has an invalid annotations field.")
                for annotation in annotations:
                    if not isinstance(annotation, Mapping):
                        raise ValueError(f"Branch row {row_number} contains a non-object annotation.")
                    if str(annotation.get("obj") or "") not in _SUPPORTED_ENTITY_TYPES:
                        raise ValueError(f"Branch row {row_number} contains an unsupported entity type.")
                    candidates.append(dict(annotation))
            merged["annotations"] = candidates
            exclusions = [mask for branch in (cell_row, pubtator_row)
                          for mask in (branch.get("entity_span_exclusions") or [])]
            if exclusions:
                merged["entity_span_exclusions"] = exclusions
            # Anonymous chunks are isolated rather than pooled across papers.
            if not any(merged.get(f) for f in ("_paper_scope", "canonical_id", "doc_key", "pmid", "pmcid")):
                merged["_paper_scope"] = f"anonymous-chunk-{row_number}"
            yield merged

    seen_papers = set()
    with output.open("wb") as raw_output:
        with gzip.GzipFile(filename="", fileobj=raw_output, mode="wb", compresslevel=6, mtime=0) as destination:
            for scope, paper_rows in groupby(aligned_rows(), key=paper_scope):
                if scope in seen_papers:
                    raise ValueError(f"Non-contiguous paper {scope!r} in Stage 2 branch artifacts.")
                seen_papers.add(scope)
                for merged in reconcile_document_rows(list(paper_rows)):
                    annotations = merged["annotations"]
                    assert_no_cell_gene_overlaps(annotations)
                    for annotation in annotations:
                        _count_annotation(counts, annotation)
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
