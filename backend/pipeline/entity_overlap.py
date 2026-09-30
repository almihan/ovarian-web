"""Shared longest-span entity overlap policy.

Within one chunk, longer [start, end) spans replace overlapping shorter spans
regardless of entity type, source, lock flag, or normalization method. Equal
span conflicts prefer cell, then hormone, then gene/protein. Equal-length
crossing spans use deterministic evidence ordering.
Touching half-open spans do not overlap. No gene-specific exceptions exist.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping



ENTITY_OVERLAP_POLICY = "longest-span-exact-cell-hormone-gene-v3"

_GENE_TYPES = {
    "gene",
    "protein",
    "gene/protein",
    "gene protein",
    "gene or protein",
    "gene or gene product",
}
_CELL_TYPES = {
    "cell",
    "cell type",
}


def _normalized_type_text(value: Any) -> str:
    raw = str(value or "").strip().casefold()
    return " ".join(raw.replace("_", " ").replace("-", " ").split())


def annotation_entity_type(annotation: Mapping[str, Any]) -> str:
    """Return ``cell``, ``gene``, ``hormone``, or a normalized unknown value."""

    unknown = ""
    for field in ("obj", "entity_type", "source_entity_type"):
        normalized = _normalized_type_text(annotation.get(field))
        if not normalized:
            continue
        if normalized in _GENE_TYPES:
            return "gene"
        if normalized in _CELL_TYPES:
            return "cell"
        if normalized == "hormone":
            return "hormone"
        if not unknown:
            unknown = normalized
    return unknown


def annotation_span(annotation: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return a valid half-open annotation span, or ``None``."""

    try:
        start = int(annotation.get("start"))
        end = int(annotation.get("end"))
    except (TypeError, ValueError):
        return None
    if start < 0 or end <= start:
        return None
    return start, end


def spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Return whether two half-open spans overlap."""

    return left[0] < right[1] and left[1] > right[0]


def annotation_evidence_priority(annotation: Mapping[str, Any]) -> int:
    """Evidence breaks ties only; it can never displace a longer span."""
    scores = {
        "hgnc_receptor_expression": 120,
        "cell_ontology_coordinated_shared_head": 115,
        "cell_ontology_exact_surface": 110,
        "cell_ontology_exact_alias": 110,
        "final_complete_cell_phrase": 110,
        "ab3p_definition_long_form": 105,
        "ab3p_definition_short_form": 105,
        "ab3p_document_short_form": 100,
        "document_hgnc_exact_surface": 100,
        "static_exact_unique": 95,
        "mesh_hormone_exact_surface": 100,
        "approved HGNC complete-set record": 95,
        "HGNC complete set exact term match": 95,
        "PubTator3": 90,
        "pubtator3": 90,
        "cell_ontology_vector_top1": 70,
    }
    return max((scores.get(str(annotation.get(key) or ""), 60)
                for key in ("recognition_source", "normalization_source")), default=60)


def prefer_longest_spans(
    raw_annotations: Iterable[Mapping[str, Any]],
    *, priority: Callable[[Mapping[str, Any]], int] | None = None,
) -> list[dict[str, Any]]:
    """Greedily retain the longest non-overlapping mentions in one chunk.

    Exact-span conflicts use cell > hormone > gene/protein, before confidence
    or source priority. A longer span still wins across all types. Identical
    span/type/ID duplicates may share provenance, never conflicting identities.
    """
    evidence = priority or annotation_evidence_priority
    rows = [dict(row) for row in raw_annotations
            if isinstance(row, Mapping) and annotation_span(row) is not None]
    type_order = {"cell": 0, "hormone": 1, "gene": 2}
    preferred_type_by_span: dict[tuple[int, int], int] = {}
    for row in rows:
        span = annotation_span(row)
        rank = type_order.get(annotation_entity_type(row), 99)
        preferred_type_by_span[span] = min(rank, preferred_type_by_span.get(span, 99))
    rows = [row for row in rows if type_order.get(annotation_entity_type(row), 99)
            == preferred_type_by_span[annotation_span(row)]]
    def identity(row: Mapping[str, Any]) -> str:
        return str(row.get("concept_id") or row.get("normalized_id")
                   or row.get("cell_ontology_id") or row.get("hormone_id") or "")
    ranked = sorted(rows, key=lambda row: (
        -(int(row["end"]) - int(row["start"])),
        -evidence(row), type_order.get(annotation_entity_type(row), 99),
        int(row["start"]), int(row["end"]), identity(row),
        str(row.get("recognition_source") or row.get("normalization_source") or ""),
    ))
    kept: list[dict[str, Any]] = []
    for row in ranked:
        span = annotation_span(row)
        overlaps = [old for old in kept if spans_overlap(span, annotation_span(old))]
        if overlaps:
            duplicate = next((old for old in overlaps
                if annotation_span(old) == span
                and annotation_entity_type(old) == annotation_entity_type(row)
                and identity(old) == identity(row)), None)
            if duplicate is not None:
                sources = set(duplicate.get("supporting_sources") or [])
                sources.update(row.get("supporting_sources") or [])
                for item in (duplicate, row):
                    source = item.get("recognition_source") or item.get("normalization_source")
                    if source:
                        sources.add(str(source))
                for key, value in row.items():
                    if duplicate.get(key) in (None, "", [], ()) and value not in (None, "", [], ()):
                        duplicate[key] = value
                if sources:
                    duplicate["supporting_sources"] = sorted(sources)
            continue
        kept.append(row)
    return sorted(kept, key=lambda row: (int(row["start"]), int(row["end"]),
                                         annotation_entity_type(row), identity(row)))


# Compatibility for callers of earlier releases; this name no longer gives
# cell annotations priority over longer gene/protein or hormone annotations.
prefer_cells_over_overlapping_genes = prefer_longest_spans


def cell_gene_overlap_count(
    raw_annotations: Iterable[Mapping[str, Any]],
) -> int:
    """Legacy counter name: return annotations removed by the overlap policy."""

    annotations = [
        dict(annotation)
        for annotation in raw_annotations
        if isinstance(annotation, Mapping)
    ]
    return len(annotations) - len(prefer_longest_spans(annotations))


def assert_no_cell_gene_overlaps(
    raw_annotations: Iterable[Mapping[str, Any]],
) -> None:
    """Raise if an annotation list still contains a cell/gene overlap."""

    annotations = [
        dict(annotation)
        for annotation in raw_annotations
        if isinstance(annotation, Mapping)
    ]
    cell_spans = [
        span
        for annotation in annotations
        if annotation_entity_type(annotation) == "cell"
        for span in [annotation_span(annotation)]
        if span is not None
    ]
    if not cell_spans:
        return
    for annotation in annotations:
        if annotation_entity_type(annotation) != "gene":
            continue
        gene_span = annotation_span(annotation)
        if gene_span is None:
            continue
        if any(spans_overlap(gene_span, cell_span) for cell_span in cell_spans):
            raise ValueError(
                "A gene/protein annotation still overlaps a cell-type annotation."
            )


def _is_annotation_list(value: list[Any]) -> bool:
    """Return whether ``value`` is an entity-annotation list.

    This deliberately recognizes both a normal ``{"annotations": [...]}``
    field and a bare top-level annotation array.  A list is considered an
    annotation list only when every member is an object and at least one member
    has both a supported entity type and a valid text span.
    """

    if not value or not all(isinstance(item, Mapping) for item in value):
        return False
    return any(
        annotation_entity_type(item) in {"cell", "gene", "hormone"}
        and annotation_span(item) is not None
        for item in value
    )


def sanitize_annotation_payload(value: Any) -> Any:
    """Recursively apply the policy to every entity-annotation list in JSON."""

    if isinstance(value, Mapping):
        return {
            str(key): sanitize_annotation_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        items: list[Any]
        if _is_annotation_list(value):
            items = prefer_longest_spans(value)
            assert_no_cell_gene_overlaps(items)
        else:
            items = value
        return [sanitize_annotation_payload(item) for item in items]
    return value


__all__ = [
    "ENTITY_OVERLAP_POLICY",
    "annotation_evidence_priority",
    "prefer_longest_spans",
    "annotation_entity_type",
    "annotation_span",
    "assert_no_cell_gene_overlaps",
    "cell_gene_overlap_count",
    "prefer_cells_over_overlapping_genes",
    "sanitize_annotation_payload",
    "spans_overlap",
]
