"""Resolve complete receptor expressions through the local HGNC reference.

A cell, gene/protein or hormone followed by 'receptor(s)' is not annotated as
its shorter base entity. Resolve the complete expression (using a supported
long form for an abbreviated base) to HGNC or discard the expression. All
names and identities come from reference records, never a special-name table.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from backend.pipeline.entity_overlap import (
    annotation_entity_type, annotation_span, spans_overlap,
)
from backend.pipeline.reference_normalization import HgncExactMatcher

RECEPTOR_POLICY = "complete-receptor-expression-hgnc-or-discard-v1"
_DASHES = "-−‐‑‒–—﹘﹣－"
_QUALIFIER = (r"(?:alpha|beta|gamma|delta|epsilon|[αβγδε]|\d+[A-Za-z]?|"
              r"type\s+\d+|isoform\s+[A-Za-z0-9]+)(?:\s+(?:chain|subunit))?")
_TAIL = r"(?:[\s" + re.escape(_DASHES) + r"]+" + _QUALIFIER + r"\b)?"
_FOLLOWS = re.compile(r"[\s" + re.escape(_DASHES) + r"]+(?:nuclear\s+)?receptors?\b" + _TAIL, re.I)
_COMPLETE = re.compile(r"\breceptors?\b" + _TAIL + r"$", re.I)
_SOURCE_FIELDS = ("base", "doc_key", "canonical_id", "pmid", "pmcid", "journal",
                  "pub_year", "section_type", "chunk_id", "offset_scope")


def reconcile_receptor_annotations(
    text: str,
    raw_annotations: Iterable[Mapping[str, Any]],
    *, matcher: HgncExactMatcher | None = None,
) -> list[dict[str, Any]]:
    """Replace receptor-headed mentions before any longest-span filtering.

    This function operates on one original chunk. A failed full-expression
    lookup removes contained base mentions too, so a discarded receptor cannot
    revert to an annotation of its hormone, ligand, or cell-type prefix.
    """
    rows = [dict(row) for row in raw_annotations if isinstance(row, Mapping)]
    expressions: dict[tuple[int, int], list[tuple[dict[str, Any], str]]] = {}
    for row in rows:
        if annotation_entity_type(row) not in {"cell", "gene", "hormone"}:
            continue
        span = annotation_span(row)
        if span is None or span[1] > len(text):
            continue
        start, end = span
        suffix = _FOLLOWS.match(text, end)
        surface = text[start:end]
        if suffix is not None:
            expressions.setdefault((start, suffix.end()), []).append((row, suffix.group()))
        elif _COMPLETE.search(surface):
            expressions.setdefault(span, []).append((row, ""))
    if not expressions:
        return rows
    if matcher is None:
        from backend.pipeline.document_entity_recovery import load_local_hgnc
        matcher = load_local_hgnc()

    # A larger receptor expression owns its complete region, including when
    # it has no HGNC match. Never rescue a smaller ligand from that region.
    regions: list[tuple[int, int]] = []
    for span in sorted(expressions, key=lambda s: (-(s[1]-s[0]), s[0])):
        if not any(spans_overlap(span, other) for other in regions):
            regions.append(span)
    additions: list[dict[str, Any]] = []
    for start, end in regions:
        surface = text[start:end]
        evidence = expressions[(start, end)]
        resolved = matcher.resolve(surface) if matcher is not None else None
        expanded = surface
        if resolved is None and matcher is not None:
            # Parenthetical/document aliases and normalized base names are
            # lookup alternatives only; the original source is never rewritten.
            matches: dict[str, tuple[Any, str, dict[str, Any]]] = {}
            for row, suffix in evidence:
                for field in ("expanded_long_form", "canonical_name", "preferred_label", "matched_term"):
                    base = str(row.get(field) or "").strip()
                    if not base:
                        continue
                    lookup = base + suffix if suffix else base
                    # For an already complete span, a ligand-only label is not
                    # evidence for a receptor identity.
                    if not re.search(r"\breceptors?\b", lookup, re.I):
                        continue
                    hit = matcher.resolve(lookup)
                    if hit is not None:
                        matches.setdefault(hit.record.hgnc_id, (hit, lookup, row))
            if len(matches) == 1:
                resolved, expanded, source_row = next(iter(matches.values()))
            else:
                source_row = evidence[0][0]
        else:
            source_row = evidence[0][0]
        if resolved is None:
            continue
        record = resolved.record
        row = {key: source_row[key] for key in _SOURCE_FIELDS if key in source_row}
        row.update({
            "obj": "gene", "entity_type": "gene", "source_entity_type": "Gene/Protein",
            "start": start, "end": end, "mention": surface, "offset_scope": "chunk",
            "concept_id": record.hgnc_id, "normalized_id": record.hgnc_id,
            "hgnc_id": record.hgnc_id, "canonical_id_type": "hgnc",
            "preferred_label": record.symbol, "canonical_name": record.name,
            "matched_term": resolved.matched_term, "term_kind": resolved.term_kind,
            "resource_version": resolved.resource_version, "label_source": "HGNC",
            "normalization_system": "HGNC", "normalization_status": "canonical_hgnc",
            "normalization_source": "hgnc_receptor_expression",
            "recognition_source": "hgnc_receptor_expression",
            "identified_source": "reference_recovery", "entity_role": "receptor",
            "expanded_long_form": expanded,
            "reference_tax_id": "9606", "taxonomy_status": "not_validated_hgnc_mapping_only",
        })
        if record.entrez_id:
            row["ncbi_gene_id"] = f"NCBIGene:{record.entrez_id}"
        if record.uniprot_ids:
            row["uniprot_ids"] = list(record.uniprot_ids)
        if source_row.get("document_start") is not None:
            offset = int(source_row["document_start"]) - int(source_row["start"])
            row.update(document_start=offset+start, document_end=offset+end)
        additions.append(row)
    remaining = [row for row in rows if not (
        (span := annotation_span(row)) is not None
        and any(a <= span[0] < span[1] <= b for a, b in regions))]
    return [*remaining, *additions]
