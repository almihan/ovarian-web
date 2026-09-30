"""Reconcile one paper after both Stage 2 branches.

A fixed registry of unambiguous mentions is searched across the same paper.
Validated cell seeds are retained before later additions can replace their
source occurrence, so overlap cleanup cannot erase document-level evidence.
New mentions are never used recursively as seeds. Receptors must resolve to
HGNC, and the common longest-span rule owns every overlap. No species filter
or special-gene override is applied. Final annotations remain text-free.
"""
from __future__ import annotations

import collections
import logging
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence

from backend.pipeline.entity_span_rules import (
    DASHES, apply_document_constraints, apply_span_exclusions, full_context_cell_spans,
    gene_term_matches, is_excluded_gene_identity, is_gene_group_annotation,
    is_generic_gene_surface, is_noncell_surface, row_span, sanitize_source_annotations,
    source_span_exclusions, token_aligned,
)

logger = logging.getLogger(__name__)
RECOVERY_VERSION = "same-paper-longest-cell-recovery-nonrecursive-v6"
_MAX_EXAMPLES = 100


def paper_scope(row: Mapping[str, Any], *, fallback: str = "") -> str:
    for field in ("_paper_scope", "canonical_id", "doc_key", "pmcid", "pmid"):
        if row.get(field) not in (None, ""):
            return f"{field}:{row[field]}"
    # Missing paper identity must never pool an entire collection of rows.
    return f"isolated:{fallback or row.get('base') or row.get('chunk_id') or id(row)}"


def _kind(row: Mapping) -> str:
    kind = str(row.get("obj") or row.get("entity_type") or "").casefold()
    return "cell" if kind in {"cell", "cell_type", "cell type"} else ("gene" if kind == "protein" else kind)


def _identifier(row: Mapping) -> str:
    return str(row.get("concept_id") or row.get("normalized_id") or "")


def _surface_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).translate(str.maketrans({c: "-" for c in DASHES}))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _literal_pattern(value: str) -> re.Pattern:
    """Preserve spaces/punctuation; allow dash glyphs, not word concatenation."""
    parts = []
    for piece in re.split(r"(\s+)", value.strip()):
        if piece.isspace():
            parts.append(r"\s+")
        else:
            parts.extend("[" + re.escape(DASHES) + "]" if c in DASHES else re.escape(c) for c in piece)
    return re.compile(r"(?<!\w)" + "".join(parts) + r"(?!\w)", re.I)


def _safe_seed(row: Mapping, resources: Any) -> bool:
    kind, identifier = _kind(row), _identifier(row)
    mention = str(row.get("mention") or "")
    if not identifier or not mention or is_gene_group_annotation(row) or is_excluded_gene_identity(row):
        return False
    if not any(c.isalpha() for c in mention) or len(mention.strip()) < 2:
        return False
    if str(row.get("recognition_source") or "") == "document_repeat_recovery":
        return False
    if str(row.get("normalization_source") or "") == "coordinated_receptor_gene":
        return False  # 'glucocorticoid' with an omitted receptor head is not a global gene surface.
    if kind == "cell":
        if is_noncell_surface(mention) or re.search(r"\s+(?:and|or)\s|/", mention, re.I):
            return False
        resolved = resources.resolve_cell(mention)
        if resolved.candidate is not None:
            return resolved.candidate.concept_id == identifier
        if identifier not in resources.cell_labels:
            return False
        from backend.cellexlink_lite.normalization import cell_min_cosine
        try:
            return float(row.get("ontology_raw_cosine", -1)) >= cell_min_cosine()
        except (TypeError, ValueError):
            return False
    if kind == "gene":
        if is_generic_gene_surface(mention):
            return False
        if not identifier.startswith("HGNC:"):
            return False
        match = resources.genes.resolve(mention) if resources.genes else None
        if match is not None:
            return identifier == match.record.hgnc_id
        known_id = bool(resources.genes and identifier in resources.genes.records_by_hgnc)
        return bool(known_id and (row.get("definition_id") or
                    str(row.get("identified_source") or "").lower() == "pubtator3"))
    if kind == "hormone":
        return identifier.startswith("MESH:") and not is_generic_gene_surface(mention)
    return False


def reconcile_document_rows(rows: Sequence[Mapping[str, Any]], *, resources: Any = None) -> list[dict]:
    """Reconcile a bounded, single-paper batch, returning text-free chunk rows."""
    if not rows:
        return []
    scopes = {paper_scope(row) for row in rows}
    if len(scopes) != 1:
        raise ValueError("Document repeat recovery cannot mix paper identities.")
    scope = next(iter(scopes))
    from backend.pipeline.document_entity_recovery import default_cell_span_resources
    from backend.pipeline.entity_artifacts import _resolve_annotation_conflicts
    resources = resources or default_cell_span_resources()
    values = [dict(row) for row in rows]
    texts = [row.get("_source_text", row.get("chunk")) for row in values]
    # Ignore persisted masks produced by the retired species-validation policy.
    for parent in values:
        parent.pop("excluded_nonhuman_gene_spans", None)
        parent.pop("protected_gene_audit", None)
    audits = []
    # Snapshot independently accepted cell occurrences, not repeated output.
    # Later exact/longer annotations can displace a source span, but must not
    # erase its valid cell identity from the same-paper repetition registry.
    cell_seed_rows: list[list[dict]] = [[] for _ in values]

    def blocked(index: int, annotation: Mapping, reason: str, origin: str) -> None:
        audit = audits[index]
        audit["blocked_reason_counts"][reason] += 1
        if len(audit["blocked_occurrences"]) < _MAX_EXAMPLES:
            audit["blocked_occurrences"].append({
                "start": annotation.get("start"), "end": annotation.get("end"),
                "mention": annotation.get("mention"), "candidate_type": _kind(annotation),
                "concept_id": _identifier(annotation), "reason": reason, "origin": origin})
        else:
            audit["blocked_examples_omitted"] += 1

    def rejection(index: int, annotation: Mapping) -> str | None:
        parent, text = values[index], texts[index]
        span = row_span(annotation)
        if span is None or not isinstance(text, str) or span[1] > len(text):
            return "invalid_or_unavailable_source_span"
        kind = _kind(annotation)
        # An imported stale span covering multiple fully written genes must
        # not occupy the whole IL-2/IL-21 combination. Valid complete receptor
        # expressions and single resource names remain intact.
        surface = text[slice(*span)]
        if (kind == "gene" and resources.genes and not resources.genes.resolve(surface)
                and re.search(r"[,/]|\b(?:and|or)\b", surface)
                and len(resources.genes.find(surface)) > 1):
            return "multiple_genes_in_one_span"
        if kind == "cell" and resources.resolve_cell(surface).candidate is None:
            score = annotation.get("ontology_raw_cosine")
            if score is not None:
                from backend.cellexlink_lite.normalization import cell_min_cosine
                try:
                    if not math.isfinite(float(score)) or float(score) < cell_min_cosine():
                        return "low_confidence_cell_normalization"
                except (TypeError, ValueError):
                    return "invalid_cell_normalization_score"
        if not token_aligned(text, *span):
            return "partial_token"
        if is_excluded_gene_identity(annotation) or is_gene_group_annotation(annotation):
            return "unsupported_gene_identity"
        if kind == "gene" and not _identifier(annotation).startswith("HGNC:"):
            return "missing_hgnc_identity"
        masks = [*(parent.get("entity_span_exclusions") or []), *source_span_exclusions(text)]
        for mask in masks:
            masked = row_span(mask)
            if masked and kind in mask.get("blocked_types", []) and span[0] < masked[1] and span[1] > masked[0]:
                return str(mask.get("reason") or "context_exclusion")
        if not apply_document_constraints([annotation], parent.get("document_abbreviation_constraints") or []):
            return "document_abbreviation_conflict"
        if not sanitize_source_annotations(text, [annotation],
                known_cell=lambda v: resources.resolve_cell(v).status == "resolved_target",
                known_gene=lambda v: bool(resources.genes and resources.genes.resolve(v)),
                cell_surface_allowed=resources.cell_surface_allowed):
            return "unsafe_source_surface_or_cell_fragment"
        return None

    # Sanitize the actual merged input before it can become a propagation seed.
    for index, (parent, text) in enumerate(zip(values, texts)):
        original = [dict(a) for a in parent.get("annotations", []) if isinstance(a, Mapping)]
        audits.append({"version": RECOVERY_VERSION, "document_scope": scope,
            "source_text_available": isinstance(text, str), "input_annotations": len(original),
            "exact_gene_added": 0, "complete_cell_phrase_added": 0, "repeated_added": 0,
            "identity_corrections": 0, "ambiguous_surface_keys": 0,
            "blocked_reason_counts": collections.Counter(), "blocked_occurrences": [],
            "blocked_examples_omitted": 0})
        if not isinstance(text, str):
            # Old text-free branches cannot be searched; do not guess offsets.
            original = apply_span_exclusions(original, parent.get("entity_span_exclusions") or [])
            original = apply_document_constraints(original, parent.get("document_abbreviation_constraints") or [])
            parent["annotations"] = _resolve_annotation_conflicts(original)
            continue
        from backend.pipeline.receptor_annotations import reconcile_receptor_annotations
        original = reconcile_receptor_annotations(text, original, matcher=resources.genes)
        kept = []
        for annotation in original:
            span = row_span(annotation)
            if span and span[1] <= len(text):
                # Offsets, not stale mention strings, define the source surface.
                annotation["mention"] = text[slice(*span)]
            reason = rejection(index, annotation)
            if reason:
                blocked(index, annotation, reason, "merged_input")
            else:
                kept.append(annotation)
        parent["annotations"] = _resolve_annotation_conflicts(kept, text=text)
        cell_seed_rows[index].extend(dict(row) for row in parent["annotations"]
                                     if _kind(row) == "cell")

    def add(index: int, annotation: dict, origin: str) -> bool:
        reason = rejection(index, annotation)
        if reason:
            blocked(index, annotation, reason, origin)
            return False
        old_rows = values[index]["annotations"]
        span = row_span(annotation)
        overlaps = [old for old in old_rows if row_span(old) and
                    span[0] < int(old["end"]) and span[1] > int(old["start"])]
        if any(row_span(old) == span and _kind(old) == _kind(annotation) and
               _identifier(old) == _identifier(annotation) for old in overlaps):
            return False
        # A newly found longer phrase may replace an earlier shorter mention.
        # Receptor processing also discards an unmatched full expression rather
        # than letting its shorter base survive during repeated-mention rescue.
        reconciled = _resolve_annotation_conflicts([*old_rows, annotation], text=texts[index])
        values[index]["annotations"] = reconciled
        if not any(row_span(row) == span and _kind(row) == _kind(annotation)
                   and _identifier(row) == _identifier(annotation) for row in reconciled):
            blocked(index, annotation, "longer_or_preferred_entity_overlap", origin)
            return False
        if _kind(annotation) == "cell" and origin != "repeated_added":
            cell_seed_rows[index].append(dict(annotation))
        audits[index][origin] += 1
        return True

    # Last-mile exact audit: handles a missing branch occurrence, not just a
    # missing alias. Runs before seed creation and respects the same vetoes.
    for index, (parent, text) in enumerate(zip(values, texts)):
        if not isinstance(text, str) or str(parent.get("section_type") or "").upper() in {"TITLE", "METADATA"}:
            continue
        for a, b, candidate in resources.exact_target_spans(text):
            if candidate.entity_type != "cell":
                continue
            coordinated = candidate.term_kind == "coordinated_shared_head"
            annotation = {**candidate.to_dict(), "obj": "cell", "normalized_id": candidate.concept_id,
                "start": a, "end": b, "mention": text[a:b],
                "recognition_source": ("cell_ontology_coordinated_shared_head" if coordinated
                                       else "final_complete_cell_phrase"),
                "normalization_status": "normalized",
                "normalization_source": ("cell_ontology_coordinated_shared_head" if coordinated
                    else "static_cell_surface" if candidate.term_kind == "static_abbreviation"
                    else "cell_ontology_exact_surface"),
                "normalization_system": "Cell Ontology"}
            add(index, annotation, "complete_cell_phrase_added")
        if resources.genes is None:
            continue
        for hit in resources.genes.find(text):
            record = hit.record
            annotation = {"obj": "gene", "entity_type": "gene", "start": hit.start, "end": hit.end,
                "mention": hit.mention, "concept_id": record.hgnc_id, "normalized_id": record.hgnc_id,
                "hgnc_id": record.hgnc_id, "ncbi_gene_id": f"NCBIGene:{record.entrez_id}",
                "preferred_label": record.symbol, "canonical_name": record.name,
                "uniprot_ids": list(record.uniprot_ids), "matched_term": hit.matched_term,
                "term_kind": hit.term_kind, "resource_version": hit.resource_version,
                "normalization_source": "final_document_hgnc_exact_audit", "normalization_system": "HGNC",
                "normalization_status": "normalized", "recognition_source": "final_document_hgnc_exact_audit",
                "identified_source": "exact_match", "reference_tax_id": "9606",
                "taxonomy_status": "not_validated_hgnc_mapping_only"}
            # Whole approved symbols may correct an alias-ID error, never a
            # hormone/cell identity. Span length is still resolved centrally.
            if not rejection(index, annotation) and resources.genes.resolve_approved_symbol(hit.mention):
                for old in list(parent["annotations"]):
                    if (_kind(old) == "gene"
                            and row_span(old) == (hit.start, hit.end)
                            and _identifier(old) != record.hgnc_id):
                        parent["annotations"].remove(old)
                        annotation["previous_normalized_id"] = _identifier(old)
                        annotation["identity_correction"] = "whole_approved_symbol"
                        audits[index]["identity_corrections"] += 1
            add(index, annotation, "exact_gene_added")

    registry: dict[str, list[tuple[int, dict]]] = collections.defaultdict(list)
    for index, parent in enumerate(values):
        if not isinstance(texts[index], str):
            continue
        seen_seeds: set[tuple] = set()
        for annotation in [*cell_seed_rows[index], *parent["annotations"]]:
            signature = (row_span(annotation), _kind(annotation), _identifier(annotation))
            if signature in seen_seeds:
                continue
            if _safe_seed(annotation, resources):
                seen_seeds.add(signature)
                registry[_surface_key(str(annotation["mention"]))].append((index, dict(annotation)))
    # Cross-type collisions are also ambiguous: one occurrence of a cell and a
    # hormone with the same surface is not a license to copy either arbitrarily.
    ambiguous = {key for key, seeds in registry.items() if len({(_kind(a), _identifier(a)) for _, a in seeds}) != 1}
    for audit in audits:
        audit["ambiguous_surface_keys"] = len(ambiguous)
    # Freeze the complete registry before searching any target chunk. This is
    # bidirectional over the paper and cannot depend on which chunk came first.
    # Longer surfaces are considered first, with all overlap decisions still
    # delegated to the common resolver in add().
    for key, seeds in sorted(registry.items(), key=lambda item: (-len(item[0]), item[0])):
        if key in ambiguous:
            continue
        seed_index, seed = seeds[0]
        pattern = _literal_pattern(str(seed["mention"]))
        for index, (parent, text) in enumerate(zip(values, texts)):
            if not isinstance(text, str) or str(parent.get("section_type") or "").upper() in {"TITLE", "METADATA"}:
                continue
            for match in pattern.finditer(text):
                # Case-sensitive short Latin qualifiers remain significant.
                if _kind(seed) == "gene" and not gene_term_matches(match.group(), str(seed["mention"])):
                    continue
                candidate = {k: v for k, v in seed.items() if k not in {
                    "start", "end", "mention", "mention_id", "chunk_id", "base", "doc_key", "canonical_id",
                    "pmid", "pmcid", "document_start", "document_end", "section_type", "locked",
                    "definition_id", "seed_evidence", "recognition_source", "supporting_sources"}}
                candidate.update({"obj": _kind(seed), "entity_type": _kind(seed), "start": match.start(),
                    "end": match.end(), "mention": match.group(), "offset_scope": "chunk",
                    "recognition_source": "document_repeat_recovery",
                    "normalization_source": "document_unique_validated_surface",
                    "seed_evidence": {"paper": scope, "chunk_id": values[seed_index].get("chunk_id"),
                        "start": seed["start"], "end": seed["end"], "concept_id": _identifier(seed),
                        "recognition_source": seed.get("recognition_source"),
                        "normalization_source": seed.get("normalization_source")}})
                # Copy only identity, not an Ab3P definition's occurrence offsets.
                add(index, candidate, "repeated_added")

    totals = collections.Counter()
    for parent, text, audit in zip(values, texts, audits):
        parent["annotations"] = _resolve_annotation_conflicts(parent["annotations"],
            text=text if isinstance(text, str) else None)
        audit["final_annotations"] = len(parent["annotations"])
        audit["blocked_reason_counts"] = dict(audit["blocked_reason_counts"])
        parent["entity_recovery_audit"] = audit
        for key in ("exact_gene_added", "complete_cell_phrase_added", "repeated_added", "identity_corrections"):
            totals[key] += audit[key]
        parent.pop("_source_text", None)
        parent.pop("_paper_scope", None)
        parent.pop("chunk", None)
    logger.info("[DOCUMENT_ENTITY_RECOVERY] paper=%s chunks=%d exact_added=%d repeats_added=%d "
                "full_cells_added=%d identity_corrections=%d ambiguous_surfaces=%d "
                "blocked=%s",
        scope, len(values), totals["exact_gene_added"], totals["repeated_added"],
        totals["complete_cell_phrase_added"], totals["identity_corrections"], len(ambiguous),
        dict(sum((collections.Counter(a["blocked_reason_counts"]) for a in audits), collections.Counter())))
    return values
