"""Deterministic, document-scoped recovery around the neural recognizers.

Ab3P remains the document abbreviation detector. Resource-validated explicit
parentheses supplement non-initialisms such as progesterone (P4), which an
initial-letter detector can miss. Every output span slices the original text.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.pipeline.entity_lexicons import (
    ExactResolution, ExactTargetIndex, TargetEntityCandidate, hormone_candidates,
    load_cell_candidates,
)
from backend.pipeline.entity_span_rules import (
    DASHES, gene_surface_key, is_noncell_surface, iter_compact_matches,
    key_prefixes, plain_surface_key, cd56_nk_spans, source_span_exclusions,
    apply_span_exclusions, sanitize_source_annotations, full_context_cell_spans,
)
from backend.pipeline.entity_text_normalization import canonical_resource_key
from backend.pipeline.cell_surface_matching import cell_term_matches
from backend.pipeline.cell_coordination import CoordinatedCellResolver
from backend.pipeline.reference_normalization import HgncExactMatcher, build_hgnc_exact_matcher

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HGNC_PATH = _ROOT / "data/reference_data/hgnc_complete_set.txt"


@lru_cache(maxsize=4)
def _cached_hgnc(path: str, size: int, mtime: int) -> HgncExactMatcher:
    return build_hgnc_exact_matcher(Path(path))


def load_local_hgnc(path: Path = DEFAULT_HGNC_PATH) -> HgncExactMatcher | None:
    """Use the application's existing reference, without a second download."""
    override = os.environ.get("HGNC_REFERENCE_PATH")
    if override:
        path = Path(override).expanduser()
    elif path == DEFAULT_HGNC_PATH:
        cache_dir = os.environ.get("REFERENCE_DATA_CACHE_DIR")
        candidate = Path(cache_dir).expanduser() / "hgnc_complete_set.txt" if cache_dir else path
        if candidate.is_file():
            path = candidate
    if not path.is_file():
        logger.warning("Local HGNC reference missing at %s; local gene abbreviation recovery disabled", path)
        return None
    stat = path.stat()
    return _cached_hgnc(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def gene_candidate(match: Any) -> TargetEntityCandidate:
    return TargetEntityCandidate("gene", match.record.hgnc_id, match.record.symbol,
                                 match.matched_term, match.term_kind, match.resource_version)


class RecoveryResources:
    def __init__(self, *, ontology_path: Path, hormone_entries: Sequence,
                 abbreviations_path: Path | None = None,
                 ontology_version: str | None = None,
                 gene_matcher: HgncExactMatcher | None = None) -> None:
        self.genes = gene_matcher
        cells = load_cell_candidates(ontology_path, resource_version=ontology_version)
        self.cell_labels = {c.concept_id: c.preferred_label for c in cells}
        self.ontology_exact = ExactTargetIndex(cells)
        ontology_candidates = tuple(cells)
        ontology_version = cells[0].resource_version if cells else ontology_version
        if abbreviations_path and abbreviations_path.is_file():
            abbreviation_version = (abbreviations_path.name + "-" +
                hashlib.sha256(abbreviations_path.read_bytes()).hexdigest()[:12])
            with abbreviations_path.open(encoding="utf-8-sig", newline="") as handle:
                for line_number, row in enumerate(csv.DictReader(handle, delimiter="\t"), start=2):
                    sf, identifier = row.get("short_form", "").strip(), row.get("matched_cl_id", "").strip()
                    if sf and identifier in self.cell_labels and not is_noncell_surface(sf):
                        cells.append(TargetEntityCandidate(
                            "cell", identifier, self.cell_labels[identifier], sf,
                            "static_abbreviation", abbreviation_version,
                            {"resource_file": abbreviations_path.name,
                             "resource_line": line_number,
                             "ontology_resource_version": ontology_version or ontology_path.stem}))
        self.cell_exact = ExactTargetIndex(cells)
        hormones = list(hormone_candidates(hormone_entries))
        self.targets = ExactTargetIndex([*cells, *hormones])
        self.hormone_exact = ExactTargetIndex(hormones)
        self.scan_terms: dict[str, list[TargetEntityCandidate]] = defaultdict(list)
        for item in [*cells, *hormones]:
            term = item.matched_term
            if is_noncell_surface(term) and item.entity_type == "cell":
                continue
            if item.entity_type == "cell" and re.search(r"\s(?:and|or)\s|/", term, re.I):
                continue  # coordination is handled by the shared-head resolver
            if item.entity_type == "hormone" and (len(plain_surface_key(term)) < 4 or
                    plain_surface_key(term) in {"hormone", "hormones"}):
                continue  # short hormone codes need a document definition
            key = plain_surface_key(term)
            if key:
                self.scan_terms[key].append(item)
        self.scan_prefixes = key_prefixes(self.scan_terms)
        self.coordination = CoordinatedCellResolver(ontology_candidates, self._resolve_cell_literal)

    def cell_surface_allowed(self, text: str) -> bool:
        """Do not let vector fallback or propagation bypass an unsafe exact hit."""
        candidates = [candidate for candidate in self.scan_terms.get(plain_surface_key(text), ())
                      if candidate.entity_type == "cell"]
        if not candidates:
            return True
        return any(cell_term_matches(text, candidate.matched_term, candidate.term_kind)
                   for candidate in candidates)

    def resolve_cell(self, text: str) -> ExactResolution:
        literal = self._resolve_cell_literal(text)
        if literal.status != "unmatched":
            return literal
        return self.coordination.resolve(text)

    def _resolve_cell_literal(self, text: str) -> ExactResolution:
        if is_noncell_surface(text):
            return ExactResolution("unmatched")
        # A generic surface never implies a narrower subtype just because an
        # alias table also contains it (e.g. naive CD4 cells or CD4 Tregs).
        marker = re.fullmatch(r"\s*CD(4|8)\s*\+\s*T[ -]+cells?\s*", text, re.I)
        if marker:
            return self.ontology_exact.resolve(f"CD{marker.group(1)}-positive, alpha-beta T cell")
        surface = str(text or "").strip()
        if re.fullmatch(r"natural[ -]+killer\s*\(\s*NK\s*\)[ -]*cells?", surface, re.I):
            return self.ontology_exact.resolve("natural killer cell")
        if re.fullmatch(r"NK[ -]+subpopulations?", surface, re.I):
            return self.ontology_exact.resolve("natural killer cell")
        phenotype = next(((a, b, level) for a, b, level in cd56_nk_spans(surface)
                          if a == 0 and b == len(surface)), None)
        if phenotype:
            # Use the existing static NK phenotype mapping, never a guessed CL
            # ID and never the molecular gene alias CD56/NCAM1.
            return self.cell_exact.resolve(f"CD56{phenotype[2]} NK cells")
        if re.fullmatch(r"\s*Tregs?(?:[ -]+cells?)?\s*", text, re.I):
            return self.ontology_exact.resolve("regulatory T cell")
        result = self.ontology_exact.resolve(text)
        if result.status != "unmatched":
            return result
        return self.cell_exact.resolve(text)

    def resolve(self, text: object) -> ExactResolution:
        raw = str(text or "").strip()
        result = self.resolve_cell(raw)
        hormone = self.hormone_exact.resolve(raw)
        if result.status != "unmatched":
            return result
        if hormone.status != "unmatched":
            return hormone
        if self.genes:
            gene = self.genes.resolve(raw)
            if gene is None:
                # Nuclear progesterone receptor -> progesterone receptor.
                gene = self.genes.resolve(re.sub(r"^(?:nuclear|human)\s+", "", raw, flags=re.I))
            if gene:
                candidate = gene_candidate(gene)
                return ExactResolution("resolved_target", candidate, (candidate,), "hgnc_exact")
        return ExactResolution("unmatched")

    def exact_target_spans(self, text: str) -> list[tuple[int, int, TargetEntityCandidate]]:
        hits = []
        exclusions = source_span_exclusions(text)
        for start, end, _key in iter_compact_matches(text, self.scan_terms, self.scan_prefixes):
            surface = text[start:end]
            result = self.resolve(surface)
            if result.status != "resolved_target" or result.candidate is None:
                continue
            candidate = result.candidate
            if any(candidate.entity_type in mask["blocked_types"] and start < mask["end"] and end > mask["start"]
                   for mask in exclusions):
                continue
            if candidate.entity_type not in {"cell", "hormone"}:
                continue
            if candidate.entity_type == "cell":
                if is_noncell_surface(surface):
                    continue
                if re.fullmatch(r"killer[ -]+cells?", surface, re.I) and re.match(
                        r"\s+immunoglobulin", text[end:], re.I):
                    continue
                # Cell-resource acceptance preserves word boundaries and
                # visible abbreviation casing; compact scan keys only retrieve
                # candidates, never justify a match by themselves.
            hits.append((start, end, candidate))
        for match in re.finditer(r"(?<!\w)CD(?:4|8)\s*\+\s*T[ -]+cells?\b", text, re.I):
            result = self.resolve_cell(match.group())
            if result.candidate:
                hits.append((*match.span(), result.candidate))
        for start, end, lookup in full_context_cell_spans(text):
            result = self.resolve_cell(lookup)
            if result.candidate:
                hits.append((start, end, result.candidate))
        hits.extend(self.coordination.find(text, list(hits)))
        return [(a, b, candidate) for a, b, candidate in hits if not any(
            candidate.entity_type in mask["blocked_types"] and a < mask["end"] and b > mask["start"]
            for mask in exclusions)]


def abbreviation_surfaces(definitions: Sequence[Any]) -> tuple[str, ...]:
    """Exact defined codes plus a controlled lower-case plural -s variant."""
    output: set[str] = set()
    for definition in definitions:
        surface = str(definition.short_form).strip()
        if not surface:
            continue
        output.add(surface)
        if len(surface) > 2 and surface.endswith("s") and any(c.isupper() for c in surface[:-1]):
            output.add(surface[:-1])
    return tuple(sorted(output))


def abbreviation_pair_conflicts(short_form: str, long_form: str, resources: Any) -> bool:
    """A marker or marker list is not a new definition of an existing gene.

    Apply to both native Ab3P output and supplemental pairs. Rejected pairs
    must be omitted, not kept as unresolved document-wide exclusion masks.
    Approved symbols always retain their own identity. A contextual hormone
    abbreviation may still disambiguate a colliding historical gene alias
    (progesterone/P4 versus the EXOSC10 alias p4).
    """
    matcher = getattr(resources, "genes", None)
    # Lists are not single short forms, even if a detector returns one pair.
    if re.search(r"[,;/]|\s+(?:and|or)\s+", short_form, re.I):
        return True
    if matcher is None:
        return False
    resolution = resources.resolve(long_form)
    candidate = resolution.candidate
    approved = matcher.resolve_approved_symbol(short_form)
    if approved is not None:
        return candidate is None or (candidate.entity_type, candidate.concept_id) != (
            "gene", approved.record.hgnc_id)
    # An independently resolved alias for another gene is also a marker in a
    # gene/gene pair. Do not let IL12B (DAP10), for example, rename HCST.
    short_gene = matcher.resolve(short_form)
    if candidate is not None and candidate.entity_type == "gene" and short_gene is not None:
        return candidate.concept_id != short_gene.record.hgnc_id
    return False


def _abbreviation_letters_align(short_form: str, long_form: str) -> bool:
    """Conservative initial-letter/subsequence evidence for supplemental pairs.

    The first short-form character must start a long-form word. The remaining
    characters (including digits) must occur in order. This does not infer an
    entity ID; the complete long form must separately resolve to a resource.
    """
    short = plain_surface_key(short_form)
    long = str(long_form).casefold()
    if len(short) < 2:
        return False
    position = len(long) - 1
    for index in range(len(short) - 1, -1, -1):
        while position >= 0 and (
            long[position] != short[index]
            or (index == 0 and position > 0 and long[position - 1].isalnum())
        ):
            position -= 1
        if position < 0:
            return False
        position -= 1
    return True


def _supplement_has_identity_evidence(short_form: str, long_form: str,
                                      resources: Any, resolution: ExactResolution) -> bool:
    """Parentheses alone are insufficient: require name or abbreviation evidence."""
    candidate = resolution.candidate
    if candidate is None:
        return False
    short_resolution = resources.resolve(short_form)
    if short_resolution.candidate is not None and (
        short_resolution.candidate.entity_type, short_resolution.candidate.concept_id
    ) == (candidate.entity_type, candidate.concept_id):
        return True
    if _abbreviation_letters_align(short_form, long_form):
        return True
    # Non-initialism explicitly requested by the project. Still requires an
    # in-document progesterone (P4) definition and the MeSH long-form match;
    # never creates a global short-form mapping or rewrites source characters.
    if (candidate.entity_type == "hormone"
            and plain_surface_key(candidate.preferred_label) == "progesterone"
            and plain_surface_key(short_form) == "p4"):
        return True
    # IL12p40 -> the recorded HGNC alias "interleukin 12, p40". The subunit
    # number is preserved and must match the SAME record as the long form.
    matcher = getattr(resources, "genes", None)
    interleukin = re.fullmatch(r"il(\d+)p(\d+)", gene_surface_key(short_form))
    if matcher is not None and candidate.entity_type == "gene" and interleukin:
        alias = matcher.resolve(
            f"interleukin {interleukin.group(1)}, p{interleukin.group(2)}")
        return alias is not None and alias.record.hgnc_id == candidate.concept_id
    return False


def supplement_parenthetical_pairs(document: Any, pairs: Sequence[tuple[str, str]],
                                    resources: Any) -> tuple[list[tuple[str, str]], set[tuple[str, str]]]:
    """Supplement LF(SF) only with resource and abbreviation-identity evidence.

    Known gene symbols in parentheses retain their own identity. Multiple
    markers, measurements and arbitrary unrelated codes are not definitions.
    Valid non-initialisms such as P4 and IL12p40 remain document-scoped.
    """
    result = [(sf, lf) for sf, lf in pairs
              if not abbreviation_pair_conflicts(sf, lf, resources)]
    supplements: set[tuple[str, str]] = set()
    seen = {(sf.casefold(), lf.casefold()) for sf, lf in result}
    for chunk in document.chunks:
        text = str(chunk.source.get("chunk") or "")
        for paren in re.finditer(r"\(\s*([^()\n]{2,30})\s*\)", text):
            sf = paren.group(1).strip()
            if not re.fullmatch(r"[\wβγδ\-+./ ]{2,30}", sf):
                continue
            if not (any(c.isupper() for c in sf) or any(c.isdigit() for c in sf)):
                continue
            if not any(c.isalpha() for c in sf) or sf.casefold() in {"and", "or", "fig", "figure"}:
                continue
            if re.match(r"^\d", sf) or re.search(r"\s", sf) or not sf[0].isalpha():
                continue
            before = text[:paren.start()].rstrip()
            words = list(re.finditer(r"\S+", before))[-18:]
            for word in words:
                lf = before[word.start():]
                if re.search(r"[.;!?\n]", lf):
                    continue
                resolution = resources.resolve(lf)
                if resolution.status != "resolved_target":
                    continue
                # Do not try a shorter suffix after a resolved but conflicting
                # complete name: it cannot rescue an unrelated marker pair.
                if abbreviation_pair_conflicts(sf, lf, resources) or not (
                    _supplement_has_identity_evidence(sf, lf, resources, resolution)
                ):
                    break
                pair = (sf, lf)
                key = (sf.casefold(), lf.casefold())
                if key not in seen:
                    result.append(pair)
                    supplements.add(pair)
                    seen.add(key)
                break
    return result, supplements


def _source_row(chunk: Any, start: int, end: int, candidate: TargetEntityCandidate,
                source: str, **extra: Any) -> dict[str,Any]:
    fields = ("base","doc_key","canonical_id","pmid","pmcid","journal","pub_year","section_type","chunk_id")
    row = {key: chunk.source[key] for key in fields if key in chunk.source}
    row.update(candidate.to_dict())
    row.update({"mention": str(chunk.source.get("chunk") or "")[start:end], "start": start,
        "end": end, "document_start": chunk.document_start+start,
        "document_end": chunk.document_start+end, "offset_scope":"chunk",
        "normalization_status":"normalized", "normalization_source": source,
        "recognition_source": source, "locked": True,
        "mention_id":f"{row.get('base',chunk.chunk_id)}:{source}:{start}:{end}"})
    row.update(extra)
    return row


def enrich_gene_row(row: dict, matcher: HgncExactMatcher | None) -> dict:
    if row.get("entity_type") != "gene":
        return row
    row["reference_tax_id"] = "9606"
    row["taxonomy_status"] = "not_validated_hgnc_mapping_only"
    identifier = str(row.get("concept_id") or "")
    if matcher and identifier in matcher.records_by_hgnc:
        record = matcher.records_by_hgnc[identifier]
        if "receptor" in str(row.get("expanded_long_form") or row.get("matched_term") or record.name).casefold():
            row["entity_role"] = "receptor"
        row.update({"hgnc_id":identifier,"normalized_id":identifier,
                    "ncbi_gene_id":f"NCBIGene:{record.entrez_id}",
                    "canonical_name":record.name,"uniprot_ids":list(record.uniprot_ids),
                    "normalization_system":"HGNC", "identified_source":"document_recovery"})
    return row


def recover_document_entities(document: Any, context: Any, resources: RecoveryResources
                              ) -> list[dict[str,Any]]:
    """Exact cells/hormones plus abbreviation-expanded and coordinated genes."""
    from backend.pipeline.abbreviation_prepass import select_contextual_definition
    from backend.pipeline.entity_text_normalization import compile_surface_pattern, short_form_key_candidates
    rows: list[dict[str,Any]] = []
    for chunk in document.chunks:
        text = str(chunk.source.get("chunk") or "")
        if str(chunk.section).upper() in {"TITLE", "METADATA"}:
            continue
        for start,end,candidate in resources.exact_target_spans(text):
            # Document definitions override a static/resource abbreviation,
            # including a local non-target or differently typed redefinition.
            definitions = next((context.definitions_by_key[k]
                for k in short_form_key_candidates(text[start:end])
                if context.definitions_by_key.get(k)), None)
            if definitions:
                selected = select_contextual_definition(definitions,
                    occurrence_document_start=chunk.document_start+start,
                    chunk_id=chunk.chunk_id, section=chunk.section)
                if selected is None or not selected.resolved or (
                    selected.entity_type, selected.concept_id) != (candidate.entity_type, candidate.concept_id):
                    continue
            source = ("cell_ontology_coordinated_shared_head"
                if candidate.term_kind == "coordinated_shared_head" else
                "static_cell_surface" if candidate.term_kind == "static_abbreviation" else
                "cell_ontology_exact_surface" if candidate.entity_type == "cell" else
                "mesh_hormone_exact_surface")
            rows.append(_source_row(chunk,start,end,candidate,source))
        if not resources.genes:
            continue
        # Scan every eligible chunk, independently of PubTator coverage. This
        # protects complete repeated IL-21/IL-2/IL-15/DAP12 mentions even when
        # the remote branch returns a clipped or differently normalized span.
        # Normalization uses HGNC reference identity without a species veto.
        for hit in resources.genes.find(text):
            rows.append(enrich_gene_row(_source_row(chunk, hit.start, hit.end,
                gene_candidate(hit), "document_hgnc_exact_surface"), resources.genes))
        gene_spans: list[tuple[int,int,str,str]] = []
        # Infix abbreviation notation is one gene expression, not two fragments.
        for match in re.finditer(r"(?<!\w)interleukin\s*\(\s*IL\s*\)\s*["+re.escape(DASHES)+r"]?\s*(\d+[A-Za-zβγ]?)\b", text, re.I):
            gene_spans.append((*match.span(), f"interleukin {match.group(1)}", "infix_abbreviation_gene"))
        for match in re.finditer(r"(?<!\w)IL\s*["+re.escape(DASHES)+r"]?\s*(\d+[A-Za-zβγ]?)\s+receptors?\b", text, re.I):
            gene_spans.append((*match.span(), f"interleukin {match.group(1)} receptor", "expanded_receptor_gene"))
        # Coordination shares the head in lookup only, never in the source span.
        for match in re.finditer(r"(?<!\w)([A-Za-z]+)\s+and\s+([A-Za-z]+)\s+(receptors?)\b", text, re.I):
            left = f"{match.group(1)} receptor"
            right = f"{match.group(2)} receptor"
            if resources.genes.resolve(left) and resources.genes.resolve(right):
                gene_spans.append((*match.span(1),left,"coordinated_receptor_gene"))
                gene_spans.append((match.start(2),match.end(3),right,"coordinated_receptor_gene"))
        for key, definitions in context.definitions_by_key.items():
            surfaces = abbreviation_surfaces(definitions)
            for surface in surfaces:
                for match in compile_surface_pattern(surface).finditer(text):
                    suffix = re.match(r"\s+(?:nuclear\s+)?receptors?\b",text[match.end():],re.I)
                    if not suffix:
                        continue
                    selected = select_contextual_definition(definitions,
                        occurrence_document_start=chunk.document_start+match.start(),
                        chunk_id=chunk.chunk_id, section=chunk.section)
                    if selected and selected.resolved:
                        expanded = re.sub(r"\s+receptors?$","",selected.long_form,flags=re.I)+" receptor"
                        gene_spans.append((match.start(),match.end()+suffix.end(),expanded,"expanded_receptor_gene"))
        for start,end,expanded,source in gene_spans:
            result = resources.genes.resolve(expanded)
            if result:
                rows.append(enrich_gene_row(_source_row(chunk,start,end,gene_candidate(result),source,
                    expanded_long_form=expanded, entity_role="receptor" if "receptor" in source else "gene"),resources.genes))
    # Apply source exclusions to all recovered rows, including expanded names.
    rows = [row for chunk in document.chunks for row in sanitize_source_annotations(
        str(chunk.source.get("chunk") or ""),
        [value for value in rows if str(value.get("chunk_id")) == str(chunk.chunk_id)],
        cell_surface_allowed=resources.cell_surface_allowed)]
    # Exact longest same-type spans win before they are used to protect NER.
    selected: list[dict] = []
    for row in sorted(rows,key=lambda r:(-(r['end']-r['start']),r['start'])):
        if any(str(k.get('chunk_id'))==str(row.get('chunk_id')) and k['entity_type']==row['entity_type']
               and row['start'] < k['end'] and row['end'] > k['start'] for k in selected):
            continue
        selected.append(row)
    return selected


@lru_cache(maxsize=4)
def default_cell_span_resources() -> RecoveryResources:
    from backend.cellexlink_lite.resources import DEFAULT_ONTOLOGY_PATH, DEFAULT_ABBREVIATIONS_PATH
    return RecoveryResources(ontology_path=DEFAULT_ONTOLOGY_PATH, hormone_entries=[],
                             abbreviations_path=DEFAULT_ABBREVIATIONS_PATH,
                             gene_matcher=load_local_hgnc())


def document_abbreviation_constraints(document: Any, context: Any,
                                      matcher: HgncExactMatcher | None = None) -> list[dict]:
    """Per-occurrence constraints that survive independent branch execution.

    No text is rewritten. Unresolved/conflicting local definitions block a
    coincident fallback annotation; resolved definitions constrain its identity.
    Larger expressions, such as P4 receptor, remain eligible for expansion.
    """
    from backend.pipeline.abbreviation_prepass import select_contextual_definition
    from backend.pipeline.entity_text_normalization import compile_surface_pattern
    constraints: dict[tuple[str, int, int], dict] = {}
    for definitions in context.definitions_by_key.values():
        for surface in abbreviation_surfaces(definitions):
            for match in compile_surface_pattern(surface).finditer(document.text):
                chunk = document.chunk_for_span(match.start(), match.end())
                if chunk is None:
                    continue
                selected = select_contextual_definition(definitions,
                    occurrence_document_start=match.start(), chunk_id=chunk.chunk_id,
                    section=chunk.section)
                start, end = match.start()-chunk.document_start, match.end()-chunk.document_start
                allowed_ids: list[str] = []
                allowed_type = ""
                if selected is not None and selected.resolved:
                    allowed_type = selected.entity_type or ""
                    allowed_ids.append(selected.concept_id)
                    if matcher and selected.concept_id in matcher.records_by_hgnc:
                        record = matcher.records_by_hgnc[selected.concept_id]
                        allowed_ids.append(f"NCBIGene:{record.entrez_id}")
                constraints[(chunk.chunk_id,start,end)] = {
                    "chunk_id":chunk.chunk_id,"start":start,"end":end,
                    "allowed_type":allowed_type,"allowed_ids":allowed_ids,
                    "definition_status":selected.resolution_status if selected else "ambiguous",
                }
    return list(constraints.values())
