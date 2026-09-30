"""Source-preserving biomedical matching and cell span validation.

No rewritten text is sent to the relation model: start/end always slice the
original chunk. Normalized keys are only used for resource lookup.
"""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left
from functools import lru_cache
from collections.abc import Callable, Container, Iterable, Iterator, Mapping
from backend.pipeline.cell_surface_matching import cell_annotation_spelling_allowed

DASHES = "-−‐‑‒–—﹘﹣－"
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
GAP = re.compile(r"[\s_," + re.escape(DASHES) + r"]*")
GREEK = {"α": "alpha", "β": "beta", "ϐ": "beta", "γ": "gamma", "δ": "delta",
         "ε": "epsilon", "κ": "kappa", "λ": "lambda", "μ": "mu", "ω": "omega"}
GREEK_CODE = {"alpha": "a", "beta": "b", "gamma": "g", "delta": "d", "epsilon": "e",
              "kappa": "k", "lambda": "l", "mu": "m", "omega": "o"}
_GREEK_SUFFIX = re.compile(r"(alpha|beta|gamma|delta|epsilon|kappa|lambda|omega)(?=\d*$)")
NONCELL = frozenset({
    "hormone", "hormones", "protein", "proteins", "gene", "genes", "cytokine", "cytokines",
    "receptor", "receptors", "molecule", "molecules", "mature", "immature", "naive", "naïve",
    "activated", "resting", "differentiated", "effector", "memory", "human", "primary",
    "cell", "cells", "cell type", "cell types", "immune", "immunity", "inflammation",
    "progesterone", "estrogen", "estrogens", "estradiol", "p4",
    "a", "an", "the", "and", "or", "in", "on", "at", "as", "of", "to", "by",
    "for", "from", "with", "without", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "not", "no", "so", "we", "our",
    "has", "have", "had", "can", "may", "will", "do", "does", "did", "than", "then",
})


def gene_surface_key(text: object) -> str:
    """Resource-backed β/beta/B, γ/gamma/G and formatting equivalence.

    Latin letters are NOT expanded globally: every resulting key still has to
    select a unique resource record. Thus no free-standing beta->gene inference.
    """
    raw = unicodedata.normalize("NFKC", str(text or "")).casefold()
    for symbol, name in GREEK.items():
        raw = raw.replace(symbol, GREEK_CODE[name])
    words = []
    for match in TOKEN.finditer(raw):
        word = match.group()
        word = GREEK_CODE.get(word, _GREEK_SUFFIX.sub(lambda m: GREEK_CODE[m.group()], word))
        words.append(word)
    return "".join(words)



GENERIC_GENE_SURFACES = frozenset({
    "gene", "genes", "protein", "proteins", "receptor", "receptors",
    "cytokine", "cytokines", "hormone", "hormones", "marker", "markers",
    "t cell receptor", "t cell receptors",
})


def is_generic_gene_surface(text: object) -> bool:
    raw = re.sub("[" + re.escape(DASHES) + "]", " ", str(text or "").casefold())
    return " ".join(raw.split()) in GENERIC_GENE_SURFACES


def _hard_punctuation(value: str) -> tuple[tuple[int, str], ...]:
    """Keep semantic punctuation; only spacing, underscores and dashes are soft.

    Recording key positions lets e.g. a comma in a real approved NAME match
    the same comma, without making a comma-separated list into one symbol.
    """
    return tuple((len(gene_surface_key(value[:i])), c)
                 for i, c in enumerate(value)
                 if not (c.isalnum() or c.isspace() or c in DASHES + "_"))


def gene_term_matches(mention: str, reference: str, term_kind: str = "") -> bool:
    """Validate a compact-key hit against the actual nomenclature spelling.

    Ordinary name-word boundaries and uppercase single-letter qualifiers are
    significant: proteins != protein S, and MS, RA != MSRA. Compact biomedical
    codes still allow IL-21/IL21, IFN-γ/IFNG and NK-p44/NKp44. Plural recovery is
    limited to the complete word 'receptors', not arbitrary symbols ending S.
    """
    del term_kind  # Every field uses the same source-preserving safeguards.
    surface = unicodedata.normalize("NFKC", str(mention or "")).strip()
    target = unicodedata.normalize("NFKC", str(reference or "")).strip()
    if not surface or not target or is_generic_gene_surface(surface):
        return False
    if re.search(r"\breceptors$", surface, re.I) and re.search(r"\breceptor$", target, re.I):
        surface = surface[:-1]
    if gene_surface_key(surface) != gene_surface_key(target):
        return False
    if _hard_punctuation(surface) != _hard_punctuation(target):
        return False
    source_tokens = [m.group() for m in TOKEN.finditer(surface)]
    target_tokens = [m.group() for m in TOKEN.finditer(target)]
    # Natural-language names must retain their word boundaries. In particular,
    # the ordinary plural noun proteins must not collide with the name protein S.
    lexical = any(t.isalpha() and len(t) > 2 and
                  (t.islower() or (t[0].isupper() and t[1:].islower()))
                  and t.casefold() not in GREEK_CODE for t in target_tokens)
    if lexical:
        if [gene_surface_key(t) for t in source_tokens] != [gene_surface_key(t) for t in target_tokens]:
            return False
        for actual, expected in zip(source_tokens, target_tokens):
            if len(expected) == 1 and expected.isascii() and expected.isalpha() and expected.isupper():
                if actual != expected:
                    return False
        return True
    # Do not assemble alphabetic words into a symbol across whitespace. An
    # explicit resource spelling can have spaces; alpha/numeric boundaries and
    # spelled Greek qualifiers may also be spaced (KIR 2DL2, IL 21, IFN gamma).
    if [gene_surface_key(t) for t in source_tokens] != [gene_surface_key(t) for t in target_tokens]:
        tokens = list(TOKEN.finditer(surface))
        for left, right in zip(tokens, tokens[1:]):
            gap = surface[left.end():right.start()]
            if gap.isspace() and left.group().isalpha() and right.group().isalpha():
                if left.group().casefold() not in GREEK_CODE and right.group().casefold() not in GREEK_CODE:
                    return False
    return True


# KIR slash shorthand shares omitted prefixes. It is not a collection of
# independent aliases L3, S1, S2 etc. Leave the WHOLE combination unannotated.
_KIR_CODE = r"(?:KIR[\s-]*)?[23]D[LSP]\d+"
_KIR_ARM = rf"(?:{_KIR_CODE}|[LSP]\d+)"
_KIR_COMBINATION = re.compile(
    rf"(?<!\w){_KIR_CODE}(?:\s*/\s*{_KIR_ARM})+(?!\w)", re.I)
_KIR_LIST_HEAD = re.compile(r"\bKIR\s+(?:combinations?|genotypes?|phenotypes?|types?)\s*\(", re.I)
_T_CELL_RECEPTOR = re.compile(
    r"(?<!\w)T[\s" + re.escape(DASHES) + r"]+cells?[\s" + re.escape(DASHES) + r"]+receptors?\b", re.I)
_CD56_NK = re.compile(
    r"(?<!\w)CD\s*56[\s" + re.escape(DASHES) + r"]*(?P<level>bright|dim)"
    r"(?:[\s" + re.escape(DASHES) + r"]+(?:NK|natural[ -]+killer)[ -]+(?:cells?|subpopulations?)|[ -]+cells?)?(?!\w)", re.I)



# References and analysis labels are context-specific exclusions, not a global
# ban on short aliases (S1/S5/GSE can have other meanings elsewhere).
# Letter-only alternatives are admitted only inside this reference grammar.
_REFERENCE_LABEL = r"(?:S\s*\d+[A-Za-z]?|[A-Z]?\d+[A-Za-z]?|[A-Za-z](?!\w))"
_REFERENCE_LIST = rf"{_REFERENCE_LABEL}(?:(?:\s*(?:,|/|[–—-]|\band\b|\bor\b)\s*){_REFERENCE_LABEL})*"
_TABLE_FIGURE = re.compile(
    rf"\b(?:(?:supplementary|supplemental|supporting)\s+)?"
    rf"(?:tables?|fig(?:s|ures?)?\.?|additional\s+files?)\s*(?:no\.?\s*)?"
    rf"(?P<label>{_REFERENCE_LIST})(?!\w)", re.I)
_REVERSED_REFERENCE = re.compile(rf"(?<!\w){_REFERENCE_LABEL}\s+(?:tables?|figures?)\b", re.I)
_GSE_ANALYSIS = re.compile(
    r"\bGSE(?:[- ]based)?\s+(?:analys(?:is|es)|methods?|approaches?|results?|enrichment)\b"
    r"|\b(?:analys(?:is|es)|enrichment)\s+(?:using|with|by)\s+GSE\b"
    r"|\bgene[- ]set\s+enrichment(?:\s+analysis)?\s*\(\s*GSE\s*\)"
    r"|\bGSE\d+\b", re.I)
_LY49 = re.compile(r"(?<!\w)(?:Ly[ -]?49[A-Za-z0-9]*|KLRA1P|KLRAP1)(?!\w)", re.I)
_NK_NONCELL_HEAD = re.compile(
    r"(?<!\w)(?:human\s+)?(?:NK|natural[ -]+killer)[ -]+"
    r"(?:(?:cells?[ -]+)?(?:(?:inhibitory|activating|surface)[ -]+)?"
    r"(?:receptors?|gene[ -]+complex|genes?)|repertoires?)\b(?:\s*\(NKC\))?", re.I)
_NK_SHARED_CELL_HEAD = re.compile(
    r"(?<!\w)NK(?=\s*(?:and|or|/)\s*(?:[BT](?:\s+and\s+[BT])?)[ -]+cells?\b)", re.I)
_NK_DEFINITION = re.compile(
    r"(?<!\w)natural[ -]+killer\s*\(\s*NK\s*\)[ -]*cells?\b", re.I)
_NK_SUBPOPULATION = re.compile(r"(?<!\w)NK[ -]+subpopulations?\b", re.I)

# Project-specific exclusion, NOT a blanket removal of all pseudogenes.
# NCBI Gene 10748 / HGNC:6372 is KLRA1P (human Ly49 pseudogene).
# https://www.ncbi.nlm.nih.gov/gene/10748
_EXCLUDED_GENE_IDS = frozenset({"HGNC:6372", "NCBIGENE:10748", "GENE:10748", "10748"})


def is_excluded_gene_identity(row: Mapping) -> bool:
    return any(str(row.get(key) or "").upper() in _EXCLUDED_GENE_IDS
               for key in ("concept_id", "normalized_id", "hgnc_id", "ncbi_gene_id", "gene_id"))


def full_context_cell_spans(text: str) -> Iterator[tuple[int, int, str]]:
    """Contiguous source phrases with a resource-resolvable cell head.

    Never combine coordinated NK and T cells into one fictitious cell type.
    """
    for match in _NK_DEFINITION.finditer(text):
        yield match.start(), match.end(), "natural killer cell"
    for match in _NK_SUBPOPULATION.finditer(text):
        yield match.start(), match.end(), "NK cells"
    for start, end, level in cd56_nk_spans(text):
        yield start, end, f"CD56{level} NK cells"


def cd56_nk_spans(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield full phenotype phrases, including an optional NK-cell head."""
    for match in _CD56_NK.finditer(text):
        yield match.start(), match.end(), match.group("level").casefold()


@lru_cache(maxsize=256)
def source_span_exclusions(text: str) -> list[dict]:
    """Context exclusions that can be persisted through a text-free merge."""
    masks = [{"start": m.start(), "end": m.end(), "blocked_types": ["cell", "gene", "protein"],
              "reason": "shared_prefix_kir_combination"}
             for m in _KIR_COMBINATION.finditer(text)]
    for head in _KIR_LIST_HEAD.finditer(text):
        close = text.find(")", head.end(), head.end() + 240)
        if close < 0:
            continue
        contents = text[head.end():close]
        residue = re.sub(_KIR_ARM, "", contents, flags=re.I)
        residue = re.sub(r"\b(?:and|or)\b|[\s,;/]", "", residue, flags=re.I)
        if not residue and re.search(_KIR_CODE, contents, re.I):
            masks.append({"start": head.start(), "end": close + 1,
                          "blocked_types": ["cell", "gene", "protein"],
                          "reason": "shared_prefix_kir_list"})
    masks.extend({"start": m.start(), "end": m.end(), "blocked_types": ["cell"],
                  "reason": "receptor_expression_not_cell"} for m in _T_CELL_RECEPTOR.finditer(text))
    for pattern, reason, kinds in (
        (_TABLE_FIGURE, "table_or_figure_reference", ["gene", "protein", "cell", "hormone"]),
        (_REVERSED_REFERENCE, "table_or_figure_reference", ["gene", "protein", "cell", "hormone"]),
        (_GSE_ANALYSIS, "analysis_method_or_accession", ["gene", "protein", "cell", "hormone"]),
        (_LY49, "excluded_ly49_human_pseudogene_or_nonhuman_family", ["gene", "protein", "cell"]),
        (_NK_NONCELL_HEAD, "nk_noncell_noun_phrase", ["cell"]),
        (_NK_SHARED_CELL_HEAD, "nk_elliptical_shared_cell_head", ["cell"]),
    ):
        masks.extend({"start": m.start(), "end": m.end(), "blocked_types": kinds,
                      "reason": reason} for m in pattern.finditer(text))
    return masks


def apply_span_exclusions(rows: Iterable[Mapping], masks: Iterable[Mapping]) -> list[dict]:
    exclusions = [m for m in masks if isinstance(m, Mapping) and row_span(m)]
    result = []
    for row in rows:
        if not isinstance(row, Mapping) or row_span(row) is None:
            continue
        start, end = row_span(row)
        kind = str(row.get("obj") or row.get("entity_type") or "").casefold()
        if kind in {"gene", "protein"} and is_excluded_gene_identity(row):
            continue
        kind = "cell" if kind in {"cell_type", "cell type"} else kind
        if any(kind in mask.get("blocked_types", []) and start < int(mask["end"])
               and end > int(mask["start"]) and
               (row.get("chunk_id") is None or mask.get("chunk_id") is None or
                str(row["chunk_id"]) == str(mask["chunk_id"])) for mask in exclusions):
            continue
        result.append(dict(row))
    return result


def sanitize_source_annotations(text: str, rows: Iterable[Mapping], *,
                                known_cell: Callable[[str], bool] | None = None,
                                known_gene: Callable[[str], bool] | None = None,
                                cell_surface_allowed: Callable[[str], bool] | None = None) -> list[dict]:
    """Validate source spans before branch priorities can hide a valid mention.

    Applies to recognizer and dictionary output alike. Source text and offsets
    are never rewritten. Without source text, persisted context masks can be
    applied separately via apply_span_exclusions.
    """
    molecular_spans = []
    if known_gene:
        code_pattern = r"(?<!\w)[A-Za-z]{1,12}[\s" + re.escape(DASHES) + r"]*\d+[A-Za-z0-9]*(?!\w)"
        molecular_spans = [(m.start(), m.end()) for m in re.finditer(code_pattern, text)
                           if known_gene(m.group()) and not (known_cell and known_cell(m.group()))]
    full_phrases = [(a, b) for a, b, label in full_context_cell_spans(text)
                    if known_cell and known_cell(label)]
    output = []
    for row in apply_span_exclusions(rows, source_span_exclusions(text)):
        start, end = row_span(row)
        if not (0 <= start < end <= len(text)):
            continue
        surface = text[start:end]
        kind = str(row.get("obj") or row.get("entity_type") or "").casefold()
        if kind in {"gene", "protein", "cell", "cell_type", "cell type"} and not token_aligned(text, start, end):
            continue
        if kind in {"gene", "protein"}:
            if is_generic_gene_surface(surface):
                continue
            # A clipped alphabetic prefix of IL-21/IFN-γ is not a complete gene.
            if re.fullmatch(r"[A-Za-z]{1,12}", surface) and re.match(
                    "[" + re.escape(DASHES) + r"]\s*(?:\d|[αβγδ])", text[end:]):
                continue
            references = [str(row.get(field) or "") for field in ("matched_term", "preferred_label", "canonical_name")]
            if any(reference and gene_surface_key(surface) == gene_surface_key(reference)
                   and not gene_term_matches(surface, reference) for reference in references):
                continue
        if kind in {"cell", "cell_type", "cell type"}:
            if not cell_annotation_spelling_allowed(surface, row):
                continue
            if (known_cell and re.search(r"\s+(?:and/or|and|or)\s+|/", surface, re.I)
                    and not known_cell(surface)):
                continue  # unsupported coordination is not one cell identity
            if (cell_surface_allowed and not row.get("definition_id")
                    and not (row.get("expanded_long_form") and
                             str(row.get("normalization_source") or "").startswith("ab3p"))
                    and not cell_surface_allowed(surface)):
                continue
            if any(a <= start < end <= b and (a, b) != (start, end)
                   for a, b in full_phrases):
                continue
            if is_noncell_surface(surface):
                continue
            if known_gene and known_gene(surface) and not (known_cell and known_cell(surface)):
                continue
            if any(a <= start < end <= b for a, b in molecular_spans):
                continue
        output.append(dict(row))
    return output


def plain_surface_key(text: object) -> str:
    raw = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(m.group() for m in TOKEN.finditer(raw))


class PrefixIndex:
    """Compact prefix membership; avoids millions of allocated prefix strings."""
    def __init__(self, keys: Iterable[str]) -> None:
        self.keys = tuple(sorted(set(keys)))

    def __contains__(self, prefix: str) -> bool:
        index = bisect_left(self.keys, prefix)
        return index < len(self.keys) and self.keys[index].startswith(prefix)


def key_prefixes(keys: Iterable[str]) -> PrefixIndex:
    return PrefixIndex(keys)


def iter_compact_matches(text: str, terms: Mapping, prefixes: Container[str], *, gene: bool = False
                         ) -> Iterator[tuple[int, int, str]]:
    """Trie-style lookup at whole Unicode token boundaries, including plurals."""
    tokens = list(TOKEN.finditer(text))
    key_fn = gene_surface_key if gene else plain_surface_key
    for i, first in enumerate(tokens):
        key = ""
        for j in range(i, min(len(tokens), i + 36)):
            token = tokens[j]
            if j > i and not GAP.fullmatch(text[tokens[j-1].end():token.start()]):
                break
            key += key_fn(token.group())
            if key in terms:
                yield first.start(), token.end(), key
            # Controlled final plural: does not modify source or enable a
            # prefix inside TH17, IL21R, P40, or another alphanumeric token.
            if token.group().casefold().endswith("s") and len(token.group()) > 3:
                singular = key[:-1]
                if singular in terms:
                    yield first.start(), token.end(), singular
            if key not in prefixes:
                break


def token_aligned(text: str, start: int, end: int) -> bool:
    if not (0 <= start < end <= len(text)):
        return False
    left = start == 0 or not (text[start-1].isalnum() and text[start].isalnum())
    right = end == len(text) or not (text[end-1].isalnum() and text[end].isalnum())
    return left and right


def is_noncell_surface(text: object) -> bool:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).casefold().split()) in NONCELL


def repair_cell_spans(text: str, spans: Iterable[tuple[int, int]],
                      known_cell: Callable[[str], bool] | None = None,
                      known_gene: Callable[[str], bool] | None = None,
                      cell_surface_allowed: Callable[[str], bool] | None = None) -> list[tuple[int, int]]:
    """Split coordinated cells, expand clipped tokens and recover a cell head.

    A repair is allowed only when the complete phrase resolves in the supplied
    cell resource; otherwise a clipped token is discarded rather than invented.
    """
    output: set[tuple[int, int]] = set()
    masks = source_span_exclusions(text)
    phenotypes = [(a, b) for a, b, label in full_context_cell_spans(text)
                  if known_cell and known_cell(label)]
    phenotypes.sort(key=lambda span: -(span[1] - span[0]))
    for start, end in spans:
        if not (0 <= start < end <= len(text)):
            continue
        if any("cell" in mask["blocked_types"] and start < mask["end"] and end > mask["start"] for mask in masks):
            continue
        for a, b in phenotypes:
            if a <= start < end <= b:
                start, end = a, b
                break
        raw = text[start:end]
        # Independent conjunction/slash arms, preserving exact source offsets.
        cuts = list(re.finditer(r"\s+(?:and|or)\s+|\s*/\s*", raw, re.I))
        if cuts:
            parts = []
            pos = 0
            for cut in cuts:
                parts.append((start+pos, start+cut.start()))
                pos = cut.end()
            parts.append((start+pos, end))
            if not (known_cell and known_cell(raw)):
                # Only resource-backed arms survive. An unsupported combined
                # phrase must not be sent to vector top-1 as one cell identity.
                parts = [(a, b) for a, b in parts if known_cell and known_cell(text[a:b])]
                output.update(repair_cell_spans(text, parts, known_cell, known_gene,
                                               cell_surface_allowed))
                continue
        a, b = start, end
        while a and text[a-1].isalnum() and text[a].isalnum():
            a -= 1
        while b < len(text) and text[b-1].isalnum() and text[b].isalnum():
            b += 1
        if (a,b) != (start,end):
            if known_cell is None or not known_cell(text[a:b]):
                continue
            start,end = a,b
        # cell -> T cell; CD4+ -> CD4+ T cells. Only adjoining components.
        head = re.search(r"(?<!\w)(?:(?:CD\d+\s*\+\s*)?(?:T|B|NK)[ -]*)$", text[:start])
        if re.fullmatch(r"cells?", text[start:end], re.I) and head:
            start = head.start()
        tail = re.match(r"\s*(?:(?:T|B|NK)[ -]+)?cells?\b", text[end:], re.I)
        if tail and not is_noncell_surface(text[start:end]) and known_cell and known_cell(text[start:end+tail.end()]):
            end += tail.end()
        if known_cell:
            # Longest ontology-backed prefix wins (e.g. mature NK cells).
            # Conjunctions are allowed only when the full phrase resolves by
            # the shared-head rule. Hard punctuation still stops expansion.
            left_tokens = list(TOKEN.finditer(text[:start]))[-8:]
            for token in left_tokens:
                prefix = text[token.start():start]
                if re.search(r"[.;:!?()]", prefix, re.I):
                    continue
                if known_cell(text[token.start():end]):
                    start = token.start()
                    break
        if any("cell" in mask["blocked_types"] and start < mask["end"] and end > mask["start"] for mask in masks):
            continue
        if known_gene and known_gene(text[start:end]) and not (known_cell and known_cell(text[start:end])):
            continue
        if not is_noncell_surface(text[start:end]) and token_aligned(text,start,end):
            output.add((start,end))
    output = {(int(row["start"]), int(row["end"])) for row in sanitize_source_annotations(text,
        [{"obj": "cell", "start": a, "end": b, "mention": text[a:b]} for a, b in output],
        known_cell=known_cell, known_gene=known_gene,
        cell_surface_allowed=cell_surface_allowed)}
    # Longest validated cell phrase owns its nested cell spans, independent of
    # input order. Coordinated arms remain separate because neither contains the other.
    kept: list[tuple[int,int]] = []
    for span in sorted(output, key=lambda x: (-(x[1]-x[0]),x[0])):
        if not any(span[0] < b and span[1] > a for a,b in kept):
            kept.append(span)
    return sorted(kept)


def row_span(row: Mapping) -> tuple[int, int] | None:
    """Reject malformed persisted offsets without crashing the merge/tag stage."""
    try:
        start, end = int(row.get("start")), int(row.get("end"))
    except (TypeError, ValueError, OverflowError):
        return None
    return (start, end) if 0 <= start < end else None


def is_gene_group_annotation(row: Mapping) -> bool:
    """Reject old family/group rows as well as newly supplied group identities."""
    identifiers = (row.get(key) for key in ("concept_id", "normalized_id", "hgnc_id"))
    return (
        any(str(value or "").upper().startswith("HGNC_GROUP:") for value in identifiers)
        or bool(row.get("hgnc_group_id"))
        or str(row.get("entity_granularity") or "").casefold() in {"gene_family", "gene_group"}
    )


def prune_cell_fragments(rows: Iterable[Mapping]) -> list[dict]:
    """Drop unsupported identities without resolving cross-type overlaps early.

    The common longest-span policy runs after receptor reclassification, so a
    short cell cannot prematurely remove a longer molecular expression.
    """
    return [dict(row) for row in rows if isinstance(row, Mapping) and row_span(row)
            and not is_gene_group_annotation(row) and not is_excluded_gene_identity(row)
            and not (str(row.get("obj") or row.get("entity_type") or "").casefold()
                     in {"cell", "cell_type", "cell type"}
                     and is_noncell_surface(row.get("mention")))]


def apply_document_constraints(rows: Iterable[Mapping], constraints: Iterable[Mapping]) -> list[dict]:
    """Apply local abbreviation identity constraints without clipping spans."""
    masks = [mask for mask in constraints if isinstance(mask, Mapping) and row_span(mask)]
    output = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        span = row_span(row)
        if span is None:
            continue
        kind = str(row.get("obj") or row.get("entity_type") or "").casefold()
        if kind == "cell_type":
            kind = "cell"
        identity = str(row.get("concept_id") or row.get("normalized_id") or "")
        blocked = False
        for mask in masks:
            if row.get("chunk_id") is not None and mask.get("chunk_id") is not None and str(row["chunk_id"]) != str(mask["chunk_id"]):
                continue
            if int(mask["start"]) <= span[0] < span[1] <= int(mask["end"]):
                if kind != mask.get("allowed_type") or identity not in mask.get("allowed_ids", []):
                    blocked = True
                    break
        if not blocked:
            output.append(dict(row))
    return output
