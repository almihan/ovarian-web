"""Human HGNC nomenclature and cached authoritative reference-file helpers.

All approved symbols/names, aliases and previous names/symbols are indexed for
unique source-preserving matches. Greek spelling and formatting equivalence
are lookup-only. Unique approved symbols outrank colliding historical aliases;
other cross-gene ambiguities are rejected. HGNC provides the required canonical
gene identity; the PubTator branch does not perform species validation.

Downloads are atomic, cached, and may use a previously valid stale file when
refresh fails. Existing HGNC-to-Entrez/UniProt cross-reference APIs are retained.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Container, Iterable, Mapping, Sequence

import requests

logger = logging.getLogger(__name__)

HGNC_COMPLETE_SET_URL = (
    "https://storage.googleapis.com/public-download-files/"
    "hgnc/tsv/tsv/hgnc_complete_set.txt"
)
HGNC_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_MULTI_VALUE_RE = re.compile(r"\s*(?:\||;)\s*")
_MESH_ID_RE = re.compile(r"^[CD]\d+$", re.IGNORECASE)
_HGNC_ID_RE = re.compile(r"^(?:HGNC:)?(\d+)$", re.IGNORECASE)
_GREEK_TO_NAME = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "ς": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
}

# Some HGNC alias/name columns can be long.  Keep DictReader from rejecting a
# valid row on platforms with a small default field limit.
try:
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
except OverflowError:
    pass


@dataclass(frozen=True, slots=True)
class CachedFileStatus:
    path: Path
    downloaded: bool
    stale_used: bool
    bytes: int


@dataclass(frozen=True, slots=True)
class HgncRecord:
    hgnc_id: str
    symbol: str
    name: str
    entrez_id: str
    uniprot_ids: tuple[str, ...]
    status: str = "Approved"
    matched_hgnc_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HgncResolutionResult:
    records: dict[str, HgncRecord]
    unresolved_ids: set[str]
    multiple_match_ids: set[str]


@dataclass(frozen=True, slots=True)
class HgncExactMatch:
    start: int
    end: int
    mention: str
    normalized_term: str
    matched_term: str
    term_kind: str
    resource_version: str
    record: HgncRecord


@dataclass(frozen=True, slots=True)
class _HgncExactTerm:
    normalized_term: str
    raw_terms: tuple[str, ...]
    term_sources: tuple[tuple[str, str], ...]
    record: HgncRecord
    require_case_match: bool


@dataclass(slots=True)
class HgncExactMatcher:
    """Token-indexed unique exact matcher built from approved HGNC rows."""

    by_first_token: dict[str, dict[int, dict[str, _HgncExactTerm]]]
    term_count: int
    ambiguous_term_count: int
    record_count: int
    resource_version: str

    compact_terms: dict[str, _HgncExactTerm] = field(default_factory=dict)
    compact_prefixes: Container[str] = field(default_factory=set)
    records_by_hgnc: dict[str, HgncRecord] = field(default_factory=dict)

    def find(self, text: str) -> list[HgncExactMatch]:
        """Match complete source tokens; normalize spelling, never source offsets.

        Hyphen/space variants and Greek letters are comparison-only changes.
        Matching never ends inside an alphanumeric token (IL21 != IL21R).
        """
        from backend.pipeline.entity_span_rules import iter_compact_matches, source_span_exclusions
        exclusions = source_span_exclusions(text)
        output: list[HgncExactMatch] = []
        for start, end, key in iter_compact_matches(
            text, self.compact_terms, self.compact_prefixes, gene=True
        ):
            term = self.compact_terms[key]
            mention = text[start:end]
            # An unresolved uppercase compound (JAK-STAT, etc.) is not evidence
            # for an isolated historical alias of one constituent gene.
            previous = re.search(r"([A-Z]{2,})[-‐‑–—]$", text[:start])
            following = re.match(r"[-‐‑–—]([A-Z]{2,})(?![a-z])", text[end:])
            if re.fullmatch(r"[A-Z]{2,}", mention) and (previous or following):
                continue
            if any("gene" in mask["blocked_types"] and start < mask["end"] and end > mask["start"] for mask in exclusions):
                continue
            sources = _compatible_hgnc_sources(mention, term)
            if not sources:
                continue
            matched_term, kind = _select_hgnc_source_term(mention, sources)
            output.append(HgncExactMatch(start, end, mention, key, matched_term,
                                         kind, self.resource_version, term.record))
        selected: list[HgncExactMatch] = []
        for hit in sorted(output, key=lambda h: (-(h.end-h.start), h.start)):
            if not any(hit.start < old.end and hit.end > old.start for old in selected):
                selected.append(hit)
        return sorted(selected, key=lambda h: (h.start, h.end))

    def resolve_approved_symbol(self, text: str) -> HgncExactMatch | None:
        """Return a whole approved symbol, never an alias or a gene-group name.

        Use the same Greek/formatting comparison as the main matcher. This
        separate lookup prevents an abbreviation from changing STAT1's own ID.
        """
        from backend.pipeline.entity_span_rules import gene_surface_key
        raw = str(text or "").strip()
        key = gene_surface_key(raw)
        term = self.compact_terms.get(key)
        if term is None:
            return None
        symbols = [(value, kind) for value, kind in _compatible_hgnc_sources(raw, term)
                   if kind == "symbol"]
        if not symbols:
            return None
        matched, kind = _select_hgnc_source_term(raw, symbols)
        return HgncExactMatch(0, len(raw), raw, key, matched, kind,
                             self.resource_version, term.record)

    def resolve(self, text: str) -> HgncExactMatch | None:
        """Unique whole-expression lookup, including controlled receptor plurals."""
        from backend.pipeline.entity_span_rules import gene_surface_key
        raw = str(text or "").strip()
        variants = [raw]
        if raw.casefold().endswith("receptors"):
            variants.append(raw[:-1])
        for variant in variants:
            key = gene_surface_key(variant)
            term = self.compact_terms.get(key)
            if term is not None:
                # Whole-expression resolution is also used for document-defined
                # aliases (P4/p4); unprompted scanning retains stricter short-code
                # casing. Name structure and single-letter qualifiers stay strict.
                sources = _compatible_hgnc_sources(raw, term, enforce_short_case=False)
                if not sources:
                    continue
                matched, kind = _select_hgnc_source_term(raw, sources)
                return HgncExactMatch(0, len(raw), raw, key, matched, kind,
                                     self.resource_version, term.record)
        return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_lookup_text(value: Any) -> str:
    """Normalize text for conservative exact matching across punctuation forms."""

    text = unicodedata.normalize("NFKD", _clean_text(value)).casefold()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    greek_expanded: list[str] = []
    for ch in text:
        replacement = _GREEK_TO_NAME.get(ch)
        if replacement is None:
            greek_expanded.append(ch)
        else:
            greek_expanded.extend((" ", replacement, " "))
    text = "".join(greek_expanded)
    text = text.replace("&", " and ")
    text = re.sub(r"[-‐‑‒–—−_/+]", " ", text)
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_mesh_id(value: Any) -> str:
    text = _clean_text(value).upper().rstrip("/")
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    for prefix in ("MESH:", "MSH:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text if _MESH_ID_RE.fullmatch(text) else ""


def normalize_hgnc_id(value: Any) -> str:
    match = _HGNC_ID_RE.fullmatch(_clean_text(value))
    return f"HGNC:{match.group(1)}" if match else ""


def _header_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _clean_text(value).casefold()).strip("_")


def _normalized_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        _header_key(key): _clean_text(value)
        for key, value in row.items()
        if key is not None
    }


def _row_value(row: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = row.get(_header_key(name), "")
        if value:
            return value
    return ""


def _split_multi_value(value: Any) -> tuple[str, ...]:
    text = _clean_text(value)
    if not text:
        return ()
    seen: set[str] = set()
    output: list[str] = []
    for raw in _MULTI_VALUE_RE.split(text):
        item = _clean_text(raw)
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return tuple(output)


def _split_hgnc_multi_value(value: Any) -> tuple[str, ...]:
    """Split HGNC complete-set list fields without breaking name punctuation.

    HGNC list columns use ``|`` as their separator. Semicolons may be part of
    an approved/previous name, so treating ``;`` as a delimiter can create
    unsafe fragments such as the standalone word ``receptor``.
    """

    text = _clean_text(value)
    if not text:
        return ()
    seen: set[str] = set()
    output: list[str] = []
    for raw in re.split(r"\s*\|\s*", text):
        item = _clean_text(raw)
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return tuple(output)


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, min(30.0, float(retry_after)))
            except ValueError:
                pass
    return min(12.0, 0.75 * (2 ** max(0, attempt - 1)))


def ensure_cached_file(
    *,
    session: requests.Session,
    url: str,
    path: Path,
    max_age_seconds: int,
    request_timeout: int,
    max_attempts: int,
    minimum_bytes: int,
) -> CachedFileStatus:
    """Return a fresh-enough local file, downloading it atomically when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_ok = path.is_file() and path.stat().st_size >= minimum_bytes
    if existing_ok:
        age = max(0.0, time.time() - path.stat().st_mtime)
        if age <= max_age_seconds:
            return CachedFileStatus(
                path=path,
                downloaded=False,
                stale_used=False,
                bytes=path.stat().st_size,
            )

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    last_error: Exception | None = None
    last_response: requests.Response | None = None
    try:
        for attempt in range(1, max(1, max_attempts) + 1):
            temporary.unlink(missing_ok=True)
            try:
                with session.get(
                    url,
                    stream=True,
                    timeout=(20, request_timeout),
                ) as response:
                    last_response = response
                    if response.status_code in _RETRYABLE_STATUS:
                        raise requests.HTTPError(
                            f"HTTP {response.status_code}", response=response
                        )
                    response.raise_for_status()
                    total = 0
                    with temporary.open("wb") as destination:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if not block:
                                continue
                            destination.write(block)
                            total += len(block)
                        destination.flush()
                        os.fsync(destination.fileno())
                    if total < minimum_bytes:
                        raise RuntimeError(
                            f"Downloaded reference file is unexpectedly small "
                            f"({total} bytes): {url}"
                        )
                    os.replace(temporary, path)
                    return CachedFileStatus(
                        path=path,
                        downloaded=True,
                        stale_used=False,
                        bytes=total,
                    )
            except (requests.RequestException, OSError, RuntimeError) as exc:
                last_error = exc
                if attempt < max(1, max_attempts):
                    time.sleep(_retry_delay(last_response, attempt))

        if existing_ok:
            logger.warning(
                "Reference-data refresh failed; using stale cached file %s: %s",
                path,
                last_error,
            )
            return CachedFileStatus(
                path=path,
                downloaded=False,
                stale_used=True,
                bytes=path.stat().st_size,
            )
        raise RuntimeError(f"Could not download reference data from {url}: {last_error}")
    finally:
        temporary.unlink(missing_ok=True)


def ensure_hgnc_reference(
    *,
    cache_dir: Path,
    session: requests.Session,
    request_timeout: int,
    max_attempts: int,
) -> CachedFileStatus:
    return ensure_cached_file(
        session=session,
        url=HGNC_COMPLETE_SET_URL,
        path=cache_dir / "hgnc_complete_set.txt",
        max_age_seconds=HGNC_MAX_AGE_SECONDS,
        request_timeout=request_timeout,
        max_attempts=max_attempts,
        minimum_bytes=1_000_000,
    )


def _iter_tsv(path: Path, *, compressed: bool) -> Iterable[dict[str, str]]:
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise RuntimeError(f"Reference file has no TSV header: {path}")
        for raw in reader:
            if raw:
                yield _normalized_row(raw)


def _hgnc_record_from_row(row: Mapping[str, str]) -> HgncRecord | None:
    hgnc_id = normalize_hgnc_id(_row_value(row, "hgnc_id", "hgnc id"))
    symbol = _row_value(row, "symbol", "approved_symbol")
    name = _row_value(row, "name", "approved_name")
    status = _row_value(row, "status") or "Approved"
    entrez_id = _row_value(row, "entrez_id", "entrez gene id", "ncbi_gene_id")
    match = re.search(r"\d+", entrez_id)
    entrez_id = match.group(0) if match else ""
    if not hgnc_id or not symbol:
        return None
    if status and status.casefold() != "approved":
        return None
    uniprot_ids = tuple(
        item.upper()
        for item in _split_multi_value(
            _row_value(row, "uniprot_ids", "uniprot id(supplied by uniprot)")
        )
        if item
    )
    return HgncRecord(
        hgnc_id=hgnc_id,
        symbol=symbol,
        name=name,
        entrez_id=entrez_id,
        uniprot_ids=uniprot_ids,
        status=status or "Approved",
        matched_hgnc_ids=(hgnc_id,),
    )


def _hgnc_id_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"\d+", value)
    return (int(match.group(0)) if match else 2**63 - 1, value)


def _merge_hgnc_records(records: Iterable[HgncRecord]) -> HgncRecord:
    ordered = sorted(records, key=lambda item: _hgnc_id_sort_key(item.hgnc_id))
    if not ordered:
        raise ValueError("At least one HGNC record is required.")
    primary = ordered[0]
    hgnc_ids = tuple(
        sorted(
            {
                hgnc_id
                for record in ordered
                for hgnc_id in (record.matched_hgnc_ids or (record.hgnc_id,))
                if hgnc_id
            },
            key=_hgnc_id_sort_key,
        )
    )
    uniprot_ids = tuple(
        sorted(
            {accession for record in ordered for accession in record.uniprot_ids}
        )
    )
    return HgncRecord(
        hgnc_id=primary.hgnc_id,
        symbol=primary.symbol,
        name=primary.name,
        entrez_id=primary.entrez_id,
        uniprot_ids=uniprot_ids,
        status=primary.status,
        matched_hgnc_ids=hgnc_ids or (primary.hgnc_id,),
    )


def resolve_hgnc_by_entrez(
    hgnc_path: Path,
    entrez_ids: Sequence[str],
) -> HgncResolutionResult:
    """Match NCBI Gene IDs directly to approved HGNC ``entrez_id`` rows.

    Any approved HGNC match makes the record eligible for retention. When more
    than one approved HGNC row shares the same Entrez ID, the rows are merged
    deterministically and their UniProt accessions are combined rather than
    discarding the match as ambiguous.
    """

    wanted = {
        match.group(0)
        for value in entrez_ids
        if (match := re.search(r"\d+", _clean_text(value)))
    }
    matches: dict[str, dict[str, HgncRecord]] = {value: {} for value in wanted}
    if not wanted:
        return HgncResolutionResult({}, set(), set())

    for row in _iter_tsv(hgnc_path, compressed=False):
        record = _hgnc_record_from_row(row)
        if record is None or record.entrez_id not in wanted:
            continue
        matches[record.entrez_id][record.hgnc_id] = record

    records: dict[str, HgncRecord] = {}
    unresolved: set[str] = set()
    multiple: set[str] = set()
    for entrez_id in wanted:
        candidates = matches.get(entrez_id, {})
        if candidates:
            records[entrez_id] = _merge_hgnc_records(candidates.values())
            if len(candidates) > 1:
                multiple.add(entrez_id)
        else:
            unresolved.add(entrez_id)
    return HgncResolutionResult(records, unresolved, multiple)


@dataclass(frozen=True, slots=True)
class _NormalizedToken:
    value: str
    start: int
    end: int


def _normalized_tokens_with_spans(text: str) -> list[_NormalizedToken]:
    """Apply ``normalize_lookup_text``-compatible tokenization with offsets."""

    tokens: list[_NormalizedToken] = []
    current: list[str] = []
    current_start = -1
    current_end = -1

    def flush() -> None:
        nonlocal current, current_start, current_end
        if current:
            tokens.append(
                _NormalizedToken(
                    value="".join(current),
                    start=current_start,
                    end=current_end,
                )
            )
        current = []
        current_start = -1
        current_end = -1

    for raw_index, raw_character in enumerate(text):
        decomposed = unicodedata.normalize("NFKD", raw_character).casefold()
        emitted: list[str] = []
        for character in decomposed:
            if unicodedata.combining(character):
                continue
            greek = _GREEK_TO_NAME.get(character)
            if greek is not None:
                emitted.extend((" ", greek, " "))
            elif character == "&":
                emitted.extend((" ", "and", " "))
            elif character.isascii() and character.isalnum():
                emitted.append(character)
            else:
                emitted.append(" ")

        for item in emitted:
            for character in item:
                if character.isascii() and character.isalnum():
                    if current_start < 0:
                        current_start = raw_index
                    current.append(character)
                    current_end = raw_index + 1
                else:
                    flush()
    flush()
    return tokens


def _compact_case_text(value: str) -> str:
    output: list[str] = []
    for raw_character in unicodedata.normalize("NFKD", _clean_text(value)):
        if unicodedata.combining(raw_character):
            continue
        greek = _GREEK_TO_NAME.get(raw_character.casefold())
        if greek is not None:
            if raw_character.isupper():
                output.append(greek.upper())
            else:
                output.append(greek)
        elif raw_character.isascii() and raw_character.isalnum():
            output.append(raw_character)
    return "".join(output)


def _short_exact_case_match(mention: str, raw_terms: Sequence[str]) -> bool:
    from backend.pipeline.entity_span_rules import gene_surface_key
    mention_key = gene_surface_key(mention)
    if any(_compact_case_text(mention) == _compact_case_text(term) for term in raw_terms):
        return True
    # Permit IL-12β, IFN-γ, NKp44 etc., but do not turn ordinary lowercase
    # prose into short gene symbols solely because punctuation was removed.
    capitals = sum(character.isascii() and character.isupper() for character in mention)
    return capitals >= 2 and any(mention_key == gene_surface_key(term) for term in raw_terms)



def _compatible_hgnc_sources(mention: str, term: _HgncExactTerm, *,
                             enforce_short_case: bool = True) -> tuple[tuple[str, str], ...]:
    """A compact key proposes candidates; literal source structure validates them."""
    from backend.pipeline.entity_span_rules import gene_term_matches
    sources = tuple((raw, field) for raw, field in term.term_sources
                    if gene_term_matches(mention, raw, field))
    if enforce_short_case and term.require_case_match and not _short_exact_case_match(mention, [raw for raw, _ in sources]):
        return ()
    return sources

def _select_hgnc_source_term(
    mention: str,
    term_sources: Sequence[tuple[str, str]],
) -> tuple[str, str]:
    """Return the resource term/field that best explains one exact span."""

    mention_key = _compact_case_text(mention)
    for raw_term, source_field in term_sources:
        if mention_key and mention_key == _compact_case_text(raw_term):
            return raw_term, source_field
    if term_sources:
        return term_sources[0]
    return mention, "hgnc_exact_term"


def _hgnc_raw_terms(
    row: Mapping[str, str],
    record: HgncRecord,
) -> tuple[tuple[str, str], ...]:
    terms: list[tuple[str, str]] = [
        (record.symbol, "symbol"),
        (record.name, "name"),
    ]
    for field in ("alias_symbol", "alias_name", "prev_symbol", "prev_name"):
        terms.extend(
            (value, field)
            for value in _split_hgnc_multi_value(_row_value(row, field))
        )
    seen: set[str] = set()
    output: list[tuple[str, str]] = []
    for term, source_field in terms:
        cleaned = _clean_text(term)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            output.append((cleaned, source_field))
    return tuple(output)


def build_hgnc_exact_matcher(hgnc_path: Path) -> HgncExactMatcher:
    """Index approved human HGNC symbols, names, aliases and previous names.

    Short aliases are retained. All normalized-key collisions are collected
    BEFORE uniqueness is tested, so spelling variants cannot choose an arbitrary
    gene. HGNC is a human nomenclature resource; species filtering of PubTator
    identifiers is a separate operation. Gene-group columns are deliberately
    not indexed: only whole individual-gene naming-field matches can resolve.
    """
    from backend.pipeline.entity_span_rules import gene_surface_key, key_prefixes, is_excluded_gene_identity
    records_by_entrez: dict[str, dict[str, HgncRecord]] = {}
    term_candidates: dict[str, dict[str, set[tuple[str, str]]]] = {}
    stopwords = {"and", "or", "in", "as", "is", "it", "at", "on", "of", "to", "an",
                 "the", "for", "by", "if", "not", "no", "are", "be", "all", "may", "was",
                 "has", "had", "with", "hormone", "hormones", "cell", "cells", "mature"}
    for row in _iter_tsv(hgnc_path, compressed=False):
        record = _hgnc_record_from_row(row)
        if record is None or not record.entrez_id or is_excluded_gene_identity({"hgnc_id": record.hgnc_id}):
            continue
        records_by_entrez.setdefault(record.entrez_id, {})[record.hgnc_id] = record
        for raw_term, source_field in _hgnc_raw_terms(row, record):
            key = gene_surface_key(raw_term)
            if not key or key.isdigit() or len(key) < 2 or key in stopwords:
                continue
            term_candidates.setdefault(key, {}).setdefault(record.entrez_id, set()).add(
                (raw_term, source_field))
    merged = {gid: _merge_hgnc_records(records.values())
              for gid, records in records_by_entrez.items()}
    compact_terms: dict[str, _HgncExactTerm] = {}
    ambiguous = 0
    for key, candidates in term_candidates.items():
        if len(candidates) != 1:
            # An approved symbol outranks another gene's historical alias.
            # Example: IL-21 is also an old alias of IL17C/IL22, but IL21 is
            # the current approved symbol. Alias-only collisions remain unresolved.
            approved = {gid: sources for gid, sources in candidates.items()
                        if any(kind == "symbol" for _, kind in sources)}
            if len(approved) == 1:
                candidates = approved
            else:
                ambiguous += 1
                continue
        gid, sources = next(iter(candidates.items()))
        sources = tuple(sorted(sources, key=lambda item: (len(item[0]), item)))
        compact_terms[key] = _HgncExactTerm(
            key, tuple(dict.fromkeys(term for term, _ in sources)), sources,
            merged[gid], len(key) <= 6,
        )
    digest = hashlib.sha256(hgnc_path.read_bytes()).hexdigest()[:12]
    return HgncExactMatcher(
        by_first_token={}, term_count=len(compact_terms), ambiguous_term_count=ambiguous,
        record_count=sum(len(records) for records in records_by_entrez.values()),
        resource_version=f"hgnc-human-token-context-v5-{digest}",
        compact_terms=compact_terms, compact_prefixes=key_prefixes(compact_terms),
        records_by_hgnc={record.hgnc_id: record for record in merged.values()},
    )


def reference_download_stats(
    statuses: Sequence[CachedFileStatus],
    *,
    prefix: str,
) -> dict[str, int]:
    return {
        f"{prefix}_reference_files": len(statuses),
        f"{prefix}_reference_files_downloaded": sum(
            1 for status in statuses if status.downloaded
        ),
        f"{prefix}_reference_stale_files_used": sum(
            1 for status in statuses if status.stale_used
        ),
        f"{prefix}_reference_bytes": sum(status.bytes for status in statuses),
    }


__all__ = [
    "HGNC_COMPLETE_SET_URL",
    "CachedFileStatus",
    "HgncExactMatch",
    "HgncExactMatcher",
    "HgncRecord",
    "HgncResolutionResult",
    "build_hgnc_exact_matcher",
    "ensure_hgnc_reference",
    "normalize_hgnc_id",
    "normalize_lookup_text",
    "normalize_mesh_id",
    "reference_download_stats",
    "resolve_hgnc_by_entrez",
]
