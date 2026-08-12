"""Low-memory PubTator3 gene and hormone annotation for Stage 2.

The worker reads one paper at a time and requests PubTator3 BioC JSON in bounded
batches. Every parseable PubTator3 ``Gene``, ``Gene/Protein``, and ``Protein``
annotation is first retained in an ephemeral provisional sidecar. Unique NCBI
Gene IDs are then resolved once through NCBI Gene ESummary, and only records
whose authoritative metadata has ``tax_id=9606`` are written to the Stage 2
artifact. PubTator3 Chemical annotations are normalized to MeSH and retained
only when the MeSH concept belongs to the biological Hormones hierarchy
(D06.472), or when a MeSH supplementary concept maps to that hierarchy. Sparse,
text-free per-paper sidecars and a small SQLite cache keep Railway memory,
storage, and repeated network traffic low.
"""

from __future__ import annotations

import collections
import difflib
import gzip
import json
import logging
import os
import re
import sqlite3
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import requests
from requests.adapters import HTTPAdapter

from backend.pipeline.taxonomy import HUMAN_TAX_ID, HUMAN_TAX_NAME, normalize_tax_id

logger = logging.getLogger(__name__)

PUBTATOR3_ABSTRACT_EXPORT = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/"
    "publications/export/biocjson"
)
PUBTATOR3_PMC_EXPORT = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/"
    "publications/pmc_export/biocjson"
)
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
MESH_RDF_SPARQL = "https://id.nlm.nih.gov/mesh/sparql"
MESH_RDF_RESOURCE = "https://id.nlm.nih.gov/mesh"

# Official MeSH biological hierarchy for Hormones. The second tree attached to
# D006728 is a pharmacologic-action hierarchy and is intentionally not used for
# ordinary descriptor membership. Supplementary concepts are retained when they
# map to a descriptor in this branch or explicitly carry D006728 as a
# pharmacological action.
MESH_HORMONE_DESCRIPTOR_ID = "D006728"
MESH_HORMONE_TREE_PREFIX = "D06.472"
_MESH_HORMONE_CACHE_KIND = "mesh-hormone-d06.472-rdf-v2"
_GENE_METADATA_CACHE_SOURCE = "ncbi-gene-esummary-taxonomy-v1"
_HORMONE_LABEL_CACHE_SOURCE = "mesh-rdf-authoritative-label-v1"

PUBTATOR3_ANNOTATIONS_FILENAME = "pubtator3_annotations.jsonl.gz"
PUBTATOR3_PROVISIONAL_FILENAME = ".pubtator3_annotations.provisional.jsonl"
PUBTATOR3_PIPELINE_VERSION = "pubtator3-ncbi-human-gene-hormone-v7"

SOURCE_FIELDS = (
    "base",
    "doc_key",
    "canonical_id",
    "pmid",
    "pmcid",
    "journal",
    "pub_year",
    "section_type",
    "chunk_id",
)

_ENTITY_TYPES = {
    "gene": "gene",
    "gene/protein": "gene",
    "protein": "gene",
    "chemical": "chemical",
    "chemicals": "chemical",
    "chemical entity": "chemical",
    "drug": "chemical",
    "drug/chemical": "chemical",
    "drug chemical": "chemical",
}
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _option_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_pmid(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    value = match.group(0).lstrip("0") or "0"
    return value if value != "0" and len(value) <= 9 else None


def _normalize_pmcid(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    match = re.search(r"(?i)\bPMC\s*(\d+)\b", text)
    if match:
        return f"PMC{match.group(1)}"
    return None


def _sanitize_character(character: str) -> str:
    if character in {"|", "\x00", "\ufffd"}:
        return " "
    if character.isspace() and character != " ":
        return " "
    if unicodedata.category(character).startswith("C"):
        return " "
    return character


def sanitize_text_with_boundaries(text: str) -> tuple[str, list[int]]:
    """Return Stage 1-compatible text plus a raw-boundary to clean-boundary map."""

    raw = text or ""
    transformed = [_sanitize_character(character) for character in raw]
    non_space = [index for index, character in enumerate(transformed) if not character.isspace()]
    if not non_space:
        return "", [0] * (len(raw) + 1)

    first = non_space[0]
    last = non_space[-1]
    output: list[str] = []
    boundaries = [0] * (len(raw) + 1)
    previous_was_space = False

    for index, character in enumerate(transformed):
        boundaries[index] = len(output)
        if index < first or index > last:
            boundaries[index + 1] = len(output)
            continue
        if character.isspace():
            if not previous_was_space:
                output.append(" ")
            previous_was_space = True
        else:
            output.append(character)
            previous_was_space = False
        boundaries[index + 1] = len(output)

    return "".join(output), boundaries


def sanitize_text(text: str) -> str:
    return sanitize_text_with_boundaries(text)[0]


def _open_jsonl(path: Path):
    if path.suffix.casefold() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_jsonl(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_no}.")
            yield row


def _atomic_write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as raw:
            temp_path = Path(raw.name)
            count = 0
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
            ) as destination:
                for row in rows:
                    destination.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode(
                            "utf-8"
                        )
                    )
                    destination.write(b"\n")
                    count += 1
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temp_path, path)
        return count
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def _batched(values: Sequence[str], size: int) -> Iterator[tuple[str, ...]]:
    size = max(1, int(size))
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


class RequestPacer:
    def __init__(self, requests_per_second: float = 3.0) -> None:
        self.minimum_delay = 1.0 / max(0.1, float(requests_per_second))
        self.last_request_at = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < self.minimum_delay:
            time.sleep(self.minimum_delay - elapsed)
        self.last_request_at = time.monotonic()


@dataclass(frozen=True)
class PubTator3Config:
    batch_size: int = 20
    request_timeout: int = 120
    max_attempts: int = 4
    required: bool = True
    resolve_preferred_labels: bool = True
    ncbi_tool: str = "ovarian_network_web"
    ncbi_email: str = ""

    @classmethod
    def from_options(cls, options: Mapping[str, Any] | None) -> "PubTator3Config":
        raw = options or {}
        return cls(
            batch_size=max(1, min(100, int(raw.get("pubtator_batch_size") or 20))),
            request_timeout=max(
                20, min(300, int(raw.get("pubtator_request_timeout") or 120))
            ),
            max_attempts=max(1, min(8, int(raw.get("pubtator_max_attempts") or 4))),
            required=_option_bool(raw.get("pubtator_required"), default=True),
            resolve_preferred_labels=_option_bool(
                raw.get("pubtator_resolve_preferred_labels"), default=True
            ),
            ncbi_tool=_clean_text(raw.get("ncbi_tool")) or "ovarian_network_web",
            ncbi_email=_clean_text(raw.get("ncbi_email")),
        )


@dataclass
class _RequestMetrics:
    requests: int = 0
    retries: int = 0
    failed_requests: int = 0


@dataclass
class _GeneIdentifierMetrics:
    """Counts PubTator3 gene identifiers parsed before metadata filtering."""

    parsed_identifiers: int = 0
    scoped_identifiers: int = 0
    unscoped_identifiers: int = 0
    invalid_identifiers: int = 0


@dataclass(frozen=True, slots=True)
class ParsedGeneIdentifier:
    gene_id: str
    pubtator_tax_id: str = ""


@dataclass(frozen=True, slots=True)
class GeneMetadata:
    gene_id: str
    tax_id: str
    tax_name: str
    symbol: str = ""
    status: str = ""


@dataclass(frozen=True, slots=True)
class GeneMetadataResult:
    records: dict[str, GeneMetadata]
    human_records: dict[str, GeneMetadata]
    non_human_ids: set[str]
    unresolved_ids: set[str]
    cache_hits: int = 0


def _build_session(config: PubTator3Config) -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    contact = f" ({config.ncbi_email})" if config.ncbi_email else ""
    session.headers.update(
        {
            "User-Agent": f"{config.ncbi_tool}/1.0{contact}",
            "Accept": (
                "application/sparql-results+json, application/json, "
                "application/rdf+json, application/x-ndjson, "
                "application/xml, text/xml;q=0.9"
            ),
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, min(30.0, float(retry_after)))
            except ValueError:
                pass
    return min(12.0, 0.75 * (2 ** max(0, attempt - 1)))


def _request_bytes(
    session: requests.Session,
    pacer: RequestPacer,
    metrics: _RequestMetrics,
    *,
    method: str,
    url: str,
    context: str,
    config: PubTator3Config,
    params: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    accepted_statuses: set[int] | None = None,
) -> requests.Response:
    accepted = accepted_statuses or set()
    last_error: Exception | None = None
    last_response: requests.Response | None = None

    for attempt in range(1, config.max_attempts + 1):
        pacer.wait()
        metrics.requests += 1
        try:
            response = session.request(
                method,
                url,
                params=dict(params or {}),
                data=dict(data or {}),
                timeout=(20, config.request_timeout),
            )
            last_response = response
            if response.status_code in accepted:
                return response
            if response.status_code not in _RETRYABLE_STATUS:
                response.raise_for_status()
                return response
            last_error = requests.HTTPError(
                f"{context} returned HTTP {response.status_code}", response=response
            )
        except requests.RequestException as exc:
            last_error = exc

        if attempt < config.max_attempts:
            metrics.retries += 1
            time.sleep(_retry_delay(last_response, attempt))

    metrics.failed_requests += 1
    if last_error is not None:
        raise RuntimeError(f"{context} failed after {config.max_attempts} attempts: {last_error}")
    raise RuntimeError(f"{context} failed without a response.")


def _decode_json_values(content: bytes) -> list[Any]:
    text = content.decode("utf-8-sig")
    stripped = text.strip()
    if not stripped:
        return []
    try:
        return [json.loads(stripped)]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    values: list[Any] = []
    position = 0
    while position < len(text):
        while position < len(text) and (text[position].isspace() or text[position] == ","):
            position += 1
        if position >= len(text):
            break
        value, end = decoder.raw_decode(text, position)
        values.append(value)
        position = end
    return values


def _collect_bioc_documents(value: Any) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if not isinstance(item, dict):
            return
        passages = item.get("passages") or item.get("passage")
        if isinstance(passages, list):
            documents.append(item)
            return
        for child in item.values():
            if isinstance(child, (list, dict)):
                visit(child)

    visit(value)
    return documents


def _parse_bioc_documents(content: bytes) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for payload in _decode_json_values(content):
        documents.extend(_collect_bioc_documents(payload))
    return documents


def _identifier_values(document: Mapping[str, Any]) -> Iterator[Any]:
    yield document.get("id")
    yield document.get("pmid")
    yield document.get("pmcid")
    containers: list[Mapping[str, Any]] = []
    infons = document.get("infons")
    if isinstance(infons, Mapping):
        containers.append(infons)
    passages = document.get("passages") or document.get("passage") or []
    for passage in passages:
        if not isinstance(passage, Mapping):
            continue
        containers.append(passage)
        passage_infons = passage.get("infons")
        if isinstance(passage_infons, Mapping):
            containers.append(passage_infons)
    for container in containers:
        for key, value in container.items():
            key_text = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
            if "pmid" in key_text or "pmc" in key_text or key_text in {
                "articleid",
                "identifier",
            }:
                yield value


def _map_documents(
    documents: Sequence[dict[str, Any]],
    requested: Sequence[str],
    *,
    kind: str,
    pmid_to_pmcid: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    requested_set = set(requested)
    mapped: dict[str, dict[str, Any]] = {}
    unmatched: list[dict[str, Any]] = []

    for document in documents:
        candidates: list[str] = []
        for raw in _identifier_values(document):
            if kind == "pmcid":
                pmcid = _normalize_pmcid(raw)
                if pmcid:
                    candidates.append(pmcid)
                pmid = _normalize_pmid(raw)
                if pmid and pmid_to_pmcid and pmid in pmid_to_pmcid:
                    candidates.append(str(pmid_to_pmcid[pmid]))
            else:
                pmid = _normalize_pmid(raw)
                if pmid:
                    candidates.append(pmid)
        resolved = next((value for value in candidates if value in requested_set), None)
        if resolved is None and kind == "pmcid":
            raw_id = _clean_text(document.get("id"))
            if raw_id.isdigit():
                suffix = f"PMC{raw_id}"
                if suffix in requested_set:
                    resolved = suffix
        if resolved is None:
            unmatched.append(document)
        else:
            mapped.setdefault(resolved, document)

    if len(requested_set) == 1 and not mapped and len(unmatched) == 1:
        mapped[next(iter(requested_set))] = unmatched[0]
    return mapped


def _passage_infons(passage: Mapping[str, Any]) -> Mapping[str, Any]:
    infons = passage.get("infons")
    return infons if isinstance(infons, Mapping) else {}


def _section_type(passage: Mapping[str, Any]) -> str:
    infons = _passage_infons(passage)
    for key in (
        "section_type",
        "sectionType",
        "section",
        "section_name",
        "sectionName",
        "type",
    ):
        value = _clean_text(infons.get(key))
        if value:
            return value.upper()
    return "UNKNOWN"


def _candidate_similarity(left: str, right: str) -> float:
    left_folded = left.casefold()
    right_folded = right.casefold()
    if left_folded == right_folded:
        return 1.0
    longest = max(len(left_folded), len(right_folded), 1)
    shortest = min(len(left_folded), len(right_folded))
    if shortest / longest < 0.82:
        return 0.0
    if left_folded in right_folded or right_folded in left_folded:
        return shortest / longest
    return difflib.SequenceMatcher(
        None, left_folded, right_folded, autojunk=False
    ).ratio()


def _match_passages_to_chunks(
    document: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    *,
    allowed_chunk_indexes: set[int] | None = None,
) -> list[tuple[Mapping[str, Any], int]]:
    allowed = (
        set(range(len(chunks))) if allowed_chunk_indexes is None else set(allowed_chunk_indexes)
    )
    unused = set(allowed)
    exact_by_section: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    exact_any: dict[str, list[int]] = collections.defaultdict(list)
    by_section: dict[str, list[int]] = collections.defaultdict(list)

    for index in sorted(allowed):
        chunk = chunks[index]
        text = str(chunk.get("chunk") or "")
        section = _clean_text(chunk.get("section_type")).upper() or "UNKNOWN"
        exact_by_section[(section, text)].append(index)
        exact_any[text].append(index)
        by_section[section].append(index)

    passages = document.get("passages") or document.get("passage") or []
    matches: list[tuple[Mapping[str, Any], int]] = []
    for passage in passages:
        if not isinstance(passage, Mapping):
            continue
        raw_text = passage.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
        clean_text = sanitize_text(raw_text)
        if not clean_text:
            continue
        section = _section_type(passage)

        chunk_index: int | None = None
        for candidate in exact_by_section.get((section, clean_text), []):
            if candidate in unused:
                chunk_index = candidate
                break
        if chunk_index is None:
            for candidate in exact_any.get(clean_text, []):
                if candidate in unused:
                    chunk_index = candidate
                    break

        if chunk_index is None:
            candidates = [index for index in by_section.get(section, []) if index in unused]
            if not candidates:
                candidates = sorted(unused)
            best_score = 0.0
            best_index: int | None = None
            for candidate in candidates:
                chunk_text = str(chunks[candidate].get("chunk") or "")
                score = _candidate_similarity(clean_text, chunk_text)
                if score > best_score:
                    best_score = score
                    best_index = candidate
            threshold = 0.94 if max(len(clean_text), 1) >= 80 else 0.88
            if best_index is not None and best_score >= threshold:
                chunk_index = best_index

        if chunk_index is None:
            continue
        unused.remove(chunk_index)
        matches.append((passage, chunk_index))
    return matches


def _entity_type(infons: Mapping[str, Any]) -> str | None:
    raw = _clean_text(infons.get("type") or infons.get("entity_type"))
    normalized = re.sub(r"[_-]+", " ", raw.casefold())
    return _ENTITY_TYPES.get(normalized)


def _annotation_identifier(infons: Mapping[str, Any]) -> Any:
    for key, value in infons.items():
        normalized = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
        if normalized in {
            "identifier",
            "identifiers",
            "databaseid",
            "databaseidentifier",
            "conceptid",
            "id",
        }:
            return value
    return None


def _split_identifiers(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, (list, tuple, set)):
        values: list[str] = []
        for value in raw:
            values.extend(_split_identifiers(value))
        return tuple(dict.fromkeys(values))
    text = _clean_text(raw)
    if not text or text in {"-", "None", "null", "N/A"}:
        return ()
    return tuple(
        dict.fromkeys(
            part.strip()
            for part in re.split(r"[,;|]", text)
            if part.strip() and part.strip() not in {"-", "None", "null", "N/A"}
        )
    )


def _parse_gene_identifiers(
    raw: Any,
    *,
    metrics: _GeneIdentifierMetrics | None = None,
) -> tuple[ParsedGeneIdentifier, ...]:
    """Parse all usable PubTator3 NCBI Gene IDs without species filtering.

    PubTator3 may emit a plain Gene ID, such as ``3558`` or
    ``NCBIGene:3558``, or a taxon-scoped form such as ``9606:3558``. The taxon
    prefix is retained only for diagnostics. NCBI Gene metadata is the sole
    source used later to decide whether a gene is human.
    """

    parsed_by_gene_id: dict[str, ParsedGeneIdentifier] = {}
    for value in _split_identifiers(raw):
        cleaned = re.sub(
            r"(?i)^(?:NCBI\s*Gene|NCBIGene|GeneID|Gene)\s*[:#]?\s*", "", value
        ).strip()
        tax_id = ""
        gene_id = ""

        if re.fullmatch(r"\d+", cleaned):
            gene_id = normalize_tax_id(cleaned)
            if metrics is not None:
                metrics.unscoped_identifiers += 1
        else:
            pair = re.fullmatch(r"(\d+)\s*:\s*(\d+)", cleaned)
            if pair is not None:
                tax_id = normalize_tax_id(pair.group(1))
                gene_id = normalize_tax_id(pair.group(2))
                if metrics is not None:
                    metrics.scoped_identifiers += 1

        if not gene_id:
            if metrics is not None:
                metrics.invalid_identifiers += 1
            continue

        if metrics is not None:
            metrics.parsed_identifiers += 1
        current = parsed_by_gene_id.get(gene_id)
        if current is None or (not current.pubtator_tax_id and tax_id):
            parsed_by_gene_id[gene_id] = ParsedGeneIdentifier(
                gene_id=gene_id,
                pubtator_tax_id=tax_id,
            )

    return tuple(parsed_by_gene_id.values())


def _normalize_gene_ids(raw: Any) -> tuple[str, ...]:
    """Return all parseable NCBI Gene IDs, irrespective of organism."""

    return tuple(item.gene_id for item in _parse_gene_identifiers(raw))


def _normalize_chemical_ids(raw: Any) -> tuple[str, ...]:
    output: list[str] = []
    for value in _split_identifiers(raw):
        cleaned = re.sub(r"(?i)^(?:MESH|MeSH)\s*:\s*", "", value).strip().upper()
        match = re.fullmatch(r"([CD]\d+)", cleaned)
        if match:
            output.append(match.group(1))
    return tuple(dict.fromkeys(output))


def _find_occurrences(text: str, query: str) -> list[int]:
    if not query:
        return []
    positions: list[int] = []
    start = 0
    while True:
        position = text.find(query, start)
        if position < 0:
            break
        positions.append(position)
        start = position + 1
    if positions:
        return positions
    folded_text = text.casefold()
    folded_query = query.casefold()
    start = 0
    while True:
        position = folded_text.find(folded_query, start)
        if position < 0:
            break
        positions.append(position)
        start = position + 1
    return positions


def _location_span(
    passage: Mapping[str, Any],
    annotation: Mapping[str, Any],
    location: Mapping[str, Any],
    chunk_text: str,
) -> tuple[int, int] | None:
    raw_text = str(passage.get("text") or "")
    clean_passage, boundaries = sanitize_text_with_boundaries(raw_text)
    try:
        offset = int(location.get("offset"))
        length = int(location.get("length"))
    except (TypeError, ValueError):
        return None
    if length <= 0:
        return None
    try:
        passage_offset = int(passage.get("offset") or 0)
    except (TypeError, ValueError):
        passage_offset = 0

    raw_candidates = (offset - passage_offset, offset)
    raw_start: int | None = next(
        (
            candidate
            for candidate in raw_candidates
            if 0 <= candidate <= len(raw_text) and candidate + length <= len(raw_text)
        ),
        None,
    )
    if raw_start is None:
        raw_start = max(0, min(len(raw_text), offset - passage_offset))
    raw_end = max(raw_start, min(len(raw_text), raw_start + length))
    expected_start = boundaries[raw_start]
    expected_end = boundaries[raw_end]

    annotation_text = sanitize_text(str(annotation.get("text") or ""))
    if not annotation_text:
        annotation_text = clean_passage[expected_start:expected_end]
    positions = _find_occurrences(chunk_text, annotation_text)
    if positions:
        start = min(positions, key=lambda value: abs(value - expected_start))
        return start, start + len(annotation_text)

    if clean_passage == chunk_text and expected_end > expected_start:
        return expected_start, expected_end
    return None


def _source_projection(chunk: Mapping[str, Any]) -> dict[str, Any]:
    return {key: chunk.get(key) for key in SOURCE_FIELDS if chunk.get(key) is not None}


def _annotation_rows_for_passage(
    passage: Mapping[str, Any],
    chunk: Mapping[str, Any],
    *,
    concept_ids: MutableMapping[str, set[str]],
    gene_identifier_metrics: _GeneIdentifierMetrics | None = None,
) -> list[dict[str, Any]]:
    chunk_text = str(chunk.get("chunk") or "")
    annotations = passage.get("annotations") or passage.get("annotation") or []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        infons = annotation.get("infons")
        infons = infons if isinstance(infons, Mapping) else {}
        entity_type = _entity_type(infons)
        if entity_type is None:
            continue
        raw_identifier = _annotation_identifier(infons)
        if entity_type == "gene":
            parsed_gene_ids = _parse_gene_identifiers(
                raw_identifier,
                metrics=gene_identifier_metrics,
            )
            identifiers_with_tax = tuple(
                (item.gene_id, item.pubtator_tax_id) for item in parsed_gene_ids
            )
        else:
            identifiers_with_tax = tuple(
                (identifier, "")
                for identifier in _normalize_chemical_ids(raw_identifier)
            )
        if not identifiers_with_tax:
            continue
        locations = annotation.get("locations") or annotation.get("location") or []
        if isinstance(locations, Mapping):
            locations = [locations]
        if not isinstance(locations, list):
            continue

        for location in locations:
            if not isinstance(location, Mapping):
                continue
            span = _location_span(passage, annotation, location, chunk_text)
            if span is None:
                continue
            start, end = span
            if not (0 <= start < end <= len(chunk_text)):
                continue
            mention = chunk_text[start:end]
            for identifier, pubtator_tax_id in identifiers_with_tax:
                key = (entity_type, identifier)
                concept_ids[entity_type].add(identifier)
                concept_id = (
                    f"NCBIGene:{identifier}"
                    if entity_type == "gene"
                    else f"MESH:{identifier}"
                )
                signature = (entity_type, start, end, mention, concept_id)
                if signature in seen:
                    continue
                seen.add(signature)
                row = _source_projection(chunk)
                row.update(
                    {
                        "entity_type": entity_type,
                        "start": start,
                        "end": end,
                        "mention": mention,
                        "concept_id": concept_id,
                        "normalization_source": "PubTator3",
                    }
                )
                if entity_type == "gene":
                    row["gene_id"] = identifier
                    if pubtator_tax_id:
                        row["pubtator_tax_id"] = pubtator_tax_id
                else:
                    row["chemical_id"] = identifier
                rows.append(row)
    return rows


@dataclass
class _EntryState:
    entry: dict[str, Any]
    provisional_path: Path
    pmid: str | None = None
    pmcid: str | None = None
    chunk_count: int = 0
    title_abstract_chunks: set[int] = field(default_factory=set)
    covered_chunks: set[int] = field(default_factory=set)
    document_seen: bool = False
    annotations_written: int = 0


def _process_document_for_state(
    document: Mapping[str, Any],
    state: _EntryState,
    *,
    concept_ids: MutableMapping[str, set[str]],
    gene_identifier_metrics: _GeneIdentifierMetrics,
    only_uncovered: bool,
) -> tuple[int, int, int]:
    chunks = list(_iter_jsonl(Path(str(state.entry["chunk_path"]))))
    allowed = set(range(len(chunks))) - state.covered_chunks if only_uncovered else None
    matches = _match_passages_to_chunks(document, chunks, allowed_chunk_indexes=allowed)
    rows: list[dict[str, Any]] = []
    for passage, chunk_index in matches:
        state.covered_chunks.add(chunk_index)
        rows.extend(
            _annotation_rows_for_passage(
                passage,
                chunks[chunk_index],
                concept_ids=concept_ids,
                gene_identifier_metrics=gene_identifier_metrics,
            )
        )
    written = _append_jsonl(state.provisional_path, rows) if rows else 0
    state.document_seen = True
    state.annotations_written += written
    return len(matches), written, len(chunks)


class LabelCache:
    """Small persistent cache for gene metadata, labels, and hormones."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS preferred_labels (
                entity_type TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                preferred_label TEXT NOT NULL,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(entity_type, concept_id)
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS hormone_membership (
                identifier_type TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                is_hormone INTEGER NOT NULL,
                evidence TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(identifier_type, concept_id)
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS gene_metadata (
                source TEXT NOT NULL,
                gene_id TEXT NOT NULL,
                tax_id TEXT NOT NULL,
                tax_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(source, gene_id)
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def get_many(
        self,
        entity_type: str,
        concept_ids: Iterable[str],
        *,
        source: str,
    ) -> dict[str, str]:
        values = tuple(dict.fromkeys(concept_ids))
        found: dict[str, str] = {}
        for batch in _batched(values, 500):
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"""
                SELECT concept_id, preferred_label
                FROM preferred_labels
                WHERE entity_type = ?
                  AND source = ?
                  AND concept_id IN ({placeholders})
                """,
                (entity_type, source, *batch),
            ).fetchall()
            found.update({str(row[0]): str(row[1]) for row in rows})
        return found

    def put_many(self, entity_type: str, labels: Mapping[str, str], source: str) -> None:
        rows = [
            (entity_type, concept_id, label, source)
            for concept_id, label in labels.items()
            if concept_id and label
        ]
        if not rows:
            return
        self.connection.executemany(
            """
            INSERT INTO preferred_labels (
                entity_type, concept_id, preferred_label, source
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(entity_type, concept_id) DO UPDATE SET
                preferred_label = excluded.preferred_label,
                source = excluded.source,
                updated_at = CURRENT_TIMESTAMP
            """,
            rows,
        )
        self.connection.commit()

    def get_gene_metadata(
        self,
        gene_ids: Iterable[str],
        *,
        source: str,
    ) -> dict[str, GeneMetadata]:
        values = tuple(dict.fromkeys(gene_ids))
        found: dict[str, GeneMetadata] = {}
        for batch in _batched(values, 500):
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"""
                SELECT gene_id, tax_id, tax_name, symbol, status
                FROM gene_metadata
                WHERE source = ? AND gene_id IN ({placeholders})
                """,
                (source, *batch),
            ).fetchall()
            for gene_id, tax_id, tax_name, symbol, status in rows:
                normalized_gene_id = normalize_tax_id(gene_id)
                normalized_tax_id = normalize_tax_id(tax_id)
                if not normalized_gene_id or not normalized_tax_id:
                    continue
                found[normalized_gene_id] = GeneMetadata(
                    gene_id=normalized_gene_id,
                    tax_id=normalized_tax_id,
                    tax_name=_clean_text(tax_name),
                    symbol=_clean_text(symbol),
                    status=_clean_text(status),
                )
        return found

    def put_gene_metadata(
        self,
        records: Mapping[str, GeneMetadata],
        *,
        source: str,
    ) -> None:
        rows = [
            (
                source,
                record.gene_id,
                record.tax_id,
                record.tax_name,
                record.symbol,
                record.status,
            )
            for record in records.values()
            if record.gene_id and record.tax_id
        ]
        if not rows:
            return
        self.connection.executemany(
            """
            INSERT INTO gene_metadata (
                source, gene_id, tax_id, tax_name, symbol, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, gene_id) DO UPDATE SET
                tax_id = excluded.tax_id,
                tax_name = excluded.tax_name,
                symbol = excluded.symbol,
                status = excluded.status,
                updated_at = CURRENT_TIMESTAMP
            """,
            rows,
        )
        self.connection.commit()

    def get_hormone_membership(
        self,
        identifier_type: str,
        concept_ids: Iterable[str],
    ) -> dict[str, tuple[bool, str]]:
        values = tuple(dict.fromkeys(concept_ids))
        found: dict[str, tuple[bool, str]] = {}
        for batch in _batched(values, 500):
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"""
                SELECT concept_id, is_hormone, evidence
                FROM hormone_membership
                WHERE identifier_type = ? AND concept_id IN ({placeholders})
                """,
                (identifier_type, *batch),
            ).fetchall()
            found.update(
                {
                    str(row[0]): (bool(int(row[1])), str(row[2]))
                    for row in rows
                }
            )
        return found

    def put_hormone_membership(
        self,
        identifier_type: str,
        classifications: Mapping[str, tuple[bool, str]],
    ) -> None:
        rows = [
            (identifier_type, concept_id, int(is_hormone), evidence)
            for concept_id, (is_hormone, evidence) in classifications.items()
            if concept_id and evidence
        ]
        if not rows:
            return
        self.connection.executemany(
            """
            INSERT INTO hormone_membership (
                identifier_type, concept_id, is_hormone, evidence
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(identifier_type, concept_id) DO UPDATE SET
                is_hormone = excluded.is_hormone,
                evidence = excluded.evidence,
                updated_at = CURRENT_TIMESTAMP
            """,
            rows,
        )
        self.connection.commit()


def _ncbi_common_params(config: PubTator3Config) -> dict[str, str]:
    params = {"tool": config.ncbi_tool}
    if config.ncbi_email:
        params["email"] = config.ncbi_email
    return params


def _gene_metadata_from_esummary_record(
    requested_gene_id: str,
    record: Mapping[str, Any],
) -> GeneMetadata | None:
    gene_id = normalize_tax_id(record.get("uid") or requested_gene_id)
    if not gene_id or gene_id != requested_gene_id:
        return None

    organism = record.get("organism")
    organism = organism if isinstance(organism, Mapping) else {}
    tax_id = normalize_tax_id(
        organism.get("taxid")
        or organism.get("tax_id")
        or record.get("taxid")
        or record.get("tax_id")
    )
    if not tax_id:
        return None

    tax_name = _clean_text(
        organism.get("scientificname")
        or organism.get("scientific_name")
        or record.get("taxname")
        or record.get("tax_name")
    )
    if tax_id == HUMAN_TAX_ID and not tax_name:
        tax_name = HUMAN_TAX_NAME

    return GeneMetadata(
        gene_id=gene_id,
        tax_id=tax_id,
        tax_name=tax_name,
        symbol=_clean_text(record.get("name")),
        status=_clean_text(record.get("status")),
    )


def _resolve_human_gene_metadata(
    gene_ids: Sequence[str],
    *,
    cache: LabelCache,
    session: requests.Session,
    pacer: RequestPacer,
    metrics: _RequestMetrics,
    config: PubTator3Config,
) -> GeneMetadataResult:
    """Resolve every unique Gene ID once and select human records by metadata."""

    requested = tuple(
        dict.fromkeys(
            normalized
            for value in gene_ids
            if (normalized := normalize_tax_id(value))
        )
    )
    records = cache.get_gene_metadata(
        requested,
        source=_GENE_METADATA_CACHE_SOURCE,
    )
    cache_hits = len(records)
    missing = tuple(gene_id for gene_id in requested if gene_id not in records)

    for batch in _batched(missing, 200):
        params = {
            "db": "gene",
            "id": ",".join(batch),
            "retmode": "json",
            **_ncbi_common_params(config),
        }
        response = _request_bytes(
            session,
            pacer,
            metrics,
            method="GET",
            url=NCBI_ESUMMARY,
            context="NCBI Gene taxonomy metadata lookup",
            config=config,
            params=params,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "NCBI Gene taxonomy metadata lookup returned invalid JSON."
            ) from exc
        result = payload.get("result") if isinstance(payload, Mapping) else None
        if not isinstance(result, Mapping):
            raise RuntimeError(
                "NCBI Gene taxonomy metadata lookup returned an invalid payload."
            )
        uids = result.get("uids") or []
        if not isinstance(uids, (list, tuple)):
            raise RuntimeError(
                "NCBI Gene taxonomy metadata lookup returned an invalid UID list."
            )

        resolved_batch: dict[str, GeneMetadata] = {}
        batch_set = set(batch)
        for uid in uids:
            requested_gene_id = normalize_tax_id(uid)
            if requested_gene_id not in batch_set:
                continue
            record = result.get(str(uid))
            if not isinstance(record, Mapping) or record.get("error"):
                continue
            metadata = _gene_metadata_from_esummary_record(
                requested_gene_id,
                record,
            )
            if metadata is not None:
                resolved_batch[metadata.gene_id] = metadata

        records.update(resolved_batch)
        cache.put_gene_metadata(
            resolved_batch,
            source=_GENE_METADATA_CACHE_SOURCE,
        )

    unresolved_ids = set(requested) - set(records)
    human_records = {
        gene_id: metadata
        for gene_id, metadata in records.items()
        if metadata.tax_id == HUMAN_TAX_ID
    }
    non_human_ids = {
        gene_id
        for gene_id, metadata in records.items()
        if metadata.tax_id != HUMAN_TAX_ID
    }
    return GeneMetadataResult(
        records=records,
        human_records=human_records,
        non_human_ids=non_human_ids,
        unresolved_ids=unresolved_ids,
        cache_hits=cache_hits,
    )


def _sparql_binding_value(binding: Mapping[str, Any], key: str) -> str:
    value = binding.get(key)
    if not isinstance(value, Mapping):
        return ""
    return _clean_text(value.get("value"))


def _mesh_uri_tail(value: str) -> str:
    cleaned = _clean_text(value).rstrip("/")
    if not cleaned:
        return ""
    return cleaned.rsplit("/", 1)[-1]


def _mesh_uri_values(value: str) -> tuple[str, ...]:
    output: list[str] = []
    for item in str(value or "").split("|"):
        identifier = _mesh_uri_tail(item)
        if identifier:
            output.append(identifier)
    return tuple(dict.fromkeys(output))


@dataclass(frozen=True)
class MeshRecord:
    mesh_id: str
    label: str
    record_type: str
    tree_numbers: tuple[str, ...] = ()
    mapped_descriptor_ids: tuple[str, ...] = ()
    pharmacological_action_ids: tuple[str, ...] = ()


def _rdf_json_values(
    properties: Mapping[str, Any],
    predicate_name: str,
) -> tuple[str, ...]:
    values: list[str] = []
    for predicate, raw_items in properties.items():
        local_name = str(predicate).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        if local_name != predicate_name:
            continue
        items = raw_items if isinstance(raw_items, list) else [raw_items]
        for item in items:
            if not isinstance(item, Mapping):
                continue
            value = _clean_text(item.get("value"))
            if value:
                values.append(value)
    return tuple(dict.fromkeys(values))


def _parse_mesh_rdf_json_record(
    payload: Any,
    mesh_id: str,
) -> MeshRecord | None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("MeSH RDF resource returned an invalid JSON object.")

    properties: Mapping[str, Any] | None = None
    for subject, raw_properties in payload.items():
        if _mesh_uri_tail(str(subject)).upper() != mesh_id:
            continue
        if isinstance(raw_properties, Mapping):
            properties = raw_properties
            break
    if properties is None:
        return None

    labels = _rdf_json_values(properties, "label")
    tree_numbers = tuple(
        value
        for value in (
            _mesh_uri_tail(uri)
            for uri in _rdf_json_values(properties, "treeNumber")
        )
        if re.fullmatch(r"[A-Z]\d+(?:\.\d+)*", value)
    )
    mapped_values = (
        *_rdf_json_values(properties, "mappedTo"),
        *_rdf_json_values(properties, "preferredMappedTo"),
    )
    mapped_descriptor_ids = tuple(
        dict.fromkeys(
            value.upper()
            for value in (_mesh_uri_tail(uri) for uri in mapped_values)
            if re.fullmatch(r"(?i)D\d+", value)
        )
    )
    pharmacological_action_ids = tuple(
        dict.fromkeys(
            value.upper()
            for value in (
                _mesh_uri_tail(uri)
                for uri in _rdf_json_values(properties, "pharmacologicalAction")
            )
            if re.fullmatch(r"(?i)D\d+", value)
        )
    )
    return MeshRecord(
        mesh_id=mesh_id,
        label=labels[0] if labels else "",
        record_type="supplementary" if mesh_id.startswith("C") else "descriptor",
        tree_numbers=tuple(dict.fromkeys(tree_numbers)),
        mapped_descriptor_ids=mapped_descriptor_ids,
        pharmacological_action_ids=pharmacological_action_ids,
    )


def _fetch_mesh_record_direct(
    mesh_id: str,
    *,
    session: requests.Session,
    pacer: RequestPacer,
    metrics: _RequestMetrics,
    config: PubTator3Config,
) -> MeshRecord | None:
    response = _request_bytes(
        session,
        pacer,
        metrics,
        method="GET",
        url=f"{MESH_RDF_RESOURCE}/{mesh_id}.json",
        context=f"MeSH RDF resource lookup for {mesh_id}",
        config=config,
        accepted_statuses={404},
    )
    if response.status_code == 404:
        return None
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"MeSH RDF resource returned invalid JSON for {mesh_id}."
        ) from exc
    return _parse_mesh_rdf_json_record(payload, mesh_id)


def _mesh_sparql_query(mesh_ids: Sequence[str]) -> str:
    values = " ".join(
        f"<http://id.nlm.nih.gov/mesh/{mesh_id}>"
        for mesh_id in dict.fromkeys(mesh_ids)
    )
    return f"""
PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?record
       (SAMPLE(?recordLabel) AS ?label)
       (GROUP_CONCAT(DISTINCT STR(?tree); separator="|") AS ?trees)
       (GROUP_CONCAT(DISTINCT STR(?mapped); separator="|") AS ?mappedIds)
       (GROUP_CONCAT(DISTINCT STR(?action); separator="|") AS ?actions)
FROM <http://id.nlm.nih.gov/mesh>
WHERE {{
  VALUES ?record {{ {values} }}
  ?record rdfs:label ?recordLabel .
  FILTER(LANG(?recordLabel) = "" || LANGMATCHES(LANG(?recordLabel), "en"))
  OPTIONAL {{ ?record meshv:treeNumber ?tree . }}
  OPTIONAL {{
    {{ ?record meshv:mappedTo ?mapped . }}
    UNION
    {{ ?record meshv:preferredMappedTo ?mapped . }}
  }}
  OPTIONAL {{ ?record meshv:pharmacologicalAction ?action . }}
}}
GROUP BY ?record
""".strip()


def _fetch_mesh_records(
    mesh_ids: Sequence[str],
    *,
    session: requests.Session,
    pacer: RequestPacer,
    metrics: _RequestMetrics,
    config: PubTator3Config,
) -> dict[str, MeshRecord]:
    """Retrieve MeSH hierarchy data from the official MeSH RDF endpoint.

    Entrez MeSH EFetch commonly returns document-summary XML rather than the
    full DescriptorRecord/SupplementalRecord XML used in the annual MeSH files.
    Parsing that response as full records leaves every identifier unresolved.
    MeSH RDF exposes labels, tree numbers, SCR mappings, and pharmacological
    actions directly, so it is the authoritative API for this classification.
    """

    records: dict[str, MeshRecord] = {}
    candidates = tuple(dict.fromkeys(mesh_ids))
    for batch in _batched(candidates, 20):
        batch_records: dict[str, MeshRecord] = {}
        sparql_error: Exception | None = None

        try:
            response = _request_bytes(
                session,
                pacer,
                metrics,
                method="GET",
                url=MESH_RDF_SPARQL,
                context="MeSH RDF hormone-classification lookup",
                config=config,
                params={
                    "query": _mesh_sparql_query(batch),
                    "format": "JSON",
                    "year": "current",
                    "limit": "1000",
                },
            )
            try:
                payload = response.json()
            except ValueError as exc:
                raise RuntimeError("MeSH RDF returned invalid JSON.") from exc

            results = payload.get("results") if isinstance(payload, Mapping) else None
            bindings = (
                results.get("bindings") if isinstance(results, Mapping) else None
            )
            if not isinstance(bindings, list):
                raise RuntimeError("MeSH RDF returned an invalid SPARQL result.")

            for binding in bindings:
                if not isinstance(binding, Mapping):
                    continue
                mesh_id = _mesh_uri_tail(
                    _sparql_binding_value(binding, "record")
                ).upper()
                if mesh_id not in batch:
                    continue

                tree_numbers = tuple(
                    value
                    for value in _mesh_uri_values(
                        _sparql_binding_value(binding, "trees")
                    )
                    if re.fullmatch(r"[A-Z]\d+(?:\.\d+)*", value)
                )
                mapped_descriptor_ids = tuple(
                    value.upper()
                    for value in _mesh_uri_values(
                        _sparql_binding_value(binding, "mappedIds")
                    )
                    if re.fullmatch(r"(?i)D\d+", value)
                )
                pharmacological_action_ids = tuple(
                    value.upper()
                    for value in _mesh_uri_values(
                        _sparql_binding_value(binding, "actions")
                    )
                    if re.fullmatch(r"(?i)D\d+", value)
                )
                batch_records[mesh_id] = MeshRecord(
                    mesh_id=mesh_id,
                    label=_sparql_binding_value(binding, "label"),
                    record_type=(
                        "supplementary"
                        if mesh_id.startswith("C")
                        else "descriptor"
                    ),
                    tree_numbers=tree_numbers,
                    mapped_descriptor_ids=mapped_descriptor_ids,
                    pharmacological_action_ids=pharmacological_action_ids,
                )
        except Exception as exc:
            sparql_error = exc
            logger.warning(
                "MeSH RDF SPARQL lookup failed for %s; using direct resource "
                "lookups: %s",
                batch,
                exc,
            )

        missing = [mesh_id for mesh_id in batch if mesh_id not in batch_records]
        direct_errors: list[Exception] = []
        for mesh_id in missing:
            try:
                record = _fetch_mesh_record_direct(
                    mesh_id,
                    session=session,
                    pacer=pacer,
                    metrics=metrics,
                    config=config,
                )
            except Exception as exc:
                direct_errors.append(exc)
                logger.warning(
                    "Direct MeSH RDF lookup failed for %s: %s",
                    mesh_id,
                    exc,
                )
                continue
            if record is not None:
                batch_records[mesh_id] = record

        if (
            not batch_records
            and missing
            and len(direct_errors) == len(missing)
        ):
            cause = sparql_error or direct_errors[-1]
            raise RuntimeError(
                "MeSH RDF batch and direct identifier lookups both failed."
            ) from cause

        records.update(batch_records)

    return records


def _is_hormone_tree_number(tree_number: str) -> bool:
    value = _clean_text(tree_number)
    return value == MESH_HORMONE_TREE_PREFIX or value.startswith(
        f"{MESH_HORMONE_TREE_PREFIX}."
    )


@dataclass
class _MeshHormoneResult:
    hormone_ids: set[str] = field(default_factory=set)
    labels: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    unresolved_ids: set[str] = field(default_factory=set)
    cache_hits: int = 0


def _fetch_mesh_records_resiliently(
    mesh_ids: Sequence[str],
    *,
    session: requests.Session,
    pacer: RequestPacer,
    metrics: _RequestMetrics,
    config: PubTator3Config,
) -> tuple[dict[str, MeshRecord], set[str]]:
    records: dict[str, MeshRecord] = {}
    failed: set[str] = set()
    for batch in _batched(tuple(dict.fromkeys(mesh_ids)), 40):
        try:
            records.update(
                _fetch_mesh_records(
                    batch,
                    session=session,
                    pacer=pacer,
                    metrics=metrics,
                    config=config,
                )
            )
        except Exception as exc:
            logger.warning("MeSH hormone-classification lookup failed for %s: %s", batch, exc)
            failed.update(batch)
    return records, failed


def _classify_mesh_hormones(
    chemical_ids: Sequence[str],
    *,
    cache: LabelCache,
    session: requests.Session,
    pacer: RequestPacer,
    metrics: _RequestMetrics,
    config: PubTator3Config,
) -> _MeshHormoneResult:
    candidates = tuple(dict.fromkeys(chemical_ids))
    result = _MeshHormoneResult()
    if not candidates:
        return result

    classifications = cache.get_hormone_membership(_MESH_HORMONE_CACHE_KIND, candidates)
    result.cache_hits = len(classifications)
    missing = [mesh_id for mesh_id in candidates if mesh_id not in classifications]
    candidate_records, failed = _fetch_mesh_records_resiliently(
        missing,
        session=session,
        pacer=pacer,
        metrics=metrics,
        config=config,
    )

    resolved_now: dict[str, tuple[bool, str]] = {}
    descriptor_membership: dict[str, tuple[bool, str]] = {}
    for mesh_id, record in candidate_records.items():
        if record.label:
            result.labels[mesh_id] = record.label
        if record.record_type != "descriptor":
            continue
        hormone_tree = next(
            (tree for tree in record.tree_numbers if _is_hormone_tree_number(tree)),
            None,
        )
        membership = (
            (True, f"MeSH hormone tree {hormone_tree}")
            if hormone_tree
            else (False, "outside MeSH biological hormone tree D06.472")
        )
        classifications[mesh_id] = membership
        descriptor_membership[mesh_id] = membership
        resolved_now[mesh_id] = membership

    mapped_ids = tuple(
        dict.fromkeys(
            mapped_id
            for record in candidate_records.values()
            if record.record_type == "supplementary"
            for mapped_id in record.mapped_descriptor_ids
        )
    )
    cached_mapped = cache.get_hormone_membership(_MESH_HORMONE_CACHE_KIND, mapped_ids)
    descriptor_membership.update(cached_mapped)
    missing_mapped = [
        mesh_id for mesh_id in mapped_ids if mesh_id not in descriptor_membership
    ]
    mapped_records, mapped_failed = _fetch_mesh_records_resiliently(
        missing_mapped,
        session=session,
        pacer=pacer,
        metrics=metrics,
        config=config,
    )
    failed.update(mapped_failed)
    mapped_resolved: dict[str, tuple[bool, str]] = {}
    for mesh_id, record in mapped_records.items():
        if record.record_type != "descriptor":
            continue
        hormone_tree = next(
            (tree for tree in record.tree_numbers if _is_hormone_tree_number(tree)),
            None,
        )
        membership = (
            (True, f"MeSH hormone tree {hormone_tree}")
            if hormone_tree
            else (False, "outside MeSH biological hormone tree D06.472")
        )
        descriptor_membership[mesh_id] = membership
        mapped_resolved[mesh_id] = membership

    for mesh_id, record in candidate_records.items():
        if mesh_id in classifications:
            continue
        if record.record_type != "supplementary":
            continue
        if MESH_HORMONE_DESCRIPTOR_ID in record.pharmacological_action_ids:
            membership = (
                True,
                f"MeSH pharmacological action {MESH_HORMONE_DESCRIPTOR_ID}",
            )
        else:
            mapped = [
                (mapped_id, descriptor_membership.get(mapped_id))
                for mapped_id in record.mapped_descriptor_ids
            ]
            positive = next(
                (
                    mapped_id
                    for mapped_id, membership in mapped
                    if membership is not None and membership[0]
                ),
                None,
            )
            if positive:
                membership = (True, f"MeSH mapped hormone descriptor {positive}")
            elif mapped and all(membership is not None for _, membership in mapped):
                membership = (False, "supplementary concept maps outside D06.472")
            elif mapped:
                result.unresolved_ids.add(mesh_id)
                continue
            else:
                membership = (False, "supplementary concept has no hormone mapping")
        classifications[mesh_id] = membership
        resolved_now[mesh_id] = membership

    result.unresolved_ids.update(
        mesh_id
        for mesh_id in candidates
        if mesh_id not in classifications
    )
    result.unresolved_ids.update(mesh_id for mesh_id in failed if mesh_id in candidates)
    cache.put_hormone_membership(
        _MESH_HORMONE_CACHE_KIND, {**mapped_resolved, **resolved_now}
    )

    for mesh_id, (is_hormone, evidence) in classifications.items():
        if mesh_id not in candidates or not is_hormone:
            continue
        result.hormone_ids.add(mesh_id)
        result.evidence[mesh_id] = evidence
    positive_labels = {
        mesh_id: label
        for mesh_id, label in result.labels.items()
        if mesh_id in result.hormone_ids and label
    }
    cache.put_many("hormone", positive_labels, _HORMONE_LABEL_CACHE_SOURCE)
    return result


def _resolve_hormone_labels(
    hormone_ids: Sequence[str],
    *,
    session: requests.Session,
    pacer: RequestPacer,
    metrics: _RequestMetrics,
    config: PubTator3Config,
) -> dict[str, str]:
    records = _fetch_mesh_records(
        hormone_ids,
        session=session,
        pacer=pacer,
        metrics=metrics,
        config=config,
    )
    return {
        mesh_id: record.label
        for mesh_id, record in records.items()
        if record.label
    }


def _resolve_labels(
    concept_ids: Mapping[str, set[str]],
    *,
    gene_metadata: Mapping[str, GeneMetadata],
    label_cache_path: Path,
    session: requests.Session,
    pacer: RequestPacer,
    metrics: _RequestMetrics,
    config: PubTator3Config,
    mesh_labels: Mapping[str, str],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str], dict[str, int]]:
    """Resolve display labels without a second NCBI Gene request."""

    labels: dict[tuple[str, str], str] = {}
    sources: dict[tuple[str, str], str] = {}
    stats = {
        "label_cache_hits": 0,
        "gene_labels_resolved": 0,
        "hormone_labels_resolved": 0,
        "label_fallbacks": 0,
    }

    for concept_id, metadata in gene_metadata.items():
        if concept_id not in concept_ids.get("gene", set()):
            continue
        symbol = _clean_text(metadata.symbol)
        if not symbol:
            continue
        labels[("gene", concept_id)] = symbol
        sources[("gene", concept_id)] = "NCBI Gene ESummary"
        stats["gene_labels_resolved"] += 1

    cache = LabelCache(label_cache_path)
    try:
        current_mesh_labels = {
            concept_id: _clean_text(label)
            for concept_id, label in mesh_labels.items()
            if concept_id in concept_ids.get("hormone", set()) and _clean_text(label)
        }
        for concept_id, label in current_mesh_labels.items():
            labels[("hormone", concept_id)] = label
            sources[("hormone", concept_id)] = "MeSH"
        cache.put_many("hormone", current_mesh_labels, _HORMONE_LABEL_CACHE_SOURCE)

        cached_hormones = cache.get_many(
            "hormone",
            concept_ids.get("hormone", set()),
            source=_HORMONE_LABEL_CACHE_SOURCE,
        )
        for concept_id, label in cached_hormones.items():
            key = ("hormone", concept_id)
            if key not in labels:
                labels[key] = label
                sources[key] = "MeSH"
                stats["label_cache_hits"] += 1

        if config.resolve_preferred_labels:
            missing_hormones = sorted(
                concept_id
                for concept_id in concept_ids.get("hormone", set())
                if ("hormone", concept_id) not in labels
            )
            try:
                resolved_hormones = _resolve_hormone_labels(
                    missing_hormones,
                    session=session,
                    pacer=pacer,
                    metrics=metrics,
                    config=config,
                )
            except Exception as exc:
                logger.warning("MeSH RDF preferred-label lookup failed: %s", exc)
                resolved_hormones = {}
            for concept_id, label in resolved_hormones.items():
                labels[("hormone", concept_id)] = label
                sources[("hormone", concept_id)] = "MeSH"
            cache.put_many("hormone", resolved_hormones, _HORMONE_LABEL_CACHE_SOURCE)
            stats["hormone_labels_resolved"] = len(resolved_hormones)

        stats["label_fallbacks"] = sum(
            1
            for entity_type in ("gene", "hormone")
            for concept_id in concept_ids.get(entity_type, set())
            if (entity_type, concept_id) not in labels
        )
    finally:
        cache.close()
    return labels, sources, stats

def _finalize_entry(
    state: _EntryState,
    *,
    labels: Mapping[tuple[str, str], str],
    label_sources: Mapping[tuple[str, str], str],
    human_gene_metadata: Mapping[str, GeneMetadata],
    non_human_gene_ids: set[str],
    unresolved_gene_ids: set[str],
    hormone_ids: set[str],
    hormone_evidence: Mapping[str, str],
    unresolved_hormone_ids: set[str],
) -> dict[str, int]:
    output_path = Path(
        str(
            state.entry.get("pubtator_annotations_path")
            or state.provisional_path.parent / PUBTATOR3_ANNOTATIONS_FILENAME
        )
    )

    counts = {
        "gene": 0,
        "hormone": 0,
        "total": 0,
        "gene_raw": 0,
        "gene_non_human_discarded": 0,
        "gene_unresolved_discarded": 0,
        "gene_tax_hint_mismatches": 0,
        "chemical_raw": 0,
        "chemical_discarded": 0,
        "chemical_unresolved": 0,
    }

    def rows() -> Iterator[dict[str, Any]]:
        if not state.provisional_path.is_file():
            return
        seen: set[tuple[Any, ...]] = set()
        for raw_row in _iter_jsonl(state.provisional_path):
            row = dict(raw_row)
            raw_entity_type = str(row.get("entity_type") or "")
            identifier = str(
                (
                    row.get("gene_id")
                    if raw_entity_type == "gene"
                    else row.get("chemical_id")
                )
                or ""
            )
            raw_signature = (
                row.get("base"),
                row.get("chunk_id"),
                raw_entity_type,
                row.get("start"),
                row.get("end"),
                row.get("concept_id"),
            )
            if raw_signature in seen:
                continue
            seen.add(raw_signature)

            if raw_entity_type == "chemical":
                counts["chemical_raw"] += 1
                if identifier not in hormone_ids:
                    counts["chemical_discarded"] += 1
                    if identifier in unresolved_hormone_ids:
                        counts["chemical_unresolved"] += 1
                    continue
                entity_type = "hormone"
                row["entity_type"] = entity_type
                row["hormone_id"] = identifier
                # Retained for compatibility with code that already consumes MeSH IDs.
                row["chemical_id"] = identifier
                row["source_entity_type"] = "Chemical"
                row["hormone_classification_source"] = hormone_evidence.get(
                    identifier,
                    "MeSH biological hormone hierarchy D06.472",
                )
            elif raw_entity_type == "gene":
                counts["gene_raw"] += 1
                metadata = human_gene_metadata.get(identifier)
                if metadata is None:
                    if identifier in non_human_gene_ids:
                        counts["gene_non_human_discarded"] += 1
                    elif identifier in unresolved_gene_ids:
                        counts["gene_unresolved_discarded"] += 1
                    else:
                        counts["gene_unresolved_discarded"] += 1
                    continue
                entity_type = "gene"
                pubtator_tax_id = normalize_tax_id(row.pop("pubtator_tax_id", ""))
                if pubtator_tax_id and pubtator_tax_id != metadata.tax_id:
                    counts["gene_tax_hint_mismatches"] += 1
                row["gene_id"] = metadata.gene_id
                row["concept_id"] = f"NCBIGene:{metadata.gene_id}"
                row["tax_id"] = metadata.tax_id
                row["tax_name"] = metadata.tax_name or HUMAN_TAX_NAME
                row["taxonomy_source"] = "NCBI Gene ESummary"
                if metadata.status:
                    row["gene_record_status"] = metadata.status
            else:
                continue

            key = (entity_type, identifier)
            preferred_label = _clean_text(labels.get(key))
            if preferred_label:
                row["preferred_label"] = preferred_label
                row["label_source"] = label_sources.get(key) or (
                    "NCBI Gene" if entity_type == "gene" else "MeSH"
                )
            else:
                row["preferred_label"] = _clean_text(row.get("mention"))
                row["label_source"] = "mention"
            counts[entity_type] += 1
            counts["total"] += 1
            yield row

    _atomic_write_gzip_jsonl(output_path, rows())
    state.provisional_path.unlink(missing_ok=True)
    return counts


def run_pubtator3_annotations(
    entries: Sequence[Mapping[str, Any]],
    *,
    options: Mapping[str, Any] | None = None,
    label_cache_path: Path,
) -> dict[str, Any]:
    """Extract genes and MeSH-filtered hormones for every Stage 1 paper entry."""

    started = time.monotonic()
    config = PubTator3Config.from_options(options)
    session = _build_session(config)
    pacer = RequestPacer(3.0)
    metrics = _RequestMetrics()
    gene_identifier_metrics = _GeneIdentifierMetrics()

    states: list[_EntryState] = []
    pmcid_to_states: dict[str, list[int]] = collections.defaultdict(list)
    pmid_to_states: dict[str, list[int]] = collections.defaultdict(list)
    pmid_to_pmcid: dict[str, str] = {}

    for index, raw_entry in enumerate(entries):
        entry = dict(raw_entry)
        parent = Path(str(entry["chunk_path"])).parent
        entry.setdefault(
            "pubtator_annotations_path", str(parent / PUBTATOR3_ANNOTATIONS_FILENAME)
        )
        provisional = parent / PUBTATOR3_PROVISIONAL_FILENAME
        provisional.unlink(missing_ok=True)

        pmid: str | None = None
        pmcid: str | None = None
        title_abstract_chunks: set[int] = set()
        chunk_count = 0
        for chunk_index, chunk in enumerate(_iter_jsonl(Path(str(entry["chunk_path"])))):
            chunk_count += 1
            if pmcid is None:
                pmcid = _normalize_pmcid(chunk.get("pmcid"))
            if pmid is None:
                pmid = _normalize_pmid(chunk.get("pmid"))
            section = _clean_text(chunk.get("section_type")).upper()
            if section in {"TITLE", "ABSTRACT"}:
                title_abstract_chunks.add(chunk_index)

        states.append(
            _EntryState(
                entry=entry,
                provisional_path=provisional,
                pmid=pmid,
                pmcid=pmcid,
                chunk_count=chunk_count,
                title_abstract_chunks=title_abstract_chunks,
            )
        )
        if pmcid:
            pmcid_to_states[pmcid].append(index)
        if pmid:
            pmid_to_states[pmid].append(index)
        if pmid and pmcid:
            pmid_to_pmcid[pmid] = pmcid

    concept_ids: dict[str, set[str]] = {"gene": set(), "chemical": set()}
    stats: dict[str, Any] = {
        "pubtator_pipeline_version": PUBTATOR3_PIPELINE_VERSION,
        "pubtator_papers_total": len(states),
        "pubtator_pmcids_requested": len(pmcid_to_states),
        "pubtator_pmids_requested": 0,
        "pubtator_documents_received": 0,
        "pubtator_papers_covered": 0,
        "pubtator_chunks_matched": 0,
        "pubtator_failed_batches": 0,
        "gene_mentions": 0,
        "hormone_count": 0,
        "pubtator_annotation_count": 0,
    }

    try:
        pmcids = tuple(pmcid_to_states)
        for batch in _batched(pmcids, config.batch_size):
            try:
                response = _request_bytes(
                    session,
                    pacer,
                    metrics,
                    method="GET",
                    url=PUBTATOR3_PMC_EXPORT,
                    context="PubTator3 PMC full-text export",
                    config=config,
                    params={"pmcids": ",".join(batch)},
                    accepted_statuses={400, 404},
                )
                if response.status_code in {400, 404}:
                    continue
                documents = _parse_bioc_documents(response.content)
                mapped = _map_documents(
                    documents,
                    batch,
                    kind="pmcid",
                    pmid_to_pmcid=pmid_to_pmcid,
                )
            except Exception as exc:
                logger.warning("PubTator3 PMC batch failed for %s: %s", batch, exc)
                stats["pubtator_failed_batches"] += 1
                continue
            stats["pubtator_documents_received"] += len(mapped)
            for pmcid, document in mapped.items():
                for state_index in pmcid_to_states.get(pmcid, []):
                    matched, written, _ = _process_document_for_state(
                        document,
                        states[state_index],
                        concept_ids=concept_ids,
                        gene_identifier_metrics=gene_identifier_metrics,
                        only_uncovered=False,
                    )
                    stats["pubtator_chunks_matched"] += matched
                    stats["pubtator_annotation_count"] += written

        needed_pmids: list[str] = []
        for pmid, state_indexes in pmid_to_states.items():
            if any(
                states[state_index].title_abstract_chunks
                - states[state_index].covered_chunks
                for state_index in state_indexes
            ):
                needed_pmids.append(pmid)
        stats["pubtator_pmids_requested"] = len(needed_pmids)

        for batch in _batched(tuple(needed_pmids), config.batch_size):
            try:
                response = _request_bytes(
                    session,
                    pacer,
                    metrics,
                    method="GET",
                    url=PUBTATOR3_ABSTRACT_EXPORT,
                    context="PubTator3 PubMed abstract export",
                    config=config,
                    params={"pmids": ",".join(batch)},
                    accepted_statuses={400, 404},
                )
                if response.status_code in {400, 404}:
                    continue
                documents = _parse_bioc_documents(response.content)
                mapped = _map_documents(documents, batch, kind="pmid")
            except Exception as exc:
                logger.warning("PubTator3 PMID batch failed for %s: %s", batch, exc)
                stats["pubtator_failed_batches"] += 1
                continue
            stats["pubtator_documents_received"] += len(mapped)
            for pmid, document in mapped.items():
                for state_index in pmid_to_states.get(pmid, []):
                    matched, written, _ = _process_document_for_state(
                        document,
                        states[state_index],
                        concept_ids=concept_ids,
                        gene_identifier_metrics=gene_identifier_metrics,
                        only_uncovered=True,
                    )
                    stats["pubtator_chunks_matched"] += matched
                    stats["pubtator_annotation_count"] += written

        attempted_identifiers = bool(pmcid_to_states or pmid_to_states)
        any_document = any(state.document_seen for state in states)
        if attempted_identifiers and config.required and not any_document:
            raise RuntimeError(
                "PubTator3 returned no usable document for this Stage 2 job. "
                "Set PUBTATOR3_REQUIRED=false only when a cell-only fallback is acceptable."
            )

        classification_cache = LabelCache(label_cache_path)
        try:
            gene_metadata_result = _resolve_human_gene_metadata(
                sorted(concept_ids["gene"]),
                cache=classification_cache,
                session=session,
                pacer=pacer,
                metrics=metrics,
                config=config,
            )
            mesh_result = _classify_mesh_hormones(
                sorted(concept_ids["chemical"]),
                cache=classification_cache,
                session=session,
                pacer=pacer,
                metrics=metrics,
                config=config,
            )
        finally:
            classification_cache.close()

        if config.required and concept_ids["gene"] and not gene_metadata_result.records:
            raise RuntimeError(
                "NCBI Gene metadata lookup resolved none of the PubTator3 Gene IDs; "
                "refusing to publish a zero-gene Stage 2 artifact."
            )

        if (
            config.required
            and concept_ids["chemical"]
            and mesh_result.unresolved_ids == concept_ids["chemical"]
        ):
            raise RuntimeError(
                "MeSH hormone classification failed for every PubTator3 chemical "
                "identifier; refusing to publish an unverified hormone result."
            )

        final_concept_ids: dict[str, set[str]] = {
            "gene": set(gene_metadata_result.human_records),
            "hormone": set(mesh_result.hormone_ids),
        }
        labels, label_sources, label_stats = _resolve_labels(
            final_concept_ids,
            gene_metadata=gene_metadata_result.human_records,
            label_cache_path=label_cache_path,
            session=session,
            pacer=pacer,
            metrics=metrics,
            config=config,
            mesh_labels=mesh_result.labels,
        )

        final_counts = {
            "gene": 0,
            "hormone": 0,
            "total": 0,
            "gene_raw": 0,
            "gene_non_human_discarded": 0,
            "gene_unresolved_discarded": 0,
            "gene_tax_hint_mismatches": 0,
            "chemical_raw": 0,
            "chemical_discarded": 0,
            "chemical_unresolved": 0,
        }
        for state in states:
            entry_counts = _finalize_entry(
                state,
                labels=labels,
                label_sources=label_sources,
                human_gene_metadata=gene_metadata_result.human_records,
                non_human_gene_ids=gene_metadata_result.non_human_ids,
                unresolved_gene_ids=gene_metadata_result.unresolved_ids,
                hormone_ids=mesh_result.hormone_ids,
                hormone_evidence=mesh_result.evidence,
                unresolved_hormone_ids=mesh_result.unresolved_ids,
            )
            for key in final_counts:
                final_counts[key] += int(entry_counts[key])

        stats["pubtator_annotation_count"] = final_counts["total"]
        stats["pubtator_papers_covered"] = sum(1 for state in states if state.document_seen)
        stats["pubtator_papers_uncovered"] = (
            len(states) - int(stats["pubtator_papers_covered"])
        )
        total_chunks = sum(state.chunk_count for state in states)
        covered_chunks = sum(len(state.covered_chunks) for state in states)
        stats["pubtator_chunks_total"] = total_chunks
        stats["pubtator_chunks_covered"] = covered_chunks
        stats["pubtator_chunks_uncovered"] = max(0, total_chunks - covered_chunks)
        stats["gene_mentions"] = final_counts["gene"]
        stats["hormone_count"] = final_counts["hormone"]
        stats["pubtator_gene_mentions_raw"] = final_counts["gene_raw"]
        stats["raw_unique_gene_ids"] = len(concept_ids["gene"])
        stats["unique_gene_ids"] = len(gene_metadata_result.human_records)
        stats["human_tax_id"] = HUMAN_TAX_ID
        stats["gene_identifiers_parsed"] = gene_identifier_metrics.parsed_identifiers
        stats["gene_identifiers_with_pubtator_tax_id"] = (
            gene_identifier_metrics.scoped_identifiers
        )
        stats["gene_identifiers_without_pubtator_tax_id"] = (
            gene_identifier_metrics.unscoped_identifiers
        )
        stats["invalid_gene_identifiers_discarded"] = (
            gene_identifier_metrics.invalid_identifiers
        )
        stats["gene_metadata_cache_hits"] = gene_metadata_result.cache_hits
        stats["gene_metadata_records_resolved"] = len(gene_metadata_result.records)
        stats["human_gene_ids_retained"] = len(gene_metadata_result.human_records)
        stats["non_human_gene_ids_discarded"] = len(
            gene_metadata_result.non_human_ids
        )
        stats["unresolved_gene_ids_discarded"] = len(
            gene_metadata_result.unresolved_ids
        )
        stats["non_human_gene_mentions_discarded"] = final_counts[
            "gene_non_human_discarded"
        ]
        stats["unresolved_gene_mentions_discarded"] = final_counts[
            "gene_unresolved_discarded"
        ]
        stats["pubtator_gene_tax_hint_mismatches"] = final_counts[
            "gene_tax_hint_mismatches"
        ]
        # Compatibility aliases for summaries created by the earlier filter.
        stats["human_gene_identifiers_retained"] = len(
            gene_metadata_result.human_records
        )
        stats["non_human_gene_identifiers_discarded"] = len(
            gene_metadata_result.non_human_ids
        )
        stats["unscoped_gene_identifiers_discarded"] = 0
        stats["unverified_gene_mentions_discarded"] = final_counts[
            "gene_unresolved_discarded"
        ]
        stats["unique_hormone_ids"] = len(mesh_result.hormone_ids)
        stats["pubtator_chemical_mentions_raw"] = final_counts["chemical_raw"]
        stats["non_hormone_chemical_mentions_discarded"] = final_counts[
            "chemical_discarded"
        ]
        stats["unresolved_hormone_chemical_mentions_discarded"] = final_counts[
            "chemical_unresolved"
        ]
        stats["hormone_mesh_classification_cache_hits"] = mesh_result.cache_hits
        stats["unresolved_hormone_mesh_ids"] = len(mesh_result.unresolved_ids)
        stats.update(label_stats)
        stats["pubtator_requests"] = metrics.requests
        stats["pubtator_retries"] = metrics.retries
        stats["pubtator_failed_requests"] = metrics.failed_requests
        stats["pubtator_elapsed_seconds"] = round(time.monotonic() - started, 2)
        return stats
    finally:
        session.close()


__all__ = [
    "HUMAN_TAX_ID",
    "HUMAN_TAX_NAME",
    "MESH_HORMONE_DESCRIPTOR_ID",
    "MESH_HORMONE_TREE_PREFIX",
    "PUBTATOR3_ANNOTATIONS_FILENAME",
    "PUBTATOR3_PIPELINE_VERSION",
    "PubTator3Config",
    "run_pubtator3_annotations",
    "sanitize_text",
    "sanitize_text_with_boundaries",
]
