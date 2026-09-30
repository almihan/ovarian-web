"""PubTator3 gene/protein and MeSH hormone annotation for Stage 2.

Gene IDs map directly to approved HGNC records. No species validation or human
NCBI reference download is performed; unmapped genes are discarded. General
HGNC exact-name recovery remains enabled, with no hard-coded gene overrides.
Chemicals are retained only when their MeSH IDs belong to the local hormone
lexicon. Complete receptor expressions are reconciled before branch merging.
"""

from __future__ import annotations

import collections
import difflib
import gzip
import json
import logging
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import requests
from requests.adapters import HTTPAdapter

from backend.pipeline.entity_lexicons import (
    DEFAULT_HORMONE_LEXICON_PATH,
    HormoneLexiconEntry,
    MESH_HORMONE_DESCRIPTOR_ID,
    MESH_HORMONE_RESOURCE_VERSION,
    MESH_HORMONE_TREE_PREFIX,
    ensure_hormone_lexicon,
)
from backend.pipeline.reference_normalization import (
    CachedFileStatus,
    HgncExactMatcher,
    HgncRecord,
    build_hgnc_exact_matcher,
    ensure_hgnc_reference,
    normalize_mesh_id,
    reference_download_stats,
    resolve_hgnc_by_entrez,
)

logger = logging.getLogger(__name__)

PUBTATOR3_ABSTRACT_EXPORT = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/"
    "publications/export/biocjson"
)
PUBTATOR3_PMC_EXPORT = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/"
    "publications/pmc_export/biocjson"
)
# Hormone classification is a local set-membership lookup against the bundled
# 2026 MeSH descriptors under D06.472. It intentionally does not call MeSH RDF.

PUBTATOR3_ANNOTATIONS_FILENAME = "pubtator3_annotations.jsonl.gz"
PUBTATOR3_PROVISIONAL_FILENAME = ".pubtator3_annotations.provisional.jsonl"
PUBTATOR3_PIPELINE_VERSION = "pubtator3-hgnc-only-panel-safe-v21-get100"

# PubTator3 publication-export endpoints are queried with comma-separated IDs.
# Keep each GET request at or below 100 identifiers so the request remains
# within the service's supported URL/query size and does not fail as an empty
# form-POST response.
PUBTATOR3_DEFAULT_BATCH_SIZE = 100
PUBTATOR3_MAX_GET_BATCH_SIZE = 100

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
    batch_size: int = PUBTATOR3_DEFAULT_BATCH_SIZE
    request_timeout: int = 120
    max_attempts: int = 4
    required: bool = True
    resolve_preferred_labels: bool = True
    ncbi_tool: str = "ovarian_network_ncbigene_mesh"
    ncbi_email: str = ""
    reference_data_cache_dir: str = ""

    @classmethod
    def from_options(cls, options: Mapping[str, Any] | None) -> "PubTator3Config":
        raw = options or {}
        return cls(
            batch_size=max(
                1,
                min(
                    PUBTATOR3_MAX_GET_BATCH_SIZE,
                    int(
                        raw.get("pubtator_batch_size")
                        or PUBTATOR3_DEFAULT_BATCH_SIZE
                    ),
                ),
            ),
            request_timeout=max(
                20, min(300, int(raw.get("pubtator_request_timeout") or 120))
            ),
            max_attempts=max(1, min(8, int(raw.get("pubtator_max_attempts") or 4))),
            required=_option_bool(raw.get("pubtator_required"), default=True),
            resolve_preferred_labels=_option_bool(
                raw.get("pubtator_resolve_preferred_labels"), default=True
            ),
            ncbi_tool=(
                _clean_text(raw.get("ncbi_tool"))
                or "ovarian_network_ncbigene_mesh"
            ),
            ncbi_email=_clean_text(raw.get("ncbi_email")),
            reference_data_cache_dir=_clean_text(
                raw.get("reference_data_cache_dir")
            ),
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
    invalid_identifiers: int = 0


@dataclass(frozen=True, slots=True)
class HormoneCanonicalMetadata:
    mesh_id: str
    preferred_label: str


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
                "application/json, application/x-ndjson, "
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


def _request_pubtator_export_batch(
    session: requests.Session,
    pacer: RequestPacer,
    metrics: _RequestMetrics,
    *,
    url: str,
    identifier_field: str,
    identifiers: Sequence[str],
    context: str,
    config: PubTator3Config,
) -> requests.Response:
    """Send one PubTator3 export GET request for a bounded ID batch."""

    if identifier_field not in {"pmids", "pmcids"}:
        raise ValueError("identifier_field must be 'pmids' or 'pmcids'.")
    return _request_bytes(
        session,
        pacer,
        metrics,
        method="GET",
        url=url,
        context=context,
        config=config,
        params={identifier_field: ",".join(identifiers)},
        accepted_statuses={400, 404},
    )


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
) -> tuple[str, ...]:
    """Parse numeric Gene IDs, ignoring optional taxonomy prefixes.

    Both ``9606:3558`` and another taxon-scoped ID are parsed the same way.
    Acceptance depends only on a subsequent approved HGNC mapping.
    """

    parsed_gene_ids: list[str] = []
    seen: set[str] = set()
    for value in _split_identifiers(raw):
        cleaned = re.sub(
            r"(?i)^(?:NCBI\s*Gene|NCBIGene|GeneID|Gene)\s*[:#]?\s*", "", value
        ).strip()
        pair = re.fullmatch(r"(?:(\d+)\s*:\s*)?(\d+)", cleaned)
        gene_id = ""
        if pair is not None:
            gene_id = pair.group(2).lstrip("0") or "0"
            if gene_id == "0":
                gene_id = ""

        if not gene_id:
            if metrics is not None:
                metrics.invalid_identifiers += 1
            continue

        if metrics is not None:
            metrics.parsed_identifiers += 1
        if gene_id not in seen:
            seen.add(gene_id)
            parsed_gene_ids.append(gene_id)

    return tuple(parsed_gene_ids)


def _normalize_gene_ids(raw: Any) -> tuple[str, ...]:
    """Return parseable NCBI Gene IDs without a species filter."""

    return _parse_gene_identifiers(raw)


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
        source_entity_type = _clean_text(
            infons.get("type") or infons.get("entity_type")
        )
        entity_type = _entity_type(infons)
        if entity_type is None:
            continue
        raw_identifier = _annotation_identifier(infons)
        if entity_type == "gene":
            identifiers = _parse_gene_identifiers(
                raw_identifier,
                metrics=gene_identifier_metrics,
            )
        else:
            identifiers = _normalize_chemical_ids(raw_identifier)
        if not identifiers:
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
            for identifier in identifiers:
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
                        "source_entity_type": source_entity_type,
                    }
                )
                if entity_type == "gene":
                    row["gene_id"] = identifier
                    row["identified_source"] = "pubtator3"
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
    text_mode: str = ""
    chunk_count: int = 0
    abstract_chunks: set[int] = field(default_factory=set)
    annotatable_chunks: set[int] = field(default_factory=set)
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
    allowed_chunk_indexes: set[int] | None = None,
) -> tuple[int, int, int]:
    chunks = list(_iter_jsonl(Path(str(state.entry["chunk_path"]))))
    allowed = set(
        state.annotatable_chunks
        if allowed_chunk_indexes is None
        else allowed_chunk_indexes & state.annotatable_chunks
    )
    if only_uncovered:
        allowed -= state.covered_chunks
    matches = _match_passages_to_chunks(
        document,
        chunks,
        allowed_chunk_indexes=allowed,
    )
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


@dataclass
class _MeshHormoneResult:
    hormone_ids: set[str] = field(default_factory=set)
    labels: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    unresolved_ids: set[str] = field(default_factory=set)
    cache_hits: int = 0
    bundle_entries: int = 0


def _classify_mesh_hormones(
    chemical_ids: Sequence[str],
    *,
    hormone_entries: Sequence[HormoneLexiconEntry],
) -> _MeshHormoneResult:
    """Classify PubTator3 chemicals by exact ID membership in the local bundle.

    The downloaded bundle contains only MeSH descriptor IDs from the biological
    Hormones tree. Therefore, an ID absent from the bundle is deterministically
    discarded as a non-hormone; no remote lookup or unresolved retry is needed.
    """

    by_id = {entry.mesh_id: entry for entry in hormone_entries}
    result = _MeshHormoneResult(bundle_entries=len(by_id))
    for mesh_id in dict.fromkeys(chemical_ids):
        entry = by_id.get(mesh_id)
        if entry is None:
            continue
        result.hormone_ids.add(mesh_id)
        result.labels[mesh_id] = entry.preferred_label
        result.evidence[mesh_id] = (
            f"local MeSH hormone descriptor bundle {MESH_HORMONE_TREE_PREFIX}"
        )
    return result


def _build_mesh_hormone_metadata(
    hormone_ids: Iterable[str],
    mesh_labels: Mapping[str, str],
) -> dict[str, HormoneCanonicalMetadata]:
    """Keep every verified hormone under its authoritative MeSH identifier."""

    return {
        mesh_id: HormoneCanonicalMetadata(
            mesh_id=mesh_id,
            preferred_label=_clean_text(mesh_labels.get(mesh_id)),
        )
        for mesh_id in sorted(set(hormone_ids))
    }


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _chunk_span_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("base"),
        row.get("doc_key"),
        row.get("canonical_id"),
        row.get("chunk_id"),
        row.get("section_type"),
    )


def _row_span_identity(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Return a stable chunk-and-offset identity for exact span comparison."""

    try:
        start = int(row.get("start"))
        end = int(row.get("end"))
    except (TypeError, ValueError):
        return None
    if start < 0 or end <= start:
        return None
    return (*_chunk_span_key(row), start, end)


def _validate_existing_gene_rows(
    state: _EntryState, chunks: Sequence[Mapping[str, Any]], *,
    matcher: HgncExactMatcher, hgnc_gene_metadata: MutableMapping[str, HgncRecord],
    concept_ids: MutableMapping[str, set[str]],
) -> dict[str, int]:
    """Validate PubTator spans BEFORE they reserve space against exact recovery.

    Complete approved symbols can
    correct a conflicting alias-based identity; a clipped or multi-gene span
    must not suppress correctly spelled IL-2/IL-21 etc. in the source text.
    """
    from backend.pipeline.entity_span_rules import gene_surface_key, sanitize_source_annotations
    stats = {"gene_mentions_rejected_unsafe_source_span": 0,
             "gene_mentions_corrected_approved_symbol": 0,
             "gene_mentions_replaced_compound_span": 0}
    if not state.provisional_path.is_file():
        return stats
    by_chunk = {_chunk_span_key(chunk): chunk for chunk in chunks}
    kept = []
    changed = False
    for raw_row in _iter_jsonl(state.provisional_path):
        row = dict(raw_row)
        if row.get("entity_type") != "gene":
            kept.append(row)
            continue
        chunk = by_chunk.get(_chunk_span_key(row))
        if chunk is None or not isinstance(chunk.get("chunk"), str):
            kept.append(row)
            continue
        text = chunk["chunk"]
        gene_ids = _parse_gene_identifiers(row.get("gene_id"))
        old_id = gene_ids[0] if gene_ids else ""
        record = hgnc_gene_metadata.get(old_id)
        validation = dict(row)
        if record and not validation.get("matched_term"):
            surface_key = gene_surface_key(row.get("mention"))
            for term in (record.symbol, record.name):
                if surface_key == gene_surface_key(term):
                    validation["matched_term"] = term
                    break
        if not sanitize_source_annotations(text, [validation]):
            stats["gene_mentions_rejected_unsafe_source_span"] += 1
            changed = True
            continue
        start, end = int(row["start"]), int(row["end"])
        surface = text[start:end]
        # An entire compound of fully written gene names is not one gene.
        # Shared-prefix KIR compounds have already been excluded, not split.
        whole = matcher.resolve(surface)
        parts = matcher.find(surface) if whole is None and re.search(r"[,/]|\b(?:and|or)\b", surface) else []
        if len(parts) > 1:
            stats["gene_mentions_replaced_compound_span"] += 1
            changed = True
            continue
        approved = matcher.resolve_approved_symbol(surface)
        if approved and old_id != approved.record.entrez_id:
            replacement = approved.record
            row.update({"gene_id": replacement.entrez_id,
                        "concept_id": replacement.hgnc_id, "normalized_id": replacement.hgnc_id,
                        "hgnc_id": replacement.hgnc_id, "ncbi_gene_id": f"NCBIGene:{replacement.entrez_id}",
                        "pubtator_original_gene_id": old_id,
                        "identity_correction": "whole_approved_symbol_over_conflicting_alias",
                        "matched_term": approved.matched_term, "term_kind": approved.term_kind,
                        "resource_version": approved.resource_version, "identified_source": "exact_match",
                        "tax_id": "9606"})
            hgnc_gene_metadata[replacement.entrez_id] = replacement
            concept_ids["gene"].add(replacement.entrez_id)
            stats["gene_mentions_corrected_approved_symbol"] += 1
            changed = True
        kept.append(row)
    if changed:
        temporary = state.provisional_path.with_suffix(".source-validation.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for row in kept:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(temporary, state.provisional_path)
        finally:
            temporary.unlink(missing_ok=True)
    return stats


def _append_hgnc_exact_gene_matches(
    states: Sequence[_EntryState],
    *,
    matcher: HgncExactMatcher,
    concept_ids: MutableMapping[str, set[str]],
    hgnc_gene_metadata: MutableMapping[str, HgncRecord],
) -> dict[str, int]:
    """Add unique exact HGNC mentions that PubTator3 did not identify.

    Source-valid PubTator3 spans reserve space only against equal or shorter
    exact candidates after approved-symbol identity correction. Clipped/compound spans do not
    suppress exact recovery. Among fallback
    candidates, the longest non-overlapping exact term is retained. Verified
    hormone precedence is applied during finalization, where a gene occupying
    the exact same span as a retained MeSH hormone is removed.
    """

    stats = {
        "hgnc_exact_terms_indexed": matcher.term_count,
        "hgnc_exact_ambiguous_terms_excluded": matcher.ambiguous_term_count,
        "hgnc_exact_records_indexed": matcher.record_count,
        "hgnc_exact_candidates_found": 0,
        "hgnc_exact_mentions_added": 0,
        "hgnc_exact_mentions_skipped_existing_gene_overlap": 0,
        "hgnc_exact_mentions_skipped_fallback_overlap": 0,
        "hgnc_exact_unique_gene_ids_added": 0,
        "gene_mentions_rejected_unsafe_source_span": 0,
        "gene_mentions_corrected_approved_symbol": 0,
        "gene_mentions_replaced_compound_span": 0,
    }
    gene_ids_before = set(hgnc_gene_metadata)

    for state in states:
        chunks = list(_iter_jsonl(Path(str(state.entry["chunk_path"]))))
        validation_stats = _validate_existing_gene_rows(state, chunks, matcher=matcher,
            hgnc_gene_metadata=hgnc_gene_metadata, concept_ids=concept_ids)
        for key, value in validation_stats.items():
            stats[key] += value
        existing_by_chunk: dict[tuple[Any, ...], list[tuple[int, int]]] = (
            collections.defaultdict(list)
        )
        if state.provisional_path.is_file():
            for row in _iter_jsonl(state.provisional_path):
                if str(row.get("entity_type") or "") != "gene":
                    continue
                try:
                    start = int(row.get("start"))
                    end = int(row.get("end"))
                except (TypeError, ValueError):
                    continue
                if 0 <= start < end:
                    existing_by_chunk[_chunk_span_key(row)].append((start, end))

        additions: list[dict[str, Any]] = []
        for chunk_index in sorted(state.annotatable_chunks):
            if not (0 <= chunk_index < len(chunks)):
                continue
            chunk = chunks[chunk_index]
            chunk_text = str(chunk.get("chunk") or "")
            if not chunk_text:
                continue
            chunk_key = _chunk_span_key(chunk)
            existing_spans = list(existing_by_chunk.get(chunk_key, ()))
            candidates = matcher.find(chunk_text)
            stats["hgnc_exact_candidates_found"] += len(candidates)

            selected: list[Any] = []
            for candidate in sorted(
                candidates,
                key=lambda item: (
                    -(item.end - item.start),
                    item.start,
                    item.end,
                    item.record.entrez_id,
                ),
            ):
                span = (candidate.start, candidate.end)
                if any(_spans_overlap(span, occupied) and
                       occupied[1] - occupied[0] >= span[1] - span[0]
                       for occupied in existing_spans):
                    stats["hgnc_exact_mentions_skipped_existing_gene_overlap"] += 1
                    continue
                if any(
                    _spans_overlap(span, (kept.start, kept.end))
                    for kept in selected
                ):
                    stats["hgnc_exact_mentions_skipped_fallback_overlap"] += 1
                    continue
                selected.append(candidate)

            for candidate in sorted(selected, key=lambda item: (item.start, item.end)):
                gene_id = candidate.record.entrez_id
                if not gene_id:
                    continue
                row = _source_projection(chunk)
                row.update(
                    {
                        "entity_type": "gene",
                        "start": candidate.start,
                        "end": candidate.end,
                        "mention": candidate.mention,
                        "concept_id": candidate.record.hgnc_id,
                        "normalized_id": candidate.record.hgnc_id,
                        "hgnc_id": candidate.record.hgnc_id,
                        "ncbi_gene_id": f"NCBIGene:{gene_id}",
                        "gene_id": gene_id,
                        "normalization_source": "HGNC complete set exact term match",
                        "matched_term": candidate.matched_term,
                        "term_kind": candidate.term_kind,
                        "resource_version": candidate.resource_version,
                        "source_entity_type": "Gene/Protein",
                        "identified_source": "exact_match",
                        "tax_id": "9606",
                        "taxonomy_source": "HGNC human nomenclature reference",
                    }
                )
                additions.append(row)
                concept_ids["gene"].add(gene_id)
                hgnc_gene_metadata.setdefault(gene_id, candidate.record)
                existing_spans.append((candidate.start, candidate.end))
                stats["hgnc_exact_mentions_added"] += 1

        if additions:
            _append_jsonl(state.provisional_path, additions)
            state.annotations_written += len(additions)

    stats["hgnc_exact_unique_gene_ids_added"] = len(
        set(hgnc_gene_metadata) - gene_ids_before
    )
    return stats


def _finalize_entry(
    state: _EntryState,
    *,
    hgnc_gene_metadata: Mapping[str, HgncRecord],
    hormone_metadata: Mapping[str, HormoneCanonicalMetadata],
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
        "gene_without_hgnc_mapping_discarded": 0,
        "gene_same_span_hormone_discarded": 0,
        "gene_mentions_from_pubtator": 0,
        "gene_mentions_from_hgnc_exact": 0,
        "gene_mentions_with_uniprot": 0,
        "chemical_raw": 0,
        "chemical_discarded": 0,
        "chemical_unresolved": 0,
        "hormone_mesh_mentions": 0,
    }

    verified_hormone_spans: set[tuple[Any, ...]] = set()
    if state.provisional_path.is_file():
        for raw_row in _iter_jsonl(state.provisional_path):
            if str(raw_row.get("entity_type") or "") != "chemical":
                continue
            identifier = normalize_mesh_id(raw_row.get("chemical_id"))
            if identifier not in hormone_metadata:
                continue
            span_identity = _row_span_identity(raw_row)
            if span_identity is not None:
                verified_hormone_spans.add(span_identity)

    def rows() -> Iterator[dict[str, Any]]:
        if not state.provisional_path.is_file():
            return
        seen: set[tuple[Any, ...]] = set()
        for raw_row in _iter_jsonl(state.provisional_path):
            row = dict(raw_row)
            raw_entity_type = str(row.get("entity_type") or "")
            if raw_entity_type == "gene":
                gene_ids = _parse_gene_identifiers(row.get("gene_id"))
                identifier = gene_ids[0] if gene_ids else ""
            else:
                identifier = normalize_mesh_id(row.get("chemical_id"))
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
                canonical = hormone_metadata.get(identifier)
                if canonical is None:
                    counts["chemical_discarded"] += 1
                    if identifier in unresolved_hormone_ids:
                        counts["chemical_unresolved"] += 1
                    continue

                mesh_curie = f"MESH:{identifier}"
                preferred_label = (
                    canonical.preferred_label or _clean_text(row.get("mention"))
                )
                row["entity_type"] = "hormone"
                row["source_entity_type"] = (
                    _clean_text(row.get("source_entity_type")) or "Chemical"
                )
                row["source_concept_id"] = mesh_curie
                row["pubtator_mesh_id"] = identifier
                row["mesh_id"] = mesh_curie
                row["chemical_id"] = identifier
                row["concept_id"] = mesh_curie
                row["normalized_id"] = mesh_curie
                row["hormone_id"] = mesh_curie
                row["canonical_id_type"] = "mesh"
                row["canonical_name"] = preferred_label
                row["preferred_label"] = preferred_label
                row["label_source"] = "MeSH"
                row["normalization_source"] = (
                    "PubTator3 MeSH chemical -> local MeSH hormone bundle"
                )
                row["normalization_status"] = "canonical_mesh_hormone"
                row["hormone_classification_source"] = hormone_evidence.get(
                    identifier,
                    "local MeSH hormone descriptor bundle D06.472",
                )
                counts["hormone_mesh_mentions"] += 1

            elif raw_entity_type == "gene":
                counts["gene_raw"] += 1
                span_identity = _row_span_identity(row)
                if span_identity in verified_hormone_spans:
                    counts["gene_same_span_hormone_discarded"] += 1
                    continue
                hgnc = hgnc_gene_metadata.get(identifier)
                identified_source = _clean_text(row.get("identified_source")).casefold()
                from_hgnc_exact = identified_source == "exact_match"
                if hgnc is None:
                    counts["gene_without_hgnc_mapping_discarded"] += 1
                    continue
                # HGNC is a naming reference, not verification of source species.
                for field in ("tax_id", "tax_name", "tax_ids", "taxonomy_source"):
                    row.pop(field, None)
                row["reference_tax_id"] = "9606"
                row["taxonomy_status"] = "not_validated_hgnc_mapping_only"
                ncbi_curie = f"NCBIGene:{identifier}"
                hgnc_curie = hgnc.hgnc_id
                for legacy_field in (
                    "gene_id",
                    "pubtator_gene_id",
                    "recognition_source",
                    "hgnc_exact_match_term",
                    "hgnc_ids",
                    "UniProt",
                ):
                    row.pop(legacy_field, None)
                row["entity_type"] = "gene"
                row["source_concept_id"] = ncbi_curie
                row["source_entity_type"] = "Gene/Protein"
                if from_hgnc_exact:
                    counts["gene_mentions_from_hgnc_exact"] += 1
                    row["identified_source"] = "exact_match"
                else:
                    counts["gene_mentions_from_pubtator"] += 1
                    row["identified_source"] = "pubtator3"

                # Both recognition paths publish the same identity schema.
                # HGNC is canonical; NCBI Gene and UniProt remain attached
                # cross-reference metadata from the approved HGNC record.
                row["ncbi_gene_id"] = ncbi_curie
                row["hgnc_id"] = hgnc_curie
                row["concept_id"] = hgnc_curie
                row["normalized_id"] = hgnc_curie
                row["canonical_id_type"] = "hgnc"
                row["canonical_name"] = hgnc.name
                row["preferred_label"] = hgnc.symbol or _clean_text(row.get("mention"))
                row["label_source"] = "HGNC"
                row["normalization_source"] = "approved HGNC complete-set record"
                row["normalization_status"] = "canonical_hgnc"
                if hgnc.uniprot_ids:
                    uniprot_ids = list(hgnc.uniprot_ids)
                    row["uniprot_ids"] = uniprot_ids
                    counts["gene_mentions_with_uniprot"] += 1
                else:
                    row.pop("uniprot_ids", None)
            else:
                continue

            counts[row["entity_type"]] += 1
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
    """Extract taxonomy-verified human genes and MeSH-normalized hormones."""

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
        text_mode = ""
        abstract_chunks: set[int] = set()
        annotatable_chunks: set[int] = set()
        chunk_count = 0
        for chunk_index, chunk in enumerate(_iter_jsonl(Path(str(entry["chunk_path"])))):
            chunk_count += 1
            if pmcid is None:
                pmcid = _normalize_pmcid(chunk.get("pmcid"))
            if pmid is None:
                pmid = _normalize_pmid(chunk.get("pmid"))
            if not text_mode:
                text_mode = _clean_text(chunk.get("text_mode")).casefold()
            section = _clean_text(chunk.get("section_type")).upper()
            chunk_text = _clean_text(chunk.get("chunk"))
            if section == "ABSTRACT" and chunk_text:
                abstract_chunks.add(chunk_index)
            if section not in {"TITLE", "METADATA"} and chunk_text:
                annotatable_chunks.add(chunk_index)

        states.append(
            _EntryState(
                entry=entry,
                provisional_path=provisional,
                pmid=pmid,
                pmcid=pmcid,
                text_mode=text_mode,
                chunk_count=chunk_count,
                abstract_chunks=abstract_chunks,
                annotatable_chunks=annotatable_chunks,
            )
        )
        # Never request text for a metadata-only paper. Abstract-only jobs use
        # the PubMed abstract endpoint directly instead of downloading PMC full
        # text during entity extraction.
        if pmcid and annotatable_chunks and text_mode != "abstract":
            pmcid_to_states[pmcid].append(index)
        if pmid and abstract_chunks:
            pmid_to_states[pmid].append(index)
        if pmid and pmcid and annotatable_chunks and text_mode != "abstract":
            pmid_to_pmcid[pmid] = pmcid

    concept_ids: dict[str, set[str]] = {"gene": set(), "chemical": set()}
    stats: dict[str, Any] = {
        "pubtator_pipeline_version": PUBTATOR3_PIPELINE_VERSION,
        "pubtator_batch_size": config.batch_size,
        "pubtator_request_method": "GET",
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
                response = _request_pubtator_export_batch(
                    session,
                    pacer,
                    metrics,
                    url=PUBTATOR3_PMC_EXPORT,
                    identifier_field="pmcids",
                    identifiers=batch,
                    context="PubTator3 PMC full-text export",
                    config=config,
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
                states[state_index].abstract_chunks
                - states[state_index].covered_chunks
                for state_index in state_indexes
            ):
                needed_pmids.append(pmid)
        stats["pubtator_pmids_requested"] = len(needed_pmids)

        for batch in _batched(tuple(needed_pmids), config.batch_size):
            try:
                response = _request_pubtator_export_batch(
                    session,
                    pacer,
                    metrics,
                    url=PUBTATOR3_ABSTRACT_EXPORT,
                    identifier_field="pmids",
                    identifiers=batch,
                    context="PubTator3 PubMed abstract export",
                    config=config,
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
                        allowed_chunk_indexes=states[state_index].abstract_chunks,
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

        reference_cache_dir = (
            Path(config.reference_data_cache_dir).expanduser().resolve()
            if config.reference_data_cache_dir
            else label_cache_path.parent / "reference_data"
        )
        pubtator_gene_ids = set(concept_ids["gene"])
        hgnc_reference_status: CachedFileStatus | None = None
        hgnc_reference_error = ""
        hgnc_gene_records: dict[str, HgncRecord] = {}
        hgnc_unresolved_pubtator_gene_ids = set(pubtator_gene_ids)
        hgnc_multiple_match_ids: set[str] = set()
        hgnc_exact_stats: dict[str, int] = {
            "hgnc_exact_terms_indexed": 0,
            "hgnc_exact_ambiguous_terms_excluded": 0,
            "hgnc_exact_records_indexed": 0,
            "hgnc_exact_candidates_found": 0,
            "hgnc_exact_mentions_added": 0,
            "hgnc_exact_mentions_skipped_existing_gene_overlap": 0,
            "hgnc_exact_mentions_skipped_fallback_overlap": 0,
            "hgnc_exact_unique_gene_ids_added": 0,
            "gene_mentions_rejected_unsafe_source_span": 0,
            "gene_mentions_corrected_approved_symbol": 0,
            "gene_mentions_replaced_compound_span": 0,
        }

        # Load the HGNC naming and numeric-ID cross-references.
        # No species evidence is requested or used as an annotation veto.
        exact_matcher: HgncExactMatcher | None = None
        if any(state.annotatable_chunks for state in states):
            try:
                hgnc_reference_status = ensure_hgnc_reference(
                    cache_dir=reference_cache_dir,
                    session=session,
                    request_timeout=config.request_timeout,
                    max_attempts=config.max_attempts,
                )
                hgnc_result = resolve_hgnc_by_entrez(
                    hgnc_reference_status.path,
                    sorted(pubtator_gene_ids),
                )
                hgnc_gene_records.update(hgnc_result.records)
                hgnc_unresolved_pubtator_gene_ids = hgnc_result.unresolved_ids
                hgnc_multiple_match_ids = hgnc_result.multiple_match_ids
                exact_matcher = build_hgnc_exact_matcher(hgnc_reference_status.path)
            except Exception as exc:
                hgnc_reference_error = str(exc)
                logger.warning(
                    "HGNC reference data unavailable; genes cannot be normalized: %s", exc)
                raise RuntimeError("The HGNC reference is required for gene normalization.") from exc

        # No species checks: approved HGNC identity is the only gene-ID gate.
        if exact_matcher is not None:
            try:
                hgnc_exact_stats = _append_hgnc_exact_gene_matches(
                    states, matcher=exact_matcher, concept_ids=concept_ids,
                    hgnc_gene_metadata=hgnc_gene_records)
            except Exception:
                # Do not publish a silently partial result after a local recovery
                # programming/I/O failure; its traceback identifies the real cause.
                logger.exception("HGNC exact recovery failed after loading reference data")
                raise

        hormone_label_cache_hits = 0
        hormone_labels_fetched = 0
        try:
            hormone_entries, hormone_bundle_source = ensure_hormone_lexicon(
                DEFAULT_HORMONE_LEXICON_PATH
            )
        except Exception as exc:
            raise RuntimeError(
                "The local MeSH hormone bundle could not be loaded from "
                f"{DEFAULT_HORMONE_LEXICON_PATH}."
            ) from exc

        mesh_result = _classify_mesh_hormones(
            sorted(concept_ids["chemical"]),
            hormone_entries=hormone_entries,
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

        hormone_metadata = _build_mesh_hormone_metadata(
            mesh_result.hormone_ids,
            mesh_result.labels,
        )

        final_counts = {
            "gene": 0,
            "hormone": 0,
            "total": 0,
            "gene_raw": 0,
            "gene_without_hgnc_mapping_discarded": 0,
            "gene_same_span_hormone_discarded": 0,
            "gene_mentions_from_pubtator": 0,
            "gene_mentions_from_hgnc_exact": 0,
            "gene_mentions_with_uniprot": 0,
            "chemical_raw": 0,
            "chemical_discarded": 0,
            "chemical_unresolved": 0,
            "hormone_mesh_mentions": 0,
        }
        for state in states:
            entry_counts = _finalize_entry(
                state,
                hgnc_gene_metadata=hgnc_gene_records,
                hormone_metadata=hormone_metadata,
                hormone_evidence=mesh_result.evidence,
                unresolved_hormone_ids=mesh_result.unresolved_ids,
            )
            for key in final_counts:
                final_counts[key] += int(entry_counts[key])

        hgnc_reference_stats: dict[str, Any] = {
            "hgnc_reference_files": 0,
            "hgnc_reference_files_downloaded": 0,
            "hgnc_reference_stale_files_used": 0,
            "hgnc_reference_bytes": 0,
            "hgnc_reference_error": hgnc_reference_error,
        }
        if hgnc_reference_status is not None:
            hgnc_reference_stats.update(
                reference_download_stats((hgnc_reference_status,), prefix="hgnc")
            )

        stats["pubtator_annotation_count"] = final_counts["total"]
        eligible_states = [state for state in states if state.annotatable_chunks]
        stats["pubtator_papers_eligible"] = len(eligible_states)
        stats["pubtator_papers_without_text"] = len(states) - len(eligible_states)
        stats["pubtator_papers_covered"] = sum(
            1 for state in eligible_states if state.document_seen
        )
        stats["pubtator_papers_uncovered"] = (
            len(eligible_states) - int(stats["pubtator_papers_covered"])
        )
        source_rows = sum(state.chunk_count for state in states)
        eligible_chunks = sum(len(state.annotatable_chunks) for state in states)
        covered_chunks = sum(len(state.covered_chunks) for state in states)
        stats["pubtator_source_rows_total"] = source_rows
        stats["pubtator_chunks_total"] = eligible_chunks
        stats["pubtator_chunks_without_text"] = max(0, source_rows - eligible_chunks)
        stats["pubtator_chunks_covered"] = covered_chunks
        stats["pubtator_chunks_uncovered"] = max(0, eligible_chunks - covered_chunks)

        stats["gene_mentions"] = final_counts["gene"]
        stats["gene_mentions_from_pubtator"] = final_counts[
            "gene_mentions_from_pubtator"
        ]
        stats["gene_mentions_from_hgnc_exact_match"] = final_counts[
            "gene_mentions_from_hgnc_exact"
        ]
        stats["pubtator_gene_mentions_raw"] = (
            final_counts["gene_raw"] - final_counts["gene_mentions_from_hgnc_exact"]
        )
        stats["pubtator_raw_unique_gene_ids"] = len(pubtator_gene_ids)
        stats["unique_hgnc_gene_ids"] = len(
            {record.hgnc_id for record in hgnc_gene_records.values()}
        )
        stats["unique_ncbi_gene_ids"] = len(hgnc_gene_records)
        stats["pubtator_gene_ids_matched_by_hgnc_entrez"] = len(
            set(pubtator_gene_ids) & set(hgnc_gene_records)
        )
        stats["pubtator_gene_ids_without_hgnc_entrez_match"] = len(
            hgnc_unresolved_pubtator_gene_ids
        )
        stats["pubtator_gene_ids_with_multiple_hgnc_rows_retained"] = len(
            hgnc_multiple_match_ids
        )
        stats["gene_mentions_without_hgnc_mapping_discarded"] = final_counts[
            "gene_without_hgnc_mapping_discarded"
        ]
        stats["gene_mentions_removed_for_same_span_mesh_hormone"] = final_counts[
            "gene_same_span_hormone_discarded"
        ]
        stats["gene_mentions_with_uniprot"] = final_counts[
            "gene_mentions_with_uniprot"
        ]
        stats["unique_gene_uniprot_ids"] = len(
            {
                accession
                for record in hgnc_gene_records.values()
                for accession in record.uniprot_ids
            }
        )
        stats["gene_reference_filter"] = "approved HGNC mapping required; no species validation"
        stats["gene_taxonomy_validation"] = False
        stats["gene_identifiers_parsed"] = gene_identifier_metrics.parsed_identifiers
        stats["invalid_gene_identifiers_discarded"] = (
            gene_identifier_metrics.invalid_identifiers
        )
        stats.update(hgnc_exact_stats)

        stats["hormone_count"] = final_counts["hormone"]
        stats["unique_hormone_mesh_ids"] = len(mesh_result.hormone_ids)
        stats["pubtator_chemical_mentions_raw"] = final_counts["chemical_raw"]
        stats["chemical_mentions_not_retained_as_hormones"] = final_counts[
            "chemical_discarded"
        ]
        stats["unresolved_hormone_chemical_mentions_discarded"] = final_counts[
            "chemical_unresolved"
        ]
        stats["hormone_mentions_with_mesh_id"] = final_counts[
            "hormone_mesh_mentions"
        ]
        stats["hormone_mesh_classification_cache_hits"] = 0
        stats["hormone_mesh_bundle_source"] = hormone_bundle_source
        stats["hormone_mesh_bundle_path"] = str(DEFAULT_HORMONE_LEXICON_PATH)
        stats["hormone_mesh_bundle_entries"] = mesh_result.bundle_entries
        stats["hormone_mesh_bundle_matches"] = len(mesh_result.hormone_ids)
        stats["hormone_mesh_resource_version"] = MESH_HORMONE_RESOURCE_VERSION
        stats["unresolved_hormone_mesh_ids"] = len(mesh_result.unresolved_ids)
        stats["hormone_identifier_policy"] = (
            "MeSH descriptor ID must occur in the local D06.472 hormone bundle"
        )
        stats["label_cache_hits"] = hormone_label_cache_hits
        stats["gene_labels_resolved"] = len(hgnc_gene_records)
        stats["hormone_labels_resolved"] = sum(
            1 for item in hormone_metadata.values() if item.preferred_label
        )
        stats["hormone_labels_fetched"] = hormone_labels_fetched
        stats["label_fallbacks"] = sum(
            1 for item in hormone_metadata.values() if not item.preferred_label
        )
        stats.update(hgnc_reference_stats)
        stats["pubtator_requests"] = metrics.requests
        stats["pubtator_retries"] = metrics.retries
        stats["pubtator_failed_requests"] = metrics.failed_requests
        stats["pubtator_elapsed_seconds"] = round(time.monotonic() - started, 2)
        return stats
    finally:
        session.close()


__all__ = [
    "MESH_HORMONE_DESCRIPTOR_ID",
    "MESH_HORMONE_TREE_PREFIX",
    "PUBTATOR3_ANNOTATIONS_FILENAME",
    "PUBTATOR3_PIPELINE_VERSION",
    "PubTator3Config",
    "run_pubtator3_annotations",
    "sanitize_text",
    "sanitize_text_with_boundaries",
]
