"""Paper retrieval for the ovarian-network web application.

The pipeline keeps one shared corpus and one reusable chunk file per paper.

1. Parse one comma-separated mixed user field containing positive keywords,
   ``not <term>`` exclusions, PMIDs, and PMCIDs.
2. Search PubMed once and select at most ``DEFAULT_KEYWORD_LIMIT`` keyword
   results. Explicit/default PMIDs and PMCIDs are still added to that selection.
3. Download PubMed metadata only for identifiers not already in the corpus.
4. Reuse locally stored PMC full text and cached chunks.
5. Retrieve missing full text with a fast PubTator3 batch request, then use the
   official NCBI BioC PMC and Europe PMC full-text APIs as per-paper fallbacks.
6. Cache definitive "not available" results briefly, but keep transient service
   failures retryable. A retriever-version change automatically rechecks legacy
   negative cache entries.
7. Store compact BioC JSON with gzip compression and rebuild only papers that
   have just been upgraded from abstract-only text.
8. Stream one combined ``chunks.jsonl`` without storing a duplicate job copy.

Author metadata is not collected or saved. Cached chunk files are never opened
for legacy-author cleanup.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import logging
import re
import shutil
import sqlite3
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# UPDATED INPUT PARSER: comma-separated keywords, NOT exclusions, PMIDs, and PMCIDs.
InputType = Literal["keywords", "pmid", "pmcid"]
ProgressCallback = Callable[[str, int, str, dict[str, Any]], None]

# The legacy input_type argument is retained for API compatibility, but the
# single user field is always parsed using this mixed comma-separated format.
USER_INPUT_MODE = "mixed_comma_separated_v2"

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_ESEARCH = f"{EUTILS_BASE}/esearch.fcgi"
PUBMED_EFETCH = f"{EUTILS_BASE}/efetch.fcgi"
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULLTEXT_XML = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
)
PUBTATOR3_PMC_EXPORT = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/"
    "publications/pmc_export/biocjson"
)
NCBI_BIOC_PMC_EXPORT = (
    "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/"
    "BioC_json/{pmcid}/unicode"
)

# PubMed can return at most 10,000 IDs in one ESearch response, but this
# application intentionally selects a much smaller fixed result set for speed.
PUBMED_ESEARCH_API_CAP = 10_000
DEFAULT_KEYWORD_LIMIT = 2000
DEFAULT_REQUEST_TIMEOUT = 30
MAX_REQUEST_TIMEOUT = 300

# PubTator3 accepts comma-separated PMCIDs and is retained as the fast path.
# Missing documents and service errors are checked through the two per-paper
# fallback APIs instead of being permanently classified from one batch result.
FULLTEXT_BATCH_SIZE = 20
FULLTEXT_MAX_CONSECUTIVE_SERVICE_ERRORS = 3
FULLTEXT_REQUEST_TIMEOUT = 90
METADATA_HTTP_ATTEMPTS = 3
FULLTEXT_HTTP_ATTEMPTS = 3
FULLTEXT_RETRIEVER_VERSION = "pmc-fulltext-v2"
FULLTEXT_NEGATIVE_CACHE_DAYS = 7
TRANSIENT_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})

# Visible marker for this rewritten mixed-input implementation.
INPUT_PARSER_VERSION = "mixed-comma-v2"

# Curated identifiers retained from the supplied scripts.
DEFAULT_PMIDS: tuple[str, ...] = (
    "24042431",
    "31778080",
    "30744632",
    "27881593",
    "26843151",
    "27567355",
    "27225433",
    "25643098",
    "24606085",
    "24875398",
    "24581581",
    "19159403",
    "35568223",
    "36792027",
    "34630328",
)

DEFAULT_PMCIDS: tuple[str, ...] = (
    "PMC3873842",
    "PMC7054152",
    "PMC6371574",
    "PMC5265695",
    "PMC4763493",
    "PMC5071150",
    "PMC4919523",
    "PMC4315337",
    "PMC4037725",
    "PMC4119548",
    "PMC4008697",
    "PMC2722955",
    "PMC8493253",
)

# PubMed is the only keyword-discovery source in this web workflow. Europe PMC
# is used later only to resolve metadata for new explicit PMCIDs, so its query is
# built dynamically in ``fetch_epmc_records_for_pmcids`` rather than stored as a
# static DEFAULT_EPMC_QUERY.
#
# The query is split into reusable semantic blocks. This keeps ovarian context,
# immune/inflammatory context, human evidence, and exclusions independently
# testable and lets user keywords broaden the biological concept block without
# turning the search into a general, non-ovarian PubMed search.
DEFAULT_PUBMED_PMID_QUERY = r"""
(
  24042431[PMID]
  OR 31778080[PMID]
  OR 30744632[PMID]
  OR 27881593[PMID]
  OR 26843151[PMID]
  OR 27567355[PMID]
  OR 27225433[PMID]
  OR 25643098[PMID]
  OR 24606085[PMID]
  OR 24875398[PMID]
  OR 24581581[PMID]
  OR 19159403[PMID]
  OR 35568223[PMID]
  OR 36792027[PMID]
  OR 34630328[PMID]
)
""".strip()


DEFAULT_PUBMED_OVARIAN_CONTEXT_QUERY = r"""
(
  "Ovary"[Mesh]
  OR "Ovulation"[Mesh]
  OR "Anovulation"[Mesh]
  OR "Ovarian Follicle"[Mesh]
  OR "Follicular Phase"[Mesh]
  OR "Corpus Luteum"[Mesh]
  OR "Granulosa Cells"[Mesh]
  OR "Theca Cells"[Mesh]
  OR "Follicular Fluid"[Mesh]
  OR "Cumulus Cells"[Mesh]
  OR "Luteal Cells"[Mesh]

  OR ovary[Title/Abstract]
  OR ovarian[Title/Abstract]
  OR ovulation[Title/Abstract]
  OR ovulatory[Title/Abstract]
  OR anovulation[Title/Abstract]
  OR anovulatory[Title/Abstract]
  OR "ovarian follicle"[Title/Abstract]
  OR "ovarian follicles"[Title/Abstract]
  OR "follicular phase"[Title/Abstract]
  OR "corpus luteum"[Title/Abstract]
  OR "corpora lutea"[Title/Abstract]
  OR "granulosa cell"[Title/Abstract]
  OR "granulosa cells"[Title/Abstract]
  OR "theca cell"[Title/Abstract]
  OR "theca cells"[Title/Abstract]
  OR "cumulus cell"[Title/Abstract]
  OR "cumulus cells"[Title/Abstract]
  OR "cumulus oophorus"[Title/Abstract]
  OR "luteal cell"[Title/Abstract]
  OR "luteal cells"[Title/Abstract]
  OR "granulosa-lutein cell"[Title/Abstract]
  OR "granulosa-lutein cells"[Title/Abstract]
  OR "follicular fluid"[Title/Abstract]

  OR "ovarian stroma"[Title/Abstract]
  OR "ovarian stromal cell"[Title/Abstract]
  OR "ovarian stromal cells"[Title/Abstract]
  OR "ovarian fibroblast"[Title/Abstract]
  OR "ovarian fibroblasts"[Title/Abstract]
  OR "ovarian surface epithelium"[Title/Abstract]
  OR "ovarian surface epithelial cell"[Title/Abstract]
  OR "ovarian surface epithelial cells"[Title/Abstract]
)
""".strip()


DEFAULT_PUBMED_IMMUNE_INFLAMMATION_QUERY = r"""
(
  "Inflammation"[Mesh]
  OR "Immunity"[Mesh]
  OR "Cytokines"[Mesh]
  OR "Chemokines"[Mesh]
  OR "Leukocytes"[Mesh]
  OR "Lymphocytes"[Mesh]
  OR "T-Lymphocytes"[Mesh]
  OR "B-Lymphocytes"[Mesh]
  OR "Killer Cells, Natural"[Mesh]
  OR "Monocytes"[Mesh]
  OR "Macrophages"[Mesh]
  OR "Neutrophils"[Mesh]
  OR "Eosinophils"[Mesh]
  OR "Dendritic Cells"[Mesh]
  OR "Mast Cells"[Mesh]
  OR "Inflammasomes"[Mesh]
  OR "Prostaglandins"[Mesh]

  OR inflamm*[Title/Abstract]
  OR immune[Title/Abstract]
  OR immunity[Title/Abstract]
  OR immunologic*[Title/Abstract]
  OR immunomodulat*[Title/Abstract]
  OR immunoregul*[Title/Abstract]
  OR cytokine*[Title/Abstract]
  OR chemokine*[Title/Abstract]
  OR inflammasome*[Title/Abstract]
  OR interleukin*[Title/Abstract]
  OR interferon*[Title/Abstract]
  OR prostaglandin*[Title/Abstract]
  OR "tumor necrosis factor"[Title/Abstract]
  OR "tumour necrosis factor"[Title/Abstract]

  OR "immune cell"[Title/Abstract]
  OR "immune cells"[Title/Abstract]
  OR "immune infiltration"[Title/Abstract]
  OR "immune infiltrate"[Title/Abstract]
  OR "immune infiltrates"[Title/Abstract]
  OR "infiltrating immune cells"[Title/Abstract]

  OR leukocyte*[Title/Abstract]
  OR "white blood cell"[Title/Abstract]
  OR "white blood cells"[Title/Abstract]
  OR macrophage*[Title/Abstract]
  OR monocyte*[Title/Abstract]
  OR neutrophil*[Title/Abstract]
  OR granulocyte*[Title/Abstract]
  OR eosinophil*[Title/Abstract]
  OR lymphocyte*[Title/Abstract]

  OR "T cell"[Title/Abstract]
  OR "T cells"[Title/Abstract]
  OR "T-cell"[Title/Abstract]
  OR "T-cells"[Title/Abstract]
  OR "T lymphocyte"[Title/Abstract]
  OR "T lymphocytes"[Title/Abstract]
  OR "T-lymphocyte"[Title/Abstract]
  OR "T-lymphocytes"[Title/Abstract]
  OR Treg[Title/Abstract]
  OR Tregs[Title/Abstract]

  OR "B cell"[Title/Abstract]
  OR "B cells"[Title/Abstract]
  OR "B-cell"[Title/Abstract]
  OR "B-cells"[Title/Abstract]
  OR "B lymphocyte"[Title/Abstract]
  OR "B lymphocytes"[Title/Abstract]
  OR "B-lymphocyte"[Title/Abstract]
  OR "B-lymphocytes"[Title/Abstract]

  OR "natural killer cell"[Title/Abstract]
  OR "natural killer cells"[Title/Abstract]
  OR "NK cell"[Title/Abstract]
  OR "NK cells"[Title/Abstract]
  OR "dendritic cell"[Title/Abstract]
  OR "dendritic cells"[Title/Abstract]
  OR "mast cell"[Title/Abstract]
  OR "mast cells"[Title/Abstract]
)
""".strip()


DEFAULT_PUBMED_DIRECT_OVARIAN_INFLAMMATION_QUERY = r"""
(
  "Oophoritis"[Mesh]
  OR oophoritis[Title/Abstract]
  OR "ovarian inflammation"[Title/Abstract]
  OR "inflammation of the ovary"[Title/Abstract]
)
""".strip()


# ``female*`` is intentionally not used as human evidence because it also
# matches phrases such as "female mice" and "female rats".
DEFAULT_PUBMED_HUMAN_QUERY = r"""
(
  Humans[Mesh]
  OR human*[Title/Abstract]
  OR woman[Title/Abstract]
  OR women[Title/Abstract]
  OR patient*[Title/Abstract]
  OR participant*[Title/Abstract]
)
""".strip()


DEFAULT_PUBMED_TOPIC_QUERY = (
    f"("
    f"("
    f"({DEFAULT_PUBMED_OVARIAN_CONTEXT_QUERY}) "
    f"AND "
    f"({DEFAULT_PUBMED_IMMUNE_INFLAMMATION_QUERY})"
    f") "
    f"OR "
    f"({DEFAULT_PUBMED_DIRECT_OVARIAN_INFLAMMATION_QUERY})"
    f") "
    f"AND "
    f"({DEFAULT_PUBMED_HUMAN_QUERY})"
)


# This is the standard indexed animal-only pattern. Papers indexed with both
# Animals and Humans remain eligible, as requested.
DEFAULT_PUBMED_ANIMAL_EXCLUSION_QUERY = r"""
(
  Animals[Mesh]
  NOT Humans[Mesh]
)
""".strip()


# Exclude ovarian-neoplasm records and unindexed cancer-focused titles. The
# MeSH branches remove indexed ovarian/fallopian-tube neoplasm papers, while the
# title branches cover recent records that have not yet received MeSH indexing.
# Exact ovarian-tumor phrases are listed instead of generic ``tumor*`` so papers
# about tumor necrosis factor in normal ovarian biology are not removed.
DEFAULT_PUBMED_CANCER_EXCLUSION_QUERY = r"""
(
  "Ovarian Neoplasms"[Mesh]
  OR "Fallopian Tube Neoplasms"[Mesh]
  OR
  (
    "Peritoneal Neoplasms"[Mesh]
    AND
    (
      ovarian[Title/Abstract]
      OR "fallopian tube"[Title/Abstract]
      OR "primary peritoneal"[Title/Abstract]
      OR tubo-ovarian[Title/Abstract]
      OR tuboovarian[Title/Abstract]
    )
  )
  OR
  (
    (
      ovary[Title]
      OR ovarian[Title]
      OR "fallopian tube"[Title]
      OR "primary peritoneal"[Title]
      OR tubo-ovarian[Title]
      OR tuboovarian[Title]
    )
    AND
    (
      cancer*[Title]
      OR carcinoma*[Title]
      OR adenocarcinoma*[Title]
      OR neoplasm*[Title]
      OR malignan*[Title]
      OR oncolog*[Title]
      OR tumorigen*[Title]
      OR tumourigen*[Title]
    )
  )
  OR "fallopian tube tumor"[Title]
  OR "fallopian tube tumors"[Title]
  OR "fallopian tube tumour"[Title]
  OR "fallopian tube tumours"[Title]
  OR "primary peritoneal tumor"[Title]
  OR "primary peritoneal tumors"[Title]
  OR "primary peritoneal tumour"[Title]
  OR "primary peritoneal tumours"[Title]
  OR "granulosa cell tumor"[Title]
  OR "granulosa cell tumors"[Title]
  OR "granulosa cell tumour"[Title]
  OR "granulosa cell tumours"[Title]
  OR "high-grade serous carcinoma"[Title]
  OR "high grade serous carcinoma"[Title]
  OR "high-grade serous ovarian cancer"[Title]
  OR "high grade serous ovarian cancer"[Title]
  OR "high-grade serous ovarian carcinoma"[Title]
  OR "high grade serous ovarian carcinoma"[Title]
  OR "low-grade serous ovarian cancer"[Title]
  OR "low grade serous ovarian cancer"[Title]
  OR "low-grade serous ovarian carcinoma"[Title]
  OR "low grade serous ovarian carcinoma"[Title]
  OR EOC[Title]
  OR HGSOC[Title]
  OR LGSOC[Title]
)
""".strip()


# Retained as a combined compatibility constant for external imports and query
# introspection. The actual search expression below uses separate ``AND NOT``
# clauses so each exclusion remains easy to inspect.
DEFAULT_PUBMED_EXCLUSION_QUERY = (
    f"({DEFAULT_PUBMED_ANIMAL_EXCLUSION_QUERY}) OR "
    f"({DEFAULT_PUBMED_CANCER_EXCLUSION_QUERY})"
)


DEFAULT_PUBMED_QUERY = (
    f"("
    f"("
    f"({DEFAULT_PUBMED_PMID_QUERY}) "
    f"OR "
    f"({DEFAULT_PUBMED_TOPIC_QUERY})"
    f") "
    f"AND NOT ({DEFAULT_PUBMED_ANIMAL_EXCLUSION_QUERY}) "
    f"AND NOT ({DEFAULT_PUBMED_CANCER_EXCLUSION_QUERY})"
    f")"
)


# Literal entries already covered by the default exclusions. These values are
# used only to deduplicate and resolve conflicts in the mixed user-input field;
# the actual PubMed exclusions are defined above.
DEFAULT_PUBMED_EXCLUSION_TERMS: tuple[str, ...] = (
    "animals",
    "ovarian neoplasm",
    "ovarian neoplasms",
    "ovarian cancer",
    "ovarian cancers",
    "ovarian carcinoma",
    "ovarian carcinomas",
    "ovarian adenocarcinoma",
    "ovarian adenocarcinomas",
    "epithelial ovarian cancer",
    "epithelial ovarian carcinoma",
    "granulosa cell tumor",
    "granulosa cell tumors",
    "granulosa cell tumour",
    "granulosa cell tumours",
    "EOC",
    "HGSOC",
    "LGSOC",
    "high-grade serous carcinoma",
    "high grade serous carcinoma",
    "high-grade serous ovarian carcinoma",
    "high grade serous ovarian carcinoma",
    "high-grade serous ovarian cancer",
    "high grade serous ovarian cancer",
    "low-grade serous ovarian carcinoma",
    "low grade serous ovarian carcinoma",
    "low-grade serous ovarian cancer",
    "low grade serous ovarian cancer",
    "serous ovarian carcinoma",
    "serous ovarian cancer",
    "fallopian tube neoplasm",
    "fallopian tube neoplasms",
    "fallopian tube cancer",
    "fallopian tube carcinoma",
    "primary peritoneal neoplasm",
    "primary peritoneal neoplasms",
    "primary peritoneal cancer",
    "primary peritoneal carcinoma",
)


DEFAULT_QUERY_LABEL = (
    "Human ovarian and ovulatory immune/inflammatory biology, including ovarian "
    "somatic and immune cells, excluding animal-only and ovarian-neoplasm papers"
)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PMCID_RE = re.compile(r"(?i)(?:PMCID\s*[:=]?\s*)?PMC\s*(\d+)(?:\.\d+)?")
_MIXED_PMID_ITEM_RE = re.compile(
    r"(?i)^(?:PMID\s*[:=]?\s*)?(\d{1,9})(?:\s*\[\s*PMID\s*\])?$"
)
_MIXED_PMCID_ITEM_RE = re.compile(r"(?i)^PMC\s*(\d+)(?:\.\d+)?$")
_MIXED_PMCID_LABEL_ITEM_RE = re.compile(
    r"(?i)^PMCID\s*[:=]?\s*(?:PMC\s*)?(\d+)(?:\.\d+)?$"
)
_NOT_ITEM_RE = re.compile(r"(?i)^NOT(?:\s+|\s*:\s*)(.+)$")
_QUERY_FIELD_RE = re.compile(r"\[[^\]]+\]")
_QUOTED_QUERY_RE = re.compile(r'"([^"]+)"')
_QUERY_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\*?")
_QUERY_BOOLEAN_WORDS = {"and", "or", "not"}
_SIMPLE_KEYWORD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+/-]*\*?$")

SKIP_SECTION_EXACT = {
    "REF",
    "METHODS",
    "ABBR",
    "SUPPL",
    "COMP_INT",
    "CASE",
    "APPENDIX",
    "ACK_FUND",
}
SKIP_SECTION_SUBSTR = {
    "author",
    "auth",
    "affiliation",
    "correspond",
    "table",
    "tabel",
    "fig",
    "figure",
}
SKIP_TYPE_SUBSTR = {
    "table",
    "fig",
    "figure",
    "author",
    "affiliation",
    "correspond",
    "ref",
}


class RetrievalError(RuntimeError):
    """A concise, user-facing retrieval failure."""


@dataclass(frozen=True)
class ParsedUserInput:
    """Classified comma-separated entries before default-value deduplication."""

    items: tuple[str, ...]
    keywords: tuple[str, ...]
    exclusions: tuple[str, ...]
    pmids: tuple[str, ...]
    pmcids: tuple[str, ...]


@dataclass(frozen=True)
class EffectiveInputs:
    input_type: InputType
    raw_user_input: str
    user_keywords: tuple[str, ...]
    user_exclusions: tuple[str, ...]
    user_keyword_was_redundant: bool
    pmids: tuple[str, ...]
    pmcids: tuple[str, ...]
    user_keyword_count: int
    user_exclusion_count: int
    user_pmid_count: int
    user_pmcid_count: int
    duplicate_user_keyword_count: int
    duplicate_user_exclusion_count: int
    duplicate_user_pmid_count: int
    duplicate_user_pmcid_count: int


@dataclass(frozen=True)
class RetrievalResult:
    job_id: str
    summary_path: Path
    chunk_paths: tuple[Path, ...]
    stats: dict[str, Any]
    # New jobs stream ``chunk_paths`` as one download instead of storing a
    # duplicated per-job chunks file. This remains for callers that handled the
    # earlier result shape.
    chunks_path: Path | None = None


@dataclass(frozen=True)
class FulltextBatchResult:
    """In-memory outcome of the multi-source full-text stage."""

    documents: dict[str, bytes]
    unavailable_now: tuple[str, ...]
    failed_now: tuple[str, ...]
    requests_made: int
    service_error_batches: int
    document_sources: dict[str, str] = field(default_factory=dict)
    pubtator_requests: int = 0
    ncbi_bioc_requests: int = 0
    epmc_requests: int = 0


@dataclass(frozen=True)
class FulltextProgress:
    """Lightweight progress snapshot that never copies downloaded article text."""

    downloaded_count: int
    unavailable_count: int
    failed_count: int
    requests_made: int
    service_error_batches: int
    pubtator_requests: int = 0
    ncbi_bioc_requests: int = 0
    epmc_requests: int = 0


class RequestPacer:
    """Space request starts without sleeping more than necessary."""

    def __init__(self, minimum_delay: float) -> None:
        self.minimum_delay = max(0.0, float(minimum_delay))
        self._last_request_at = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.minimum_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()


class ProgressReporter:
    """Throttle database-facing callbacks while preserving stage changes."""

    def __init__(
        self,
        callback: ProgressCallback | None,
        *,
        minimum_interval: float = 0.8,
    ) -> None:
        self.callback = callback
        self.minimum_interval = minimum_interval
        self._last_emit = 0.0
        self._last_stage: str | None = None

    def emit(
        self,
        stage: str,
        progress: int,
        message: str,
        stats: dict[str, Any],
        *,
        force: bool = False,
    ) -> None:
        if self.callback is None:
            return
        now = time.monotonic()
        stage_changed = stage != self._last_stage
        if not force and not stage_changed and now - self._last_emit < self.minimum_interval:
            return
        self._last_emit = now
        self._last_stage = stage
        self.callback(stage, max(0, min(100, int(progress))), message, dict(stats))


# ---------------------------------------------------------------------------
# Normalization and user input
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u2028", "\n").replace("\u2029", "\n").replace("\u0085", "\n")
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def sanitize_query(value: str) -> str:
    text = clean_text(value).replace("\n", " ")
    return re.sub(r"\s+", " ", _CONTROL_RE.sub(" ", text)).strip()


def normalize_pmid(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    candidate = match.group(0).lstrip("0") or "0"
    return candidate if candidate != "0" and len(candidate) <= 9 else None


def normalize_pmcid(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    match = _PMCID_RE.search(text)
    if match:
        return f"PMC{match.group(1)}"
    bare = re.fullmatch(r"\s*(?:PMCID\s*[:=]?\s*)?(\d+)(?:\.\d+)?\s*", text, re.I)
    return f"PMC{bare.group(1)}" if bare else None


def normalize_doi(value: Any) -> str | None:
    if value is None:
        return None
    doi = str(value).strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = doi.removeprefix("doi:").strip()
    return doi or None


def unique_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def split_mixed_user_input(raw: str) -> tuple[str, ...]:
    """Split one comma-separated field while respecting quoted phrases."""

    text = clean_text(raw).replace("\n", " ")
    text = _CONTROL_RE.sub(" ", text).strip()
    if not text:
        return ()

    try:
        row = next(csv.reader([text], skipinitialspace=True, strict=True))
    except csv.Error as exc:
        raise RetrievalError(
            "The search input could not be parsed. Separate entries with commas "
            "and close any quotation marks."
        ) from exc

    return tuple(
        cleaned
        for value in row
        if (cleaned := sanitize_query(value).strip())
    )


def _parse_mixed_pmid_item(item: str) -> str | None:
    match = _MIXED_PMID_ITEM_RE.fullmatch(item)
    return normalize_pmid(match.group(1)) if match else None


def _parse_mixed_pmcid_item(item: str) -> str | None:
    match = _MIXED_PMCID_ITEM_RE.fullmatch(item)
    if not match:
        match = _MIXED_PMCID_LABEL_ITEM_RE.fullmatch(item)
    return f"PMC{match.group(1)}" if match else None


def normalize_keyword_item(value: str) -> str:
    """Normalize one literal user keyword or phrase without making it a query."""

    keyword = sanitize_query(value).strip()
    if len(keyword) >= 2 and keyword[0] == keyword[-1] and keyword[0] in {'"', "'"}:
        keyword = keyword[1:-1].strip()
    keyword = sanitize_query(keyword)
    if not keyword or not re.search(r"[A-Za-z0-9]", keyword):
        raise RetrievalError("Each keyword or exclusion must contain a word or number.")
    return keyword


def parse_mixed_user_input(raw: str) -> ParsedUserInput:
    """Classify comma-separated keywords, NOT terms, PMIDs, and PMCIDs.

    Rules:
    - ``not cancer`` is an exclusion keyword;
    - ``PMID:12345678`` or bare ``12345678`` is a PMID;
    - ``PMC1234567`` or ``PMCID:PMC1234567`` is a PMCID;
    - every other entry is a positive keyword or phrase.
    """

    items = split_mixed_user_input(raw)
    keywords: list[str] = []
    exclusions: list[str] = []
    pmids: list[str] = []
    pmcids: list[str] = []

    for item in items:
        if item.casefold() == "not" or re.fullmatch(r"(?i)NOT\s*:", item):
            raise RetrievalError(
                "An exclusion beginning with 'not' must include a term, for example "
                "'not cancer'."
            )

        not_match = _NOT_ITEM_RE.fullmatch(item)
        if not_match:
            excluded = normalize_keyword_item(not_match.group(1))
            if _parse_mixed_pmid_item(excluded) or _parse_mixed_pmcid_item(excluded):
                raise RetrievalError(
                    "The 'not' prefix can be used only with keywords, not with PMID "
                    "or PMCID values."
                )
            exclusions.append(excluded)
            continue

        pmcid = _parse_mixed_pmcid_item(item)
        if pmcid:
            pmcids.append(pmcid)
            continue

        pmid = _parse_mixed_pmid_item(item)
        if pmid:
            pmids.append(pmid)
            continue

        # A bare number is always interpreted as a PMID in mixed input. Do not
        # silently convert an invalid long numeric identifier into a keyword.
        if re.fullmatch(r"\d+", item):
            raise RetrievalError(
                f"'{item}' is not a valid PMID. A bare PMID must contain 1 to 9 "
                "digits. Prefix PMC identifiers with 'PMC' or 'PMCID:'."
            )

        if re.match(r"(?i)^(?:PMID|PMCID)\b|^PMC", item):
            raise RetrievalError(
                f"'{item}' looks like an identifier but is not a valid PMID or PMCID."
            )

        keywords.append(normalize_keyword_item(item))

    return ParsedUserInput(
        items=items,
        keywords=tuple(keywords),
        exclusions=tuple(exclusions),
        pmids=tuple(pmids),
        pmcids=tuple(pmcids),
    )


def extract_keyword_atoms(query: str) -> set[str]:
    """Return comparable terms without PubMed field tags or booleans."""

    text = sanitize_query(query).lower()
    text = _PMCID_RE.sub(" ", text)
    text = _QUERY_FIELD_RE.sub(" ", text)
    atoms: set[str] = set()

    for match in _QUOTED_QUERY_RE.finditer(text):
        phrase = re.sub(r"\s+", " ", match.group(1)).strip()
        if phrase:
            atoms.add(phrase)
            atoms.update(
                word
                for word in _QUERY_WORD_RE.findall(phrase)
                if word not in _QUERY_BOOLEAN_WORDS
            )

    unquoted = _QUOTED_QUERY_RE.sub(" ", text)
    atoms.update(
        word
        for word in _QUERY_WORD_RE.findall(unquoted)
        if word not in _QUERY_BOOLEAN_WORDS
    )
    return atoms


def keyword_atom_is_covered(atom: str, default_atoms: set[str]) -> bool:
    if atom in default_atoms:
        return True
    if " " in atom:
        return False

    plain_atom = atom.removesuffix("*")
    for default_atom in default_atoms:
        if " " in default_atom:
            continue
        plain_default = default_atom.removesuffix("*")
        if default_atom.endswith("*") and plain_atom.startswith(plain_default):
            return True
        if atom.endswith("*") and plain_default.startswith(plain_atom):
            return True
    return False


def keyword_item_is_covered(item: str, default_query: str) -> bool:
    user_atoms = extract_keyword_atoms(item)
    if not user_atoms:
        return False
    default_atoms = extract_keyword_atoms(default_query)
    return all(keyword_atom_is_covered(atom, default_atoms) for atom in user_atoms)


def _keyword_identity(value: str) -> str:
    return re.sub(r"\s+", " ", normalize_keyword_item(value).casefold()).strip()


def _select_new_keyword_items(
    values: Sequence[str],
    *,
    default_query: str | None = None,
    default_items: Sequence[str] = (),
) -> tuple[tuple[str, ...], int]:
    selected: list[str] = []
    seen: set[str] = set()
    duplicate_count = 0
    default_identities = {_keyword_identity(item) for item in default_items}

    for value in values:
        identity = _keyword_identity(value)
        covered_by_default_query = bool(
            default_query and keyword_item_is_covered(value, default_query)
        )
        if identity in seen or identity in default_identities or covered_by_default_query:
            duplicate_count += 1
            continue
        seen.add(identity)
        selected.append(value)

    return tuple(selected), duplicate_count


def _select_new_identifiers(
    values: Sequence[str],
    *,
    defaults: Sequence[str],
) -> tuple[tuple[str, ...], int]:
    selected: list[str] = []
    seen: set[str] = set()
    default_set = set(defaults)
    duplicate_count = 0

    for value in values:
        if value in default_set or value in seen:
            duplicate_count += 1
            continue
        seen.add(value)
        selected.append(value)

    return tuple(selected), duplicate_count


def keyword_to_pubmed_clause(keyword: str) -> str:
    """Convert one literal keyword item to a Title/Abstract PubMed clause."""

    value = normalize_keyword_item(keyword)
    if _SIMPLE_KEYWORD_RE.fullmatch(value):
        return f"{value}[Title/Abstract]"

    # Comma-separated items are intentionally treated as literal phrases rather
    # than advanced Boolean expressions. This keeps the mixed input predictable.
    phrase = sanitize_query(value.replace('"', " "))
    return f'"{phrase}"[Title/Abstract]'


def build_augmented_pubmed_query(
    user_keywords: Sequence[str] = (),
    user_exclusions: Sequence[str] = (),
) -> str:
    """Build the default ovarian query plus optional user additions.

    Positive user terms broaden the immune/inflammatory concept block while
    retaining ovarian and human context. This prevents a term such as
    ``macrophage`` from becoming a separate, general PubMed search. User
    exclusions apply to the discovered topic branch; the curated PMID branch is
    retained as an intentional seed set. Animal-only and ovarian-neoplasm
    exclusions apply to the complete ESearch expression.
    """

    if not user_keywords and not user_exclusions:
        return DEFAULT_PUBMED_QUERY

    immune_query = f"({DEFAULT_PUBMED_IMMUNE_INFLAMMATION_QUERY})"
    if user_keywords:
        user_query = " OR ".join(
            f"({keyword_to_pubmed_clause(item)})" for item in user_keywords
        )
        immune_query = f"({immune_query}) OR ({user_query})"

    topic_query = (
        f"("
        f"("
        f"({DEFAULT_PUBMED_OVARIAN_CONTEXT_QUERY}) "
        f"AND "
        f"({immune_query})"
        f") "
        f"OR "
        f"({DEFAULT_PUBMED_DIRECT_OVARIAN_INFLAMMATION_QUERY})"
        f") "
        f"AND "
        f"({DEFAULT_PUBMED_HUMAN_QUERY})"
    )

    topic_branch = f"({topic_query})"
    if user_exclusions:
        user_exclusion_query = " OR ".join(
            f"({keyword_to_pubmed_clause(item)})" for item in user_exclusions
        )
        topic_branch = f"({topic_branch}) AND NOT ({user_exclusion_query})"

    return (
        f"("
        f"("
        f"({DEFAULT_PUBMED_PMID_QUERY}) "
        f"OR "
        f"({topic_branch})"
        f") "
        f"AND NOT ({DEFAULT_PUBMED_ANIMAL_EXCLUSION_QUERY}) "
        f"AND NOT ({DEFAULT_PUBMED_CANCER_EXCLUSION_QUERY})"
        f")"
    )


def build_effective_inputs(input_type: InputType, raw_user_input: str) -> EffectiveInputs:
    """Combine defaults with one mixed comma-separated user field.

    ``input_type`` is retained for compatibility with the existing API and
    frontend, but all values are parsed in mixed mode regardless of the selected
    legacy type.
    """

    raw = sanitize_query(raw_user_input)
    parsed = parse_mixed_user_input(raw_user_input)

    # Resolve exclusions first. An exclusion has priority when the same term is
    # also submitted as a positive keyword, and default exclusions must never be
    # reintroduced through the positive side of the query.
    added_exclusions, duplicate_exclusion_count = _select_new_keyword_items(
        parsed.exclusions,
        default_items=DEFAULT_PUBMED_EXCLUSION_TERMS,
    )
    added_keywords, duplicate_keyword_count = _select_new_keyword_items(
        parsed.keywords,
        default_query=DEFAULT_PUBMED_TOPIC_QUERY,
    )

    # Do not put the same literal term on both sides of the query. We compare
    # explicit items rather than all atoms in DEFAULT_PUBMED_EXCLUSION_QUERY;
    # this matters because a compound default rule such as
    # ``(ovarian AND cancer*)`` must not make a positive generic ``cancer`` item
    # look globally blocked.
    blocked_keyword_identities = {
        _keyword_identity(item)
        for item in (*DEFAULT_PUBMED_EXCLUSION_TERMS, *added_exclusions)
    }
    filtered_keywords: list[str] = []
    for keyword in added_keywords:
        if _keyword_identity(keyword) in blocked_keyword_identities:
            duplicate_keyword_count += 1
            continue
        filtered_keywords.append(keyword)
    added_keywords = tuple(filtered_keywords)

    added_pmids, duplicate_pmid_count = _select_new_identifiers(
        parsed.pmids,
        defaults=DEFAULT_PMIDS,
    )
    added_pmcids, duplicate_pmcid_count = _select_new_identifiers(
        parsed.pmcids,
        defaults=DEFAULT_PMCIDS,
    )

    had_keyword_entries = bool(parsed.keywords or parsed.exclusions)
    keyword_was_redundant = (
        had_keyword_entries and not added_keywords and not added_exclusions
    )

    return EffectiveInputs(
        input_type=input_type,
        raw_user_input=raw,
        user_keywords=added_keywords,
        user_exclusions=added_exclusions,
        user_keyword_was_redundant=keyword_was_redundant,
        pmids=(*DEFAULT_PMIDS, *added_pmids),
        pmcids=(*DEFAULT_PMCIDS, *added_pmcids),
        user_keyword_count=len(added_keywords),
        user_exclusion_count=len(added_exclusions),
        user_pmid_count=len(added_pmids),
        user_pmcid_count=len(added_pmcids),
        duplicate_user_keyword_count=duplicate_keyword_count,
        duplicate_user_exclusion_count=duplicate_exclusion_count,
        duplicate_user_pmid_count=duplicate_pmid_count,
        duplicate_user_pmcid_count=duplicate_pmcid_count,
    )


def defaults_payload(keyword_limit: int = DEFAULT_KEYWORD_LIMIT) -> dict[str, Any]:
    """Return browser-facing retrieval defaults."""

    safe_limit = max(1, min(int(keyword_limit), PUBMED_ESEARCH_API_CAP))
    return {
        "keyword_query_label": DEFAULT_QUERY_LABEL,
        "default_pmid_count": len(DEFAULT_PMIDS),
        "default_pmcid_count": len(DEFAULT_PMCIDS),
        "keyword_result_limit": safe_limit,
        "pubmed_result_mode": "limited_relevance_results",
        "user_inputs_are_added": True,
        "user_input_mode": USER_INPUT_MODE,
        "input_parser_version": INPUT_PARSER_VERSION,
        "user_input_separator": ",",
        "user_exclusion_prefix": "not ",
        "bare_numeric_identifier_type": "pmid",
        "pmcid_prefix_required_in_mixed_input": True,
        "user_input_example": (
            "macrophage, granulosa cells, not cancer, PMID:12345678, PMC1234567"
        ),
        "keyword_search_mode": "single_augmented_query_with_exclusions",
        "metadata_reuse": True,
        "metadata_retry_count": METADATA_HTTP_ATTEMPTS - 1,
        "fulltext_batch_retrieval": True,
        "fulltext_fallback_retrieval": True,
        "fulltext_attempt_mode": "retry_transient_refresh_negative_cache",
        "fulltext_failed_requests_are_retried": True,
        "fulltext_negative_cache_days": FULLTEXT_NEGATIVE_CACHE_DAYS,
        "fulltext_retriever_version": FULLTEXT_RETRIEVER_VERSION,
        "fulltext_storage": "compact_bioc_json_gzip",
        "chunk_storage": "jsonl_gzip_streamed_as_ndjson",
        "chunk_reuse_mode": "paper_id_with_fulltext_upgrade",
        "paper_change_checks": False,
    }


# ---------------------------------------------------------------------------
# Identifier-aware JSONL corpus
# ---------------------------------------------------------------------------


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    # Author fields are not part of this application's data model. This is an
    # in-memory normalization only; cached chunk files are never scanned.
    for unsupported_field in ("author", "authors", "author_string", "authorString"):
        normalized.pop(unsupported_field, None)

    normalized["pmid"] = normalize_pmid(record.get("pmid"))
    normalized["pmcid"] = normalize_pmcid(record.get("pmcid"))
    normalized["doi"] = normalize_doi(record.get("doi"))
    for field in ("title", "abstract", "journal"):
        normalized[field] = clean_text(record.get(field))
    normalized["pub_year"] = clean_text(record.get("pub_year")) or None

    sources = record.get("sources") or []
    if isinstance(sources, str):
        sources = [sources]
    source = clean_text(record.get("source"))
    normalized["sources"] = sorted(
        {clean_text(item) for item in (*sources, source) if clean_text(item)}
    )

    if "fulltext_checked" in record:
        normalized["fulltext_checked"] = bool(record.get("fulltext_checked"))
    if clean_text(record.get("fulltext_status")):
        normalized["fulltext_status"] = clean_text(record.get("fulltext_status"))
    if clean_text(record.get("fulltext_checked_at")):
        normalized["fulltext_checked_at"] = clean_text(record.get("fulltext_checked_at"))
    if clean_text(record.get("fulltext_retriever_version")):
        normalized["fulltext_retriever_version"] = clean_text(
            record.get("fulltext_retriever_version")
        )
    if clean_text(record.get("fulltext_source")):
        normalized["fulltext_source"] = clean_text(record.get("fulltext_source"))
    return normalized


def merge_records(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = normalize_record(existing)
    candidate = normalize_record(incoming)

    for field in ("pmid", "pmcid", "doi", "journal", "pub_year"):
        if not merged.get(field) and candidate.get(field):
            merged[field] = candidate[field]
    if not merged.get("title") and candidate.get("title"):
        merged["title"] = candidate["title"]
    if len(candidate.get("abstract") or "") > len(merged.get("abstract") or ""):
        merged["abstract"] = candidate["abstract"]

    merged["sources"] = sorted(
        set(merged.get("sources") or []) | set(candidate.get("sources") or [])
    )
    for field in (
        "fulltext_path",
        "fulltext_bytes",
        "fulltext_checked",
        "fulltext_status",
        "fulltext_checked_at",
        "fulltext_retriever_version",
        "fulltext_source",
        "fulltext_uncompressed_bytes",
    ):
        if candidate.get(field) not in (None, ""):
            merged[field] = candidate[field]

    merged.setdefault("first_seen_at", existing.get("first_seen_at") or utc_now_iso())
    merged["last_updated_at"] = utc_now_iso()
    if existing.get("canonical_id"):
        merged["canonical_id"] = existing["canonical_id"]
    return merged


class CorpusStore:
    """Small JSONL store with PMID/PMCID/DOI deduplication."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, dict[str, Any]] = {}
        self.id_index: dict[str, str] = {}
        self._load()

    @staticmethod
    def identifier_keys(record: dict[str, Any]) -> list[str]:
        return [
            f"{field}:{record[field]}"
            for field in ("pmid", "pmcid", "doi")
            if record.get(field)
        ]

    @staticmethod
    def new_canonical_id(record: dict[str, Any]) -> str:
        for field in ("doi", "pmcid", "pmid"):
            if record.get(field):
                return f"{field}:{record[field]}"
        source = clean_text(record.get("source")) or "unknown"
        source_id = clean_text(record.get("source_id")) or hashlib.sha1(
            json.dumps(record, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        return f"source:{source}:{source_id}"

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise TypeError("Corpus rows must be JSON objects.")
                    record = normalize_record(payload)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Skipping malformed corpus line in %s", self.path)
                    continue
                canonical_id = clean_text(record.get("canonical_id"))
                if not canonical_id:
                    canonical_id = self.new_canonical_id(record)
                    record["canonical_id"] = canonical_id
                if canonical_id in self.records:
                    self.records[canonical_id] = merge_records(
                        self.records[canonical_id], record
                    )
                else:
                    self.records[canonical_id] = record
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self.id_index.clear()
        for canonical_id, record in self.records.items():
            self._index_record(canonical_id, record)

    def _index_record(self, canonical_id: str, record: dict[str, Any]) -> None:
        for key in self.identifier_keys(record):
            self.id_index[key] = canonical_id

    def _drop_record_index(self, canonical_id: str, record: dict[str, Any]) -> None:
        for key in self.identifier_keys(record):
            if self.id_index.get(key) == canonical_id:
                self.id_index.pop(key, None)

    def add_if_new(
        self,
        incoming: dict[str, Any],
        *,
        prefer_existing_ids: set[str] | None = None,
    ) -> tuple[str, bool]:
        """Add a paper once and retain existing metadata on later matches."""

        record = normalize_record(incoming)
        matches = unique_preserving_order(
            self.id_index[key]
            for key in self.identifier_keys(record)
            if key in self.id_index
        )

        if not matches:
            canonical_id = self.new_canonical_id(record)
            base_id = canonical_id
            suffix = 2
            while canonical_id in self.records:
                canonical_id = f"{base_id}:{suffix}"
                suffix += 1
            now = utc_now_iso()
            record["canonical_id"] = canonical_id
            record.setdefault("first_seen_at", now)
            record["last_updated_at"] = now
            self.records[canonical_id] = record
            self._index_record(canonical_id, record)
            return canonical_id, True

        preferred = prefer_existing_ids or set()
        primary_id = next(
            (candidate for candidate in matches if candidate in preferred),
            matches[0],
        )
        primary = self.records[primary_id]
        self._drop_record_index(primary_id, primary)

        # Merge every matching record, including a match that precedes the chosen
        # primary in ``matches``. This repairs older split identities and carries
        # forward all full-text cache fields, not only the original subset.
        for other_id in matches:
            if other_id == primary_id or other_id not in self.records:
                continue
            other = self.records.pop(other_id)
            self._drop_record_index(other_id, other)
            primary = merge_records(primary, other)

        # Later metadata lookups must be able to enrich an earlier explicit-ID
        # placeholder. ``merge_records`` keeps existing values while filling a
        # missing title/journal and preferring a longer abstract.
        primary = merge_records(primary, record)
        primary["canonical_id"] = primary_id
        self.records[primary_id] = primary
        self._index_record(primary_id, primary)
        return primary_id, False

    def upsert(self, incoming: dict[str, Any]) -> str:
        canonical_id, _is_new = self.add_if_new(incoming)
        return canonical_id

    def get(self, canonical_id: str) -> dict[str, Any] | None:
        return self.records.get(canonical_id)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for canonical_id in sorted(self.records):
                handle.write(json.dumps(self.records[canonical_id], ensure_ascii=False) + "\n")
        temporary.replace(self.path)


# ---------------------------------------------------------------------------
# HTTP clients and PubMed/Europe PMC parsing
# ---------------------------------------------------------------------------


def build_retry_session(user_agent: str) -> requests.Session:
    """Create a pooled session; retry policy is handled explicitly below.

    Keeping retries in one helper avoids accidental multiplication of adapter
    retries and application retries, and makes transient failures testable.
    """

    adapter = HTTPAdapter(
        max_retries=Retry(total=0),
        pool_connections=4,
        pool_maxsize=4,
    )
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json, application/xml, text/xml;q=0.9, */*;q=0.8",
        }
    )
    return session


def ncbi_params(email: str, tool: str, api_key: str | None) -> dict[str, str]:
    params = {"email": email, "tool": tool}
    if api_key:
        params["api_key"] = api_key
    return params


def raise_for_response(response: requests.Response, context: str) -> None:
    if response.status_code < 400:
        return
    preview = clean_text(response.text[:500])
    raise RetrievalError(
        f"{context} returned HTTP {response.status_code}. "
        f"{preview or 'No response details were provided.'}"
    )


def _retry_after_seconds(response: requests.Response | None) -> float | None:
    if response is None:
        return None
    raw_value = clean_text(response.headers.get("Retry-After"))
    if not raw_value:
        return None
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return None


def perform_request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    context: str,
    timeout: int,
    pacer: RequestPacer | None = None,
    max_attempts: int = METADATA_HTTP_ATTEMPTS,
    retry_statuses: Iterable[int] = TRANSIENT_HTTP_STATUSES,
    accepted_statuses: Iterable[int] = (),
    **kwargs: Any,
) -> requests.Response:
    """Perform one HTTP request with bounded retries for transient failures.

    Connection errors, truncated/chunked responses, timeouts, and selected HTTP
    statuses are retried with 1, 2, ... seconds of backoff. Client errors are
    returned only when explicitly listed in ``accepted_statuses``; otherwise
    they fail immediately with a concise ``RetrievalError``.
    """

    attempts = max(1, int(max_attempts))
    retryable = {int(value) for value in retry_statuses}
    accepted = {int(value) for value in accepted_statuses}
    last_error: Exception | None = None
    last_response: requests.Response | None = None

    for attempt in range(1, attempts + 1):
        if pacer is not None:
            pacer.wait()
        try:
            response = session.request(
                method.upper(),
                url,
                timeout=timeout,
                **kwargs,
            )
            last_response = response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(float(attempt))
            continue

        if response.status_code in accepted or response.status_code < 400:
            return response
        if response.status_code not in retryable:
            raise_for_response(response, context)

        last_error = RetrievalError(
            f"{context} returned transient HTTP {response.status_code}."
        )
        if attempt >= attempts:
            break
        delay = _retry_after_seconds(response)
        time.sleep(delay if delay is not None else float(attempt))

    detail = clean_text(str(last_error))
    if not detail and last_response is not None:
        detail = f"HTTP {last_response.status_code}"
    raise RetrievalError(
        f"{context} failed after {attempts} attempts."
        + (f" {detail}" if detail else "")
    ) from last_error


def _request_esearch(
    session: requests.Session,
    pacer: RequestPacer,
    *,
    query: str,
    email: str,
    tool: str,
    api_key: str | None,
    timeout: int,
    retmax: int,
) -> tuple[int, tuple[str, ...]]:
    """Run one limited PubMed ESearch request and decode its JSON response."""

    safe_limit = max(1, min(int(retmax), PUBMED_ESEARCH_API_CAP))
    params: dict[str, str] = {
        "db": "pubmed",
        "term": sanitize_query(query),
        "retmode": "json",
        "retmax": str(safe_limit),
        "sort": "relevance",
        **ncbi_params(email, tool, api_key),
    }
    # Always use POST for ESearch. The built-in Boolean query is several
    # thousand characters before URL encoding and exceeds common safe URL lengths
    # after encoding. The previous raw-length threshold therefore selected GET
    # and allowed the end of the query -- including exclusion filters -- to be
    # truncated by an intermediary. A form-encoded POST preserves the complete
    # PubMed expression for both the default query and user-augmented queries.
    response = perform_request_with_retries(
        session,
        "POST",
        PUBMED_ESEARCH,
        context="PubMed search",
        timeout=timeout,
        pacer=pacer,
        data=params,
    )
    try:
        result = response.json()["esearchresult"]
        total = int(result.get("count", 0))
        ids = unique_preserving_order(
            normalize_pmid(value) or "" for value in result.get("idlist", [])
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RetrievalError("PubMed returned an unreadable search response.") from exc
    return total, ids


def search_pubmed_limited(
    session: requests.Session,
    pacer: RequestPacer,
    *,
    query: str,
    limit: int,
    email: str,
    tool: str,
    api_key: str | None,
    timeout: int,
) -> tuple[int, tuple[str, ...]]:
    """Return the total hit count and only the selected limited PMID set."""

    return _request_esearch(
        session,
        pacer,
        query=query,
        email=email,
        tool=tool,
        api_key=api_key,
        timeout=timeout,
        retmax=limit,
    )


def xml_text(element: ET.Element | None) -> str:
    return clean_text("".join(element.itertext())) if element is not None else ""


def parse_pubmed_xml(xml_bytes: bytes) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise RetrievalError("PubMed returned malformed XML metadata.") from exc

    records: list[dict[str, Any]] = []
    for article_root in root.findall(".//PubmedArticle"):
        citation = article_root.find("./MedlineCitation")
        article = article_root.find("./MedlineCitation/Article")
        if citation is None:
            continue

        pmid = normalize_pmid(citation.findtext("PMID"))
        title = xml_text(article.find("ArticleTitle") if article is not None else None)

        abstract_parts: list[str] = []
        if article is not None:
            for abstract_element in article.findall("./Abstract/AbstractText"):
                text = xml_text(abstract_element)
                if not text:
                    continue
                label = clean_text(
                    abstract_element.attrib.get("Label")
                    or abstract_element.attrib.get("NlmCategory")
                )
                abstract_parts.append(f"{label}: {text}" if label else text)

        journal = ""
        pub_year: str | None = None
        if article is not None:
            journal_element = article.find("Journal")
            if journal_element is not None:
                journal = xml_text(journal_element.find("Title"))
                year = xml_text(journal_element.find("./JournalIssue/PubDate/Year"))
                medline_date = xml_text(
                    journal_element.find("./JournalIssue/PubDate/MedlineDate")
                )
                if year:
                    pub_year = year
                elif medline_date:
                    match = re.search(r"\b(18|19|20)\d{2}\b", medline_date)
                    pub_year = match.group(0) if match else None


        identifiers: dict[str, str] = {}
        for identifier in article_root.findall("./PubmedData/ArticleIdList/ArticleId"):
            id_type = clean_text(identifier.attrib.get("IdType")).lower()
            value = clean_text(identifier.text)
            if id_type and value:
                identifiers[id_type] = value

        records.append(
            {
                "source": "pubmed",
                "source_id": pmid or "",
                "pmid": pmid,
                "pmcid": normalize_pmcid(identifiers.get("pmc")),
                "doi": normalize_doi(identifiers.get("doi")),
                "title": title,
                "abstract": "\n\n".join(abstract_parts),
                "journal": journal,
                "pub_year": pub_year,
                "sources": ["pubmed"],
            }
        )
    return records


def batched(values: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    safe_size = max(1, int(size))
    for start in range(0, len(values), safe_size):
        yield values[start : start + safe_size]


def fetch_pubmed_records(
    session: requests.Session,
    pacer: RequestPacer,
    *,
    pmids: Sequence[str],
    batch_size: int,
    email: str,
    tool: str,
    api_key: str | None,
    timeout: int,
) -> Iterator[tuple[list[dict[str, Any]], int, int]]:
    total = len(pmids)
    processed = 0
    for batch in batched(pmids, min(max(1, batch_size), 200)):
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            **ncbi_params(email, tool, api_key),
        }
        response = perform_request_with_retries(
            session,
            "GET",
            PUBMED_EFETCH,
            context="PubMed metadata download",
            timeout=timeout,
            pacer=pacer,
            params=params,
        )
        records = parse_pubmed_xml(response.content)
        processed += len(batch)
        yield records, processed, total


def fetch_epmc_records_for_pmcids(
    session: requests.Session,
    *,
    pmcids: Sequence[str],
    timeout: int,
) -> Iterator[list[dict[str, Any]]]:
    for batch in batched(pmcids, 40):
        params = {
            "query": " OR ".join(f'PMCID:"{pmcid}"' for pmcid in batch),
            "format": "json",
            "resultType": "core",
            "pageSize": str(len(batch)),
            "synonym": "FALSE",
        }
        response = perform_request_with_retries(
            session,
            "GET",
            EPMC_SEARCH,
            context="Europe PMC identifier lookup",
            timeout=timeout,
            params=params,
        )
        try:
            results = (response.json().get("resultList") or {}).get("result") or []
        except (ValueError, AttributeError, json.JSONDecodeError) as exc:
            raise RetrievalError("Europe PMC returned unreadable identifier metadata.") from exc

        yield [
            {
                "source": "europepmc",
                "source_id": clean_text(result.get("id")),
                "pmid": normalize_pmid(result.get("pmid")),
                "pmcid": normalize_pmcid(result.get("pmcid")),
                "doi": normalize_doi(result.get("doi")),
                "title": clean_text(result.get("title")),
                "abstract": clean_text(result.get("abstractText")),
                "journal": clean_text(result.get("journalTitle")),
                "pub_year": clean_text(result.get("pubYear")) or None,
                "sources": ["europepmc"],
            }
            for result in results
        ]


# ---------------------------------------------------------------------------
# PMC BioC full text and chunks
# ---------------------------------------------------------------------------


def collect_bioc_documents(value: Any) -> list[dict[str, Any]]:
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
        for key in ("collection", "documents", "document"):
            if key in item:
                visit(item[key])

    visit(value)
    return documents


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def atomic_write_gzip_bytes(
    path: Path,
    data: bytes,
    *,
    compresslevel: int = 6,
) -> int:
    """Write deterministic gzip data atomically and return compressed bytes."""

    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=buffer,
        compresslevel=max(1, min(int(compresslevel), 9)),
        mtime=0,
    ) as handle:
        handle.write(data)
    compressed = buffer.getvalue()
    atomic_write_bytes(path, compressed)
    return len(compressed)


def build_fulltext_session(user_agent: str) -> requests.Session:
    """Create a pooled full-text session with explicit application retries."""

    adapter = HTTPAdapter(
        max_retries=Retry(total=0),
        pool_connections=4,
        pool_maxsize=4,
    )
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": (
                "application/json, application/x-ndjson, application/xml, "
                "text/xml;q=0.9, */*;q=0.8"
            ),
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def _decode_json_values(content: bytes) -> list[Any]:
    """Decode JSON, NDJSON, or several adjacent JSON values."""

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The full-text response was not UTF-8 JSON.") from exc

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


def _passage_infons(passage: dict[str, Any]) -> dict[str, Any]:
    infons = passage.get("infons")
    return infons if isinstance(infons, dict) else {}


def _passage_section_type(passage: dict[str, Any]) -> str:
    infons = _passage_infons(passage)
    for key in (
        "section_type",
        "sectionType",
        "section",
        "section_name",
        "sectionName",
    ):
        value = clean_text(infons.get(key))
        if value:
            return value
    return "UNKNOWN"


def _passage_type(passage: dict[str, Any]) -> str:
    infons = _passage_infons(passage)
    for key in ("type", "passage_type", "passageType"):
        value = clean_text(infons.get(key))
        if value:
            return value
    return ""


def _document_has_fulltext(document: dict[str, Any]) -> bool:
    """Return true only when a BioC document contains body-level text."""

    nonempty_passages = 0
    total_characters = 0
    unknown_characters = 0
    body_types = {"paragraph", "paragrpah", "body", "text", "section", "subsection"}
    nonbody_sections = {"", "UNKNOWN", "TITLE", "ABSTRACT", "FRONT", "META"}

    for passage in document.get("passages") or document.get("passage") or []:
        if not isinstance(passage, dict):
            continue
        raw_text = passage.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
        nonempty_passages += 1
        total_characters += len(raw_text)
        section = _passage_section_type(passage).upper()
        passage_type = _passage_type(passage).lower()

        if section in {"TITLE", "ABSTRACT", "FRONT", "META"}:
            continue
        if passage_type in body_types:
            return True
        if section not in nonbody_sections:
            return True
        if section == "UNKNOWN" and passage_type not in {"title", "abstract", "front"}:
            unknown_characters += len(raw_text)

    # Some BioC producers omit section/type infons. Multiple substantial
    # passages are still distinguishable from a title-and-abstract response.
    return (
        nonempty_passages >= 3
        and total_characters >= 1000
        and unknown_characters >= 400
    )


def _strict_pmcid(value: Any, *, key_implies_pmcid: bool = False) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    match = re.search(r"(?i)\bPMC\s*(\d+)\b", text)
    if match:
        return f"PMC{match.group(1)}"
    if key_implies_pmcid and re.fullmatch(r"\d+", text):
        return f"PMC{text}"
    return None


def _identifier_containers(document: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield document
    infons = document.get("infons")
    if isinstance(infons, dict):
        yield infons
    for passage in document.get("passages") or document.get("passage") or []:
        if not isinstance(passage, dict):
            continue
        yield passage
        passage_infons = passage.get("infons")
        if isinstance(passage_infons, dict):
            yield passage_infons


def _document_pmcid(
    document: dict[str, Any],
    requested: set[str],
    pmid_to_pmcid: dict[str, str] | None = None,
) -> str | None:
    """Resolve a requested PMCID from either PMC- or PMID-labelled BioC data."""

    normalized_pmid_map = {
        pmid: pmcid
        for raw_pmid, raw_pmcid in (pmid_to_pmcid or {}).items()
        if (pmid := normalize_pmid(raw_pmid))
        and (pmcid := normalize_pmcid(raw_pmcid)) in requested
    }

    for value in (document.get("pmcid"), document.get("id")):
        pmcid = _strict_pmcid(value)
        if pmcid in requested:
            return pmcid

    direct_document_id = normalize_pmid(document.get("id"))
    if direct_document_id and direct_document_id in normalized_pmid_map:
        return normalized_pmid_map[direct_document_id]

    for container in _identifier_containers(document):
        for key, value in container.items():
            key_text = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
            if "pmc" in key_text:
                pmcid = _strict_pmcid(value, key_implies_pmcid=True)
                if pmcid in requested:
                    return pmcid
            if "pmid" in key_text:
                pmid = normalize_pmid(value)
                if pmid and pmid in normalized_pmid_map:
                    return normalized_pmid_map[pmid]

            # URLs and generic identifier fields can still contain a literal
            # PMC accession even when their key is not standardized.
            pmcid = _strict_pmcid(value)
            if pmcid in requested:
                return pmcid

    # Compatibility for BioC payloads whose numeric document ID is the numeric
    # portion of a PMCID and for which no PMID mapping is available.
    document_id = clean_text(document.get("id"))
    if re.fullmatch(r"\d+", document_id):
        suffix_match = f"PMC{document_id}"
        if suffix_match in requested:
            return suffix_match
    return None


def _compact_passage(passage: dict[str, Any]) -> dict[str, Any] | None:
    raw_text = passage.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    infons = _passage_infons(passage)
    compact_infons: dict[str, Any] = {}
    section_type = _passage_section_type(passage)
    passage_type = _passage_type(passage)
    if section_type and section_type != "UNKNOWN":
        compact_infons["section_type"] = section_type
    if passage_type:
        compact_infons["type"] = passage_type

    # Preserve a human-readable section name when it differs from the broad
    # section type. No annotations or relations are retained.
    for key in ("section", "section_name", "sectionName"):
        value = clean_text(infons.get(key))
        if value and value != section_type:
            compact_infons["section"] = value
            break

    compact: dict[str, Any] = {
        "offset": passage.get("offset", 0),
        "infons": compact_infons,
        "text": raw_text,
    }
    if isinstance(passage.get("sentences"), list):
        # Sentences are deliberately omitted: passage text is sufficient for the
        # retrieval-stage chunker and avoids duplicating the same text.
        pass
    return compact


def _compact_bioc_document(
    document: dict[str, Any],
    *,
    pmcid: str,
) -> dict[str, Any]:
    passages = [
        compact
        for passage in document.get("passages") or document.get("passage") or []
        if isinstance(passage, dict)
        and (compact := _compact_passage(passage)) is not None
    ]
    compact_document: dict[str, Any] = {
        "id": pmcid,
        "infons": {"pmcid": pmcid},
        "passages": passages,
    }
    return compact_document


def _serialize_compact_bioc_document(
    document: dict[str, Any],
    *,
    pmcid: str,
) -> bytes:
    compact = _compact_bioc_document(document, pmcid=pmcid)
    return json.dumps([compact], ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _parse_pubtator3_batch(
    content: bytes,
    requested_pmcids: Sequence[str],
    pmid_to_pmcid: dict[str, str] | None = None,
) -> dict[str, bytes]:
    requested = set(
        unique_preserving_order(normalize_pmcid(value) or "" for value in requested_pmcids)
    )
    payloads = _decode_json_values(content)
    documents: list[dict[str, Any]] = []
    for payload in payloads:
        documents.extend(collect_bioc_documents(payload))

    mapped: dict[str, bytes] = {}
    unmatched: list[dict[str, Any]] = []
    for document in documents:
        if not _document_has_fulltext(document):
            continue
        pmcid = _document_pmcid(document, requested, pmid_to_pmcid)
        if pmcid is None:
            unmatched.append(document)
            continue
        mapped.setdefault(
            pmcid,
            _serialize_compact_bioc_document(document, pmcid=pmcid),
        )

    # A single-document response that omits identifiers is unambiguous when the
    # request itself contains exactly one PMCID.
    if len(requested) == 1 and not mapped and len(unmatched) == 1:
        pmcid = next(iter(requested))
        mapped[pmcid] = _serialize_compact_bioc_document(
            unmatched[0],
            pmcid=pmcid,
        )
    return mapped


def _request_pubtator3_fulltext_batch(
    session: requests.Session,
    pacer: RequestPacer,
    *,
    pmcids: Sequence[str],
    timeout: int,
    pmid_to_pmcid: dict[str, str] | None = None,
) -> tuple[dict[str, bytes], tuple[str, ...], tuple[str, ...]]:
    """Return found documents, unresolved IDs, and service-failed IDs."""

    requested = unique_preserving_order(normalize_pmcid(value) or "" for value in pmcids)
    if not requested:
        return {}, (), ()

    try:
        response = perform_request_with_retries(
            session,
            "GET",
            PUBTATOR3_PMC_EXPORT,
            context="PubTator3 full-text batch",
            timeout=timeout,
            pacer=pacer,
            max_attempts=FULLTEXT_HTTP_ATTEMPTS,
            accepted_statuses={400, 404},
            params={"pmcids": ",".join(requested)},
        )
    except RetrievalError as exc:
        logger.warning("PubTator3 full-text batch failed: %s", exc)
        return {}, (), requested

    # A batch-level 400/404 does not prove that each individual paper lacks full
    # text. All IDs are left unresolved for the single-paper fallbacks.
    if response.status_code in {400, 404}:
        return {}, requested, ()

    content = response.content or b""
    if not content or content.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        return {}, (), requested

    try:
        documents = _parse_pubtator3_batch(
            content,
            requested,
            pmid_to_pmcid=pmid_to_pmcid,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse PubTator3 full-text batch: %s", exc)
        return {}, (), requested

    unresolved = tuple(pmcid for pmcid in requested if pmcid not in documents)
    return documents, unresolved, ()


@dataclass(frozen=True)
class _SingleFulltextOutcome:
    content: bytes | None
    status: Literal["downloaded", "unavailable", "failed"]
    source: str


def _response_looks_unavailable(content: bytes) -> bool:
    text = content[:1000].decode("utf-8", errors="ignore").casefold()
    return any(
        marker in text
        for marker in (
            "not found",
            "no document",
            "not available",
            "no full text",
            "does not exist",
        )
    )


def _parse_single_bioc_document(content: bytes, *, pmcid: str) -> bytes | None:
    payloads = _decode_json_values(content)
    documents: list[dict[str, Any]] = []
    for payload in payloads:
        documents.extend(collect_bioc_documents(payload))
    full_documents = [document for document in documents if _document_has_fulltext(document)]
    if not full_documents:
        return None

    requested = {pmcid}
    for document in full_documents:
        if _document_pmcid(document, requested) == pmcid:
            return _serialize_compact_bioc_document(document, pmcid=pmcid)
    # The request URL itself identifies the document, so a single valid result is
    # safe even when the payload omits the PMCID.
    return _serialize_compact_bioc_document(full_documents[0], pmcid=pmcid)


def _request_ncbi_bioc_fulltext(
    session: requests.Session,
    pacer: RequestPacer,
    *,
    pmcid: str,
    timeout: int,
) -> _SingleFulltextOutcome:
    source = "ncbi_bioc_pmc"
    try:
        response = perform_request_with_retries(
            session,
            "GET",
            NCBI_BIOC_PMC_EXPORT.format(pmcid=pmcid),
            context=f"NCBI BioC PMC full text for {pmcid}",
            timeout=timeout,
            pacer=pacer,
            max_attempts=FULLTEXT_HTTP_ATTEMPTS,
            accepted_statuses={400, 404},
        )
    except RetrievalError as exc:
        logger.warning("NCBI BioC PMC request failed for %s: %s", pmcid, exc)
        return _SingleFulltextOutcome(None, "failed", source)

    if response.status_code in {400, 404}:
        return _SingleFulltextOutcome(None, "unavailable", source)
    content = response.content or b""
    if not content:
        return _SingleFulltextOutcome(None, "unavailable", source)
    if content.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        return _SingleFulltextOutcome(None, "failed", source)

    try:
        compact = _parse_single_bioc_document(content, pmcid=pmcid)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse NCBI BioC PMC response for %s: %s", pmcid, exc)
        status: Literal["unavailable", "failed"] = (
            "unavailable" if _response_looks_unavailable(content) else "failed"
        )
        return _SingleFulltextOutcome(None, status, source)
    if compact is None:
        status = "unavailable" if _response_looks_unavailable(content) else "unavailable"
        return _SingleFulltextOutcome(None, status, source)
    return _SingleFulltextOutcome(compact, "downloaded", source)


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _nearest_section_title(
    element: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> str:
    current = parent_map.get(element)
    while current is not None:
        if _xml_local_name(current.tag) == "sec":
            for child in current:
                if _xml_local_name(child.tag) == "title":
                    return xml_text(child)
        current = parent_map.get(current)
    return ""


def _jats_section_type(section_title: str) -> str:
    normalized = clean_text(section_title).casefold()
    if not normalized:
        return "BODY"
    mappings = (
        (("introduction", "background"), "INTRO"),
        (("method", "material", "experimental"), "METHODS"),
        (("result", "finding"), "RESULTS"),
        (("discussion",), "DISCUSSION"),
        (("conclusion",), "CONCLUSION"),
        (("case report", "case presentation"), "CASE"),
        (("supplement",), "SUPPLEMENT"),
        (("reference", "bibliograph"), "REFERENCES"),
    )
    for needles, label in mappings:
        if any(needle in normalized for needle in needles):
            return label
    return "BODY"


def _element_has_skipped_ancestor(
    element: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> bool:
    skipped = {
        "table-wrap",
        "table-wrap-group",
        "fig",
        "ref-list",
        "ack",
        "fn-group",
        "supplementary-material",
        "permissions",
    }
    current = parent_map.get(element)
    while current is not None:
        if _xml_local_name(current.tag) in skipped:
            return True
        current = parent_map.get(current)
    return False


def _parse_epmc_fulltext_xml(content: bytes, *, pmcid: str) -> bytes | None:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError("Europe PMC returned malformed full-text XML.") from exc

    if _xml_local_name(root.tag) == "error":
        return None

    parent_map = {child: parent for parent in root.iter() for child in parent}
    passages: list[dict[str, Any]] = []
    offset = 0

    def append_passage(text: str, section_type: str, passage_type: str, section: str = "") -> None:
        nonlocal offset
        text = text.strip()
        if not text:
            return
        infons: dict[str, Any] = {
            "section_type": section_type,
            "type": passage_type,
        }
        if section and section.upper() != section_type:
            infons["section"] = section
        passages.append({"offset": offset, "infons": infons, "text": text})
        offset += len(text) + 2

    article_title = next(
        (xml_text(element) for element in root.iter() if _xml_local_name(element.tag) == "article-title"),
        "",
    )
    append_passage(article_title, "TITLE", "title")

    abstract_elements = [
        element for element in root.iter() if _xml_local_name(element.tag) == "abstract"
    ]
    seen_abstract_texts: set[str] = set()
    for abstract in abstract_elements:
        paragraph_elements = [
            element for element in abstract.iter() if _xml_local_name(element.tag) == "p"
        ]
        if paragraph_elements:
            for paragraph in paragraph_elements:
                text = xml_text(paragraph)
                identity = text.casefold()
                if text and identity not in seen_abstract_texts:
                    seen_abstract_texts.add(identity)
                    append_passage(text, "ABSTRACT", "abstract")
        else:
            text = xml_text(abstract)
            identity = text.casefold()
            if text and identity not in seen_abstract_texts:
                seen_abstract_texts.add(identity)
                append_passage(text, "ABSTRACT", "abstract")

    body_elements = [element for element in root.iter() if _xml_local_name(element.tag) == "body"]
    body_paragraph_count = 0
    for body in body_elements:
        for paragraph in body.iter():
            if _xml_local_name(paragraph.tag) != "p":
                continue
            if _element_has_skipped_ancestor(paragraph, parent_map):
                continue
            text = xml_text(paragraph)
            if not text:
                continue
            section = _nearest_section_title(paragraph, parent_map)
            append_passage(text, _jats_section_type(section), "paragraph", section)
            body_paragraph_count += 1

    if body_paragraph_count == 0:
        return None
    document = {"id": pmcid, "infons": {"pmcid": pmcid}, "passages": passages}
    return _serialize_compact_bioc_document(document, pmcid=pmcid)


def _request_epmc_fulltext(
    session: requests.Session,
    pacer: RequestPacer,
    *,
    pmcid: str,
    timeout: int,
) -> _SingleFulltextOutcome:
    source = "europe_pmc_fulltext_xml"
    try:
        response = perform_request_with_retries(
            session,
            "GET",
            EPMC_FULLTEXT_XML.format(pmcid=pmcid),
            context=f"Europe PMC full text for {pmcid}",
            timeout=timeout,
            pacer=pacer,
            max_attempts=FULLTEXT_HTTP_ATTEMPTS,
            accepted_statuses={400, 404},
        )
    except RetrievalError as exc:
        logger.warning("Europe PMC full-text request failed for %s: %s", pmcid, exc)
        return _SingleFulltextOutcome(None, "failed", source)

    if response.status_code in {400, 404}:
        return _SingleFulltextOutcome(None, "unavailable", source)
    content = response.content or b""
    if not content:
        return _SingleFulltextOutcome(None, "unavailable", source)
    if content.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        return _SingleFulltextOutcome(None, "failed", source)

    try:
        compact = _parse_epmc_fulltext_xml(content, pmcid=pmcid)
    except ValueError as exc:
        logger.warning("Could not parse Europe PMC full text for %s: %s", pmcid, exc)
        return _SingleFulltextOutcome(None, "failed", source)
    if compact is None:
        return _SingleFulltextOutcome(None, "unavailable", source)
    return _SingleFulltextOutcome(compact, "downloaded", source)


def fetch_pubtator3_fulltext_batches(
    session: requests.Session,
    pacer: RequestPacer,
    *,
    pmcids: Sequence[str],
    timeout: int,
    batch_size: int = FULLTEXT_BATCH_SIZE,
    pmid_to_pmcid: dict[str, str] | None = None,
    on_progress: Callable[[int, int, FulltextProgress], None] | None = None,
) -> FulltextBatchResult:
    """Retrieve full text through a batch fast path and resilient fallbacks."""

    requested = unique_preserving_order(normalize_pmcid(value) or "" for value in pmcids)
    if not requested:
        return FulltextBatchResult({}, (), (), 0, 0)

    documents: dict[str, bytes] = {}
    document_sources: dict[str, str] = {}
    unavailable: set[str] = set()
    failed: set[str] = set()
    pubtator_requests = 0
    ncbi_bioc_requests = 0
    epmc_requests = 0
    service_error_batches = 0
    consecutive_service_errors = 0
    batches = list(batched(requested, max(1, min(int(batch_size), FULLTEXT_BATCH_SIZE))))
    progress_total = len(batches) + len(requested)
    last_progress_step = 0

    def progress_snapshot() -> FulltextProgress:
        return FulltextProgress(
            downloaded_count=len(documents),
            unavailable_count=len(unavailable),
            failed_count=len(failed),
            requests_made=pubtator_requests + ncbi_bioc_requests + epmc_requests,
            service_error_batches=service_error_batches,
            pubtator_requests=pubtator_requests,
            ncbi_bioc_requests=ncbi_bioc_requests,
            epmc_requests=epmc_requests,
        )

    for batch_index, batch in enumerate(batches, start=1):
        found, _unresolved_now, failed_now = _request_pubtator3_fulltext_batch(
            session,
            pacer,
            pmcids=batch,
            timeout=timeout,
            pmid_to_pmcid=pmid_to_pmcid,
        )
        pubtator_requests += 1
        documents.update(found)
        document_sources.update({pmcid: "pubtator3" for pmcid in found})

        if failed_now and not found:
            service_error_batches += 1
            consecutive_service_errors += 1
        else:
            consecutive_service_errors = 0

        if on_progress is not None:
            on_progress(batch_index, progress_total, progress_snapshot())
            last_progress_step = batch_index

        if consecutive_service_errors >= FULLTEXT_MAX_CONSECUTIVE_SERVICE_ERRORS:
            logger.warning(
                "Opening the PubTator3 circuit after %s consecutive failed batches; "
                "remaining papers will use per-paper fallbacks.",
                consecutive_service_errors,
            )
            break

    unresolved = [pmcid for pmcid in requested if pmcid not in documents]
    ncbi_circuit_open = False
    epmc_circuit_open = False
    ncbi_consecutive_failures = 0
    epmc_consecutive_failures = 0

    for fallback_index, pmcid in enumerate(unresolved, start=1):
        ncbi_status = "skipped" if ncbi_circuit_open else "not_attempted"
        epmc_status = "skipped" if epmc_circuit_open else "not_attempted"

        ncbi_outcome: _SingleFulltextOutcome | None = None
        if not ncbi_circuit_open:
            ncbi_outcome = _request_ncbi_bioc_fulltext(
                session,
                pacer,
                pmcid=pmcid,
                timeout=timeout,
            )
            ncbi_bioc_requests += 1
            ncbi_status = ncbi_outcome.status
            if ncbi_outcome.status == "failed":
                ncbi_consecutive_failures += 1
                if ncbi_consecutive_failures >= FULLTEXT_MAX_CONSECUTIVE_SERVICE_ERRORS:
                    ncbi_circuit_open = True
            else:
                ncbi_consecutive_failures = 0

        if ncbi_outcome is not None and ncbi_outcome.status == "downloaded":
            assert ncbi_outcome.content is not None
            documents[pmcid] = ncbi_outcome.content
            document_sources[pmcid] = ncbi_outcome.source
        else:
            epmc_outcome: _SingleFulltextOutcome | None = None
            if not epmc_circuit_open:
                epmc_outcome = _request_epmc_fulltext(
                    session,
                    pacer,
                    pmcid=pmcid,
                    timeout=timeout,
                )
                epmc_requests += 1
                epmc_status = epmc_outcome.status
                if epmc_outcome.status == "failed":
                    epmc_consecutive_failures += 1
                    if epmc_consecutive_failures >= FULLTEXT_MAX_CONSECUTIVE_SERVICE_ERRORS:
                        epmc_circuit_open = True
                else:
                    epmc_consecutive_failures = 0

            if epmc_outcome is not None and epmc_outcome.status == "downloaded":
                assert epmc_outcome.content is not None
                documents[pmcid] = epmc_outcome.content
                document_sources[pmcid] = epmc_outcome.source
            elif ncbi_status == "unavailable" and epmc_status == "unavailable":
                unavailable.add(pmcid)
            else:
                # A skipped source due to an open circuit or any transient failure
                # means availability is still unknown and must remain retryable.
                failed.add(pmcid)

        if pmcid in documents:
            unavailable.discard(pmcid)
            failed.discard(pmcid)

        if on_progress is not None:
            last_progress_step = len(batches) + fallback_index
            on_progress(last_progress_step, progress_total, progress_snapshot())

    classified = set(documents) | unavailable | failed
    failed.update(set(requested) - classified)
    if on_progress is not None and last_progress_step < progress_total:
        # Always complete the stage, including the common case where the batch
        # fast path resolves most papers and few or no fallback requests are made.
        on_progress(progress_total, progress_total, progress_snapshot())
    return FulltextBatchResult(
        documents=documents,
        unavailable_now=tuple(sorted(unavailable)),
        failed_now=tuple(sorted(failed)),
        requests_made=pubtator_requests + ncbi_bioc_requests + epmc_requests,
        service_error_batches=service_error_batches,
        document_sources=document_sources,
        pubtator_requests=pubtator_requests,
        ncbi_bioc_requests=ncbi_bioc_requests,
        epmc_requests=epmc_requests,
    )


def _parse_iso_datetime(value: Any) -> dt.datetime | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def should_attempt_fulltext(
    record: dict[str, Any],
    *,
    now: dt.datetime | None = None,
) -> bool:
    """Decide whether a PMCID should be checked in the current job."""

    if not normalize_pmcid(record.get("pmcid")):
        return False
    if clean_text(record.get("fulltext_retriever_version")) != FULLTEXT_RETRIEVER_VERSION:
        return True

    status = clean_text(record.get("fulltext_status")).casefold()
    if status in {"failed", "pending_retry", "service_error", ""}:
        return True
    if status == "downloaded":
        # The caller checks the actual file first. A missing downloaded file must
        # be repaired rather than trusted from metadata alone.
        return True
    if status != "not_available":
        return not bool(record.get("fulltext_checked"))

    checked_at = _parse_iso_datetime(record.get("fulltext_checked_at"))
    if checked_at is None:
        return True
    current = now or dt.datetime.now(dt.timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=dt.timezone.utc)
    age = current.astimezone(dt.timezone.utc) - checked_at
    return age >= dt.timedelta(days=FULLTEXT_NEGATIVE_CACHE_DAYS)

def iter_bioc_passages(path: Path) -> Iterator[dict[str, Any]]:
    try:
        if path.suffix.casefold() == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Could not parse cached BioC file %s: %s", path, exc)
        return

    for document in collect_bioc_documents(payload):
        for passage in document.get("passages") or document.get("passage") or []:
            if not isinstance(passage, dict):
                continue
            raw_text = passage.get("text")
            if not isinstance(raw_text, str) or not raw_text.strip():
                continue
            infons = passage.get("infons")
            if not isinstance(infons, dict):
                infons = {}
            yield {
                "text": raw_text,
                "section_type": _passage_section_type(passage),
                "passage_type": _passage_type(passage),
                "infons": infons,
            }


def should_keep_passage(passage: dict[str, Any]) -> bool:
    section = clean_text(passage.get("section_type"))
    passage_type = clean_text(passage.get("passage_type"))
    infons = passage.get("infons") or {}
    section_upper = section.upper()
    section_lower = section.lower()
    type_lower = passage_type.lower()

    if section_upper in SKIP_SECTION_EXACT:
        return False
    if any(token in section_lower for token in SKIP_SECTION_SUBSTR):
        return False
    if any(token in type_lower for token in SKIP_TYPE_SUBSTR):
        return False
    if any(
        str(key).lower().startswith(("name_", "aff_"))
        or "affiliation" in str(key).lower()
        or "orcid" in str(key).lower()
        for key in infons
    ):
        return False
    if section_upper == "ABSTRACT":
        return type_lower in {"abstract", "paragraph", ""}
    if section_upper == "TITLE":
        return type_lower in {"title", "front", ""}
    return type_lower in {
        "paragraph",
        "paragrpah",
        "body",
        "text",
        "section",
        "subsection",
        "",
    }


def sanitize_chunk_text(text: str) -> str:
    output: list[str] = []
    for character in text or "":
        if character in {"|", "\x00", "\ufffd"}:
            output.append(" ")
        elif character.isspace() and character != " ":
            output.append(" ")
        elif unicodedata.category(character).startswith("C"):
            output.append(" ")
        else:
            output.append(character)
    return re.sub(r"\s+", " ", "".join(output)).strip()


def safe_doc_key(record: dict[str, Any]) -> str:
    raw = record.get("pmcid") or record.get("pmid") or record.get("canonical_id") or "unknown"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw))[:200]


def make_chunk_base(doc_key: str, chunk_id: int) -> str:
    return hashlib.md5(f"{doc_key}::{chunk_id}".encode("utf-8")).hexdigest()


def _resolve_fulltext_path(
    record: dict[str, Any],
    papers_root: Path,
) -> Path | None:
    papers_root = papers_root.resolve()
    relative_value = clean_text(record.get("fulltext_path"))
    if relative_value:
        candidate = (papers_root / relative_value).resolve()
        if candidate.is_relative_to(papers_root) and candidate.is_file():
            return candidate

    # Compatibility fallback for an existing full-text file whose corpus row
    # predates the fulltext_path field. This is a local lookup only; it never
    # triggers a download for an existing paper.
    pmcid = normalize_pmcid(record.get("pmcid"))
    if pmcid:
        for filename in (f"{pmcid}.bioc.json.gz", f"{pmcid}.bioc.json"):
            candidate = (papers_root / "fulltext_bioc" / filename).resolve()
            if candidate.is_relative_to(papers_root) and candidate.is_file():
                return candidate
    return None


def _paper_cache_key(record: dict[str, Any]) -> str:
    canonical_id = clean_text(record.get("canonical_id")) or safe_doc_key(record)
    return hashlib.sha256(canonical_id.encode("utf-8")).hexdigest()[:24]


def cached_chunk_path(
    record: dict[str, Any],
    *,
    chunks_dir: Path,
) -> Path:
    return chunks_dir / _paper_cache_key(record) / "chunks.jsonl.gz"


class ChunkCacheIndex:
    """SQLite index for one reusable chunk file per paper identity."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA busy_timeout=30000;")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_chunks (
                canonical_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                chunk_path TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (canonical_id)
            )
            """
        )
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_chunks_latest
            ON paper_chunks(canonical_id, created_at DESC)
            """
        )
        self.connection.commit()

    def __enter__(self) -> "ChunkCacheIndex":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None


    def get_latest(self, canonical_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT * FROM paper_chunks
            WHERE canonical_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (canonical_id,),
        ).fetchone()
        return self._row_to_dict(row)

    def candidates(self, canonical_id: str) -> tuple[dict[str, Any], ...]:
        rows = self.connection.execute(
            """
            SELECT * FROM paper_chunks
            WHERE canonical_id = ?
            ORDER BY created_at DESC
            """,
            (canonical_id,),
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def upsert(
        self,
        *,
        canonical_id: str,
        chunk_path: str,
        chunk_count: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO paper_chunks (
                canonical_id, source_type, chunk_path, chunk_count, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(canonical_id)
            DO UPDATE SET
                source_type = excluded.source_type,
                chunk_path = excluded.chunk_path,
                chunk_count = excluded.chunk_count,
                created_at = excluded.created_at
            """,
            (
                canonical_id,
                "gzip_jsonl" if chunk_path.casefold().endswith(".gz") else "jsonl",
                chunk_path,
                int(chunk_count),
                utc_now_iso(),
            ),
        )


def count_jsonl_rows(path: Path) -> int:
    if path.suffix.casefold() == ".gz":
        handle_context = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        handle_context = path.open("r", encoding="utf-8", errors="replace")
    with handle_context as handle:
        return sum(1 for line in handle if line.strip())


def migrate_legacy_chunk_to_gzip(
    legacy_path: Path,
    compressed_path: Path,
) -> Path:
    """Compress one legacy JSONL cache file atomically on first reuse."""

    if legacy_path.suffix.casefold() == ".gz" or not legacy_path.is_file():
        return legacy_path
    if compressed_path.is_file():
        legacy_path.unlink(missing_ok=True)
        return compressed_path

    compressed_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = compressed_path.with_name(compressed_path.name + ".tmp")
    try:
        with legacy_path.open("rb") as source, temporary.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_output,
                compresslevel=3,
                mtime=0,
            ) as compressed_output:
                shutil.copyfileobj(source, compressed_output, length=1024 * 1024)
        temporary.replace(compressed_path)
        legacy_path.unlink(missing_ok=True)
        return compressed_path
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        logger.warning("Could not compress legacy chunk cache %s: %s", legacy_path, exc)
        return legacy_path



def build_paper_chunks(
    record: dict[str, Any],
    *,
    papers_root: Path,
    output_path: Path,
) -> int:
    """Build exactly one paper's chunk file atomically."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    rows_written = 0
    doc_key = safe_doc_key(record)
    paper_metadata = {
        key: value
        for key, value in {
            "title": clean_text(record.get("title")) or None,
            "doi": record.get("doi"),
            "sources": record.get("sources") or None,
        }.items()
        if value not in (None, "", [])
    }

    def write_chunks(output: Any) -> None:
        nonlocal rows_written

        def write_chunk(section_type: str, text: str) -> None:
            nonlocal rows_written
            chunk_id = rows_written + 1
            row: dict[str, Any] = {
                "base": make_chunk_base(doc_key, chunk_id),
                "doc_key": doc_key,
                "canonical_id": record.get("canonical_id"),
                "pmid": record.get("pmid"),
                "pmcid": record.get("pmcid"),
                "journal": record.get("journal"),
                "pub_year": record.get("pub_year"),
                "section_type": section_type,
                "chunk_id": chunk_id,
                "chunk": text,
            }
            if chunk_id == 1 and paper_metadata:
                row["paper_metadata"] = paper_metadata
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1

        fulltext_path = _resolve_fulltext_path(record, papers_root)
        if fulltext_path is not None:
            for passage in iter_bioc_passages(fulltext_path):
                if not should_keep_passage(passage):
                    continue
                text = sanitize_chunk_text(passage["text"])
                if text:
                    write_chunk(passage["section_type"], text)

        if rows_written == 0:
            title = sanitize_chunk_text(clean_text(record.get("title")))
            abstract = sanitize_chunk_text(clean_text(record.get("abstract")))
            if title:
                write_chunk("TITLE", title)
            if abstract:
                write_chunk("ABSTRACT", abstract)

    try:
        if output_path.suffix.casefold() == ".gz":
            with temporary.open("wb") as raw_output:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw_output,
                    compresslevel=6,
                    mtime=0,
                ) as compressed_output:
                    with io.TextIOWrapper(
                        compressed_output,
                        encoding="utf-8",
                        newline="\n",
                    ) as text_output:
                        write_chunks(text_output)
        else:
            with temporary.open("w", encoding="utf-8", newline="\n") as output:
                write_chunks(output)
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    if output_path.suffix.casefold() == ".gz":
        output_path.with_suffix("").unlink(missing_ok=True)
    return rows_written


def prepare_cached_chunks(
    records: Sequence[dict[str, Any]],
    *,
    papers_root: Path,
    chunks_dir: Path,
    cache_index: ChunkCacheIndex,
    new_canonical_ids: set[str],
    rebuild_canonical_ids: set[str] | None = None,
    on_progress: Callable[[int, int, dict[str, int]], None] | None = None,
) -> tuple[tuple[Path, ...], dict[str, int]]:
    """Reuse chunks, generate new papers, and rebuild repaired full-text papers."""

    rebuild_ids = rebuild_canonical_ids or set()
    counters = {
        "chunk_papers_reused": 0,
        "chunk_papers_generated_new": 0,
        "chunk_papers_rebuilt_fulltext": 0,
        "chunk_papers_missing_cache": 0,
        "chunk_papers_regenerated_missing_cache": 0,
        "chunk_rows_reused": 0,
        "chunk_rows_generated": 0,
        "chunks_written": 0,
    }
    chunk_paths: list[Path] = []
    seen_paths: set[Path] = set()
    total = len(records)
    papers_root_resolved = papers_root.resolve()
    chunks_root_resolved = chunks_dir.resolve()

    for index, record in enumerate(records, start=1):
        canonical_id = clean_text(record.get("canonical_id"))
        if not canonical_id:
            canonical_id = CorpusStore.new_canonical_id(record)
            record["canonical_id"] = canonical_id

        force_fulltext_rebuild = canonical_id in rebuild_ids
        output_path: Path | None = None
        cached_count = 0

        if not force_fulltext_rebuild:
            preferred_path = cached_chunk_path(record, chunks_dir=chunks_dir).resolve()
            for cached in cache_index.candidates(canonical_id):
                raw_path = clean_text(cached.get("chunk_path"))
                if not raw_path:
                    continue
                candidate = (papers_root_resolved / raw_path).resolve()
                if candidate.is_relative_to(chunks_root_resolved) and candidate.is_file():
                    output_path = candidate
                    cached_count = int(cached.get("chunk_count") or 0)
                    break

            if output_path is None:
                for candidate in (preferred_path, preferred_path.with_suffix("")):
                    if candidate.is_relative_to(chunks_root_resolved) and candidate.is_file():
                        output_path = candidate
                        break

            if (
                output_path is not None
                and output_path.suffix.casefold() != ".gz"
                and preferred_path.is_relative_to(chunks_root_resolved)
            ):
                output_path = migrate_legacy_chunk_to_gzip(
                    output_path,
                    preferred_path,
                )

        if force_fulltext_rebuild:
            output_path = cached_chunk_path(record, chunks_dir=chunks_dir).resolve()
            chunk_count = build_paper_chunks(
                record,
                papers_root=papers_root,
                output_path=output_path,
            )
            if canonical_id in new_canonical_ids:
                counters["chunk_papers_generated_new"] += 1
            else:
                counters["chunk_papers_rebuilt_fulltext"] += 1
            counters["chunk_rows_generated"] += chunk_count
        elif output_path is not None:
            chunk_count = cached_count or count_jsonl_rows(output_path)
            counters["chunk_papers_reused"] += 1
            counters["chunk_rows_reused"] += chunk_count
        elif canonical_id in new_canonical_ids:
            output_path = cached_chunk_path(record, chunks_dir=chunks_dir).resolve()
            chunk_count = build_paper_chunks(
                record,
                papers_root=papers_root,
                output_path=output_path,
            )
            counters["chunk_papers_generated_new"] += 1
            counters["chunk_rows_generated"] += chunk_count
        else:
            counters["chunk_papers_missing_cache"] += 1
            output_path = cached_chunk_path(record, chunks_dir=chunks_dir).resolve()
            chunk_count = build_paper_chunks(
                record,
                papers_root=papers_root,
                output_path=output_path,
            )
            counters["chunk_papers_regenerated_missing_cache"] += 1
            counters["chunk_rows_generated"] += chunk_count

        relative_path = str(output_path.relative_to(papers_root_resolved))
        cache_index.upsert(
            canonical_id=canonical_id,
            chunk_path=relative_path,
            chunk_count=chunk_count,
        )
        counters["chunks_written"] += chunk_count
        if output_path not in seen_paths:
            chunk_paths.append(output_path)
            seen_paths.add(output_path)

        if on_progress is not None:
            on_progress(index, total, dict(counters))

    return tuple(chunk_paths), counters


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def progress_for_fraction(start: int, end: int, current: int, total: int) -> int:
    if total <= 0:
        return end
    ratio = max(0.0, min(1.0, current / total))
    return round(start + (end - start) * ratio)


def _record_has_metadata(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    sources = set(record.get("sources") or [])
    return bool(
        sources.intersection({"pubmed", "europepmc"})
        or clean_text(record.get("title"))
        or clean_text(record.get("abstract"))
        or clean_text(record.get("journal"))
    )


def run_paper_retrieval(
    *,
    job_id: str,
    input_type: InputType,
    user_input: str,
    papers_root: Path,
    ncbi_email: str,
    ncbi_tool: str,
    ncbi_api_key: str | None = None,
    keyword_limit: int = DEFAULT_KEYWORD_LIMIT,
    batch_size: int = 200,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    progress_callback: ProgressCallback | None = None,
    exclude_pmids: Sequence[str] = (),
    exclude_pmcids: Sequence[str] = (),
    exclude_canonical_ids: Sequence[str] = (),
) -> RetrievalResult:
    """Retrieve papers while reusing metadata, full text, and chunk caches.

    Missing full text is checked through multiple official endpoints. Definitive
    negative results are cached for a limited period; transient failures remain
    eligible for the next job rather than becoming permanent abstract-only rows.
    """

    normalized_email = clean_text(ncbi_email)
    if not normalized_email or "@" not in normalized_email:
        raise RetrievalError(
            "Set NCBI_EMAIL in .env to a real contact email before running retrieval."
        )
    normalized_tool = clean_text(ncbi_tool) or "ovarian_network_web"

    started_monotonic = time.monotonic()
    started_at = utc_now_iso()
    effective = build_effective_inputs(input_type, user_input)
    effective_query = build_augmented_pubmed_query(
        effective.user_keywords, effective.user_exclusions
    )
    excluded_pmids = {
        normalized
        for value in exclude_pmids
        if (normalized := normalize_pmid(value))
    }
    excluded_pmcids = {
        normalized
        for value in exclude_pmcids
        if (normalized := normalize_pmcid(value))
    }
    excluded_canonical_ids = {
        clean_text(value) for value in exclude_canonical_ids if clean_text(value)
    }
    reporter = ProgressReporter(progress_callback)
    metadata_timeout = max(5, min(int(request_timeout), MAX_REQUEST_TIMEOUT))

    papers_root = papers_root.expanduser().resolve()
    fulltext_dir = papers_root / "fulltext_bioc"
    chunks_dir = papers_root / "chunks_by_paper"
    chunk_cache_path = papers_root / "chunk_cache.sqlite"
    job_dir = papers_root / "jobs" / job_id
    corpus_path = papers_root / "corpus.jsonl"
    summary_path = job_dir / "summary.json"
    for directory in (papers_root, fulltext_dir, chunks_dir, job_dir):
        directory.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "paper_count": 0,
        "abstract_count": 0,
        "metadata_reused_cache": 0,
        "metadata_downloaded_new": 0,
        "metadata_missing": 0,
        "fulltexts_downloaded": 0,
        "papers_without_pmcid": 0,
        # Compatibility aliases used by the existing frontend.
        "fulltext_available": 0,
        "fulltext_downloaded_new": 0,
        "fulltext_reused_cache": 0,
        "fulltext_not_available": 0,
        "fulltext_not_available_current_run": 0,
        "fulltext_pending_retry": 0,
        "fulltext_service_error_batches": 0,
        "fulltext_batch_requests": 0,
        "fulltext_errors": 0,
        "without_pmcid": 0,
        "chunk_papers_reused": 0,
        "chunk_papers_generated_new": 0,
        "chunk_papers_rebuilt_fulltext": 0,
        "chunk_papers_missing_cache": 0,
        "chunk_papers_regenerated_missing_cache": 0,
        "chunk_rows_reused": 0,
        "chunk_rows_generated": 0,
        "chunks_written": 0,
        "chunk_part_count": 0,
        "papers_in_download": 0,
        "search_total_hits": 0,
        "search_selected": 0,
        "search_result_limit": max(1, min(int(keyword_limit), PUBMED_ESEARCH_API_CAP)),
        "search_all_matches": False,
        "metadata_retry_count": METADATA_HTTP_ATTEMPTS - 1,
        "metadata_request_timeout_seconds": metadata_timeout,
        "fulltext_request_timeout_seconds": min(
            metadata_timeout, FULLTEXT_REQUEST_TIMEOUT
        ),
        "fulltext_attempt_mode": "retry_transient_refresh_negative_cache",
        "fulltext_retriever_version": FULLTEXT_RETRIEVER_VERSION,
        "fulltext_negative_cache_days": FULLTEXT_NEGATIVE_CACHE_DAYS,
        "fulltext_checked_once": 0,
        "fulltext_already_checked": 0,
        "fulltext_failed": 0,
        "fulltext_pubtator_requests": 0,
        "fulltext_ncbi_bioc_requests": 0,
        "fulltext_epmc_requests": 0,
        "fulltext_total_requests": 0,
        "user_keyword_count": effective.user_keyword_count,
        "user_exclusion_count": effective.user_exclusion_count,
        "user_pmid_count": effective.user_pmid_count,
        "user_pmcid_count": effective.user_pmcid_count,
        "duplicate_user_keyword_count": effective.duplicate_user_keyword_count,
        "duplicate_user_exclusion_count": effective.duplicate_user_exclusion_count,
        "duplicate_user_pmid_count": effective.duplicate_user_pmid_count,
        "duplicate_user_pmcid_count": effective.duplicate_user_pmcid_count,
        "keyword_query_augmented": bool(
            effective.user_keywords or effective.user_exclusions
        ),
        "keyword_query_duplicate_ignored": effective.user_keyword_was_redundant,
        "new_paper_count": 0,
        "existing_paper_count": 0,
        "baseline_excluded_pmid_count": len(excluded_pmids),
        "baseline_excluded_pmcid_count": len(excluded_pmcids),
        "baseline_excluded_canonical_count": len(excluded_canonical_ids),
        "baseline_papers_removed_from_run": 0,
        "elapsed_seconds": 0.0,
    }

    reporter.emit(
        "preparing",
        3,
        "Preparing the shared corpus and your additions...",
        stats,
        force=True,
    )

    metadata_session = build_retry_session(f"{normalized_tool} ({normalized_email})")
    fulltext_session = build_fulltext_session(f"{normalized_tool} ({normalized_email})")
    ncbi_pacer = RequestPacer(0.11 if ncbi_api_key else 0.36)
    fulltext_pacer = RequestPacer(0.36)
    store = CorpusStore(corpus_path)
    initial_canonical_ids = set(store.records)
    job_ids: list[str] = []
    job_id_set: set[str] = set()
    metadata_reused_ids: set[str] = set()
    metadata_downloaded_ids: set[str] = set()

    def add_job_id(canonical_id: str | None) -> None:
        if (
            canonical_id
            and canonical_id in store.records
            and canonical_id not in job_id_set
        ):
            job_id_set.add(canonical_id)
            job_ids.append(canonical_id)

    def include(record: dict[str, Any], *, metadata_downloaded: bool = False) -> str:
        normalized_record = normalize_record(record)
        candidate_id = CorpusStore.new_canonical_id(normalized_record)
        if (
            normalized_record.get("pmid") in excluded_pmids
            or normalized_record.get("pmcid") in excluded_pmcids
            or candidate_id in excluded_canonical_ids
        ):
            stats["baseline_papers_removed_from_run"] = int(
                stats.get("baseline_papers_removed_from_run") or 0
            ) + 1
            return ""
        canonical_id, is_new = store.add_if_new(
            normalized_record,
            prefer_existing_ids=initial_canonical_ids,
        )
        if canonical_id in excluded_canonical_ids:
            return ""
        add_job_id(canonical_id)
        if metadata_downloaded and canonical_id:
            metadata_downloaded_ids.add(canonical_id)
            metadata_reused_ids.discard(canonical_id)
        elif not is_new and canonical_id:
            metadata_reused_ids.add(canonical_id)
        return canonical_id

    def current_records() -> list[dict[str, Any]]:
        return [record for canonical_id in job_ids if (record := store.get(canonical_id))]

    def refresh_counts() -> None:
        records = current_records()
        active_ids = {clean_text(record.get("canonical_id")) for record in records}
        stats["paper_count"] = len(records)
        stats["abstract_count"] = sum(
            1 for record in records if clean_text(record.get("abstract"))
        )
        stats["metadata_reused_cache"] = len(metadata_reused_ids & active_ids)
        stats["metadata_downloaded_new"] = len(metadata_downloaded_ids & active_ids)

    try:
        safe_keyword_limit = max(
            1, min(int(keyword_limit), PUBMED_ESEARCH_API_CAP)
        )
        search_message = (
            f"Searching up to {safe_keyword_limit} PubMed results for the augmented query..."
            if effective.user_keywords or effective.user_exclusions
            else f"Searching up to {safe_keyword_limit} PubMed results for the built-in query..."
        )
        reporter.emit("searching", 8, search_message, stats, force=True)
        search_total, search_ids = search_pubmed_limited(
            metadata_session,
            ncbi_pacer,
            query=effective_query,
            limit=safe_keyword_limit,
            email=normalized_email,
            tool=normalized_tool,
            api_key=ncbi_api_key,
            timeout=metadata_timeout,
        )
        stats["search_total_hits"] = search_total
        stats["search_selected"] = len(search_ids)

        selected_pmids = unique_preserving_order(
            pmid
            for pmid in (*effective.pmids, *search_ids)
            if pmid not in excluded_pmids
        )
        selected_pmcids = unique_preserving_order(
            pmcid for pmcid in effective.pmcids if pmcid not in excluded_pmcids
        )
        pmids_to_fetch: list[str] = []
        for pmid in selected_pmids:
            cached_id = store.id_index.get(f"pmid:{pmid}")
            cached_record = store.get(cached_id) if cached_id else None
            if cached_id and _record_has_metadata(cached_record):
                add_job_id(cached_id)
                metadata_reused_ids.add(cached_id)
            else:
                # A prior explicit-ID placeholder must not block a later metadata
                # repair attempt.
                add_job_id(cached_id)
                pmids_to_fetch.append(pmid)

        refresh_counts()
        reporter.emit(
            "metadata",
            18,
            (
                f"Reusing metadata for {len(metadata_reused_ids)} papers and "
                f"downloading {len(pmids_to_fetch)} new PubMed records..."
            ),
            stats,
            force=True,
        )
        if pmids_to_fetch:
            for records, processed, total in fetch_pubmed_records(
                metadata_session,
                ncbi_pacer,
                pmids=pmids_to_fetch,
                batch_size=batch_size,
                email=normalized_email,
                tool=normalized_tool,
                api_key=ncbi_api_key,
                timeout=metadata_timeout,
            ):
                for record in records:
                    include(record, metadata_downloaded=True)
                refresh_counts()
                reporter.emit(
                    "metadata",
                    progress_for_fraction(18, 32, processed, total),
                    f"Downloaded metadata for {processed} of {total} new PubMed records...",
                    stats,
                )
        else:
            reporter.emit(
                "metadata",
                32,
                "All selected PubMed metadata was already available locally.",
                stats,
                force=True,
            )

        # Preserve explicit PMIDs even when PubMed returns no metadata.
        for pmid in selected_pmids:
            cached_id = store.id_index.get(f"pmid:{pmid}")
            if cached_id:
                add_job_id(cached_id)
            else:
                include(
                    {
                        "source": "explicit_pmid",
                        "source_id": pmid,
                        "pmid": pmid,
                        "sources": ["explicit_pmid"],
                    }
                )

        unresolved_pmcids: list[str] = []
        for pmcid in selected_pmcids:
            cached_id = store.id_index.get(f"pmcid:{pmcid}")
            cached_record = store.get(cached_id) if cached_id else None
            if cached_id and _record_has_metadata(cached_record):
                add_job_id(cached_id)
                metadata_reused_ids.add(cached_id)
            else:
                add_job_id(cached_id)
                unresolved_pmcids.append(pmcid)

        if unresolved_pmcids:
            reporter.emit(
                "metadata",
                34,
                f"Resolving metadata for {len(unresolved_pmcids)} new PMC identifiers...",
                stats,
                force=True,
            )
            for records in fetch_epmc_records_for_pmcids(
                metadata_session,
                pmcids=unresolved_pmcids,
                timeout=metadata_timeout,
            ):
                for record in records:
                    include(record, metadata_downloaded=True)

        # Preserve explicit PMCIDs even when Europe PMC returns no metadata.
        for pmcid in selected_pmcids:
            cached_id = store.id_index.get(f"pmcid:{pmcid}")
            if cached_id:
                add_job_id(cached_id)
            else:
                include(
                    {
                        "source": "explicit_pmcid",
                        "source_id": pmcid,
                        "pmcid": pmcid,
                        "sources": ["explicit_pmcid"],
                    }
                )

        refresh_counts()
        job_records = current_records()
        stats["metadata_missing"] = sum(
            1 for record in job_records if not _record_has_metadata(record)
        )
        records_with_pmcid = [record for record in job_records if record.get("pmcid")]
        papers_without_pmcid = len(job_records) - len(records_with_pmcid)
        stats["papers_without_pmcid"] = papers_without_pmcid
        stats["without_pmcid"] = papers_without_pmcid

        new_job_ids = {
            clean_text(record.get("canonical_id"))
            for record in job_records
            if clean_text(record.get("canonical_id")) not in initial_canonical_ids
        }
        stats["new_paper_count"] = len(new_job_ids)
        stats["existing_paper_count"] = len(job_records) - len(new_job_ids)

        fulltext_available_ids: set[str] = set()
        missing_fulltext_records: list[dict[str, Any]] = []
        for record in job_records:
            canonical_id = clean_text(record.get("canonical_id"))
            cached_path = _resolve_fulltext_path(record, papers_root)
            if cached_path is not None:
                record["fulltext_path"] = str(cached_path.relative_to(papers_root))
                record["fulltext_bytes"] = int(cached_path.stat().st_size)
                record["fulltext_checked"] = True
                record["fulltext_status"] = "downloaded"
                record["fulltext_retriever_version"] = FULLTEXT_RETRIEVER_VERSION
                record.setdefault("fulltext_source", "local_cache")
                record.setdefault("fulltext_checked_at", utc_now_iso())
                fulltext_available_ids.add(canonical_id)
                stats["fulltext_reused_cache"] += 1
            elif record.get("pmcid"):
                if should_attempt_fulltext(record):
                    missing_fulltext_records.append(record)
                else:
                    stats["fulltext_already_checked"] += 1

        pmcid_to_records: dict[str, list[dict[str, Any]]] = {}
        pmid_to_pmcid: dict[str, str] = {}
        for record in job_records:
            pmid = normalize_pmid(record.get("pmid"))
            pmcid = normalize_pmcid(record.get("pmcid"))
            if pmid and pmcid:
                pmid_to_pmcid[pmid] = pmcid
        for record in missing_fulltext_records:
            pmcid = normalize_pmcid(record.get("pmcid"))
            if pmcid:
                pmcid_to_records.setdefault(pmcid, []).append(record)

        missing_pmcids = tuple(pmcid_to_records)
        reporter.emit(
            "fulltext",
            40,
            (
                "Retrieving missing PMC full text with batch requests and "
                f"per-paper fallbacks for {len(missing_pmcids)} papers..."
            ),
            stats,
            force=True,
        )

        def report_fulltext_progress(
            current_step: int,
            total_steps: int,
            result: FulltextProgress,
        ) -> None:
            stats["fulltexts_downloaded"] = result.downloaded_count
            stats["fulltext_downloaded_new"] = result.downloaded_count
            stats["fulltext_not_available_current_run"] = result.unavailable_count
            stats["fulltext_not_available"] = result.unavailable_count
            stats["fulltext_pending_retry"] = result.failed_count
            stats["fulltext_errors"] = result.failed_count
            stats["fulltext_failed"] = result.failed_count
            stats["fulltext_service_error_batches"] = result.service_error_batches
            stats["fulltext_batch_requests"] = result.pubtator_requests
            stats["fulltext_pubtator_requests"] = result.pubtator_requests
            stats["fulltext_ncbi_bioc_requests"] = result.ncbi_bioc_requests
            stats["fulltext_epmc_requests"] = result.epmc_requests
            stats["fulltext_total_requests"] = result.requests_made
            reporter.emit(
                "fulltext",
                progress_for_fraction(40, 84, current_step, max(1, total_steps)),
                f"Completed full-text retrieval step {current_step} of {total_steps}...",
                stats,
            )

        if missing_pmcids:
            fulltext_result = fetch_pubtator3_fulltext_batches(
                fulltext_session,
                fulltext_pacer,
                pmcids=missing_pmcids,
                pmid_to_pmcid=pmid_to_pmcid,
                timeout=min(metadata_timeout, FULLTEXT_REQUEST_TIMEOUT),
                batch_size=FULLTEXT_BATCH_SIZE,
                on_progress=report_fulltext_progress,
            )
        else:
            fulltext_result = FulltextBatchResult({}, (), (), 0, 0)
            reporter.emit(
                "fulltext",
                84,
                "All eligible PMC full text was already cached or recently checked.",
                stats,
                force=True,
            )

        repaired_chunk_ids: set[str] = set()
        downloaded_ids: set[str] = set()
        checked_at = utc_now_iso()

        for pmcid, content in fulltext_result.documents.items():
            output_path = fulltext_dir / f"{pmcid}.bioc.json.gz"
            compressed_bytes = atomic_write_gzip_bytes(output_path, content)
            legacy_path = fulltext_dir / f"{pmcid}.bioc.json"
            if legacy_path != output_path:
                legacy_path.unlink(missing_ok=True)
            relative_path = str(output_path.relative_to(papers_root))
            source = fulltext_result.document_sources.get(pmcid, "unknown")
            for record in pmcid_to_records.get(pmcid, []):
                canonical_id = clean_text(record.get("canonical_id"))
                record.update(
                    {
                        "fulltext_path": relative_path,
                        "fulltext_bytes": compressed_bytes,
                        "fulltext_uncompressed_bytes": len(content),
                        "fulltext_checked": True,
                        "fulltext_status": "downloaded",
                        "fulltext_checked_at": checked_at,
                        "fulltext_retriever_version": FULLTEXT_RETRIEVER_VERSION,
                        "fulltext_source": source,
                    }
                )
                fulltext_available_ids.add(canonical_id)
                downloaded_ids.add(canonical_id)
                repaired_chunk_ids.add(canonical_id)

        unavailable_pmcids = set(fulltext_result.unavailable_now)
        failed_pmcids = set(fulltext_result.failed_now)
        classified_pmcids = (
            set(fulltext_result.documents) | unavailable_pmcids | failed_pmcids
        )
        # Unclassified IDs reflect an interrupted or malformed retrieval stage;
        # keep them retryable rather than converting uncertainty into a permanent
        # negative cache entry.
        failed_pmcids.update(set(missing_pmcids) - classified_pmcids)

        for pmcid in unavailable_pmcids:
            for record in pmcid_to_records.get(pmcid, []):
                for field_name in (
                    "fulltext_path",
                    "fulltext_bytes",
                    "fulltext_uncompressed_bytes",
                    "fulltext_source",
                ):
                    record.pop(field_name, None)
                record.update(
                    {
                        "fulltext_checked": True,
                        "fulltext_status": "not_available",
                        "fulltext_checked_at": checked_at,
                        "fulltext_retriever_version": FULLTEXT_RETRIEVER_VERSION,
                    }
                )

        for pmcid in failed_pmcids:
            for record in pmcid_to_records.get(pmcid, []):
                for field_name in (
                    "fulltext_path",
                    "fulltext_bytes",
                    "fulltext_uncompressed_bytes",
                    "fulltext_source",
                ):
                    record.pop(field_name, None)
                record.update(
                    {
                        "fulltext_checked": False,
                        "fulltext_status": "pending_retry",
                        "fulltext_checked_at": checked_at,
                        "fulltext_retriever_version": FULLTEXT_RETRIEVER_VERSION,
                    }
                )

        stats["fulltexts_downloaded"] = len(downloaded_ids)
        stats["fulltext_downloaded_new"] = len(downloaded_ids)
        stats["fulltext_checked_once"] = len(missing_pmcids)
        stats["fulltext_not_available_current_run"] = len(unavailable_pmcids)
        stats["fulltext_not_available"] = len(unavailable_pmcids)
        stats["fulltext_pending_retry"] = len(failed_pmcids)
        stats["fulltext_errors"] = len(failed_pmcids)
        stats["fulltext_failed"] = len(failed_pmcids)
        stats["fulltext_service_error_batches"] = (
            fulltext_result.service_error_batches
        )
        stats["fulltext_batch_requests"] = fulltext_result.pubtator_requests
        stats["fulltext_pubtator_requests"] = fulltext_result.pubtator_requests
        stats["fulltext_ncbi_bioc_requests"] = fulltext_result.ncbi_bioc_requests
        stats["fulltext_epmc_requests"] = fulltext_result.epmc_requests
        stats["fulltext_total_requests"] = fulltext_result.requests_made
        stats["fulltext_available"] = len(fulltext_available_ids)

        # job_records already contains references to the shared corpus records;
        # no second upsert/reindex pass is needed here.
        job_records = current_records()
        job_records.sort(
            key=lambda record: (
                -(
                    int(record.get("pub_year"))
                    if str(record.get("pub_year") or "").isdigit()
                    else 0
                ),
                clean_text(record.get("title")).lower(),
            )
        )

        reporter.emit(
            "chunks",
            86,
            (
                "Reusing cached chunks and rebuilding only papers whose missing "
                "full text was downloaded..."
            ),
            stats,
            force=True,
        )

        def report_chunk_progress(
            current: int,
            total: int,
            chunk_stats: dict[str, int],
        ) -> None:
            stats.update(chunk_stats)
            reporter.emit(
                "chunks",
                progress_for_fraction(86, 98, current, max(1, total)),
                (
                    f"Prepared {current} of {total} papers: "
                    f"{chunk_stats['chunk_papers_reused']} reused, "
                    f"{chunk_stats['chunk_papers_generated_new']} new, and "
                    f"{chunk_stats['chunk_papers_rebuilt_fulltext']} rebuilt from "
                    "newly available full text."
                ),
                stats,
            )

        with ChunkCacheIndex(chunk_cache_path) as chunk_cache:
            chunk_paths, chunk_stats = prepare_cached_chunks(
                job_records,
                papers_root=papers_root,
                chunks_dir=chunks_dir,
                cache_index=chunk_cache,
                new_canonical_ids=new_job_ids,
                rebuild_canonical_ids=repaired_chunk_ids,
                on_progress=report_chunk_progress,
            )
        stats.update(chunk_stats)
        stats["chunk_part_count"] = len(chunk_paths)
        stats["papers_in_download"] = len(chunk_paths)
        stats["paper_count"] = len(job_records)
        stats["abstract_count"] = sum(
            1 for record in job_records if clean_text(record.get("abstract"))
        )
        stats["elapsed_seconds"] = round(time.monotonic() - started_monotonic, 2)
        store.save()

        summary = {
            "job_id": job_id,
            "status": "completed",
            "started_at": started_at,
            "completed_at": utc_now_iso(),
            "input": {
                "type": input_type,
                "parsing_mode": USER_INPUT_MODE,
                "input_parser_version": INPUT_PARSER_VERSION,
                "user_value": effective.raw_user_input,
                "user_keywords": list(effective.user_keywords),
                "user_exclusions": list(effective.user_exclusions),
                "defaults_included": True,
                "default_query_label": DEFAULT_QUERY_LABEL,
                "default_pmid_count": len(DEFAULT_PMIDS),
                "default_pmcid_count": len(DEFAULT_PMCIDS),
                "user_keyword_count": effective.user_keyword_count,
                "user_exclusion_count": effective.user_exclusion_count,
                "user_pmid_count": effective.user_pmid_count,
                "user_pmcid_count": effective.user_pmcid_count,
                "duplicate_user_keyword_count": effective.duplicate_user_keyword_count,
                "duplicate_user_exclusion_count": effective.duplicate_user_exclusion_count,
                "duplicate_user_pmid_count": effective.duplicate_user_pmid_count,
                "duplicate_user_pmcid_count": effective.duplicate_user_pmcid_count,
                "user_keyword_added": bool(effective.user_keywords),
                "user_exclusion_added": bool(effective.user_exclusions),
                "user_keyword_ignored_as_duplicate": effective.user_keyword_was_redundant,
                "keyword_search_mode": "single_augmented_query_with_exclusions",
                "keyword_limit": max(
                    1, min(int(keyword_limit), PUBMED_ESEARCH_API_CAP)
                ),
                "pubmed_result_mode": "limited_relevance_results",
                "metadata_retry_count": METADATA_HTTP_ATTEMPTS - 1,
                "metadata_request_timeout_seconds": metadata_timeout,
                "fulltext_request_timeout_seconds": min(
                    metadata_timeout, FULLTEXT_REQUEST_TIMEOUT
                ),
                "fulltext_attempt_mode": "retry_transient_refresh_negative_cache",
                "fulltext_retriever_version": FULLTEXT_RETRIEVER_VERSION,
                "fulltext_negative_cache_days": FULLTEXT_NEGATIVE_CACHE_DAYS,
                "fulltext_storage": "compact_bioc_json_gzip",
                "baseline_filter_enabled": bool(
                    excluded_pmids or excluded_pmcids or excluded_canonical_ids
                ),
            },
            "stats": stats,
            "files": {
                "storage_mode": "shared_per_paper",
                "chunk_cache_compression": "gzip",
                "download_compression": "none",
                "download_name": "chunks.jsonl",
                "chunk_parts": [
                    str(path.relative_to(papers_root)) for path in chunk_paths
                ],
            },
        }
        write_json(summary_path, summary)

        completion_message = (
            f"Finished: {stats['paper_count']} papers; "
            f"{stats['fulltexts_downloaded']} full texts downloaded; "
            f"{stats['papers_without_pmcid']} papers without a PMCID."
        )
        if stats["fulltext_failed"]:
            completion_message += (
                f" {stats['fulltext_failed']} full-text checks encountered a "
                "temporary error and remain eligible for retry."
            )
        reporter.emit(
            "completed",
            100,
            completion_message,
            stats,
            force=True,
        )

        return RetrievalResult(
            job_id=job_id,
            summary_path=summary_path,
            chunk_paths=chunk_paths,
            stats=stats,
            chunks_path=None,
        )
    finally:
        metadata_session.close()
        fulltext_session.close()
