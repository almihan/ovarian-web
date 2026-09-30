"""Document-aware Cell Ontology normalization for CellExLink cell mentions.

The active evidence hierarchy is:
1. an exact cell definition from the saved document-level Ab3P prepass;
2. an exact unique mapping in ``abbreviations.tsv``;
3. a unique exact Cell Ontology label or synonym;
4. the highest-cosine ontology term, accepted only at or above
   ``CELL_NORMALIZATION_MIN_COSINE`` (default 0.95).

There is no top-1/top-2 margin and no fuzzy abbreviation-key fallback in this
route. The prepass can recover cells, hormones and HGNC genes before NER.
Longer complete cell mentions may replace shorter locked mentions. All offsets
remain relative to the unchanged source chunk.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable as RuntimeIterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .resources import (
    CELL_ONTOLOGY_RESOURCE_VERSION,
    DEFAULT_ABBREVIATIONS_PATH,
    DEFAULT_ONTOLOGY_PATH,
)
from backend.pipeline.entity_span_rules import is_noncell_surface, token_aligned


def cell_min_cosine() -> float:
    raw = os.getenv("CELL_NORMALIZATION_MIN_COSINE", "0.95")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("CELL_NORMALIZATION_MIN_COSINE must be a number in [0, 1]") from exc
    if not 0 <= value <= 1:
        raise ValueError("CELL_NORMALIZATION_MIN_COSINE must be in [0, 1]")
    return value

from backend.pipeline.cell_surface_matching import cell_term_matches
from backend.pipeline.entity_text_normalization import (
    canonical_resource_key,
    canonical_short_form_key,
    compile_surface_pattern,
    short_form_key_candidates,
    spans_overlap,
)

logger = logging.getLogger(__name__)

# Retained as deprecated public constants for compatibility with older callers.
# The active pipeline no longer uses fuzzy abbreviation matching or a long-form
# cosine threshold. Ab3P long forms are validated by exact cell/hormone resource
# lookup in the pre-NER document pass.
AB3P_LONG_FORM_MIN_RAW_COSINE = 0.0
FUZZY_ABBREVIATION_MIN_COSINE = 0.0
FUZZY_ABBREVIATION_MIN_LENGTH = 0
FUZZY_ABBREVIATION_MAX_LENGTH = 0
EMBEDDING_CACHE_VERSION = "cellexlink-thresholded-vector-top1-v2026-09-11-f16"

NormalizerProgressCallback = Callable[
    [str, float, str, Mapping[str, Any]],
    None,
]

# Use a positive fixture shipped in pyab3p's own test data. A health check must
# test whether the binding and its packaged Ab3P resources work; it must not
# require the algorithm to recognize one particular domain example.
AB3P_HEALTHCHECK_TEXT = (
    "Respiratory syncytial viruses ( RSV ) are a subgroup of the "
    "paramyxoviruses."
)
AB3P_HEALTHCHECK_SHORT_FORM = "RSV"
AB3P_HEALTHCHECK_LONG_FORM = "Respiratory syncytial viruses"

# pyab3p loads these files from the top-level ``word_data`` package.  The
# macOS PyPI installation is built from the source distribution because no
# macOS wheel is published.  Some source installations contain Git LFS pointer
# text instead of the actual multi-megabyte WordData files; the extension then
# imports successfully but returns no abbreviation definitions.
AB3P_REQUIRED_WORD_DATA_FILES: tuple[tuple[str, int], ...] = (
    ("Ab3P_prec.dat", 1_000),
    ("cshset_wrdset3.ad", 100_000),
    ("cshset_wrdset3.ha", 1_000_000),
)
_GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class Ab3PError(RuntimeError):
    """Base exception for required Ab3P abbreviation processing."""


class Ab3PUnavailableError(Ab3PError):
    """Raised when pyab3p cannot be imported or initialized."""


class Ab3PHealthCheckError(Ab3PError):
    """Raised when the startup Ab3P health check does not produce its fixture."""


class Ab3POutputError(Ab3PError):
    """Raised when pyab3p returns an unsupported or malformed result."""


class Ab3PDocumentError(Ab3PError):
    """Raised when Ab3P fails while processing a selected document."""


_AB3P_DETECTOR: Any | None = None
_AB3P_INITIALIZATION_ERROR_MESSAGE: str | None = None
_AB3P_HEALTH_CHECK_ERROR_MESSAGE: str | None = None
_AB3P_HEALTH_CHECKED = False
_TOKEN_FINDER = re.compile(r"[^\W_]+|[^\w\s]|_|,", re.UNICODE)
_BOUNDARY_LEFT = r"(?<![A-Za-z0-9])"
_BOUNDARY_RIGHT = r"(?![A-Za-z0-9])"


# ---------------------------------------------------------------------------
# Lexical helpers retained from CellExLink's ontology linker
# ---------------------------------------------------------------------------
def split_tokens(value: object) -> list[str]:
    return _TOKEN_FINDER.findall(str(value))


def _replace_tail(word: str, suffix: str, replacement: str) -> str:
    return word[: -len(suffix)] + replacement


def normalize_token(value: object) -> str:
    word = str(value)
    if not word.endswith("s"):
        return word
    if word.endswith("viruses"):
        return _replace_tail(word, "uses", "us")
    if word.endswith("ies") and not word.endswith(("eies", "aies")):
        return _replace_tail(word, "ies", "y")
    if word.endswith("es") and not word.endswith(("aes", "ees", "oes")):
        if word.endswith("sses"):
            return _replace_tail(word, "es", "")
        return _replace_tail(word, "es", "e")
    if word.endswith(("us", "ss")):
        return word
    return _replace_tail(word, "s", "")


def plural_normalize_text(value: object) -> str:
    return " ".join(normalize_token(part) for part in split_tokens(value))


def _casefold_normalize_text(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _ontology_alias_key(value: object) -> str:
    return _casefold_normalize_text(plural_normalize_text(value))


def token_jaccard(left: object, right: object) -> float:
    left_tokens = set(_casefold_normalize_text(left).split())
    right_tokens = set(_casefold_normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def sequence_ratio(left: object, right: object) -> float:
    return SequenceMatcher(
        None, _casefold_normalize_text(left), _casefold_normalize_text(right)
    ).ratio()


def has_parenthetical_relation(left: object, right: object) -> float:
    left_text = str(left)
    right_text = str(right)
    if "(" in left_text and ")" in left_text:
        if any(
            _casefold_normalize_text(item) == _casefold_normalize_text(right_text)
            for item in re.findall(r"\(([^()]*)\)", left_text)
        ):
            return 1.0
    if "(" in right_text and ")" in right_text:
        if any(
            _casefold_normalize_text(item) == _casefold_normalize_text(left_text)
            for item in re.findall(r"\(([^()]*)\)", right_text)
        ):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Abbreviation-key rules
# ---------------------------------------------------------------------------
def canonical_abbreviation_key(text: object) -> str:
    """Backward-compatible alias for the shared exact short-form key."""

    return canonical_short_form_key(text)


def protected_signature(key: str) -> str:
    return "".join(
        character
        for character in str(key)
        if character.isdigit() or character in "+-/"
    )


def is_controlled_plural_variant(left_key: str, right_key: str) -> bool:
    return (
        left_key.endswith("S") and left_key[:-1] == right_key
    ) or (
        right_key.endswith("S") and right_key[:-1] == left_key
    )


def fuzzy_abbreviation_allowed(canonical_mention_key: str) -> bool:
    """Return False: fuzzy abbreviation-key matching is intentionally disabled."""

    return False



# ---------------------------------------------------------------------------
# Ontology and static abbreviation resources
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class TermEntry:
    name: str
    raw_name: str
    identifier: str
    preferred_label: str
    is_preferred: bool = False


@dataclass(slots=True)
class ConceptMetadata:
    preferred_label: str
    synonyms: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    namespace: str = ""


@dataclass(slots=True, frozen=True)
class StaticAbbreviationCandidate:
    short_form: str
    key: str
    identifier: str
    stable_index: int
    resource_line: int | None = None


@dataclass(slots=True)
class StaticAbbreviationLookup:
    key_to_candidates: dict[str, list[StaticAbbreviationCandidate]] = field(
        default_factory=dict
    )
    all_keys: list[str] = field(default_factory=list)
    key_to_stable_index: dict[str, int] = field(default_factory=dict)
    keys_by_signature: dict[str, list[str]] = field(default_factory=dict)

    def valid_identifiers(
        self,
        key: str,
        concept_metadata: Mapping[str, ConceptMetadata],
        *, surface: str | None = None,
    ) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for candidate in self.key_to_candidates.get(key, []):
            if surface is not None and not cell_term_matches(
                    surface, candidate.short_form, "static_abbreviation"):
                continue
            identifier = candidate.identifier
            if identifier in concept_metadata and identifier not in seen:
                seen.add(identifier)
                output.append(identifier)
        return output

    def __bool__(self) -> bool:
        return bool(self.all_keys)


def load_cell_ontology_terms(
    ontology_path: str | Path,
) -> tuple[
    list[TermEntry],
    dict[str, ConceptMetadata],
    dict[str, tuple[str, ...]],
]:
    path = Path(ontology_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cell Ontology JSONL does not exist: {path}")

    term_entries: list[TermEntry] = []
    concept_metadata: dict[str, ConceptMetadata] = {}
    alias_to_ids: dict[str, list[str]] = defaultdict(list)
    seen_entries: set[tuple[str, str]] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_no} in {path}: {exc}") from exc

            identifier = str(record.get("norm_concept_id") or "").strip()
            preferred_label = str(record.get("norm_preferred_label") or "").strip()
            synonyms = record.get("synonyms") or []
            namespace = str(record.get("namespace") or "")
            if not identifier or not preferred_label:
                continue
            if not isinstance(synonyms, list):
                raise ValueError(f"Expected synonyms list on line {line_no} in {path}")

            metadata = concept_metadata.setdefault(
                identifier,
                ConceptMetadata(
                    preferred_label=preferred_label,
                    namespace=namespace,
                ),
            )
            names = [(preferred_label, True)] + [
                (str(value).strip(), False) for value in synonyms if str(value).strip()
            ]
            for name, is_preferred in names:
                metadata.names.add(name)
                if not is_preferred:
                    metadata.synonyms.add(name)
                alias_key = _ontology_alias_key(name)
                if identifier not in alias_to_ids[alias_key]:
                    alias_to_ids[alias_key].append(identifier)
                entry_signature = (identifier, name.casefold())
                if entry_signature in seen_entries:
                    continue
                seen_entries.add(entry_signature)
                term_entries.append(
                    TermEntry(
                        name=plural_normalize_text(name),
                        raw_name=name,
                        identifier=identifier,
                        preferred_label=preferred_label,
                        is_preferred=is_preferred,
                    )
                )

    if not term_entries:
        raise ValueError(f"No usable Cell Ontology terms were loaded from {path}")
    return (
        term_entries,
        concept_metadata,
        {key: tuple(values) for key, values in alias_to_ids.items()},
    )


def load_static_abbreviations(
    abbreviations_path: str | Path | None,
) -> StaticAbbreviationLookup:
    if abbreviations_path is None:
        return StaticAbbreviationLookup()
    path = Path(abbreviations_path)
    if not path.is_file():
        raise FileNotFoundError(f"Abbreviation resource does not exist: {path}")

    key_to_candidates: dict[str, list[StaticAbbreviationCandidate]] = defaultdict(list)
    all_keys: list[str] = []
    key_to_stable_index: dict[str, int] = {}
    stable_row_index = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            fields = line.rstrip("\n").split("\t")
            if line_no == 0 and fields[:2] == ["short_form", "matched_cl_id"]:
                continue
            if len(fields) < 2:
                continue
            short_form = fields[0].strip()
            raw_identifiers = fields[1].strip()
            key = canonical_abbreviation_key(short_form)
            if not key or raw_identifiers in {"", "-", "None", "none"}:
                continue

            identifiers = [
                value.strip()
                for value in re.split(r"[,;]", raw_identifiers)
                if value.strip() not in {"", "-", "None", "none"}
            ]
            if not identifiers:
                continue
            if key not in key_to_stable_index:
                key_to_stable_index[key] = len(all_keys)
                all_keys.append(key)
            for identifier in identifiers:
                key_to_candidates[key].append(
                    StaticAbbreviationCandidate(
                        short_form=short_form,
                        key=key,
                        identifier=identifier,
                        stable_index=stable_row_index,
                        resource_line=line_no + 1,
                    )
                )
                stable_row_index += 1

    keys_by_signature: dict[str, list[str]] = defaultdict(list)
    for key in all_keys:
        keys_by_signature[protected_signature(key)].append(key)
    return StaticAbbreviationLookup(
        key_to_candidates=dict(key_to_candidates),
        all_keys=all_keys,
        key_to_stable_index=key_to_stable_index,
        keys_by_signature=dict(keys_by_signature),
    )


# ---------------------------------------------------------------------------
# Selected document text and offset mapping
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class ChunkOffset:
    chunk_id: str
    section: str
    document_start: int
    document_end: int
    chunk_order: int
    source: Mapping[str, Any]

    def contains(self, start: int, end: int) -> bool:
        return self.document_start <= start <= end <= self.document_end


@dataclass(slots=True)
class DocumentText:
    document_key: str
    text: str
    chunks: list[ChunkOffset]

    def chunk_by_id(self, chunk_id: object) -> ChunkOffset | None:
        target = str(chunk_id or "")
        for chunk in self.chunks:
            if chunk.chunk_id == target:
                return chunk
        return None

    def chunk_for_span(self, start: int, end: int) -> ChunkOffset | None:
        for chunk in self.chunks:
            if chunk.contains(start, end):
                return chunk
        return None

    def document_offset_for_mention(self, mention: Mapping[str, Any]) -> int | None:
        chunk = self.chunk_by_id(mention.get("chunk_id"))
        if chunk is None:
            return None
        try:
            start = int(mention.get("start"))
        except (TypeError, ValueError):
            return None
        return chunk.document_start + start


def build_document_text(
    chunk_records: Sequence[Mapping[str, Any]],
    *,
    document_key: str | None = None,
) -> DocumentText:
    parts: list[str] = []
    chunks: list[ChunkOffset] = []
    cursor = 0
    inferred_key = ""

    for chunk_order, record in enumerate(chunk_records):
        if chunk_order:
            parts.append("\n")
            cursor += 1
        text = str(record.get("chunk") or "")
        start = cursor
        parts.append(text)
        cursor += len(text)
        end = cursor
        chunk_id = str(record.get("chunk_id") or f"chunk-{chunk_order + 1}")
        section = str(record.get("section_type") or "")
        chunks.append(
            ChunkOffset(
                chunk_id=chunk_id,
                section=section,
                document_start=start,
                document_end=end,
                chunk_order=chunk_order,
                source=dict(record),
            )
        )
        if not inferred_key:
            inferred_key = str(
                record.get("doc_key")
                or record.get("canonical_id")
                or record.get("pmid")
                or record.get("pmcid")
                or ""
            )

    return DocumentText(
        document_key=str(document_key or inferred_key),
        text="".join(parts),
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# Ab3P definitions and final normalization evidence
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class OntologyMatch:
    identifier: str
    preferred_label: str
    matched_alias: str
    raw_cosine: float
    exact_unique_alias: bool = False
    final_score: float | None = None
    term_kind: str | None = None
    match_metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class Ab3PDefinition:
    short_form: str
    long_form: str
    key: str
    stable_index: int
    definition_id: str = ""
    definition_detector: str = "ab3p"
    location_status: str = "unlocated"
    resolution_status: str = "unmatched"
    definition_start: int | None = None
    definition_end: int | None = None
    document_start: int | None = None
    document_end: int | None = None
    long_form_document_start: int | None = None
    long_form_document_end: int | None = None
    chunk_id: str | None = None
    section: str | None = None
    chunk_start: int | None = None
    long_form_chunk_start: int | None = None
    entity_type: str | None = None
    concept_id: str | None = None
    preferred_label: str | None = None
    matched_term: str | None = None
    term_kind: str | None = None
    resource_version: str | None = None
    match_method: str | None = None
    ontology_match: OntologyMatch | None = None

    @property
    def resolved(self) -> bool:
        return bool(self.concept_id and self.entity_type)


@dataclass(slots=True)
class DocumentAbbreviationContext:
    document_key: str
    ab3p_status: str = "not_run"
    definitions: list[Ab3PDefinition] = field(default_factory=list)
    definitions_by_key: dict[str, list[Ab3PDefinition]] = field(default_factory=dict)
    all_keys: list[str] = field(default_factory=list)
    key_to_stable_index: dict[str, int] = field(default_factory=dict)
    key_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    query_embedding_cache: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def validated_definition_count(self) -> int:
        return sum(1 for item in self.definitions if item.resolved)

    @property
    def cell_definition_count(self) -> int:
        return sum(1 for item in self.definitions if item.entity_type == "cell")

    @property
    def hormone_definition_count(self) -> int:
        return sum(1 for item in self.definitions if item.entity_type == "hormone")


@dataclass(slots=True, frozen=True)
class FuzzyKeyMatch:
    """Deprecated compatibility record; fuzzy key matching is not active."""

    key: str
    cosine: float


@dataclass(slots=True)
class NormalizationDecision:
    mention: str
    normalized_id: str | None = None
    preferred_label: str | None = None
    normalization_source: str = "unresolved"
    matched_abbreviation_key: str | None = None
    matched_static_abbreviation_key: str | None = None
    expanded_long_form: str | None = None
    abbreviation_key_cosine: float | None = None
    ab3p_key_cosine: float | None = None
    ab3p_match_method: str | None = None
    ontology_raw_cosine: float | None = None
    matched_ontology_alias: str | None = None
    normalization_score: float | None = None
    matched_term: str | None = None
    term_kind: str | None = None
    resource_version: str | None = None
    match_metadata: Mapping[str, Any] | None = None

    @property
    def normalized(self) -> bool:
        return bool(self.normalized_id)

    def to_annotation_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "normalization_status": "normalized" if self.normalized else "unresolved",
            "cell_ontology_id": self.normalized_id,
            "cell_ontology_label": self.preferred_label,
            "normalization_source": self.normalization_source,
            "matched_abbreviation_key": self.matched_abbreviation_key,
            "matched_static_abbreviation_key": self.matched_static_abbreviation_key,
            "expanded_long_form": self.expanded_long_form,
            "abbreviation_key_cosine": self.abbreviation_key_cosine,
            "ab3p_key_cosine": self.ab3p_key_cosine,
            "ab3p_match_method": self.ab3p_match_method,
            "ontology_raw_cosine": self.ontology_raw_cosine,
            "matched_ontology_alias": self.matched_ontology_alias,
            "normalization_score": self.normalization_score,
            "concept_id": self.normalized_id,
            "preferred_label": self.preferred_label,
            "matched_term": self.matched_term,
            "term_kind": self.term_kind,
            "resource_version": self.resource_version,
            **dict(self.match_metadata or {}),
        }
        return fields


@dataclass(slots=True, frozen=True)
class RescuedMention:
    mention: str
    start: int
    end: int
    chunk: ChunkOffset
    decision: NormalizationDecision


# ---------------------------------------------------------------------------
# Ab3P execution and definition location
# ---------------------------------------------------------------------------
def _normalize_pyab3p_output(results: Any) -> list[tuple[str, str]]:
    """Convert pyab3p output without hiding unsupported return shapes."""

    if results is None:
        raise Ab3POutputError("pyab3p.get_abbrs() returned None instead of a result collection.")
    if isinstance(results, Mapping):
        iterable: Iterable[Any] = list(results.items())
    elif isinstance(results, RuntimeIterable) and not isinstance(
        results, (str, bytes, bytearray)
    ):
        iterable = list(results)
    else:
        raise Ab3POutputError(
            "pyab3p.get_abbrs() returned an unsupported result type: "
            f"{type(results).__name__}."
        )

    pairs: list[tuple[str, str]] = []
    malformed_indices: list[int] = []
    for item_index, item in enumerate(iterable):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            short_form = str(item[0]).strip()
            long_form = str(item[1]).strip()
        elif isinstance(item, Mapping):
            short_form = str(
                item.get("short_form")
                or item.get("short")
                or item.get("abbr")
                or item.get("abbreviation")
                or ""
            ).strip()
            long_form = str(
                item.get("long_form")
                or item.get("long")
                or item.get("expansion")
                or ""
            ).strip()
        else:
            short_form = str(
                getattr(item, "short_form", getattr(item, "short", ""))
            ).strip()
            long_form = str(
                getattr(item, "long_form", getattr(item, "long", ""))
            ).strip()
        if not short_form or not long_form:
            malformed_indices.append(item_index)
            continue
        pairs.append((short_form, long_form))

    if malformed_indices:
        preview = ", ".join(str(index) for index in malformed_indices[:5])
        suffix = "" if len(malformed_indices) <= 5 else ", ..."
        raise Ab3POutputError(
            "pyab3p.get_abbrs() returned result item(s) without both "
            f"short_form and long_form at index(es) {preview}{suffix}."
        )
    return pairs


def _resolve_ab3p_word_data_directory() -> Path:
    """Resolve the exact WordData directory that pyab3p will use."""

    try:
        import word_data  # type: ignore
    except Exception as exc:
        raise Ab3PUnavailableError(
            "Ab3P is unavailable because its required top-level word_data "
            "package cannot be imported. Install pyab3p with complete WordData "
            "resources in the same Python environment used by Uvicorn. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    package_paths = [
        Path(str(item)).expanduser().resolve()
        for item in list(getattr(word_data, "__path__", ()) or ())
        if str(item).strip()
    ]
    if package_paths:
        return package_paths[0]

    module_file = getattr(word_data, "__file__", None)
    if module_file:
        return Path(str(module_file)).expanduser().resolve().parent

    raise Ab3PUnavailableError(
        "Ab3P is unavailable because the imported word_data package has no "
        "filesystem path."
    )


def _assert_ab3p_word_data_available() -> Path:
    """Reject missing, truncated, or Git-LFS-pointer WordData resources."""

    word_data_dir = _resolve_ab3p_word_data_directory()
    missing: list[str] = []
    lfs_pointers: list[str] = []
    undersized: list[str] = []
    unreadable: list[str] = []

    for file_name, minimum_size in AB3P_REQUIRED_WORD_DATA_FILES:
        resource_path = word_data_dir / file_name
        if not resource_path.is_file():
            missing.append(file_name)
            continue

        try:
            size = resource_path.stat().st_size
            with resource_path.open("rb") as stream:
                prefix = stream.read(256).lstrip()
        except OSError as exc:
            unreadable.append(f"{file_name} ({type(exc).__name__}: {exc})")
            continue

        if prefix.startswith(_GIT_LFS_POINTER_PREFIX):
            lfs_pointers.append(file_name)
            continue
        if size < minimum_size:
            undersized.append(
                f"{file_name} ({size} bytes; expected at least {minimum_size})"
            )

    if missing or lfs_pointers or undersized or unreadable:
        details: list[str] = [f"word_data_path={word_data_dir}"]
        if missing:
            details.append("missing=" + ", ".join(missing))
        if lfs_pointers:
            details.append("git_lfs_pointers=" + ", ".join(lfs_pointers))
        if undersized:
            details.append("undersized=" + ", ".join(undersized))
        if unreadable:
            details.append("unreadable=" + ", ".join(unreadable))

        raise Ab3PUnavailableError(
            "Ab3P is installed, but its WordData resources are incomplete. "
            "The Python extension can import in this state, yet get_abbrs() "
            "returns an empty list for known-positive text. "
            + "; ".join(details)
            + ". On macOS, activate the same virtual environment used to run "
            "Uvicorn, install Git LFS if necessary (`brew install git-lfs` "
            "and `git lfs install`), then run "
            "`bash scripts/install_pyab3p_macos.sh`."
        )

    logger.info(
        "[AB3P_WORD_DATA] status=available path=%s",
        word_data_dir,
    )
    return word_data_dir


def _get_ab3p_detector() -> Any:
    """Import and initialize the required pyab3p detector exactly once."""

    global _AB3P_DETECTOR, _AB3P_INITIALIZATION_ERROR_MESSAGE
    if _AB3P_DETECTOR is not None:
        return _AB3P_DETECTOR
    if _AB3P_INITIALIZATION_ERROR_MESSAGE is not None:
        raise Ab3PUnavailableError(_AB3P_INITIALIZATION_ERROR_MESSAGE)

    try:
        import pyab3p  # type: ignore
    except Exception as exc:
        _AB3P_INITIALIZATION_ERROR_MESSAGE = (
            "Ab3P is unavailable: pyab3p could not be imported. Install the "
            "local dependencies in the same Python environment used to run "
            "Uvicorn. Original error: "
            f"{type(exc).__name__}: {exc}"
        )
        raise Ab3PUnavailableError(_AB3P_INITIALIZATION_ERROR_MESSAGE) from exc

    try:
        _assert_ab3p_word_data_available()
        detector_factory = getattr(pyab3p, "Ab3p", None)
        if not callable(detector_factory):
            raise AttributeError("the pyab3p module does not expose callable Ab3p")
        detector = detector_factory()
        if not callable(getattr(detector, "get_abbrs", None)):
            raise AttributeError("pyab3p.Ab3p() does not expose callable get_abbrs")
    except Ab3PUnavailableError as exc:
        _AB3P_INITIALIZATION_ERROR_MESSAGE = str(exc)
        raise
    except Exception as exc:
        _AB3P_INITIALIZATION_ERROR_MESSAGE = (
            "Ab3P is unavailable: pyab3p could not be initialized after its "
            "WordData resources were checked. Original error: "
            f"{type(exc).__name__}: {exc}"
        )
        raise Ab3PUnavailableError(_AB3P_INITIALIZATION_ERROR_MESSAGE) from exc

    _AB3P_DETECTOR = detector
    return detector


def ensure_ab3p_healthy() -> tuple[str, str]:
    """Fail fast unless Ab3P extracts its known-positive library fixture.

    This check verifies the imported binding, packaged word-data resources, and
    extraction engine. A document-specific phrase that Ab3P legitimately does
    not recognize is not a suitable availability check.
    """

    global _AB3P_HEALTH_CHECKED, _AB3P_HEALTH_CHECK_ERROR_MESSAGE
    if _AB3P_HEALTH_CHECKED:
        return AB3P_HEALTHCHECK_SHORT_FORM, AB3P_HEALTHCHECK_LONG_FORM
    if _AB3P_HEALTH_CHECK_ERROR_MESSAGE is not None:
        raise Ab3PHealthCheckError(_AB3P_HEALTH_CHECK_ERROR_MESSAGE)

    detector = _get_ab3p_detector()
    try:
        pairs = _normalize_pyab3p_output(
            detector.get_abbrs(AB3P_HEALTHCHECK_TEXT)
        )
    except Exception as exc:
        _AB3P_HEALTH_CHECK_ERROR_MESSAGE = (
            "Ab3P startup health check failed while processing the fixture "
            f"{AB3P_HEALTHCHECK_TEXT!r}. Original error: "
            f"{type(exc).__name__}: {exc}"
        )
        raise Ab3PHealthCheckError(_AB3P_HEALTH_CHECK_ERROR_MESSAGE) from exc

    expected_short_key = canonical_abbreviation_key(AB3P_HEALTHCHECK_SHORT_FORM)
    expected_long_form = _casefold_normalize_text(AB3P_HEALTHCHECK_LONG_FORM)
    matched_pair = next(
        (
            (short_form, long_form)
            for short_form, long_form in pairs
            if canonical_abbreviation_key(short_form) == expected_short_key
            and _casefold_normalize_text(long_form) == expected_long_form
        ),
        None,
    )
    if matched_pair is None:
        received = pairs[:5]
        _AB3P_HEALTH_CHECK_ERROR_MESSAGE = (
            "Ab3P was imported and initialized, but its known-positive startup "
            "fixture failed. Expected "
            f"{AB3P_HEALTHCHECK_SHORT_FORM} -> {AB3P_HEALTHCHECK_LONG_FORM} "
            f"from input {AB3P_HEALTHCHECK_TEXT!r}, but received {received!r}. "
            "This indicates an unhealthy pyab3p installation or missing/corrupt "
            "packaged Ab3P word-data resources, not a document with no "
            "abbreviation definitions."
        )
        raise Ab3PHealthCheckError(_AB3P_HEALTH_CHECK_ERROR_MESSAGE)

    _AB3P_HEALTH_CHECKED = True
    logger.info(
        "[AB3P_HEALTH] status=passed short_form=%s long_form=%s",
        matched_pair[0],
        matched_pair[1],
    )
    return matched_pair


def run_ab3p_for_document(
    text: str,
    *,
    document_key: str | None = None,
) -> list[tuple[str, str]]:
    """Run healthy Ab3P once and retain every reported definition."""

    document_text = str(text)
    if not document_text.strip():
        return []

    ensure_ab3p_healthy()
    detector = _get_ab3p_detector()
    try:
        return _normalize_pyab3p_output(detector.get_abbrs(document_text))
    except Exception as exc:
        label = str(document_key or "<unknown-document>")
        raise Ab3PDocumentError(
            f"Ab3P failed while processing document {label}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _find_literal_matches(text: str, literal: str) -> list[re.Match[str]]:
    if not literal:
        return []
    pattern = re.compile(re.escape(literal), flags=re.IGNORECASE)
    return list(pattern.finditer(text))


def _locate_definition_short_form(
    document_text: str,
    short_form: str,
    long_form: str,
    used_positions: set[int],
) -> tuple[int | None, int | None]:
    sf = re.escape(short_form)
    lf = re.escape(long_form)
    patterns = (
        re.compile(rf"{lf}\s*\(\s*(?P<sf>{sf})\s*\)", re.IGNORECASE),
        re.compile(rf"(?P<sf>{sf})\s*\(\s*{lf}\s*\)", re.IGNORECASE),
    )
    for pattern in patterns:
        for match in pattern.finditer(document_text):
            start, end = match.span("sf")
            if start not in used_positions:
                used_positions.add(start)
                return start, end

    # Do not assign an arbitrary short-form usage as a definition location.
    # The production prepass preserves such Ab3P output as an unlocated
    # definition instead of inventing provenance.
    return None, None


def cache_document_definitions(
    document: DocumentText,
    pairs: Sequence[tuple[str, str]],
) -> list[Ab3PDefinition]:
    definitions: list[Ab3PDefinition] = []
    used_positions: set[int] = set()
    for stable_index, (short_form, long_form) in enumerate(pairs):
        key = canonical_abbreviation_key(short_form)
        if not key:
            continue
        start, end = _locate_definition_short_form(
            document.text,
            short_form,
            long_form,
            used_positions,
        )
        chunk = (
            document.chunk_for_span(start, end)
            if start is not None and end is not None
            else None
        )
        definitions.append(
            Ab3PDefinition(
                short_form=short_form,
                long_form=long_form,
                key=key,
                stable_index=stable_index,
                document_start=start,
                document_end=end,
                chunk_id=chunk.chunk_id if chunk else None,
                section=chunk.section if chunk else None,
                chunk_start=(start - chunk.document_start) if chunk and start is not None else None,
            )
        )
    return definitions


# ---------------------------------------------------------------------------
# Encoder cache helpers
# ---------------------------------------------------------------------------
def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_torch_threads(cpu_threads: int) -> None:
    safe_threads = max(1, int(cpu_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(safe_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(safe_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    torch.set_num_threads(safe_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, safe_threads)))
    except RuntimeError:
        pass


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)



def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main linker
# ---------------------------------------------------------------------------
class CellOntologyNormalizer:
    """Document-aware SapBERT linker with explicit abbreviation evidence."""

    def __init__(
        self,
        *,
        model_name_or_path: str | Path,
        model_cache_dir: str | Path | None,
        embedding_cache_dir: str | Path,
        ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
        abbreviations_path: str | Path | None = DEFAULT_ABBREVIATIONS_PATH,
        disable_abbreviations: bool = False,
        batch_size: int = 64,
        cpu_threads: int = 2,
        trust_remote_code: bool = False,
        device: str = "auto",
        model_identity: str | None = None,
        progress_callback: NormalizerProgressCallback | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        requested_device = str(device or "auto").strip().casefold()
        if requested_device not in {"auto", "cpu"}:
            raise ValueError("device must be either 'auto' or 'cpu'")
        _set_torch_threads(cpu_threads)

        self.model_reference = str(model_name_or_path)
        self.model_identity = str(
            model_identity
            or Path(self.model_reference).name
            or self.model_reference
        )
        self.model_cache_dir = str(model_cache_dir) if model_cache_dir else None
        self.embedding_cache_dir = Path(embedding_cache_dir).expanduser().resolve()
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        self.ontology_path = Path(ontology_path).expanduser().resolve()
        self.abbreviations_path = (
            Path(abbreviations_path).expanduser().resolve()
            if abbreviations_path is not None
            else None
        )
        self.disable_abbreviations = bool(disable_abbreviations)
        self.batch_size = int(batch_size)
        self.trust_remote_code = bool(trust_remote_code)
        self.requested_device = requested_device
        self.progress_callback = progress_callback
        self.dictionary_embedding_cache_reused: bool | None = None
        self.selected_device_name = "not loaded"

        self._notify(
            "normalization_resources",
            2,
            "Reading the local Cell Ontology resource...",
            {},
        )
        (
            self.term_entries,
            self.concept_metadata,
            self.alias_to_identifiers,
        ) = load_cell_ontology_terms(self.ontology_path)
        self._notify(
            "normalization_resources",
            4,
            (
                f"Loaded {len(self.concept_metadata):,} Cell Ontology concepts "
                f"and {len(self.term_entries):,} searchable labels and synonyms."
            ),
            {
                "cell_ontology_concepts": len(self.concept_metadata),
                "cell_ontology_terms": len(self.term_entries),
            },
        )
        self.static_lookup = (
            StaticAbbreviationLookup()
            if self.disable_abbreviations
            else load_static_abbreviations(self.abbreviations_path)
        )

        self.torch: Any | None = None
        self.device: Any | None = None
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.dictionary_embeddings: np.ndarray | None = None
        self.static_key_embeddings: np.ndarray | None = None
        self._ontology_digest = _file_sha256(self.ontology_path)
        self.ontology_version = (
            f"{self.ontology_path.stem}-{self._ontology_digest[:12]}"
        )
        self._abbreviation_digest = (
            _file_sha256(self.abbreviations_path)
            if self.abbreviations_path and self.abbreviations_path.is_file()
            else "none"
        )

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @property
    def compute_device(self) -> str:
        return self.selected_device_name

    def _notify(
        self,
        stage: str,
        percent: float,
        message: str,
        stats: Mapping[str, Any] | None = None,
    ) -> None:
        callback = self.progress_callback
        if callback is None:
            return
        payload = {
            "normalization_requested_device": self.requested_device,
            "normalization_compute_device": self.selected_device_name,
            "ontology_embedding_cache_reused": (
                self.dictionary_embedding_cache_reused
            ),
            **dict(stats or {}),
        }
        try:
            callback(stage, float(percent), str(message), payload)
        except Exception as exc:
            # Progress reporting is informational and must never abort NEN.
            logger.warning("Could not report CellExLink normalization progress: %s", exc)

    @contextmanager
    def _heartbeat(
        self,
        *,
        stage: str,
        percent: float,
        message: str,
        stats: Mapping[str, Any] | None = None,
        interval_seconds: float = 5.0,
    ) -> Iterable[None]:
        """Keep the UI alive while a model-loading operation blocks Python."""

        stop = threading.Event()
        started = time.monotonic()

        def report() -> None:
            while not stop.wait(max(1.0, float(interval_seconds))):
                elapsed = int(time.monotonic() - started)
                self._notify(
                    stage,
                    percent,
                    f"{message} ({elapsed:,} seconds elapsed)",
                    {
                        **dict(stats or {}),
                        "normalization_step_elapsed_seconds": elapsed,
                    },
                )

        thread = threading.Thread(
            target=report,
            name="cellexlink-normalization-heartbeat",
            daemon=True,
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=max(1.0, float(interval_seconds)) + 1.0)

    def _resolve_device(self, torch: Any) -> Any:
        if self.requested_device == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _device_display_name(torch: Any, device: Any) -> str:
        if getattr(device, "type", "cpu") == "cuda":
            try:
                return f"CUDA: {torch.cuda.get_device_name(device)}"
            except Exception:
                return "CUDA GPU"
        if getattr(device, "type", "cpu") == "mps":
            return "Apple Metal (MPS)"
        return "CPU"

    def _prepare_encoder(self) -> None:
        if self.model_loaded:
            return
        self._notify(
            "normalization_model",
            6,
            "Importing PyTorch and Transformers for CellExLink normalization...",
            {},
        )
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = self._resolve_device(torch)
        self.selected_device_name = self._device_display_name(torch, self.device)
        self._notify(
            "normalization_model",
            8,
            f"Selected {self.selected_device_name} for CellExLink normalization.",
            {},
        )
        common_kwargs = {
            "cache_dir": self.model_cache_dir,
            "trust_remote_code": self.trust_remote_code,
        }
        if Path(self.model_reference).expanduser().exists():
            common_kwargs["local_files_only"] = True

        tokenizer_message = "Loading the saved SapBERT tokenizer from the local cache..."
        self._notify("normalization_model", 10, tokenizer_message, {})
        with self._heartbeat(
            stage="normalization_model",
            percent=10,
            message=tokenizer_message,
        ):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_reference,
                **common_kwargs,
            )

        model_message = "Loading the saved SapBERT encoder weights from the local cache..."
        self._notify("normalization_model", 14, model_message, {})
        with self._heartbeat(
            stage="normalization_model",
            percent=14,
            message=model_message,
        ):
            self.model = AutoModel.from_pretrained(
                self.model_reference,
                **common_kwargs,
            )

        move_message = f"Moving the SapBERT encoder to {self.selected_device_name}..."
        self._notify("normalization_model", 18, move_message, {})
        try:
            with self._heartbeat(
                stage="normalization_model",
                percent=18,
                message=move_message,
            ):
                self.model.to(self.device)
        except Exception as exc:
            if getattr(self.device, "type", "cpu") == "cpu":
                raise
            logger.warning(
                "Could not initialize CellExLink normalization on %s; "
                "falling back to CPU: %s",
                self.selected_device_name,
                exc,
            )
            self.device = torch.device("cpu")
            self.selected_device_name = "CPU"
            self._notify(
                "normalization_model",
                18,
                (
                    "The selected accelerator could not initialize; CellExLink "
                    "normalization is continuing on CPU."
                ),
                {"normalization_device_fallback": type(exc).__name__},
            )
            self.model.to(self.device)
        self.model.eval()
        self._notify(
            "normalization_model",
            20,
            f"SapBERT is ready on {self.selected_device_name}.",
            {"normalization_model_loaded": True},
        )

    def _encode_texts(
        self,
        texts: Sequence[str],
        *,
        purpose: str = "cell mentions",
        progress_start: float = 67,
        progress_end: float = 69,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._prepare_encoder()
        assert self.torch is not None
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.device is not None

        representations: list[np.ndarray] = []
        total = len(texts)
        batch_count = (total + self.batch_size - 1) // self.batch_size
        self._notify(
            "normalization_encoding",
            progress_start,
            f"Encoding {total:,} {purpose} in {batch_count:,} batches...",
            {
                "normalization_encoding_total": total,
                "normalization_encoding_completed": 0,
                "normalization_encoding_purpose": purpose,
            },
        )
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = [str(value) for value in texts[start : start + self.batch_size]]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    max_length=32,
                    truncation=True,
                    return_tensors="pt",
                )
                try:
                    device_tokens = {
                        key: value.to(self.device) for key, value in tokens.items()
                    }
                    output = self.model(**device_tokens)
                except RuntimeError as exc:
                    if getattr(self.device, "type", "cpu") == "cpu":
                        raise
                    logger.warning(
                        "CellExLink normalization failed on %s during encoding; "
                        "retrying on CPU: %s",
                        self.selected_device_name,
                        exc,
                    )
                    self.device = self.torch.device("cpu")
                    self.selected_device_name = "CPU"
                    self.model.to(self.device)
                    self.model.eval()
                    self._notify(
                        "normalization_encoding",
                        progress_start,
                        "Accelerator inference failed; retrying this batch on CPU.",
                        {"normalization_device_fallback": type(exc).__name__},
                    )
                    device_tokens = {
                        key: value.to(self.device) for key, value in tokens.items()
                    }
                    output = self.model(**device_tokens)
                hidden = output[0] if isinstance(output, tuple) else output.last_hidden_state
                representations.append(hidden[:, 0, :].detach().cpu().float().numpy())
                completed = min(total, start + len(batch))
                batch_percent = progress_start + (
                    (progress_end - progress_start) * completed / max(1, total)
                )
                self._notify(
                    "normalization_encoding",
                    batch_percent,
                    f"Encoded {completed:,} of {total:,} {purpose}.",
                    {
                        "normalization_encoding_total": total,
                        "normalization_encoding_completed": completed,
                        "normalization_encoding_purpose": purpose,
                    },
                )
        matrix = np.concatenate(representations, axis=0).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def _embedding_cache_paths(
        self,
        *,
        kind: str,
        resource_digest: str,
        count: int,
    ) -> tuple[Path, Path]:
        key = hashlib.sha256(
            "|".join(
                (
                    EMBEDDING_CACHE_VERSION,
                    kind,
                    self.model_identity,
                    resource_digest,
                    str(count),
                )
            ).encode("utf-8")
        ).hexdigest()[:24]
        return (
            self.embedding_cache_dir / f"{kind}-{key}.npy",
            self.embedding_cache_dir / f"{kind}-{key}.json",
        )

    def _load_or_build_embeddings(
        self,
        *,
        kind: str,
        names: Sequence[str],
        resource_digest: str,
    ) -> np.ndarray:
        cache_path, metadata_path = self._embedding_cache_paths(
            kind=kind,
            resource_digest=resource_digest,
            count=len(names),
        )
        self._notify(
            "normalization_dictionary",
            22,
            (
                f"Checking the saved {kind.replace('-', ' ')} embedding cache "
                f"for {len(names):,} terms..."
            ),
            {
                "ontology_embedding_cache_path": str(cache_path),
                "ontology_embedding_term_count": len(names),
            },
        )
        if cache_path.is_file() and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                matrix = np.load(cache_path, mmap_mode="r", allow_pickle=False)
                if (
                    metadata.get("version") == EMBEDDING_CACHE_VERSION
                    and int(metadata.get("row_count") or -1) == len(names)
                    and matrix.ndim == 2
                    and matrix.shape[0] == len(names)
                ):
                    if kind == "cell-ontology-aliases":
                        self.dictionary_embedding_cache_reused = True
                    self._notify(
                        "normalization_dictionary",
                        65,
                        (
                            f"Loaded {len(names):,} saved Cell Ontology embeddings; "
                            "no dictionary rebuild is needed."
                        ),
                        {
                            "ontology_embedding_cache_reused": True,
                            "ontology_embedding_term_count": len(names),
                        },
                    )
                    return matrix
            except (OSError, ValueError, json.JSONDecodeError):
                cache_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)

        if kind == "cell-ontology-aliases":
            self.dictionary_embedding_cache_reused = False
        self._notify(
            "normalization_dictionary",
            24,
            (
                f"No compatible saved embedding cache was found. Encoding "
                f"{len(names):,} Cell Ontology labels and synonyms once; the "
                "result will be reused by later runs."
            ),
            {
                "ontology_embedding_cache_reused": False,
                "ontology_embedding_term_count": len(names),
            },
        )
        embeddings = self._encode_texts(
            names,
            purpose="Cell Ontology labels and synonyms",
            progress_start=25,
            progress_end=62,
        )
        compact = embeddings.astype(np.float16)
        self._notify(
            "normalization_dictionary",
            63,
            "Saving the reusable Cell Ontology embedding cache...",
            {
                "ontology_embedding_cache_reused": False,
                "ontology_embedding_term_count": len(names),
            },
        )
        _atomic_save_npy(cache_path, compact)
        _atomic_write_json(
            metadata_path,
            {
                "version": EMBEDDING_CACHE_VERSION,
                "kind": kind,
                "model": self.model_identity,
                "model_source": self.model_reference,
                "resource_digest": resource_digest,
                "row_count": int(compact.shape[0]),
                "dimension": int(compact.shape[1]) if compact.ndim == 2 else 0,
                "dtype": "float16",
                "normalized": True,
            },
        )
        del embeddings, compact
        gc.collect()
        matrix = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        self._notify(
            "normalization_dictionary",
            65,
            (
                f"Saved {len(names):,} Cell Ontology embeddings. Future runs "
                "will load this cache directly."
            ),
            {
                "ontology_embedding_cache_reused": False,
                "ontology_embedding_term_count": len(names),
            },
        )
        return matrix

    def _ensure_dictionary_embeddings(self) -> np.ndarray:
        if self.dictionary_embeddings is None:
            self.dictionary_embeddings = self._load_or_build_embeddings(
                kind="cell-ontology-aliases",
                names=[entry.name for entry in self.term_entries],
                resource_digest=self._ontology_digest,
            )
        return self.dictionary_embeddings

    def prepare_dictionary_embeddings(self) -> np.ndarray:
        """Load or create the reusable Cell Ontology embedding dictionary."""

        return self._ensure_dictionary_embeddings()

    def _ensure_static_key_embeddings(self) -> np.ndarray | None:
        if not self.static_lookup.all_keys:
            return None
        if self.static_key_embeddings is None:
            self.static_key_embeddings = self._load_or_build_embeddings(
                kind="static-abbreviation-keys",
                names=self.static_lookup.all_keys,
                resource_digest=self._abbreviation_digest,
            )
        return self.static_key_embeddings

    @staticmethod
    def _topk_similarity(
        query_embeddings: np.ndarray,
        dictionary_embeddings: np.ndarray,
        *,
        topk: int,
        block_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_count = int(query_embeddings.shape[0])
        dictionary_count = int(dictionary_embeddings.shape[0])
        if dictionary_count < 1:
            return (
                np.zeros((query_count, 0), dtype=np.float32),
                np.zeros((query_count, 0), dtype=np.int64),
            )
        safe_topk = max(1, min(int(topk), dictionary_count))
        best_scores = np.full((query_count, safe_topk), -np.inf, dtype=np.float32)
        best_indices = np.full((query_count, safe_topk), -1, dtype=np.int64)

        for block_start in range(0, dictionary_count, block_size):
            block_end = min(dictionary_count, block_start + block_size)
            block = np.asarray(
                dictionary_embeddings[block_start:block_end], dtype=np.float32
            )
            similarities = query_embeddings @ block.T
            local_k = min(safe_topk, similarities.shape[1])
            local_positions = np.argpartition(
                similarities, similarities.shape[1] - local_k, axis=1
            )[:, -local_k:]
            local_scores = np.take_along_axis(similarities, local_positions, axis=1)
            local_indices = local_positions.astype(np.int64) + block_start

            combined_scores = np.concatenate((best_scores, local_scores), axis=1)
            combined_indices = np.concatenate((best_indices, local_indices), axis=1)
            keep_positions = np.argpartition(
                combined_scores,
                combined_scores.shape[1] - safe_topk,
                axis=1,
            )[:, -safe_topk:]
            best_scores = np.take_along_axis(combined_scores, keep_positions, axis=1)
            best_indices = np.take_along_axis(combined_indices, keep_positions, axis=1)

        order = np.argsort(best_scores, axis=1)[:, ::-1]
        return (
            np.take_along_axis(best_scores, order, axis=1),
            np.take_along_axis(best_indices, order, axis=1),
        )

    @property
    def cell_surface_resources(self):
        from backend.pipeline.document_entity_recovery import RecoveryResources
        resources = getattr(self, "_cell_surface_resources", None)
        if resources is None:
            resources = RecoveryResources(ontology_path=self.ontology_path,
                hormone_entries=[], ontology_version=self.ontology_version,
                abbreviations_path=(None if self.disable_abbreviations else self.abbreviations_path))
            self._cell_surface_resources = resources
        return resources

    def _exact_unique_alias(self, text: str) -> OntologyMatch | None:
        resources = self.cell_surface_resources
        result = resources.ontology_exact.resolve(text)
        if result.status == "unmatched":
            result = resources.coordination.resolve(text)
        candidate = result.candidate
        if candidate is None:
            return None
        return OntologyMatch(
            identifier=candidate.concept_id,
            preferred_label=candidate.preferred_label,
            matched_alias=candidate.matched_term,
            raw_cosine=1.0,
            exact_unique_alias=True,
            final_score=1.0,
            term_kind=candidate.term_kind,
            match_metadata=candidate.match_metadata,
        )

    def _validate_long_forms(
        self,
        long_forms: Sequence[str],
    ) -> dict[str, OntologyMatch | None]:
        """Legacy cell-only exact validator.

        The production pipeline validates cell and hormone long forms in the
        pre-NER abbreviation pass. This method remains for API compatibility and
        never performs vector validation of an Ab3P long form.
        """

        return {str(value): self._exact_unique_alias(str(value)) for value in long_forms}

    def prepare_document_context(
        self,
        document: DocumentText,
    ) -> DocumentAbbreviationContext:
        """Legacy helper; production callers load the saved pre-NER context."""

        if self.disable_abbreviations:
            return DocumentAbbreviationContext(
                document_key=document.document_key,
                ab3p_status="disabled",
            )
        pairs = run_ab3p_for_document(
            document.text,
            document_key=document.document_key,
        )
        definitions = cache_document_definitions(document, pairs)
        validation = self._validate_long_forms(
            list(dict.fromkeys(item.long_form for item in definitions))
        )
        definitions_by_key: dict[str, list[Ab3PDefinition]] = defaultdict(list)
        all_keys: list[str] = []
        key_to_stable_index: dict[str, int] = {}
        for definition in definitions:
            definition.ontology_match = validation.get(definition.long_form)
            if definition.ontology_match is not None:
                definition.entity_type = "cell"
                definition.concept_id = definition.ontology_match.identifier
                definition.preferred_label = definition.ontology_match.preferred_label
                definition.matched_term = definition.ontology_match.matched_alias
                definition.term_kind = (
                    "preferred_label"
                    if definition.ontology_match.exact_unique_alias
                    else "synonym"
                )
                definition.resource_version = CELL_ONTOLOGY_RESOURCE_VERSION
                definition.resolution_status = "resolved_target"
                definition.match_method = "strict_exact"
            definitions_by_key[definition.key].append(definition)
            if definition.key not in key_to_stable_index:
                key_to_stable_index[definition.key] = len(all_keys)
                all_keys.append(definition.key)
        return DocumentAbbreviationContext(
            document_key=document.document_key,
            ab3p_status="definitions_found" if definitions else "no_definitions",
            definitions=definitions,
            definitions_by_key=dict(definitions_by_key),
            all_keys=all_keys,
            key_to_stable_index=key_to_stable_index,
        )

    @staticmethod
    def _definition_context_rank(
        definition: Ab3PDefinition,
        *,
        mention_chunk_id: str,
        mention_section: str,
        mention_document_start: int | None,
    ) -> tuple[int, int, int]:
        """Return the contextual rank without a stable-index tie breaker.

        Definition meaning is selected by document context first.  A stable
        index is used only after verifying that equally ranked definitions do
        not disagree about resolution status, entity type, or concept ID.
        Unlocated definitions are deliberately ranked after every located
        definition.
        """

        definition_start = (
            definition.definition_start
            if definition.definition_start is not None
            else definition.document_start
        )
        if definition_start is None:
            return (4, 0, 10**12)
        if definition.chunk_id and definition.chunk_id == mention_chunk_id:
            if mention_document_start is None:
                return (0, 2, 10**12)
            return (
                0,
                0 if definition_start <= mention_document_start else 1,
                abs(definition_start - mention_document_start),
            )
        if mention_document_start is not None and definition_start <= mention_document_start:
            return (1, 0, mention_document_start - definition_start)
        if definition.section and definition.section == mention_section:
            distance = (
                abs(definition_start - mention_document_start)
                if mention_document_start is not None
                else 10**12
            )
            return (2, 0, distance)
        distance = (
            abs(definition_start - mention_document_start)
            if mention_document_start is not None
            else 10**12
        )
        return (3, 0, distance)

    def _select_definition(
        self,
        definitions: Sequence[Ab3PDefinition],
        *,
        mention: Mapping[str, Any],
        document: DocumentText,
    ) -> Ab3PDefinition | None:
        mention_start = document.document_offset_for_mention(mention)
        mention_chunk_id = str(mention.get("chunk_id") or "")
        mention_section = str(mention.get("section_type") or "")
        if not definitions:
            return None
        ranked = sorted(
            definitions,
            key=lambda item: (
                self._definition_context_rank(
                    item,
                    mention_chunk_id=mention_chunk_id,
                    mention_section=mention_section,
                    mention_document_start=mention_start,
                ),
                item.stable_index,
            ),
        )
        best_rank = self._definition_context_rank(
            ranked[0],
            mention_chunk_id=mention_chunk_id,
            mention_section=mention_section,
            mention_document_start=mention_start,
        )
        tied = [
            item
            for item in ranked
            if self._definition_context_rank(
                item,
                mention_chunk_id=mention_chunk_id,
                mention_section=mention_section,
                mention_document_start=mention_start,
            )
            == best_rank
        ]
        identities = {
            (item.resolution_status, item.entity_type, item.concept_id)
            for item in tied
        }
        if len(identities) > 1:
            return None
        return tied[0]

    def _query_key_embedding(
        self,
        query_key: str,
        *,
        context: DocumentAbbreviationContext | None = None,
    ) -> np.ndarray:
        if context is not None:
            cached = context.query_embedding_cache.get(query_key)
            if cached is not None:
                return cached
            embedded = self._encode_texts([query_key])[0]
            context.query_embedding_cache[query_key] = embedded
            return embedded
        return self._encode_texts([query_key])[0]

    def _select_fuzzy_document_key(
        self,
        *,
        canonical_mention_key: str,
        query_key: str,
        context: DocumentAbbreviationContext,
        mention: Mapping[str, Any],
        document: DocumentText,
    ) -> FuzzyKeyMatch | None:
        if not fuzzy_abbreviation_allowed(canonical_mention_key):
            return None
        signature = protected_signature(query_key)
        candidates = [
            key
            for key in context.all_keys
            if protected_signature(key) == signature and key in context.key_embeddings
        ]
        if not candidates:
            return None
        query_embedding = self._query_key_embedding(query_key, context=context)
        mention_document_start = document.document_offset_for_mention(mention)
        mention_chunk_id = str(mention.get("chunk_id") or "")
        mention_section = str(mention.get("section_type") or "")

        ranked: list[tuple[float, tuple[int, int, int], int, str]] = []
        for key in candidates:
            cosine = float(query_embedding @ context.key_embeddings[key])
            definitions = context.definitions_by_key.get(key, [])
            tie_rank = min(
                (
                    self._definition_context_rank(
                        definition,
                        mention_chunk_id=mention_chunk_id,
                        mention_section=mention_section,
                        mention_document_start=mention_document_start,
                    )
                    for definition in definitions
                ),
                default=(9, 10**12, context.key_to_stable_index.get(key, 10**12)),
            )
            ranked.append(
                (
                    -cosine,
                    tie_rank,
                    context.key_to_stable_index.get(key, 10**12),
                    key,
                )
            )
        ranked.sort()
        best_cosine = -ranked[0][0]
        if best_cosine < FUZZY_ABBREVIATION_MIN_COSINE:
            return None
        return FuzzyKeyMatch(key=ranked[0][3], cosine=best_cosine)

    def _select_fuzzy_static_key(
        self,
        *,
        canonical_mention_key: str,
        query_key: str,
        context: DocumentAbbreviationContext,
    ) -> FuzzyKeyMatch | None:
        if not fuzzy_abbreviation_allowed(canonical_mention_key):
            return None
        matrix = self._ensure_static_key_embeddings()
        if matrix is None:
            return None
        candidate_keys = self.static_lookup.keys_by_signature.get(
            protected_signature(query_key), []
        )
        if not candidate_keys:
            return None
        # Reuse the same query vector that may already have been created for
        # the document-Ab3P fuzzy step for this mention.
        query_embedding = self._query_key_embedding(query_key, context=context)
        ranked: list[tuple[float, int, str]] = []
        for key in candidate_keys:
            index = self.static_lookup.key_to_stable_index[key]
            cosine = float(query_embedding @ np.asarray(matrix[index], dtype=np.float32))
            ranked.append((-cosine, index, key))
        ranked.sort()
        best_cosine = -ranked[0][0]
        if best_cosine < FUZZY_ABBREVIATION_MIN_COSINE:
            return None
        return FuzzyKeyMatch(key=ranked[0][2], cosine=best_cosine)

    @staticmethod
    def _decision_from_definition(
        mention_text: str,
        definition: Ab3PDefinition,
        *,
        source: str,
        match_method: str,
        matched_key: str,
        abbreviation_key_cosine: float | None = None,
        matched_static_key: str | None = None,
        ab3p_key_cosine: float | None = None,
    ) -> NormalizationDecision | None:
        # Cell NEN consumes only the cell definitions produced by the shared
        # multi-entity prepass. Hormone definitions remain annotations in
        # the local sidecar and are never converted to a Cell Ontology concept.
        if definition.entity_type != "cell" or not definition.concept_id:
            return None
        match = definition.ontology_match
        return NormalizationDecision(
            mention=mention_text,
            normalized_id=definition.concept_id,
            preferred_label=definition.preferred_label,
            normalization_source=source,
            matched_abbreviation_key=matched_key,
            matched_static_abbreviation_key=matched_static_key,
            expanded_long_form=definition.long_form,
            abbreviation_key_cosine=abbreviation_key_cosine,
            ab3p_key_cosine=ab3p_key_cosine,
            ab3p_match_method=match_method,
            ontology_raw_cosine=(match.raw_cosine if match is not None else 1.0),
            matched_ontology_alias=(
                definition.matched_term
                or (match.matched_alias if match is not None else definition.long_form)
            ),
            normalization_score=(match.final_score if match is not None else 1.0),
            matched_term=definition.matched_term or definition.long_form,
            term_kind=definition.term_kind or "preferred_label",
            resource_version=definition.resource_version or CELL_ONTOLOGY_RESOURCE_VERSION,
        )

    def _resolve_with_document_definitions(
        self,
        *,
        mention: Mapping[str, Any],
        document: DocumentText,
        context: DocumentAbbreviationContext,
        query_key: str,
        canonical_mention_key: str,
        source: str,
        matched_static_key: str | None = None,
        primary_cosine: float | None = None,
    ) -> NormalizationDecision | None:
        del canonical_mention_key
        mention_text = str(mention.get("mention") or "")
        definitions = context.definitions_by_key.get(query_key, [])
        selected = self._select_definition(
            definitions,
            mention=mention,
            document=document,
        )
        if selected is None:
            return None
        return self._decision_from_definition(
            mention_text,
            selected,
            source=source,
            match_method="exact",
            matched_key=query_key,
            matched_static_key=matched_static_key,
            abbreviation_key_cosine=primary_cosine,
        )

    def _static_unique_decision(
        self,
        mention_text: str,
        *,
        identifier: str,
        source: str,
        matched_static_key: str,
        abbreviation_key_cosine: float | None,
    ) -> NormalizationDecision:
        metadata = self.concept_metadata[identifier]
        candidate = next((candidate for candidate in self.static_lookup.key_to_candidates.get(
            matched_static_key, []) if candidate.identifier == identifier and cell_term_matches(
                mention_text, candidate.short_form, "static_abbreviation")), None)
        reference_file = self.abbreviations_path.name if self.abbreviations_path else "abbreviations.tsv"
        return NormalizationDecision(
            mention=mention_text,
            normalized_id=identifier,
            preferred_label=metadata.preferred_label,
            normalization_source=source,
            matched_abbreviation_key=matched_static_key,
            matched_static_abbreviation_key=matched_static_key,
            abbreviation_key_cosine=abbreviation_key_cosine,
            matched_ontology_alias=metadata.preferred_label,
            normalization_score=1.0,
            matched_term=candidate.short_form if candidate else mention_text,
            term_kind="static_abbreviation",
            resource_version=f"{reference_file}-{self._abbreviation_digest[:12]}",
            match_metadata={"resource_file": reference_file,
                            "resource_line": candidate.resource_line if candidate else None,
                            "ontology_resource_version": self.ontology_version},
        )

    def _abbreviation_decision(
        self,
        *,
        mention: Mapping[str, Any],
        document: DocumentText,
        context: DocumentAbbreviationContext,
    ) -> NormalizationDecision | None:
        if self.disable_abbreviations:
            return None
        mention_text = str(mention.get("mention") or "")
        candidate_keys = short_form_key_candidates(mention_text)
        if not candidate_keys:
            return None

        # 1. Exact document Ab3P key, then its controlled lowercase plural
        # variant. All definitions remain in the cache and the closest contextual
        # definition is selected for this occurrence.
        for key_index, mention_key in enumerate(candidate_keys):
            selected = self._select_definition(
                context.definitions_by_key.get(mention_key, []),
                mention=mention,
                document=document,
            )
            if selected is not None:
                decision = self._decision_from_definition(
                    mention_text,
                    selected,
                    source=(
                        "ab3p_exact"
                        if key_index == 0
                        else "ab3p_plural_variant"
                    ),
                    match_method=(
                        "exact"
                        if key_index == 0
                        else "controlled_plural_variant"
                    ),
                    matched_key=mention_key,
                )
                if decision is not None:
                    return decision

        # 2. Exact abbreviations.tsv key. An ambiguous or absent static mapping
        # is unresolved here and proceeds to the thresholded top-1 Cell
        # Ontology linker. No fuzzy abbreviation-key lookup is
        # performed.
        for mention_key in candidate_keys:
            if mention_key not in self.static_lookup.key_to_candidates:
                continue
            identifiers = self.static_lookup.valid_identifiers(
                mention_key,
                self.concept_metadata,
                surface=mention_text,
            )
            if len(identifiers) == 1:
                return self._static_unique_decision(
                    mention_text,
                    identifier=identifiers[0],
                    source="static_exact_unique",
                    matched_static_key=mention_key,
                    abbreviation_key_cosine=None,
                )
            if len(identifiers) > 1:
                decision = self._resolve_with_document_definitions(
                    mention=mention,
                    document=document,
                    context=context,
                    query_key=mention_key,
                    canonical_mention_key=mention_key,
                    source="static_exact_ambiguous_ab3p",
                    matched_static_key=mention_key,
                )
                if decision is not None:
                    return decision
        return None

    def _rerank_normal_candidate(
        self,
        query_text: str,
        *,
        entry: TermEntry,
        raw_cosine: float,
    ) -> OntologyMatch:
        metadata = self.concept_metadata[entry.identifier]
        query_normalized = _casefold_normalize_text(query_text)
        exact_match = 0.0
        best_overlap = 0.0
        best_parenthetical = 0.0
        best_sequence = 0.0
        for name in metadata.names:
            if query_normalized == _casefold_normalize_text(name):
                exact_match = 1.0
            best_overlap = max(best_overlap, token_jaccard(query_text, name))
            best_parenthetical = max(
                best_parenthetical,
                has_parenthetical_relation(query_text, name),
            )
            best_sequence = max(best_sequence, sequence_ratio(query_text, name))
        preferred_overlap = token_jaccard(query_text, metadata.preferred_label)
        final_score = float(
            raw_cosine
            + 0.35 * exact_match
            + 0.20 * best_overlap
            + 0.15 * preferred_overlap
            + 0.10 * best_parenthetical
            + 0.05 * best_sequence
            + (0.03 if entry.is_preferred else 0.0)
        )
        return OntologyMatch(
            identifier=entry.identifier,
            preferred_label=metadata.preferred_label,
            matched_alias=entry.raw_name,
            raw_cosine=raw_cosine,
            exact_unique_alias=False,
            final_score=final_score,
        )

    def _normal_link_batch(self, texts: Sequence[str]) -> list[OntologyMatch | None]:
        """Unique exact cell alias first; vector top-1 must pass raw cosine gate."""
        results: list[OntologyMatch | None] = [None] * len(texts)
        pending = []
        for index, text in enumerate(texts):
            if is_noncell_surface(text) or not self.cell_surface_resources.cell_surface_allowed(text):
                continue
            try:
                exact = self._exact_unique_alias(text)
            except AttributeError:  # lightweight mocked linkers used in unit tests
                exact = None
            if exact:
                results[index] = exact
            elif not re.search(r"\s+(?:and/or|and|or)\s+|/", text, re.I):
                # An unresolved coordination must not acquire one arbitrary
                # subtype through vector similarity.
                pending.append(index)
        if not pending:
            return results
        dictionary = self._ensure_dictionary_embeddings()
        embeddings = self._encode_texts([plural_normalize_text(texts[i]) for i in pending])
        scores, indices = self._topk_similarity(embeddings, dictionary, topk=1)
        for i, score_row, index_row in zip(pending, scores, indices):
            if not len(index_row) or int(index_row[0]) < 0:
                continue
            raw_cosine = float(score_row[0])
            if not np.isfinite(raw_cosine) or raw_cosine < cell_min_cosine():
                continue
            entry = self.term_entries[int(index_row[0])]
            results[i] = OntologyMatch(identifier=entry.identifier,
                preferred_label=entry.preferred_label, matched_alias=entry.raw_name,
                raw_cosine=raw_cosine, exact_unique_alias=False, final_score=raw_cosine)
        return results

    def normalize_document_mentions(
        self,
        *,
        document: DocumentText,
        mentions: Sequence[Mapping[str, Any]],
        context: DocumentAbbreviationContext,
    ) -> list[NormalizationDecision]:
        """Normalize every NER occurrence without collapsing its context."""

        decisions: list[NormalizationDecision | None] = [None] * len(mentions)
        unresolved_indices: list[int] = []
        unresolved_texts: list[str] = []
        for index, mention in enumerate(mentions):
            mention_text = str(mention.get("mention") or "")
            if is_noncell_surface(mention_text):
                decisions[index] = NormalizationDecision(mention=mention_text,
                    normalization_source="rejected_noncell_surface")
                continue
            chunk = document.chunk_by_id(mention.get("chunk_id"))
            if chunk is not None and mention.get("end") is not None:
                text = str(chunk.source.get("chunk") or "")
                if not token_aligned(text, int(mention.get("start", 0)), int(mention["end"])):
                    decisions[index] = NormalizationDecision(mention=mention_text,
                        normalization_source="rejected_partial_token")
                    continue
            if not self.disable_abbreviations:
                definitions = next((context.definitions_by_key[k]
                    for k in short_form_key_candidates(mention_text)
                    if context.definitions_by_key.get(k)), None)
                if definitions:
                    selected = self._select_definition(definitions, mention=mention, document=document)
                    if selected is None or not selected.resolved or selected.entity_type != "cell":
                        decisions[index] = NormalizationDecision(mention=mention_text,
                            normalization_source="rejected_document_definition")
                        continue
            decision = self._abbreviation_decision(
                mention=mention,
                document=document,
                context=context,
            )
            if decision is not None and decision.normalized:
                decisions[index] = decision
            else:
                unresolved_indices.append(index)
                unresolved_texts.append(str(mention.get("mention") or ""))

        linked = self._normal_link_batch(unresolved_texts) if unresolved_texts else []
        for index, match in zip(unresolved_indices, linked):
            mention_text = str(mentions[index].get("mention") or "")
            if match is None or (not match.exact_unique_alias and (not np.isfinite(match.raw_cosine) or match.raw_cosine < cell_min_cosine())):
                decisions[index] = NormalizationDecision(mention=mention_text,
                    normalization_source="rejected_low_cell_similarity",
                    ontology_raw_cosine=match.raw_cosine if match else None,
                    matched_ontology_alias=match.matched_alias if match else None)
                continue
            decisions[index] = NormalizationDecision(
                mention=mention_text,
                normalized_id=match.identifier,
                preferred_label=match.preferred_label,
                normalization_source=("cell_ontology_coordinated_shared_head"
                    if match.term_kind == "coordinated_shared_head" else
                    "cell_ontology_exact_alias" if match.exact_unique_alias else "cell_ontology_vector_top1"),
                ontology_raw_cosine=match.raw_cosine,
                matched_ontology_alias=match.matched_alias,
                normalization_score=match.raw_cosine,
                matched_term=match.matched_alias,
                term_kind=(match.term_kind or "exact_alias" if match.exact_unique_alias else "vector_top1_alias"),
                resource_version=self.ontology_version,
                match_metadata=match.match_metadata,
            )
        return [
            decision if decision is not None else NormalizationDecision(mention="")
            for decision in decisions
        ]

    def rescue_document_cell_mentions(
        self,
        *,
        document: DocumentText,
        accepted_rows: Sequence[Mapping[str, Any]],
        context: DocumentAbbreviationContext | None = None,
    ) -> list[RescuedMention]:
        """Propagate a uniquely normalized cell surface to missed chunks once.

        Only the fixed set of already accepted annotations can seed this pass; a
        rescued mention never recursively creates more rescue candidates. Search
        the whole selected paper, including earlier chunks and the seed's chunk.
        Existing shorter annotations must not veto a recovered complete phrase:
        the worker applies the shared longest-span policy to all candidates after
        source/context checks. Only an already present identical cell occurrence
        is skipped here.
        """

        seeds_by_key: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        existing_occurrences: set[tuple[str, int, int, str]] = set()
        for row in accepted_rows:
            entity_type = str(row.get("entity_type") or row.get("obj") or "").casefold()
            if entity_type in {"cell_type", "cell"}:
                try:
                    start = int(row.get("start"))
                    end = int(row.get("end"))
                except (TypeError, ValueError):
                    continue
                raw_chunk_id = row.get("chunk_id")
                chunk_id = str(raw_chunk_id) if raw_chunk_id is not None else ""
                concept_id = str(
                    row.get("concept_id") or row.get("cell_ontology_id")
                    or row.get("normalized_id") or ""
                ).strip()
                chunk = document.chunk_by_id(chunk_id)
                if chunk is None:
                    continue
                source_text = str(chunk.source.get("chunk") or "")
                if not token_aligned(source_text, start, end):
                    continue
                # Copy identity from the accepted row, but search the real
                # source spelling rather than a stale display/lookup string.
                mention = source_text[start:end]
                existing_occurrences.add((chunk_id, start, end, concept_id))
                key = canonical_resource_key(mention)
                if is_noncell_surface(mention):
                    continue
                if re.search(r"\s+(?:and/or|and|or)\s+|/", mention, re.I):
                    continue  # local shared-head inferences are never propagation seeds
                if not self.cell_surface_resources.cell_surface_allowed(mention) and not row.get("definition_id"):
                    continue
                cosine = _optional_float(row.get("ontology_raw_cosine"))
                if cosine is not None and (not np.isfinite(cosine) or cosine < cell_min_cosine()):
                    continue
                if concept_id and key:
                    seed = dict(row)
                    seed.update(mention=mention, concept_id=concept_id)
                    seeds_by_key[key].append(seed)

        rescued: list[RescuedMention] = []
        seen: set[tuple[str, int, int, str]] = set()
        # Longest first for stable candidate generation; final overlap
        # ownership is still decided centrally, not by this iteration order.
        for key, rows in sorted(seeds_by_key.items(), key=lambda item: (-len(item[0]), item[0])):
            concept_ids = {
                str(row.get("concept_id") or row.get("cell_ontology_id") or "").strip()
                for row in rows
            }
            if len(concept_ids) != 1:
                continue
            seed = rows[0]
            surface = str(seed.get("mention") or "").strip()
            if not surface:
                continue
            pattern = compile_surface_pattern(surface)
            for match in pattern.finditer(document.text):
                chunk = document.chunk_for_span(match.start(), match.end())
                if chunk is None:
                    continue
                start = match.start() - chunk.document_start
                end = match.end() - chunk.document_start
                if not self.cell_surface_resources.cell_surface_allowed(match.group(0)):
                    continue
                concept_id = next(iter(concept_ids))
                signature = (chunk.chunk_id, start, end, concept_id)
                if signature in existing_occurrences or signature in seen:
                    continue
                # Do not skip overlapping spans. A recognized short cell head
                # can coexist temporarily with this longer recovery candidate;
                # the shared resolver removes the losing annotation afterwards.

                # Do not let generic document-surface rescue undo the more
                # precise per-occurrence Ab3P definition selection.  When this
                # surface is a known document short form, rescue is permitted
                # only if the contextually selected definition is the same
                # resolved cell concept.  A local unmatched, hormone, or
                # conflicting definition therefore blocks a farther cell seed.
                if context is not None:
                    contextual_definitions: list[Ab3PDefinition] | None = None
                    for candidate_key in short_form_key_candidates(match.group(0)):
                        values = context.definitions_by_key.get(candidate_key)
                        if values:
                            contextual_definitions = values
                            break
                    if contextual_definitions is not None:
                        selected = self._select_definition(
                            contextual_definitions,
                            mention={
                                "mention": match.group(0),
                                "chunk_id": chunk.chunk_id,
                                "section_type": chunk.section,
                                "start": start,
                                "end": end,
                            },
                            document=document,
                        )
                        if (
                            selected is None
                            or selected.resolution_status != "resolved_target"
                            or selected.entity_type != "cell"
                            or selected.concept_id != concept_id
                        ):
                            continue

                seen.add(signature)
                decision = NormalizationDecision(
                    mention=match.group(0),
                    normalized_id=concept_id,
                    preferred_label=str(
                        seed.get("preferred_label")
                        or seed.get("cell_ontology_label")
                        or ""
                    ),
                    normalization_source="document_cell_surface_rescue",
                    ontology_raw_cosine=_optional_float(seed.get("ontology_raw_cosine")),
                    matched_ontology_alias=str(
                        seed.get("matched_term")
                        or seed.get("matched_ontology_alias")
                        or surface
                    ),
                    normalization_score=_optional_float(seed.get("normalization_score")),
                    matched_term=str(
                        seed.get("matched_term")
                        or seed.get("matched_ontology_alias")
                        or surface
                    ),
                    term_kind="document_unique_surface",
                    resource_version=str(seed.get("resource_version") or self.ontology_version),
                    match_metadata={"seed_evidence": {
                        "paper": document.document_key,
                        "chunk_id": seed.get("chunk_id"),
                        "start": seed.get("start"), "end": seed.get("end"),
                        "concept_id": concept_id,
                        "recognition_source": seed.get("recognition_source"),
                        "normalization_source": seed.get("normalization_source"),
                    }},
                )
                rescued.append(
                    RescuedMention(
                        mention=match.group(0),
                        start=start,
                        end=end,
                        chunk=chunk,
                        decision=decision,
                    )
                )
        rescued.sort(
            key=lambda item: (
                item.chunk.chunk_order, item.start, item.end, item.mention.casefold()
            )
        )
        return rescued

    def rescue_ab3p_short_forms(
        self,
        *,
        document: DocumentText,
        context: DocumentAbbreviationContext,
        ner_mentions: Sequence[Mapping[str, Any]],
    ) -> list[RescuedMention]:
        """Add exact, boundary-aware Ab3P short-form spans missed by NER."""

        if self.disable_abbreviations or not context.definitions:
            return []

        existing_by_chunk: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for mention in ner_mentions:
            try:
                start = int(mention.get("start"))
                end = int(mention.get("end"))
            except (TypeError, ValueError):
                continue
            existing_by_chunk[str(mention.get("chunk_id") or "")].append((start, end))

        validated_ids_by_key: dict[str, set[str]] = defaultdict(set)
        for definition in context.definitions:
            if definition.ontology_match is not None:
                validated_ids_by_key[definition.key].add(
                    definition.ontology_match.identifier
                )

        rescued: list[RescuedMention] = []
        seen_spans: set[tuple[str, int, int, str]] = set()
        surfaces_by_key: dict[str, list[str]] = defaultdict(list)
        for definition in context.definitions:
            if definition.ontology_match is None:
                continue
            if definition.short_form not in surfaces_by_key[definition.key]:
                surfaces_by_key[definition.key].append(definition.short_form)

        for key in context.all_keys:
            concept_ids = validated_ids_by_key.get(key, set())
            if len(concept_ids) != 1:
                # Conflicting validated meanings are never rescued globally.
                continue
            for short_form in surfaces_by_key.get(key, []):
                pattern = re.compile(
                    rf"{_BOUNDARY_LEFT}{re.escape(short_form)}{_BOUNDARY_RIGHT}"
                )
                for match in pattern.finditer(document.text):
                    chunk = document.chunk_for_span(match.start(), match.end())
                    if chunk is None:
                        continue
                    chunk_start = match.start() - chunk.document_start
                    chunk_end = match.end() - chunk.document_start
                    if any(
                        chunk_start < existing_end and chunk_end > existing_start
                        for existing_start, existing_end in existing_by_chunk.get(
                            chunk.chunk_id, []
                        )
                    ):
                        continue
                    signature = (chunk.chunk_id, chunk_start, chunk_end, key)
                    if signature in seen_spans:
                        continue

                    synthetic_mention = {
                        "mention": match.group(0),
                        "chunk_id": chunk.chunk_id,
                        "section_type": chunk.section,
                        "start": chunk_start,
                        "end": chunk_end,
                    }
                    definition = self._select_definition(
                        context.definitions_by_key.get(key, []),
                        mention=synthetic_mention,
                        document=document,
                    )
                    if definition is None or definition.ontology_match is None:
                        continue
                    if definition.ontology_match.identifier not in concept_ids:
                        continue
                    decision = self._decision_from_definition(
                        match.group(0),
                        definition,
                        source="ab3p_rescue",
                        match_method="exact_span_rescue",
                        matched_key=key,
                    )
                    if decision is None:
                        continue
                    seen_spans.add(signature)
                    existing_by_chunk[chunk.chunk_id].append((chunk_start, chunk_end))
                    rescued.append(
                        RescuedMention(
                            mention=match.group(0),
                            start=chunk_start,
                            end=chunk_end,
                            chunk=chunk,
                            decision=decision,
                        )
                    )
        rescued.sort(
            key=lambda item: (
                item.chunk.chunk_order,
                item.start,
                item.end,
                item.mention.casefold(),
            )
        )
        return rescued

    def close(self) -> None:
        model = self.model
        tokenizer = self.tokenizer
        dictionary = self.dictionary_embeddings
        static_keys = self.static_key_embeddings
        self.model = None
        self.tokenizer = None
        self.dictionary_embeddings = None
        self.static_key_embeddings = None
        self.progress_callback = None
        del model, tokenizer, dictionary, static_keys
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            try:
                self.torch.cuda.ipc_collect()
            except Exception:
                pass

    def __enter__(self) -> "CellOntologyNormalizer":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


NORMALIZATION_METHODS = (
    "ab3p_definition_long_form",
    "ab3p_definition_short_form",
    "ab3p_document_short_form",
    "ab3p_exact",
    "ab3p_plural_variant",
    "static_exact_unique",
    "static_exact_ambiguous_ab3p",
    "cell_ontology_vector_top1",
    "document_cell_surface_rescue",
    "unresolved",
)



__all__ = [
    "AB3P_HEALTHCHECK_LONG_FORM",
    "AB3P_HEALTHCHECK_SHORT_FORM",
    "AB3P_HEALTHCHECK_TEXT",
    "AB3P_LONG_FORM_MIN_RAW_COSINE",
    "FUZZY_ABBREVIATION_MIN_COSINE",
    "FUZZY_ABBREVIATION_MIN_LENGTH",
    "FUZZY_ABBREVIATION_MAX_LENGTH",
    "NORMALIZATION_METHODS",
    "Ab3PDefinition",
    "Ab3PDocumentError",
    "Ab3PError",
    "Ab3PHealthCheckError",
    "Ab3POutputError",
    "Ab3PUnavailableError",
    "CellOntologyNormalizer",
    "ChunkOffset",
    "DocumentAbbreviationContext",
    "DocumentText",
    "NormalizationDecision",
    "OntologyMatch",
    "RescuedMention",
    "build_document_text",
    "cache_document_definitions",
    "canonical_abbreviation_key",
    "ensure_ab3p_healthy",
    "fuzzy_abbreviation_allowed",
    "is_controlled_plural_variant",
    "load_cell_ontology_terms",
    "load_static_abbreviations",
    "plural_normalize_text",
    "protected_signature",
    "run_ab3p_for_document",
]
