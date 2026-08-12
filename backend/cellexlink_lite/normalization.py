"""Lean CellExLink Cell Ontology normalization for chunk NER mentions.

The ontology loading, plural normalization, abbreviation handling, SapBERT CLS
encoding, cosine retrieval, and lexical reranking mirror the corresponding
CellExLink components.  Query encoding is batched and ontology embeddings are
stored as a small float16 cache, while model weights are loaded only in the
short-lived normalization worker.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .resources import DEFAULT_ABBREVIATIONS_PATH, DEFAULT_ONTOLOGY_PATH
EMBEDDING_CACHE_VERSION = "cellexlink-cls32-f16-v1"
_AB3P_DETECTOR: Any | None = None
_AB3P_UNAVAILABLE = False
DASH_PATTERN = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]"
_TOKEN_FINDER = re.compile(r"[^\W_]+|[^\w\s]|_|,", re.UNICODE)


# ---------------------------------------------------------------------------
# CellExLink lexical normalization
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


def _casefold_normalize_text(text: object) -> str:
    return " ".join(str(text).casefold().split())


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
        inside = re.findall(r"\(([^()]*)\)", left_text)
        if any(
            _casefold_normalize_text(item) == _casefold_normalize_text(right_text)
            for item in inside
        ):
            return 1.0
    if "(" in right_text and ")" in right_text:
        inside = re.findall(r"\(([^()]*)\)", right_text)
        if any(
            _casefold_normalize_text(item) == _casefold_normalize_text(left_text)
            for item in inside
        ):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Ontology resources
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


def load_cell_ontology_terms(
    ontology_path: str | Path,
) -> tuple[list[TermEntry], dict[str, ConceptMetadata]]:
    path = Path(ontology_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cell Ontology JSONL does not exist: {path}")

    term_entries: list[TermEntry] = []
    concept_metadata: dict[str, ConceptMetadata] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_no} in {path}: {exc}") from exc

            identifier = record.get("norm_concept_id")
            preferred_label = record.get("norm_preferred_label")
            synonyms = record.get("synonyms", []) or []
            namespace = str(record.get("namespace", "") or "")
            if not identifier or not preferred_label:
                continue
            if not isinstance(synonyms, list):
                raise ValueError(f"Expected synonyms list on line {line_no} in {path}")

            identifier = str(identifier)
            preferred_label = str(preferred_label)
            meta = concept_metadata.setdefault(
                identifier,
                ConceptMetadata(
                    preferred_label=preferred_label,
                    namespace=namespace,
                ),
            )
            meta.names.add(preferred_label)
            term_entries.append(
                TermEntry(
                    name=plural_normalize_text(preferred_label),
                    raw_name=preferred_label,
                    identifier=identifier,
                    preferred_label=preferred_label,
                    is_preferred=True,
                )
            )
            for synonym_value in synonyms:
                if not synonym_value:
                    continue
                synonym = str(synonym_value)
                meta.synonyms.add(synonym)
                meta.names.add(synonym)
                term_entries.append(
                    TermEntry(
                        name=plural_normalize_text(synonym),
                        raw_name=synonym,
                        identifier=identifier,
                        preferred_label=preferred_label,
                        is_preferred=False,
                    )
                )
    return term_entries, concept_metadata


# ---------------------------------------------------------------------------
# Abbreviation resources and document context
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class AbbreviationCandidate:
    short_form: str
    key: str
    identifier: str


@dataclass(slots=True)
class AbbreviationLookup:
    direct_lookup: dict[str, tuple[str, str]] = field(default_factory=dict)
    ambiguous_candidates: dict[str, list[AbbreviationCandidate]] = field(
        default_factory=dict
    )
    all_keys: list[str] = field(default_factory=list)
    key_to_candidates: dict[str, list[AbbreviationCandidate]] = field(
        default_factory=dict
    )
    row_counts: Counter = field(default_factory=Counter)

    def __bool__(self) -> bool:
        return bool(self.direct_lookup or self.ambiguous_candidates)


def normalize_abbreviation_key(text: object) -> str:
    value = str(text).strip()
    value = re.sub(DASH_PATTERN, "", value)
    return re.sub(r"\s+", "", value)


def abbreviation_variant_keys(text: object) -> list[str]:
    key = normalize_abbreviation_key(text)
    variants: list[str] = []
    for item in (key, key[:-1] if key.endswith("s") else key + "s"):
        if item and item not in variants:
            variants.append(item)
    return variants


def abbreviation_threshold_for(mention_text: object) -> float:
    length = len(normalize_abbreviation_key(mention_text))
    if length <= 4:
        return 1.0
    if length <= 7:
        return 0.95
    return 0.90


def abbreviation_sequence_ratio(left: object, right: object) -> float:
    left_norm = normalize_abbreviation_key(left)
    right_norm = normalize_abbreviation_key(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def is_abbreviation_like(text: object) -> bool:
    raw = str(text).strip()
    if not raw:
        return False
    compact = normalize_abbreviation_key(raw)
    if len(compact) <= 1:
        return False
    return (
        any(ch.isupper() for ch in raw)
        or any(ch.isdigit() for ch in raw)
        or any(ch in "+/-" for ch in raw)
        or (len(compact) <= 12 and " " not in raw)
    )


def load_abbreviation_identifier_lookup(
    abbreviations_path: str | Path | None,
) -> AbbreviationLookup:
    if abbreviations_path is None:
        return AbbreviationLookup()
    path = Path(abbreviations_path)
    if not path.is_file():
        return AbbreviationLookup()

    key_to_candidates: dict[str, list[AbbreviationCandidate]] = defaultdict(list)
    row_counts: Counter = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            fields = line.rstrip("\n").split("\t")
            if not fields or not fields[0]:
                continue
            if line_no == 0 and fields[:2] == ["short_form", "matched_cl_id"]:
                continue
            if len(fields) < 2:
                continue
            short_form = fields[0].strip()
            raw_identifier = fields[1].strip()
            if not short_form or raw_identifier in {"", "-", "None", "none"}:
                continue
            identifier = re.split(r"[,;]", raw_identifier)[0].strip()
            if identifier in {"", "-", "None", "none"}:
                continue
            key = normalize_abbreviation_key(short_form)
            if not key:
                continue
            candidate = AbbreviationCandidate(short_form, key, identifier)
            key_to_candidates[key].append(candidate)
            row_counts[key] += 1

    direct_lookup: dict[str, tuple[str, str]] = {}
    ambiguous: dict[str, list[AbbreviationCandidate]] = {}
    for key, candidates in key_to_candidates.items():
        identifiers = {candidate.identifier for candidate in candidates}
        if len(identifiers) == 1:
            best = max(candidates, key=lambda item: len(item.short_form))
            direct_lookup[key] = (best.short_form, best.identifier)
        else:
            ambiguous[key] = list(candidates)

    return AbbreviationLookup(
        direct_lookup=direct_lookup,
        ambiguous_candidates=ambiguous,
        all_keys=sorted(key_to_candidates),
        key_to_candidates=dict(key_to_candidates),
        row_counts=row_counts,
    )


def _normalize_pyab3p_output(results: Any) -> list[tuple[str, str]]:
    if isinstance(results, dict):
        return [
            (str(short).strip(), str(long).strip())
            for short, long in results.items()
            if str(short).strip() and str(long).strip()
        ]
    if not isinstance(results, list):
        return []

    pairs: list[tuple[str, str]] = []
    for item in results:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            short_form, long_form = str(item[0]).strip(), str(item[1]).strip()
        elif isinstance(item, dict):
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
        if short_form and long_form:
            pairs.append((short_form, long_form))
    return pairs


def extract_document_abbreviations(text: str) -> dict[str, str]:
    """Return normalized short-form to long-form mappings from one paper."""

    global _AB3P_DETECTOR, _AB3P_UNAVAILABLE
    if not str(text).strip() or _AB3P_UNAVAILABLE:
        return {}
    if _AB3P_DETECTOR is None:
        try:
            import pyab3p  # type: ignore

            _AB3P_DETECTOR = pyab3p.Ab3p()
        except Exception:
            _AB3P_UNAVAILABLE = True
            return {}

    try:
        results = _AB3P_DETECTOR.get_abbrs(str(text))
    except Exception:
        return {}

    lookup: dict[str, str] = {}
    for short_form, long_form in _normalize_pyab3p_output(results):
        key = normalize_abbreviation_key(short_form)
        if key:
            lookup[key] = long_form
    return lookup


# ---------------------------------------------------------------------------
# Public result records
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class NormalizationRequest:
    mention_text: str
    document_key: str = ""

    @property
    def normalized_text(self) -> str:
        return plural_normalize_text(self.mention_text)


@dataclass(slots=True)
class NormalizationCandidate:
    identifier: str
    name: str
    preferred_label: str
    embedding_score: float
    final_score: float
    source: str = "model_normal"
    is_preferred: bool = False
    matched_alias: str | None = None
    exact_synonym_match: float = 0.0
    token_overlap: float = 0.0
    preferred_overlap: float = 0.0
    parenthetical_match: float = 0.0
    sequence_ratio: float = 0.0
    abbreviation_method: str | None = None
    expanded_long_form: str | None = None
    ab3p_method: str | None = None
    ab3p_matched_key: str | None = None
    ab3p_match_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "identifier": self.identifier,
            "name": self.name,
            "preferred_label": self.preferred_label,
            "embedding_score": round(float(self.embedding_score), 8),
            "final_score": round(float(self.final_score), 8),
            "source": self.source,
            "is_preferred": self.is_preferred,
            "matched_alias": self.matched_alias,
            "exact_synonym_match": self.exact_synonym_match,
            "token_overlap": self.token_overlap,
            "preferred_overlap": self.preferred_overlap,
            "parenthetical_match": self.parenthetical_match,
            "sequence_ratio": self.sequence_ratio,
            "abbreviation_method": self.abbreviation_method,
            "expanded_long_form": self.expanded_long_form,
            "ab3p_method": self.ab3p_method,
            "ab3p_matched_key": self.ab3p_matched_key,
            "ab3p_match_score": self.ab3p_match_score,
        }
        return {key: value for key, value in result.items() if value is not None}


@dataclass(slots=True)
class NormalizationResult:
    mention_text: str
    normalized_text: str
    document_key: str
    candidates: list[NormalizationCandidate] = field(default_factory=list)

    @property
    def best(self) -> NormalizationCandidate | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_text": self.mention_text,
            "normalized_text": self.normalized_text,
            "document_key": self.document_key,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(slots=True)
class _ModelJob:
    request_index: int
    query_text: str
    mode: str
    abbreviation_method: str | None = None
    expanded_long_form: str | None = None
    ab3p_method: str | None = None
    ab3p_matched_key: str | None = None
    ab3p_match_score: float | None = None


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
        temp_path = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


class CellOntologyNormalizer:
    """Batched SapBERT linker using only the CellExLink NEN behavior needed here."""

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
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        _set_torch_threads(cpu_threads)

        self.model_reference = str(model_name_or_path)
        self.model_cache_dir = (
            str(model_cache_dir) if model_cache_dir is not None else None
        )
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
        self.trust_remote_code = trust_remote_code

        self.term_entries, self.concept_metadata = load_cell_ontology_terms(
            self.ontology_path
        )
        self.abbreviation_lookup = (
            AbbreviationLookup()
            if self.disable_abbreviations
            else load_abbreviation_identifier_lookup(self.abbreviations_path)
        )

        self.torch: Any | None = None
        self.device: Any | None = None
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.dictionary_embeddings: np.ndarray | None = None
        self.abbreviation_embeddings: np.ndarray | None = None

        self._ontology_digest = _file_sha256(self.ontology_path)
        self._abbreviation_digest = (
            _file_sha256(self.abbreviations_path)
            if self.abbreviations_path and self.abbreviations_path.is_file()
            else "none"
        )

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _prepare_encoder(self) -> None:
        if self.model_loaded:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        common_kwargs = {
            "cache_dir": self.model_cache_dir,
            "trust_remote_code": self.trust_remote_code,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_reference,
            **common_kwargs,
        )
        self.model = AutoModel.from_pretrained(
            self.model_reference,
            **common_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._prepare_encoder()
        assert self.torch is not None
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.device is not None

        all_representations: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = [str(text) for text in texts[start : start + self.batch_size]]
                tokens = self.tokenizer(
                    batch,
                    padding="max_length",
                    max_length=32,
                    truncation=True,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                output = self.model(**tokens)
                hidden = output[0] if isinstance(output, tuple) else output.last_hidden_state
                cls_representations = hidden[:, 0, :]
                all_representations.append(
                    cls_representations.detach().cpu().float().numpy()
                )
        matrix = np.concatenate(all_representations, axis=0).astype(np.float32)
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
                    self.model_reference,
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
                    return matrix
            except (OSError, ValueError, json.JSONDecodeError):
                cache_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)

        embeddings = self._encode_texts(names)
        compact = embeddings.astype(np.float16)
        _atomic_save_npy(cache_path, compact)
        _atomic_write_json(
            metadata_path,
            {
                "version": EMBEDDING_CACHE_VERSION,
                "kind": kind,
                "model": self.model_reference,
                "resource_digest": resource_digest,
                "row_count": int(compact.shape[0]),
                "dimension": int(compact.shape[1]) if compact.ndim == 2 else 0,
                "dtype": "float16",
                "normalized": True,
            },
        )
        del embeddings, compact
        gc.collect()
        return np.load(cache_path, mmap_mode="r", allow_pickle=False)

    def _ensure_dictionary_embeddings(self) -> np.ndarray:
        if self.dictionary_embeddings is None:
            self.dictionary_embeddings = self._load_or_build_embeddings(
                kind="cell-ontology-aliases",
                names=[entry.name for entry in self.term_entries],
                resource_digest=self._ontology_digest,
            )
        return self.dictionary_embeddings

    def _ensure_abbreviation_embeddings(self) -> np.ndarray | None:
        if not self.abbreviation_lookup.all_keys:
            return None
        if self.abbreviation_embeddings is None:
            self.abbreviation_embeddings = self._load_or_build_embeddings(
                kind="abbreviation-keys",
                names=self.abbreviation_lookup.all_keys,
                resource_digest=self._abbreviation_digest,
            )
        return self.abbreviation_embeddings

    @staticmethod
    def _topk_similarity(
        query_embeddings: np.ndarray,
        dictionary_embeddings: np.ndarray,
        *,
        topk: int,
        block_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_count = query_embeddings.shape[0]
        dictionary_count = dictionary_embeddings.shape[0]
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

    def _rerank_candidate(
        self,
        query_text: str,
        candidate: NormalizationCandidate,
    ) -> NormalizationCandidate:
        query_normalized = _casefold_normalize_text(query_text)
        metadata = self.concept_metadata.get(candidate.identifier)
        names = (
            set(metadata.names)
            if metadata is not None
            else {candidate.name, candidate.preferred_label}
        )
        preferred_label = (
            metadata.preferred_label if metadata is not None else candidate.preferred_label
        )

        best_overlap = 0.0
        exact_match = 0.0
        best_parenthetical = 0.0
        best_sequence = 0.0
        for name in names:
            if query_normalized == _casefold_normalize_text(name):
                exact_match = 1.0
            best_overlap = max(best_overlap, token_jaccard(query_text, name))
            best_parenthetical = max(
                best_parenthetical, has_parenthetical_relation(query_text, name)
            )
            best_sequence = max(best_sequence, sequence_ratio(query_text, name))

        preferred_overlap = token_jaccard(query_text, preferred_label)
        candidate.final_score = float(
            candidate.embedding_score
            + 0.35 * exact_match
            + 0.20 * best_overlap
            + 0.15 * preferred_overlap
            + 0.10 * best_parenthetical
            + 0.05 * best_sequence
            + (0.03 if candidate.is_preferred else 0.0)
        )
        candidate.exact_synonym_match = exact_match
        candidate.token_overlap = max(best_overlap, preferred_overlap)
        candidate.preferred_overlap = preferred_overlap
        candidate.parenthetical_match = best_parenthetical
        candidate.sequence_ratio = best_sequence
        return candidate

    def _retrieve_query_batch(
        self,
        query_texts: Sequence[str],
        *,
        topn: int,
        initial_k: int,
    ) -> list[list[NormalizationCandidate]]:
        if not query_texts:
            return []
        dictionary = self._ensure_dictionary_embeddings()
        query_embeddings = self._encode_texts(query_texts)
        scores, indices = self._topk_similarity(
            query_embeddings,
            dictionary,
            topk=min(max(initial_k, topn), len(self.term_entries)),
        )

        all_candidates: list[list[NormalizationCandidate]] = []
        for query_text, row_scores, row_indices in zip(query_texts, scores, indices):
            best_by_concept: dict[str, NormalizationCandidate] = {}
            for similarity, raw_index in zip(row_scores, row_indices):
                if raw_index < 0:
                    continue
                entry = self.term_entries[int(raw_index)]
                embedding_score = float(similarity) - 1.0
                previous = best_by_concept.get(entry.identifier)
                if previous is not None and embedding_score <= previous.embedding_score:
                    continue
                best_by_concept[entry.identifier] = NormalizationCandidate(
                    identifier=entry.identifier,
                    name=entry.raw_name,
                    preferred_label=entry.preferred_label,
                    embedding_score=embedding_score,
                    final_score=embedding_score,
                    is_preferred=entry.is_preferred,
                    matched_alias=entry.raw_name,
                )
            candidates = sorted(
                best_by_concept.values(),
                key=lambda item: item.embedding_score,
                reverse=True,
            )[:topn]
            reranked = [self._rerank_candidate(query_text, item) for item in candidates]
            reranked.sort(
                key=lambda item: (item.final_score, item.embedding_score), reverse=True
            )
            all_candidates.append(reranked)
        return all_candidates

    def _preferred_label_for(self, identifier: str, fallback: str) -> str:
        metadata = self.concept_metadata.get(identifier)
        if metadata is not None and metadata.preferred_label:
            return metadata.preferred_label
        return fallback

    @staticmethod
    def _find_document_long_form(
        document_lookup: Mapping[str, str], matched_key: str
    ) -> tuple[str | None, str | None, str | None, float | None]:
        exact = document_lookup.get(matched_key)
        if exact:
            return exact, "ab3p_exact_key", matched_key, 1.0
        for variant in abbreviation_variant_keys(matched_key):
            if variant == matched_key:
                continue
            hit = document_lookup.get(variant)
            if hit:
                return hit, "ab3p_exact_variant", variant, 1.0

        threshold = abbreviation_threshold_for(matched_key)
        best_key: str | None = None
        best_score = -1.0
        for document_key in document_lookup:
            score = abbreviation_sequence_ratio(matched_key, document_key)
            if score > best_score:
                best_key, best_score = document_key, score
        if best_key is not None and best_score >= threshold:
            return (
                document_lookup[best_key],
                "ab3p_fuzzy_shortform",
                best_key,
                float(best_score),
            )
        return None, None, None, None

    def _resolve_abbreviation_key(
        self,
        *,
        request_index: int,
        request: NormalizationRequest,
        matched_key: str,
        match_score: float,
        method: str,
        document_abbreviations: Mapping[str, Mapping[str, str]],
    ) -> tuple[NormalizationResult | None, _ModelJob | None]:
        direct = self.abbreviation_lookup.direct_lookup.get(matched_key)
        if direct is not None:
            short_form, identifier = direct
            candidate = NormalizationCandidate(
                identifier=identifier,
                name=short_form,
                preferred_label=self._preferred_label_for(identifier, short_form),
                embedding_score=float(match_score),
                final_score=float(match_score),
                source="abbreviation_direct",
                matched_alias=short_form,
                abbreviation_method=method,
            )
            return (
                NormalizationResult(
                    mention_text=request.mention_text,
                    normalized_text=request.normalized_text,
                    document_key=request.document_key,
                    candidates=[candidate],
                ),
                None,
            )

        if matched_key not in self.abbreviation_lookup.ambiguous_candidates:
            return None, None

        document_lookup = document_abbreviations.get(request.document_key, {})
        long_form, ab3p_method, ab3p_key, ab3p_score = self._find_document_long_form(
            document_lookup, matched_key
        )
        if not long_form:
            return None, None
        return (
            None,
            _ModelJob(
                request_index=request_index,
                query_text=plural_normalize_text(long_form),
                mode="ambiguous_long_form",
                abbreviation_method=method,
                expanded_long_form=long_form,
                ab3p_method=ab3p_method,
                ab3p_matched_key=ab3p_key,
                ab3p_match_score=ab3p_score,
            ),
        )

    def normalize_batch(
        self,
        requests: Sequence[NormalizationRequest],
        *,
        document_abbreviations: Mapping[str, Mapping[str, str]] | None = None,
    ) -> list[NormalizationResult]:
        """Normalize a bounded batch, vectorizing encoder work across mentions."""

        if not requests:
            return []
        document_abbreviations = document_abbreviations or {}
        results: list[NormalizationResult | None] = [None] * len(requests)
        normal_jobs: list[_ModelJob] = []
        ambiguous_jobs: list[_ModelJob] = []
        fuzzy_indices: list[int] = []
        fuzzy_keys: list[str] = []

        for index, request in enumerate(requests):
            if is_abbreviation_like(request.mention_text) and self.abbreviation_lookup:
                key = normalize_abbreviation_key(request.mention_text)
                if key in self.abbreviation_lookup.key_to_candidates:
                    result, job = self._resolve_abbreviation_key(
                        request_index=index,
                        request=request,
                        matched_key=key,
                        match_score=1.0,
                        method="exact_short_abbreviation",
                        document_abbreviations=document_abbreviations,
                    )
                    if result is not None:
                        results[index] = result
                        continue
                    if job is not None:
                        ambiguous_jobs.append(job)
                        continue
                elif abbreviation_threshold_for(request.mention_text) < 1.0:
                    fuzzy_indices.append(index)
                    fuzzy_keys.append(key)
                    continue

            normal_jobs.append(
                _ModelJob(
                    request_index=index,
                    query_text=request.normalized_text,
                    mode="normal",
                )
            )

        if fuzzy_indices:
            abbreviation_matrix = self._ensure_abbreviation_embeddings()
            if abbreviation_matrix is not None:
                query_embeddings = self._encode_texts(fuzzy_keys)
                fuzzy_scores, fuzzy_positions = self._topk_similarity(
                    query_embeddings, abbreviation_matrix, topk=1
                )
                for request_index, similarity_row, position_row in zip(
                    fuzzy_indices, fuzzy_scores, fuzzy_positions
                ):
                    similarity = float(similarity_row[0])
                    matched_position = int(position_row[0])
                    request = requests[request_index]
                    if (
                        matched_position >= 0
                        and similarity >= abbreviation_threshold_for(request.mention_text)
                    ):
                        matched_key = self.abbreviation_lookup.all_keys[matched_position]
                        result, job = self._resolve_abbreviation_key(
                            request_index=request_index,
                            request=request,
                            matched_key=matched_key,
                            match_score=similarity,
                            method="encoder_abbreviation_match",
                            document_abbreviations=document_abbreviations,
                        )
                        if result is not None:
                            results[request_index] = result
                            continue
                        if job is not None:
                            ambiguous_jobs.append(job)
                            continue
                    normal_jobs.append(
                        _ModelJob(
                            request_index=request_index,
                            query_text=request.normalized_text,
                            mode="normal",
                        )
                    )
            else:
                for request_index in fuzzy_indices:
                    request = requests[request_index]
                    normal_jobs.append(
                        _ModelJob(
                            request_index=request_index,
                            query_text=request.normalized_text,
                            mode="normal",
                        )
                    )

        def apply_model_jobs(
            jobs: Sequence[_ModelJob], *, topn: int, initial_k: int
        ) -> None:
            if not jobs:
                return
            unique_queries: list[str] = []
            query_to_index: dict[str, int] = {}
            for job in jobs:
                if job.query_text not in query_to_index:
                    query_to_index[job.query_text] = len(unique_queries)
                    unique_queries.append(job.query_text)
            linked = self._retrieve_query_batch(
                unique_queries,
                topn=topn,
                initial_k=initial_k,
            )
            for job in jobs:
                request = requests[job.request_index]
                candidates = [
                    replace(candidate)
                    for candidate in linked[query_to_index[job.query_text]]
                ]
                for candidate in candidates:
                    if job.mode == "ambiguous_long_form":
                        candidate.source = "abbreviation_ambiguous_via_long_form"
                        candidate.abbreviation_method = job.abbreviation_method
                        candidate.expanded_long_form = job.expanded_long_form
                        candidate.ab3p_method = job.ab3p_method
                        candidate.ab3p_matched_key = job.ab3p_matched_key
                        candidate.ab3p_match_score = job.ab3p_match_score
                    else:
                        candidate.source = "model_normal"
                results[job.request_index] = NormalizationResult(
                    mention_text=request.mention_text,
                    normalized_text=request.normalized_text,
                    document_key=request.document_key,
                    candidates=candidates,
                )

        apply_model_jobs(normal_jobs, topn=1, initial_k=20)
        apply_model_jobs(ambiguous_jobs, topn=5, initial_k=50)

        final_results: list[NormalizationResult] = []
        for request, result in zip(requests, results):
            final_results.append(
                result
                if result is not None
                else NormalizationResult(
                    mention_text=request.mention_text,
                    normalized_text=request.normalized_text,
                    document_key=request.document_key,
                    candidates=[],
                )
            )
        return final_results

    def close(self) -> None:
        model = self.model
        tokenizer = self.tokenizer
        dictionary = self.dictionary_embeddings
        abbreviations = self.abbreviation_embeddings
        self.model = None
        self.tokenizer = None
        self.dictionary_embeddings = None
        self.abbreviation_embeddings = None
        del model, tokenizer, dictionary, abbreviations
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


__all__ = [
    "DEFAULT_ONTOLOGY_PATH",
    "DEFAULT_ABBREVIATIONS_PATH",
    "CellOntologyNormalizer",
    "NormalizationRequest",
    "NormalizationResult",
    "NormalizationCandidate",
    "extract_document_abbreviations",
    "plural_normalize_text",
    "normalize_abbreviation_key",
    "is_abbreviation_like",
]
