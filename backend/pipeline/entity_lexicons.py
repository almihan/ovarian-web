"""Exact Cell Ontology and MeSH hormone lexicons.

These indexes supply cell/hormone targets to document_entity_recovery, which
also resolves human HGNC genes through a separate nomenclature matcher.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.pipeline.cell_surface_matching import cell_term_matches

from backend.pipeline.entity_text_normalization import (
    canonical_resource_key,
    compact_resource_key,
    resource_key_candidates,
)

MESH_HORMONE_TREE_PREFIX = "D06.472"
MESH_HORMONE_DESCRIPTOR_ID = "D006728"
MESH_HORMONE_RESOURCE_VERSION = "mesh-xml-2026-d06.472-plus-estrogens-v2"
MESH_HORMONE_BUNDLE_FILENAME = "mesh_hormone_lexicon_v2026.jsonl.gz"
DEFAULT_HORMONE_LEXICON_PATH = (
    Path(__file__).resolve().parent.parent
    / "cellexlink_lite"
    / "resources"
    / MESH_HORMONE_BUNDLE_FILENAME
)


@dataclass(slots=True, frozen=True)
class TargetEntityCandidate:
    entity_type: str
    concept_id: str
    preferred_label: str
    matched_term: str
    term_kind: str
    resource_version: str
    match_metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "concept_id": self.concept_id,
            "preferred_label": self.preferred_label,
            "matched_term": self.matched_term,
            "term_kind": self.term_kind,
            "resource_version": self.resource_version,
            **dict(self.match_metadata or {}),
        }


@dataclass(slots=True, frozen=True)
class HormoneLexiconEntry:
    mesh_id: str
    preferred_label: str
    synonyms: tuple[str, ...]
    resource_version: str = MESH_HORMONE_RESOURCE_VERSION

    @property
    def concept_id(self) -> str:
        return self.mesh_id if self.mesh_id.startswith("MESH:") else f"MESH:{self.mesh_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mesh_id": self.mesh_id.removeprefix("MESH:"),
            "mesh_heading": self.preferred_label,
            "entry_terms": list(self.synonyms),
        }


@dataclass(slots=True, frozen=True)
class ExactResolution:
    status: str
    candidate: TargetEntityCandidate | None = None
    candidates: tuple[TargetEntityCandidate, ...] = ()
    match_method: str | None = None


class ExactTargetIndex:
    """Strict and compact exact indexes with explicit ambiguity handling."""

    def __init__(self, candidates: Iterable[TargetEntityCandidate]) -> None:
        strict: dict[str, list[TargetEntityCandidate]] = defaultdict(list)
        compact: dict[str, list[TargetEntityCandidate]] = defaultdict(list)
        for candidate in candidates:
            strict_key = canonical_resource_key(candidate.matched_term)
            compact_key = compact_resource_key(candidate.matched_term)
            if strict_key:
                strict[strict_key].append(candidate)
            if compact_key:
                compact[compact_key].append(candidate)
        self.strict_index = {
            key: self._deduplicate(values) for key, values in strict.items()
        }
        self.compact_index = {
            key: self._deduplicate(values) for key, values in compact.items()
        }

    @staticmethod
    def _deduplicate(
        values: Sequence[TargetEntityCandidate],
    ) -> tuple[TargetEntityCandidate, ...]:
        output: list[TargetEntityCandidate] = []
        seen: set[tuple[str, str, str, str]] = set()
        for value in values:
            key = (value.entity_type, value.concept_id, value.matched_term, value.term_kind)
            if key not in seen:
                seen.add(key)
                output.append(value)
        return tuple(output)

    @staticmethod
    def _result(
        candidates: Sequence[TargetEntityCandidate],
        *,
        match_method: str,
    ) -> ExactResolution:
        # Ambiguity concerns identities, not multiple spellings for one ID.
        unique: dict[tuple[str, str], TargetEntityCandidate] = {}
        for candidate in candidates:
            unique.setdefault((candidate.entity_type, candidate.concept_id), candidate)
        candidates = tuple(unique.values())
        if not candidates:
            return ExactResolution(status="unmatched")
        if len(candidates) == 1:
            return ExactResolution(
                status="resolved_target",
                candidate=candidates[0],
                candidates=(candidates[0],),
                match_method=match_method,
            )
        entity_types = {candidate.entity_type for candidate in candidates}
        return ExactResolution(
            status=(
                "ambiguous_same_type"
                if len(entity_types) == 1
                else "ambiguous_cross_type"
            ),
            candidates=tuple(candidates),
            match_method=match_method,
        )

    def resolve(self, text: object) -> ExactResolution:
        # Original exact key first, then a controlled singular variant. Compact
        # lookup is secondary and accepted only when it is unique.
        strict_keys = resource_key_candidates(text)
        for index, key in enumerate(strict_keys):
            candidates = tuple(candidate for candidate in self.strict_index.get(key, ())
                if candidate.entity_type != "cell" or cell_term_matches(
                    text, candidate.matched_term, candidate.term_kind))
            if candidates:
                return self._result(
                    candidates,
                    match_method=(
                        "strict_exact" if index == 0 else "controlled_singular_exact"
                    ),
                )
        compact_keys = tuple(dict.fromkeys(key.replace(" ", "") for key in strict_keys))
        for index, key in enumerate(compact_keys):
            candidates = tuple(candidate for candidate in self.compact_index.get(key, ())
                if candidate.entity_type != "cell" or cell_term_matches(
                    text, candidate.matched_term, candidate.term_kind))
            if candidates:
                return self._result(
                    candidates,
                    match_method=(
                        "compact_exact"
                        if index == 0
                        else "controlled_singular_compact_exact"
                    ),
                )
        return ExactResolution(status="unmatched")


def _file_version(path: Path, prefix: str) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{prefix}-{digest}"


def load_cell_candidates(
    ontology_path: str | Path,
    *,
    resource_version: str | None = None,
) -> list[TargetEntityCandidate]:
    path = Path(ontology_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cell Ontology resource is missing: {path}")
    version = resource_version or _file_version(path, path.stem)
    candidates: list[TargetEntityCandidate] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid ontology JSON at {path}:{line_no}") from exc
            concept_id = str(row.get("norm_concept_id") or "").strip()
            preferred = str(row.get("norm_preferred_label") or "").strip()
            synonyms = row.get("synonyms") or []
            if not concept_id or not preferred:
                continue
            candidates.append(
                TargetEntityCandidate(
                    entity_type="cell",
                    concept_id=concept_id,
                    preferred_label=preferred,
                    matched_term=preferred,
                    term_kind="preferred_label",
                    resource_version=version,
                )
            )
            if isinstance(synonyms, list):
                for synonym in synonyms:
                    term = str(synonym or "").strip()
                    if not term:
                        continue
                    candidates.append(
                        TargetEntityCandidate(
                            entity_type="cell",
                            concept_id=concept_id,
                            preferred_label=preferred,
                            matched_term=term,
                            term_kind="synonym",
                            resource_version=version,
                        )
                    )
    return candidates


def _open_lexicon(path: Path):
    if path.suffix.casefold() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_hormone_lexicon(path: str | Path) -> list[HormoneLexiconEntry]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    entries: list[HormoneLexiconEntry] = []
    with _open_lexicon(source) as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid hormone lexicon JSON at {source}:{line_no}") from exc
            mesh_id = str(row.get("mesh_id") or "").strip().upper().removeprefix("MESH:")
            label = str(
                row.get("mesh_heading") or row.get("preferred_label") or ""
            ).strip()
            raw_synonyms = row.get("entry_terms")
            if raw_synonyms is None:
                raw_synonyms = row.get("synonyms") or []
            if isinstance(raw_synonyms, str):
                raw_synonyms = [raw_synonyms]
            if not mesh_id or not label or mesh_id == MESH_HORMONE_DESCRIPTOR_ID:
                continue
            synonyms = tuple(
                dict.fromkeys(
                    str(value or "").strip()
                    for value in raw_synonyms
                    if str(value or "").strip() and str(value or "").strip() != label
                )
            )
            entries.append(
                HormoneLexiconEntry(
                    mesh_id=mesh_id,
                    preferred_label=label,
                    synonyms=synonyms,
                    resource_version=str(
                        row.get("resource_version") or MESH_HORMONE_RESOURCE_VERSION
                    ),
                )
            )
    if not entries:
        raise ValueError(f"No usable hormone entries were found in {source}")
    return entries


def hormone_candidates(
    entries: Sequence[HormoneLexiconEntry],
) -> list[TargetEntityCandidate]:
    candidates: list[TargetEntityCandidate] = []
    for entry in entries:
        candidates.append(
            TargetEntityCandidate(
                entity_type="hormone",
                concept_id=entry.concept_id,
                preferred_label=entry.preferred_label,
                matched_term=entry.preferred_label,
                term_kind="preferred_label",
                resource_version=entry.resource_version,
            )
        )
        for synonym in entry.synonyms:
            candidates.append(
                TargetEntityCandidate(
                    entity_type="hormone",
                    concept_id=entry.concept_id,
                    preferred_label=entry.preferred_label,
                    matched_term=synonym,
                    term_kind="synonym",
                    resource_version=entry.resource_version,
                )
            )
    return candidates


def ensure_hormone_lexicon(
    bundle_path: str | Path = DEFAULT_HORMONE_LEXICON_PATH,
) -> tuple[list[HormoneLexiconEntry], str]:
    """Load the prebuilt local MeSH hormone descriptor bundle only.

    This function never calls MeSH RDF or any other remote service. The bundle
    must contain ``mesh_id``, ``mesh_heading``, and ``entry_terms`` fields.
    """

    bundle = Path(bundle_path).expanduser().resolve()
    if not bundle.is_file():
        raise FileNotFoundError(
            "The local MeSH hormone bundle is missing: "
            f"{bundle}. Run download_mesh_hormone_bundle_colab.py and place "
            "the generated mesh_hormone_lexicon_v2026.jsonl.gz file at this path."
        )
    entries = load_hormone_lexicon(bundle)
    # MeSH D004967 is a hormone class under D27.505.696.399.472.277, not
    # D06.472. Keep it distinct from estradiol (D004958). Only the actual
    # substance/class terms are added, not "Estrogen Effect" or agonist terms.
    if bundle.resolve() == DEFAULT_HORMONE_LEXICON_PATH.resolve() and not any(
        entry.concept_id == "MESH:D004967" for entry in entries
    ):
        entries.append(HormoneLexiconEntry("D004967", "Estrogens", ("Estrogen",)))
    return entries, "bundled_mesh_xml"


def build_cell_hormone_index(
    *,
    ontology_path: str | Path,
    ontology_version: str | None,
    hormone_entries: Sequence[HormoneLexiconEntry],
) -> ExactTargetIndex:
    return ExactTargetIndex(
        [
            *load_cell_candidates(
                ontology_path,
                resource_version=ontology_version,
            ),
            *hormone_candidates(hormone_entries),
        ]
    )


__all__ = [
    "DEFAULT_HORMONE_LEXICON_PATH",
    "ExactResolution",
    "ExactTargetIndex",
    "HormoneLexiconEntry",
    "MESH_HORMONE_BUNDLE_FILENAME",
    "MESH_HORMONE_DESCRIPTOR_ID",
    "MESH_HORMONE_RESOURCE_VERSION",
    "MESH_HORMONE_TREE_PREFIX",
    "TargetEntityCandidate",
    "build_cell_hormone_index",
    "ensure_hormone_lexicon",
    "hormone_candidates",
    "load_cell_candidates",
    "load_hormone_lexicon",
]
