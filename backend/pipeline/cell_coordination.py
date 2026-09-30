"""Resource-backed recovery of coordinated modifiers with a shared cell head.

There are no cell names, subtype IDs, or modifier-to-ID maps in this module.
Omitted heads are reconstructed for lookup only. Original contiguous source
spans and the evidence for every arm are retained in the annotation.
"""
from __future__ import annotations

import re
from dataclasses import replace
from functools import cached_property, lru_cache
from typing import Callable, Iterable

from backend.pipeline.cell_surface_matching import cell_words
from backend.pipeline.entity_lexicons import ExactResolution, TargetEntityCandidate

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CONJUNCTION = re.compile(r"\b(?:and/or|and|or)\b", re.I)
_ARM_SEPARATOR = re.compile(r"\s+(?:and/or|and|or)\s+|\s*,\s*", re.I)
_HARD_BOUNDARY = re.compile(r"[.;:!?()\n]")


class CoordinatedCellResolver:
    def __init__(self, ontology_terms: Iterable[TargetEntityCandidate],
                 literal_resolver: Callable[[str], ExactResolution]) -> None:
        self.terms = tuple(ontology_terms)
        self.literal = literal_resolver

    @cached_property
    def modifiers(self) -> frozenset[tuple[str, ...]]:
        """Learn descriptor phrases from ontology labels/synonyms plus a head.

        A descriptor is admitted only when removing it leaves a complete,
        independently resolvable cell name. Bare 'cell' is not an informative
        head. Descriptors are grammatical evidence, not subtype equivalences.
        """
        found: set[tuple[str, ...]] = set()
        for item in self.terms:
            term = item.matched_term
            if _CONJUNCTION.search(term) or re.search(r"[,;/()]", term):
                continue
            tokens = list(_WORD.finditer(term))
            for split in range(1, min(len(tokens), 5)):
                prefix = term[:tokens[split].start()].strip(" -_")
                head = term[tokens[split].start():]
                if cell_words(head) in {("cell",), ("cells",)}:
                    continue
                if self.literal(head).candidate is None:
                    continue
                # Fully named cell populations are not mere modifiers.
                if self.literal(prefix).candidate is not None:
                    continue
                key = cell_words(prefix)
                if key:
                    found.add(key)
        return frozenset(found)

    @lru_cache(maxsize=2048)
    def resolve(self, phrase: str) -> ExactResolution:
        raw = phrase.strip()
        conjunctions = list(_CONJUNCTION.finditer(raw))
        if not conjunctions or _HARD_BOUNDARY.search(raw):
            return ExactResolution("unmatched")
        tokens = list(_WORD.finditer(raw))
        if len(tokens) > 24:
            return ExactResolution("unmatched")
        last_conjunction = conjunctions[-1].end()
        possibilities: list[TargetEntityCandidate] = []
        for token in tokens:
            if token.start() <= last_conjunction:
                continue
            head = raw[token.start():].strip()
            base = self.literal(head)
            if base.candidate is None or cell_words(head) in {("cell",), ("cells",)}:
                continue
            prefix = raw[:token.start()].strip()
            modifiers = [part.strip() for part in _ARM_SEPARATOR.split(prefix) if part.strip()]
            if len(modifiers) < 2 or len(modifiers) > 6:
                continue
            if any(cell_words(modifier) not in self.modifiers for modifier in modifiers):
                continue
            resolved: list[TargetEntityCandidate] = []
            evidence: list[dict] = []
            ambiguous = False
            unresolved = []
            for modifier in modifiers:
                lookup = f"{modifier} {head}"
                result = self.literal(lookup)
                arm = {"modifier": modifier, "lookup": lookup, "status": result.status}
                if result.candidate is not None:
                    resolved.append(result.candidate)
                    arm.update({"concept_id": result.candidate.concept_id,
                                "matched_term": result.candidate.matched_term})
                elif result.status != "unmatched":
                    ambiguous = True
                else:
                    unresolved.append(lookup)
                evidence.append(arm)
            identifiers = {candidate.concept_id for candidate in resolved}
            if ambiguous or len(identifiers) != 1:
                continue
            # Do not reinterpret an AND-list of incompletely known populations
            # as a single cell type. OR permits an unlisted alternative modifier
            # only when ontology-backed arms provide one unambiguous subtype.
            if unresolved and any(m.group().casefold() != "or" for m in conjunctions):
                continue
            chosen = resolved[0]
            if chosen.concept_id == base.candidate.concept_id:
                continue
            possibilities.append(replace(chosen, term_kind="coordinated_shared_head",
                match_metadata={
                    **dict(chosen.match_metadata or {}),
                    "coordination_shared_head": head,
                    "coordination_arms": evidence,
                    "coordination_unresolved_arms": unresolved,
                    "coordination_rule": "ontology_shared_head_modifiers_v1",
                    "normalization_scope": ("supported_alternative_arm" if unresolved
                                             else "all_arms_same_identity"),
                }))
        if len({candidate.concept_id for candidate in possibilities}) != 1:
            return ExactResolution("unmatched")
        candidate = possibilities[0]
        return ExactResolution("resolved_target", candidate, (candidate,),
                               "coordinated_shared_head_exact")

    def find(self, text: str, literal_hits: Iterable[tuple[int, int, TargetEntityCandidate]]
             ) -> list[tuple[int, int, TargetEntityCandidate]]:
        """Extend a resource-backed rightmost cell head to compatible modifiers."""
        if not _CONJUNCTION.search(text):
            return []
        tokens = list(_WORD.finditer(text))
        found: dict[tuple[int, int], TargetEntityCandidate] = {}
        for head_start, end, candidate in literal_hits:
            if candidate.entity_type != "cell":
                continue
            left_tokens = [token for token in tokens if token.end() <= head_start][-18:]
            for token in left_tokens:
                start = token.start()
                phrase = text[start:end]
                if not _CONJUNCTION.search(phrase) or _HARD_BOUNDARY.search(phrase):
                    continue
                result = self.resolve(phrase)
                if result.candidate is not None:
                    found[(start, end)] = result.candidate
                    break  # longest compatible left extension for this head
        return [(start, end, candidate) for (start, end), candidate in sorted(found.items())]
