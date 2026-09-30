"""Conservative cell-resource spelling rules, independent of models and IDs.

Lookup keys may retrieve candidates broadly, but acceptance preserves lexical
boundaries. Static short codes also retain capitalization evidence; a resource
row is not permission to label an ordinary lowercase word in running prose.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

from backend.pipeline.entity_text_normalization import normalize_unicode

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def cell_words(value: object) -> tuple[str, ...]:
    """Accent/case-normalized words for lookup, never for source offsets."""
    text = unicodedata.normalize("NFKD", normalize_unicode(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return tuple(m.group().casefold() for m in _WORD.finditer(text))


def _word_variants(words: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    if not words:
        return ()
    result = [words]
    last = words[-1]
    if len(last) > 3 and not last.endswith(("ss", "us", "is")):
        if last.endswith("s"):
            result.append((*words[:-1], last[:-1]))
        if last.endswith("ies"):
            result.append((*words[:-1], last[:-3] + "y"))
        if last.endswith("es"):
            result.append((*words[:-1], last[:-2]))
    return tuple(result)


def _semantic_marks(value: object) -> tuple[tuple[int, str], ...]:
    text = normalize_unicode(value)
    # Plus, slash, and terminal minus markers change biological meaning.
    # Commas/parentheses must not disappear when a compact key is generated.
    pattern = r"[+/(),;:]|(?<=\w)-(?=\s|$)"
    return tuple((len(cell_words(text[:m.start()])), m.group())
                 for m in re.finditer(pattern, text))


def static_cell_surface_allowed(surface: object, reference: object) -> bool:
    """Accept code-like casing, not bare word-like short dictionary aliases.

    A document-defined short form is resolved separately from this static path.
    Multiword names and ordinary long names retain case-insensitive matching.
    Alphabetic abbreviations must preserve evidence of acronym capitalization;
    digits distinguish codes such as Th1 even when their case varies.
    """
    actual = normalize_unicode(surface).strip()
    expected = normalize_unicode(reference).strip()
    tokens = cell_words(expected)
    if len(tokens) != 1:
        return True
    letters = [c for c in expected if c.isalpha()]
    if any(c.isdigit() for c in expected):
        return True
    if sum(c.isupper() for c in letters) >= 2:
        return sum(c.isupper() for c in actual if c.isalpha()) >= 2
    # Short title-case/lowercase words are not safe free-text abbreviations.
    # They remain usable through a locally defined, ontology-resolved long form.
    if len(tokens[0]) <= 5:
        return False
    return True


def cell_term_matches(surface: object, reference: object,
                      term_kind: str = "", *, check_static_case: bool = True) -> bool:
    """Compare whole lexical tokens, allowing only controlled final plurals.

    NK-cell and NK cell have the same boundaries; B and and band do not.
    Spaces, punctuation, or coordination can never assemble a different word.
    """
    source_words, target_words = cell_words(surface), cell_words(reference)
    if not source_words or not target_words:
        return False
    if not set(_word_variants(source_words)).intersection(_word_variants(target_words)):
        return False
    if _semantic_marks(surface) != _semantic_marks(reference):
        return False
    if term_kind == "static_abbreviation" and check_static_case:
        return static_cell_surface_allowed(surface, reference)
    # Ontology short acronyms also need visible code casing. Ordinary aliases
    # such as band, rod, cone and glia are not acronyms and remain available.
    raw_reference = normalize_unicode(reference).strip()
    if (term_kind != "static_abbreviation" and len(target_words) == 1
            and raw_reference.isalpha() and raw_reference.isupper()
            and len(raw_reference) > 1):
        raw_surface = normalize_unicode(surface)
        if sum(c.isupper() for c in raw_surface if c.isalpha()) < 2:
            return False
    return True


def cell_annotation_spelling_allowed(surface: str, row: Mapping[str, Any]) -> bool:
    """Apply the same safeguards to persisted, dictionary, and neural rows."""
    # A valid local definition is evidence unavailable to the static dictionary.
    if row.get("definition_id") or (row.get("expanded_long_form") and
            str(row.get("normalization_source") or "").startswith("ab3p")):
        return True
    reference = str(row.get("matched_term") or row.get("matched_ontology_alias") or "")
    kind = str(row.get("term_kind") or "")
    source = str(row.get("normalization_source") or row.get("recognition_source") or "")
    if kind == "static_abbreviation" or source.startswith(("static_cell", "static_exact")):
        return bool(reference and cell_term_matches(surface, reference, "static_abbreviation"))
    if reference and "".join(cell_words(surface)) == "".join(cell_words(reference)):
        return cell_term_matches(surface, reference, kind)
    return True
