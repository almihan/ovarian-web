"""Shared text-normalization and exact-span helpers for Stage 2 entities.

The abbreviation pipeline intentionally keeps short-form comparison separate
from long-form/resource comparison:

* short-form keys remove whitespace, periods, and underscores;
* resource keys retain word boundaries;
* compact resource keys are used only as a secondary exact lookup;
* plural/singular transformations are secondary candidates and never replace
  the original key.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

_DASHES = "-−‐‑‒–—﹘﹣－"
_DASH_TRANSLATION = str.maketrans({character: "-" for character in _DASHES})
_ALLOWED_SYMBOLS = frozenset("+-/")
_LEFT_BOUNDARY = r"(?<![A-Za-z0-9])"
_RIGHT_BOUNDARY = r"(?![A-Za-z0-9])"


def normalize_unicode(value: object) -> str:
    """Apply NFKC and normalize Unicode dash characters to ASCII ``-``."""

    return unicodedata.normalize("NFKC", str(value or "")).translate(
        _DASH_TRANSLATION
    )


def canonical_short_form_key(value: object) -> str:
    """Return the exact comparison key used for abbreviation short forms.

    The key is Unicode-normalized, upper-cased, stripped of surrounding
    whitespace, and stripped of spaces, periods, underscores, and unsupported
    punctuation. Unicode letters, digits, and ``+``, ``-``, ``/`` are retained.
    """

    text = normalize_unicode(value).strip().upper()
    output: list[str] = []
    for character in text:
        if character.isspace() or character in "._":
            continue
        if character.isalnum() or character in _ALLOWED_SYMBOLS:
            output.append(character)
    return "".join(output)


def canonical_resource_key(value: object) -> str:
    """Return a word-boundary-preserving exact key for resource terms."""

    text = normalize_unicode(value).strip().upper()
    output: list[str] = []
    pending_space = False
    for character in text:
        if character.isalnum() or character in _ALLOWED_SYMBOLS:
            if pending_space and output and output[-1] != " ":
                output.append(" ")
            output.append(character)
            pending_space = False
        elif character.isspace() or character in "._" or unicodedata.category(
            character
        ).startswith("P"):
            pending_space = bool(output)
        else:
            pending_space = bool(output)
    return " ".join("".join(output).split())


def compact_resource_key(value: object) -> str:
    """Return the secondary compact exact key for a long-form/resource term."""

    return canonical_resource_key(value).replace(" ", "")


def _singular_word_candidates(word: str) -> tuple[str, ...]:
    """Return conservative ``-s``/``-es`` variants for one final word.

    Both removals are emitted when they are syntactically possible.  The exact
    resource index decides whether either candidate is valid, so a form such as
    ``MACROPHAGES`` can resolve through ``MACROPHAGE`` without destructively
    changing the original term, while malformed variants simply find no match.
    """

    lower = word.casefold()
    if len(word) <= 3 or lower.endswith(("ss", "us", "is")):
        return ()

    candidates: list[str] = []
    if lower.endswith("ies") and len(word) > 4:
        candidates.append(word[:-3] + ("Y" if word.isupper() else "y"))
    # Try removing the ordinary plural ``s`` before the broader ``es`` form.
    # This makes MACROPHAGES -> MACROPHAGE the first useful candidate while
    # still permitting CLASSES -> CLASS through the later ``es`` candidate.
    if lower.endswith("s") and len(word) > 3:
        candidates.append(word[:-1])
    if lower.endswith("es") and len(word) > 4:
        candidates.append(word[:-2])
    return tuple(dict.fromkeys(item for item in candidates if item and item != word))


def resource_key_candidates(value: object) -> tuple[str, ...]:
    """Return the original key followed by controlled singular candidates.

    Only the final lexical token is varied.  The original exact key is always
    tried first and is never overwritten.
    """

    original = canonical_resource_key(value)
    if not original:
        return ()
    parts = original.split()
    candidates = [original]
    for singular_last in _singular_word_candidates(parts[-1]):
        candidates.append(" ".join((*parts[:-1], singular_last)))
    return tuple(dict.fromkeys(candidates))


def short_form_key_candidates(surface: object) -> tuple[str, ...]:
    """Return the original short-form key and controlled lowercase plural forms.

    A suffix is removed only when the source surface visibly ends in lowercase
    ``s`` or ``es``. This permits ``TAMs`` -> ``TAM`` while leaving ``RAS`` and
    ``NOS`` unchanged.
    """

    raw = normalize_unicode(surface).strip()
    original = canonical_short_form_key(raw)
    if not original:
        return ()
    candidates = [original]
    if raw.endswith("es") and len(original) > 4:
        candidate = canonical_short_form_key(raw[:-2])
        if candidate:
            candidates.append(candidate)
    if raw.endswith("s") and len(original) > 3:
        candidate = canonical_short_form_key(raw[:-1])
        if candidate:
            candidates.append(candidate)
    return tuple(dict.fromkeys(candidates))


def spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and left[1] > right[0]


def span_contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] >= inner[1]


def _literal_fragment(value: str) -> str:
    """Create a case-insensitive pattern allowing normalized formatting variants."""

    pieces: list[str] = []
    for character in normalize_unicode(value):
        if character.isspace() or character in "._":
            pieces.append(r"[\s._]*")
        elif character == "-":
            pieces.append(f"[{re.escape(_DASHES)}]")
        else:
            pieces.append(re.escape(character))
    return "".join(pieces)


def compile_surface_pattern(surface: object) -> re.Pattern[str]:
    """Compile a boundary-aware exact/flexible pattern for one known surface."""

    text = str(surface or "").strip()
    if not text:
        return re.compile(r"(?!x)x")
    return re.compile(
        rf"{_LEFT_BOUNDARY}{_literal_fragment(text)}{_RIGHT_BOUNDARY}",
        flags=re.IGNORECASE,
    )


def iter_surface_matches(text: str, surfaces: tuple[str, ...]) -> Iterator[re.Match[str]]:
    """Yield unique matches for known surfaces in deterministic span order."""

    by_span: dict[tuple[int, int], re.Match[str]] = {}
    for surface in surfaces:
        for match in compile_surface_pattern(surface).finditer(text):
            by_span.setdefault(match.span(), match)
    for span in sorted(by_span):
        yield by_span[span]


__all__ = [
    "canonical_resource_key",
    "canonical_short_form_key",
    "compact_resource_key",
    "compile_surface_pattern",
    "iter_surface_matches",
    "normalize_unicode",
    "resource_key_candidates",
    "short_form_key_candidates",
    "span_contains",
    "spans_overlap",
]
