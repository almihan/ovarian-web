"""Minimal, lazily imported CellExLink runtime for chunk annotation.

Importing this package in the FastAPI process exposes only resource paths. The
NumPy, PyTorch, and Transformers inference modules are imported inside the
short-lived NER and NEN workers.
"""

from __future__ import annotations

from typing import Any

from .resources import DEFAULT_ABBREVIATIONS_PATH, DEFAULT_ONTOLOGY_PATH

_RECOGNITION_EXPORTS = {
    "ChunkNER",
    "EntitySpan",
    "reconstruct_entities",
}
_NORMALIZATION_EXPORTS = {
    "CellOntologyNormalizer",
    "NormalizationRequest",
    "NormalizationResult",
    "NormalizationCandidate",
    "extract_document_abbreviations",
    "plural_normalize_text",
}


def __getattr__(name: str) -> Any:
    if name in _RECOGNITION_EXPORTS:
        from . import recognition

        return getattr(recognition, name)
    if name in _NORMALIZATION_EXPORTS:
        from . import normalization

        return getattr(normalization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *_RECOGNITION_EXPORTS,
    *_NORMALIZATION_EXPORTS,
    "DEFAULT_ABBREVIATIONS_PATH",
    "DEFAULT_ONTOLOGY_PATH",
]
