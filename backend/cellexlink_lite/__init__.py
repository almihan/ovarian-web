"""Lazy CellExLink exports for the local entity-extraction worker.

Importing this package in FastAPI exposes resource paths without importing
NumPy, PyTorch, or Transformers. Inference modules are loaded only by the
short-lived recognition and normalization process.
"""

from __future__ import annotations

from typing import Any

from .resources import (
    CELL_HIERARCHY_RELEASE,
    CELL_ONTOLOGY_RELEASE,
    DEFAULT_ABBREVIATIONS_PATH,
    DEFAULT_HIERARCHY_PATH,
    DEFAULT_ONTOLOGY_PATH,
)

_RECOGNITION_EXPORTS = {
    "ChunkNER",
    "EntitySpan",
    "reconstruct_entities",
}
_NORMALIZATION_EXPORTS = {
    "AB3P_HEALTHCHECK_LONG_FORM",
    "AB3P_HEALTHCHECK_SHORT_FORM",
    "AB3P_HEALTHCHECK_TEXT",
    "AB3P_LONG_FORM_MIN_RAW_COSINE",
    "FUZZY_ABBREVIATION_MAX_LENGTH",
    "FUZZY_ABBREVIATION_MIN_COSINE",
    "FUZZY_ABBREVIATION_MIN_LENGTH",
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
    "plural_normalize_text",
    "protected_signature",
    "run_ab3p_for_document",
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
    "CELL_HIERARCHY_RELEASE",
    "CELL_ONTOLOGY_RELEASE",
    "DEFAULT_ABBREVIATIONS_PATH",
    "DEFAULT_HIERARCHY_PATH",
    "DEFAULT_ONTOLOGY_PATH",
]
