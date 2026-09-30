"""Canonical identifier handling shared by relation and network stages."""

from __future__ import annotations

import re
from typing import Any

_SPACE_RE = re.compile(r"\s+")
_HGNC_RE = re.compile(r"^(?:HGNC:)?(\d+)$", re.IGNORECASE)
_CHEBI_RE = re.compile(r"^(?:CHEBI:)?(\d+)$", re.IGNORECASE)
_MESH_RE = re.compile(r"^(?:MESH:)?([CD]\d+)$", re.IGNORECASE)
_NCBI_GENE_RE = re.compile(
    r"^(?:(?:NCBIGENE|GENEID|GENE):)?(\d+)$",
    re.IGNORECASE,
)
_CELL_ONTOLOGY_RE = re.compile(r"^(?:CL[:_])?(\d+)$", re.IGNORECASE)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return _SPACE_RE.sub(" ", str(value)).strip()


def canonical_identifier(
    entity_type: str,
    field: str,
    value: Any,
) -> tuple[str, str]:
    """Return ``(namespace, canonical value)`` for an entity identifier.

    The same CURIE can occur in ``concept_id``, ``normalized_id``, or a
    namespace-specific field during rolling deployments.  Canonicalizing both
    the namespace and value prevents duplicate graph nodes or relation tags.
    """

    text = _text(value)
    if not text:
        return "", ""

    match = _HGNC_RE.fullmatch(text)
    if match and (":" in text or field == "hgnc_id"):
        return "hgnc", f"HGNC:{match.group(1)}"

    match = _CHEBI_RE.fullmatch(text)
    if match and (":" in text or field == "chebi_id"):
        return "chebi", f"CHEBI:{match.group(1)}"

    match = _MESH_RE.fullmatch(text)
    if match and (
        ":" in text
        or field in {"mesh_id", "pubtator_mesh_id", "chemical_id"}
    ):
        return "mesh", f"MESH:{match.group(1).upper()}"

    match = _NCBI_GENE_RE.fullmatch(text)
    if match and (
        ":" in text
        or field in {"ncbi_gene_id", "pubtator_gene_id", "gene_id"}
    ):
        return "ncbi_gene", f"NCBIGene:{match.group(1)}"

    if entity_type == "cell":
        match = _CELL_ONTOLOGY_RE.fullmatch(text)
        if match and (
            text.upper().startswith("CL")
            or field in {"cell_ontology_id", "concept_id", "normalized_id"}
        ):
            return "cell_ontology", f"CL:{match.group(1)}"

    # A namespace-aware CURIE not explicitly handled above is still safer than
    # a bare source field. Preserve it under its declared prefix.
    if ":" in text:
        prefix = text.split(":", 1)[0].strip().casefold()
        if prefix:
            return prefix, text

    return field.casefold(), text


__all__ = ["canonical_identifier"]
