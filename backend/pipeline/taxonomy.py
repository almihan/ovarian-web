"""Small taxonomy helpers shared by Stage 2 and Stage 3."""

from __future__ import annotations

from typing import Any, Mapping

HUMAN_TAX_ID = "9606"
HUMAN_TAX_NAME = "Homo sapiens"


def normalize_tax_id(value: Any) -> str:
    """Return a canonical numeric NCBI Taxonomy ID, or an empty string."""

    text = "" if value is None else str(value).strip()
    if not text.isdigit():
        return ""
    normalized = text.lstrip("0") or "0"
    return "" if normalized == "0" else normalized


def is_verified_human_gene(annotation: Mapping[str, Any]) -> bool:
    """Return True only for a gene/protein explicitly verified as human."""

    entity_type = str(annotation.get("obj") or annotation.get("entity_type") or "").casefold()
    return entity_type in {"gene", "protein"} and normalize_tax_id(
        annotation.get("tax_id")
    ) == HUMAN_TAX_ID


__all__ = [
    "HUMAN_TAX_ID",
    "HUMAN_TAX_NAME",
    "is_verified_human_gene",
    "normalize_tax_id",
]
