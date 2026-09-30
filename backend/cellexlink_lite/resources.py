"""Paths to the compact CellExLink resources bundled with the local app."""

from __future__ import annotations

from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
CELL_ONTOLOGY_RELEASE = "2026-06-08"
CELL_ONTOLOGY_RESOURCE_VERSION = f"cell-ontology-v{CELL_ONTOLOGY_RELEASE}"
# The June lexical export has labels/synonyms but no parent relationships.
# Keep the separate is_a snapshot's actual release rather than relabeling it.
CELL_HIERARCHY_RELEASE = "2025-12-17"
DEFAULT_ONTOLOGY_PATH = RESOURCE_DIR / "cell_ontology_v2026-06-08.jsonl"
DEFAULT_HIERARCHY_PATH = RESOURCE_DIR / "cell_ontology_hierarchy_v2025-12-17.jsonl.gz"
DEFAULT_ABBREVIATIONS_PATH = RESOURCE_DIR / "abbreviations.tsv"

__all__ = [
    "RESOURCE_DIR",
    "CELL_ONTOLOGY_RELEASE",
    "CELL_ONTOLOGY_RESOURCE_VERSION",
    "CELL_HIERARCHY_RELEASE",
    "DEFAULT_ONTOLOGY_PATH",
    "DEFAULT_HIERARCHY_PATH",
    "DEFAULT_ABBREVIATIONS_PATH",
]
