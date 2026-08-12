"""Paths to the compact CellExLink resources bundled with the web service."""

from __future__ import annotations

from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
CELL_ONTOLOGY_RELEASE = "2025-12-17"
DEFAULT_ONTOLOGY_PATH = RESOURCE_DIR / "cell_ontology_v2025-12-17.jsonl"
DEFAULT_HIERARCHY_PATH = RESOURCE_DIR / "cell_ontology_hierarchy_v2025-12-17.jsonl.gz"
DEFAULT_ABBREVIATIONS_PATH = RESOURCE_DIR / "abbreviations.tsv"

__all__ = [
    "RESOURCE_DIR",
    "CELL_ONTOLOGY_RELEASE",
    "DEFAULT_ONTOLOGY_PATH",
    "DEFAULT_HIERARCHY_PATH",
    "DEFAULT_ABBREVIATIONS_PATH",
]
