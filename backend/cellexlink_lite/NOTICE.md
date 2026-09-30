# CellExLink attribution

This directory contains the minimal recognition and Cell Ontology normalization
runtime used by Ovarian Network.  It is adapted from **CellExLink**, authored by
Alimire Nabijiang and distributed under **GPL-3.0-only**.

Upstream project: `https://github.com/ShahriyariLab/CellExLink`

The web integration intentionally omits the upstream command-line, plain-text,
BioC, PubTator3, PMID/PMCID retrieval, notebook display, and general file-routing
workflows.  It accepts only the `chunks.jsonl` / `chunks.jsonl.gz` records
created by this application.

Cell mention normalization uses `cell_ontology_v2026-06-08.jsonl`, the bundled
June 8, 2026 lexicon. This JSONL contains labels and synonyms, but no parent
relationships, so it cannot supply a June 2026 hierarchy.

The separate Stage 4 cell-hierarchy resource is a compact transformation of Cell
Ontology release `2025-12-17` and retains direct `is_a` relationships for
on-demand visualization. Cell Ontology is licensed under CC BY 4.0.
