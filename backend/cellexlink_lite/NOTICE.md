# CellExLink attribution

This directory contains the minimal recognition and Cell Ontology normalization
runtime used by Ovarian Network.  It is adapted from **CellExLink**, authored by
Alimire Nabijiang and distributed under **GPL-3.0-only**.

Upstream project: `https://github.com/ShahriyariLab/CellExLink`

The web integration intentionally omits the upstream command-line, plain-text,
BioC, PubTator3, PMID/PMCID retrieval, notebook display, and general file-routing
workflows.  It accepts only the `chunks.jsonl` / `chunks.jsonl.gz` records
created by this application.

The Stage 4 cell-hierarchy resource is a compact transformation of Cell
Ontology release `2025-12-17` and retains direct `is_a` relationships for
on-demand visualization. Cell Ontology is licensed under CC BY 4.0.
