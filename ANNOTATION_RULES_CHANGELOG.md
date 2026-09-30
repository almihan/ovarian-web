# Cell-context revision — September 11, 2026

- Added ontology-derived shared-head modifier recovery for coordinated cell phrases. Normalization uses reference labels/synonyms, never a hard-coded cell ID.
- Identical spans now use **cell > hormone > gene/protein**. Longest-span priority is unchanged for different-length overlaps.
- Cell exact matching preserves lexical token boundaries; compact lookup is candidate retrieval only.
- Short word-like static aliases and lowercase uses of uppercase acronyms are no longer accepted without the separate document-definition route.
- Extended figure/table-reference masking to letter-only panels in lists.
- Static matches report the actual dictionary filename, line, and content version; the ontology version is stored separately.
- Applied safeguards in prepass, NER span repair, normalization, paper-level recovery, merge, export and relation-tag selection. Coordinated inferences are not reused as propagation seeds.
- Updated pipeline version markers. Re-run entity extraction and relation extraction; completed results and saved corpora are not rewritten automatically.
- No new tests or validation scripts were added, and no tests or model/service execution were run for this revision.

Detailed behavior: [CELL_CONTEXT_MATCHING.md](CELL_CONTEXT_MATCHING.md).

---

# Annotation rules update — September 11, 2026

## Changes

- PubTator3: direct NCBI Gene ID → HGNC mapping; no source-species validation and no NCBIGene-only canonical fallback. Unmapped genes are discarded.
- Overlaps: longest span first across all entity types; hormone wins a same-span gene/protein conflict. Applied during local selection, merging, recovery, export, relation tagging, and network preparation.
- Receptors: complete receptor expression → HGNC gene/protein, or discard. Shorter contained base annotations cannot survive an unmapped receptor expression. The new shared implementation is `backend/pipeline/receptor_annotations.py`.
- Cell phrases: a longer ontology-backed mention can replace a shorter locked/recognized cell span. The complete text is normalized to its own Cell Ontology identity.
- Removed `backend/pipeline/protected_gene_annotations.py` and all active special-name override hooks.
- Corrected the normalizer description and implementation notes to document exact-alias matching followed by cosine-thresholded vector top-1 (default 0.95). The cosine acceptance rule itself is unchanged.
- Updated pipeline/cache version markers. Existing saved corpora were preserved and require reprocessing to acquire the new annotations.

The complete project and bundled references are retained. Pre-existing tests are unchanged; no tests or validation scripts were added. Full model inference and live PubTator calls were not run.

See `IMPLEMENTATION_NOTES_AB3P_CELL_HORMONE_TOP1.md` for the current processing rules.
