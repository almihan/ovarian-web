# Document-aware entity annotation and thresholded Cell Ontology linking

## Active processing order

1. Reconstruct each paper from its selected, ordered source chunks.
2. Run Ab3P once per document, supplemented by resource-backed parenthetical definitions.
3. Resolve definitions against Cell Ontology, the local MeSH hormone lexicon, and HGNC. Preserve definition locations and conflicting meanings.
4. Recover exact resource mentions and contextually supported abbreviation occurrences.
5. Reclassify complete receptor expressions through HGNC before resolving overlaps. An unmapped receptor expression is discarded, including its shorter base annotation.
6. Run CellExLink recognition on unchanged source text. A shorter locked cell annotation does not block a longer complete cell phrase.
7. Normalize remaining cell mentions by the evidence hierarchy below.
8. Independently retrieve PubTator3 annotations and map Gene IDs directly to HGNC. Do not validate source species or download an NCBI human gene reference. Discard genes without HGNC mappings. Keep general HGNC exact-name recovery.
9. Retain PubTator chemicals only when their MeSH identifiers belong to the local hormone resource.
10. Merge branches, perform non-recursive same-paper mention recovery, and apply the shared longest-span policy. Export and relation tagging use the same policy.

## Cell normalization

The active order is:

1. Contextually selected exact document abbreviation definition.
2. Exact unique static abbreviation mapping in `abbreviations.tsv`.
3. Unique exact Cell Ontology label or synonym.
4. Highest-cosine Cell Ontology result, accepted only when its raw cosine is at least `CELL_NORMALIZATION_MIN_COSINE` (default `0.95`).

There is no top-1/top-2 margin requirement and no fuzzy abbreviation-key fallback in this route. Unaccepted vector results are not published as normalized cell annotations. Raw cosine is a similarity score, not a probability.

Complete cell expressions are matched and normalized as a whole. Adjacent modifiers may expand a recognized cell span only when the expanded phrase resolves in the cell reference. For example, the bundled ontology includes `mature NK cell` as an alias of `CL:0000824`; a longer `mature NK cells` annotation therefore replaces a nested generic `NK cells` annotation (`CL:0000623`). Identifiers are read from the ontology, not encoded in a new exception table.

## HGNC-only gene/protein normalization

PubTator Gene IDs, including taxon-prefixed numeric forms, are parsed without accepting or rejecting them by species. They are looked up using HGNC's NCBI Gene cross-references. An unmatched PubTator gene is discarded; there is no canonical NCBIGene-only fallback.

General exact recovery continues to use approved symbols, approved names, alias symbols/names, and previous symbols/names. Approved-symbol conflict correction is general, not specific to named genes. Unambiguous HGNC normalization is required. The HGNC reference is necessary for this processing; an unavailable reference raises an explicit error instead of publishing unnormalized genes.

HGNC is canonical. NCBI Gene and UniProt values, when available, are cross-reference metadata. `reference_tax_id=9606` describes the HGNC reference and does not assert that the source mention's species was verified. No taxonomy exclusion masks are applied during merging or repeat recovery.

There are no special overrides for IL-21, IL-15, IL-2, or DAP12. These names use the same reference-based rules as other genes/proteins.

## Receptor expressions

`backend/pipeline/receptor_annotations.py` implements a shared source-preserving rule for cell, gene/protein, and hormone mentions followed by `receptor` or `receptors`, including supported hyphenated forms and qualifiers.

The full expression is looked up in HGNC. Approved names are included in the exact index. For an abbreviated base, document expansions and reference-backed base names may supply lookup alternatives; an ambiguous or unmatched expression is discarded. The original source text and offsets are never rewritten.

For example, `progesterone receptor` resolves through the HGNC approved name to PGR (`HGNC:8910`). The output is one gene/protein annotation covering the whole phrase, not a hormone annotation on `progesterone`. When the complete expression does not resolve, shorter contained ligand, hormone, or cell annotations are removed too.

## Overlap policy

Within each chunk:

- Rank by span length first, regardless of entity type, recognition source, confidence, or lock flag.
- Keep the longest annotation and remove shorter overlapping annotations, including contained and partially crossing spans.
- For identical start/end offsets, prefer cell over hormone over gene/protein, before evidence or confidence. A longer overlapping span always remains first priority.
- Cell matching preserves complete lexical tokens; compact scan keys cannot turn separate words into one synonym.
- Word-like short static aliases are not automatic free-text cell annotations. Explicit document definitions remain a separate source of evidence.
- The resource-backed shared-head resolver can retain a coordinated modifier phrase, recording which expanded arms resolve and which do not. Incompatible subtype IDs are not collapsed.
- Figure/table panel lists (including letter-only panels) are excluded before annotation selection.

See [Cell context matching](CELL_CONTEXT_MATCHING.md) for the rules and examples.
- Merge provenance only for duplicates with the same span, type, and identifier. Never transfer a shorter cell's or ligand's identifier into a longer, differently normalized entity.
- Touching half-open spans do not overlap.

The receptor rule precedes overlap selection. Thus a complete receptor is not overridden by its shorter hormone prefix. Relation tagging does not emit nested entity annotations.

## Definition selection and text identity

Document definitions retain the existing contextual order: same chunk, nearest preceding definition, same section, nearest other located definition, then unlocated definition. Equally ranked conflicting meanings do not produce an assignment.

Lookup keys may normalize Unicode, dashes, case, and supported singular/plural variants. Published `start` and `end` always index the original chunk, using zero-based, end-exclusive offsets. A recovered longer mention is normalized for its complete text rather than inheriting the shorter annotation's ID.

## Applying this revision

Annotation, branch, prepass, relation, and export version markers have been updated so new processing does not reuse earlier policy signatures. Existing saved corpora and completed output files are not rewritten automatically. Restart the application and run entity extraction and relation extraction again for outputs that need these rules.

No new tests or validation scripts were added. The pre-existing tests were left unchanged, and the full model/PubTator pipeline was not executed for this revision.
