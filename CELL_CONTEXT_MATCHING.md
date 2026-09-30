# Cell-context matching

This revision extends the previous HGNC-only / longest-span / receptor update.
It does not add a new model, a new dependency, a cell-name exception list, or
hard-coded Cell Ontology identifiers.

## 1. Coordinated modifiers and a shared cell head

For a phrase of the form `modifier A or modifier B <cell head>`, the pipeline
reconstructs `modifier A <cell head>` and `modifier B <cell head>` **for resource
lookup only**. The annotation's mention and character offsets still slice the
original text, including the conjunction.

The cell head must itself resolve in the cell resources. Modifier phrases are
learned from ontology labels/synonyms: removing such a modifier must leave a
complete, independently resolvable cell name. There is no manually maintained
list of modifiers or modifier-to-ID mappings.

A full coordinated annotation requires at least one exact supported subtype
and one unique subtype ID across the resolved arms. Ambiguous or conflicting
IDs are not collapsed into one cell. For an `and` coordination, every arm must
resolve to that same ID. For an `or` coordination, an unlisted alternative
modifier is permitted only when it is an ontology-derived modifier and the
resolved arms provide a single non-generic subtype. This latter result is
explicitly recorded as an inference based on the supported alternative arm.

For the supplied example, the intended annotation is:

```json
{
  "mention": "immature or naïve NK cells",
  "entity_type": "cell",
  "concept_id": "CL:0000823",
  "preferred_label": "immature natural killer cell",
  "matched_term": "immature NK cell",
  "term_kind": "coordinated_shared_head",
  "normalization_source": "cell_ontology_coordinated_shared_head",
  "coordination_shared_head": "NK cells",
  "normalization_scope": "supported_alternative_arm"
}
```

The complete output additionally stores `coordination_arms`, with the lookup,
status, matched term and identifier of every resolved arm, and
`coordination_unresolved_arms` where applicable. An unlisted `naïve NK cells`
arm is **not** silently added to the ontology as a synonym. The longer
annotation replaces its contained `NK cells` annotation through the common
overlap policy.

Conflicting expansions, such as two modifiers that resolve to different
subtypes, do not acquire one combined identity through vector top-1. Independently
recognizable component mentions can remain. This is bounded rule-based recovery,
not a general syntactic parser: arbitrarily complex coordination is not covered.
Coordinated inferences are not propagated to unrelated occurrences or papers.

Implementation: `backend/pipeline/cell_coordination.py`, used by
`document_entity_recovery.py`, NER span repair, cell normalization, and the
post-merge exact cell recovery pass.

## 2. Identical-span conflicts

Longer spans continue to win over shorter overlapping spans regardless of
entity type, confidence or lock state. When **both start and end are identical**,
priority is:

```text
cell > hormone > gene/protein
```

Consequently, a valid cell annotation of `TH1` wins over a gene annotation of
that same occurrence. The hormone-over-gene rule is preserved where no cell
annotation has that exact span. This precedence does not override the prior
receptor-expression reclassification rule.

Implementation: `backend/pipeline/entity_overlap.py`, reused by local selection,
merging, recovery, export and relation preparation. Final tag selection uses the
same exact-span priority.

## 3. Word boundaries and figure/table references

Compact keys are used to *retrieve* resource candidates, but a cell candidate
must pass a comparison of complete lexical tokens before being accepted.
Controlled final plural forms and ordinary word-separator variants remain
supported. Significant biological punctuation is preserved.

`B and` has two lexical tokens; `band` has one. They cannot be accepted as the
same cell name, even though removing spaces would produce the same string.
The rule also applies before vector fallback and to persisted annotation rows,
so a later stage cannot recreate this compact-key collision.

Figure/table exclusion masks now include letter-only panels in a reference
list. In `Figures 2A, B and Table S1`, the figure panels and the table label are
excluded as reference labels. The exclusion is contextual; it does not globally
ban the letters B, S, or other valid biological symbols.

Implementation: `cell_surface_matching.py`, `entity_lexicons.py`, and
`entity_span_rules.py`.

## 4. Static abbreviation casing and provenance

The input project archive contains this row at line 1708 of
`backend/cellexlink_lite/resources/abbreviations.tsv`:

```text
End    CL:0000115
```

The Cell Ontology entry itself has the preferred label `endothelial cell` and
synonym `endotheliocyte`; `End` comes from the bundled static abbreviation
resource. A local dictionary outside this archive may differ.

The old case-insensitive static lookup could therefore turn ordinary `end`
into a locked endothelial-cell annotation. The resource row is not deleted or
handled with an exception. Instead, a general source-spelling rule applies:

- Single-token alphabetic acronyms require visible acronym capitalization in
  the mention. Numeric codes such as Th1 retain case-insensitive matching.
- Short word-like aliases of up to five letters, without acronym casing or
  digits, are not automatically accepted as free-text static cell mentions.
- Multiword names and ordinary longer names retain boundary-preserving
  case-insensitive matching. Ontology words such as `band` remain available as
  actual whole-word matches outside excluded reference spans.
- Explicit document definitions are a separate evidence route; their resolved
  long forms can establish a local abbreviation identity.

Unsafe static collisions are also blocked before vector fallback and repeated
mention recovery. Thus rejecting a static match cannot merely send the same
ordinary word through another route to the same unsupported annotation.

Static annotations now record the actual `resource_file`, `resource_line`,
`resource_version` (dictionary filename plus content hash), and a separate
`ontology_resource_version`. The `matched_term` is the original dictionary
spelling, not a substituted ontology label.

## Applying the update

Replace the previous project with this complete archive and restart the app.
Run entity extraction and then relation extraction again for affected papers.
Pipeline/cache version markers have been updated, but already completed saved
results and precomputed corpora are not rewritten automatically.

The previous HGNC-only mapping, no-species-validation policy, removal of the
four gene-name overrides, complete-receptor handling, and configurable cell
cosine threshold (default 0.95) are retained.

No tests or validation scripts were added. No tests, model inference, live
PubTator requests, or end-to-end pipeline execution were run for this revision.
Existing tests are retained unchanged.
