# Same-paper cell mention recovery fix

This is a focused update to `newmethod-ovarian-network-web-cell-context-updated.zip`.
The gene/HGNC, hormone, receptor, coordination, figure/table exclusion, static
abbreviation, and final overlap policies are unchanged.

## The conflicting recovery rule

`CellOntologyNormalizer.rescue_document_cell_mentions()` previously skipped a
candidate whenever it overlapped any existing cell annotation. That included
an existing shorter cell head, so a recognized `NK cells` occurrence could
prevent recovery of the full `mature NK cells` phrase at the local rescue step.
The final exact-resource recovery step already accepts the supplied sentence
in isolation; this fix removes the earlier inconsistent overlap veto and
strengthens retention of accepted same-paper evidence at the final merge.

## Updated behavior

The local pass snapshots accepted cell annotations from the selected paper and
uses their original source spans to obtain the spelling to search. It searches
all selected chunks, including earlier chunks and the seed's own chunk. It does
not require the source occurrence to precede the missed occurrence.

Only an identical occurrence (same chunk, span, and cell identity) is skipped as
a duplicate. Other overlapping candidates are allowed through to the existing
source/context checks and the shared longest-span resolver. This lets a longer
recovered cell phrase replace a shorter annotation, without inventing an
exception for any particular cell name or identifier.

The final per-paper pass retains snapshots of independently accepted cell seeds
while subsequent exact additions and overlap cleanup run. It builds a fixed,
unambiguous surface registry before searching the paper. Recovered occurrences
do not recursively create new seeds. Recovery retains original character
positions and records the source occurrence in `seed_evidence`.

For the reported sentence, the intended final cell annotation includes:

```json
{
  "mention": "mature NK cells",
  "concept_id": "CL:0000824",
  "preferred_label": "mature natural killer cell"
}
```

The nested `NK cells` annotation is removed by longest-span selection. The
identifier is taken from accepted annotation/reference data, not a hard-coded
mapping added by this update. A genuinely longer valid expression can still
win. Document abbreviation conflicts, reference-context exclusions, receptor
reclassification, lexical boundaries, and ambiguous identities remain subject
to the existing rules. Recovery is confined to the available selected text of
the same paper, not text that was never retrieved.

## Changed code

- `backend/cellexlink_lite/normalization.py`: remove the blanket overlap veto,
  search source-derived surfaces, deduplicate exact occurrences, retain seed
  evidence, and process longer surface candidates first.
- `backend/pipeline/document_repeat_recovery.py`: retain accepted cell seed
  snapshots through later additions, freeze the final registry, and search
  longer surfaces first.
- `backend/pipeline/annotation_contract.py`: update the pipeline/cache signature
  and use the recovery module's version constant directly.
- `backend/pipeline/entity_artifacts.py`: update the final output version.

## Applying this archive

Replace the prior project, restart the app, then rerun entity extraction and
relation extraction for the affected paper. Existing completed outputs and
precomputed corpora are not rewritten automatically.

No test or validation files were added; the existing tests are unchanged. The
lightweight production recovery functions were exercised on the supplied
sentence with a shorter existing annotation and a seed in a later chunk. The
longer phrase was recovered and retained after overlap selection and final
reconciliation. No neural models, live PubTator requests, or end-to-end web-app
run were executed.
