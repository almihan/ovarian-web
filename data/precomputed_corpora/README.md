These two JSONL files seed the saved collections and the union PMID index.

- `neoplastic_ovarian_prediction.jsonl`: non-neoplastic ovarian inflammation (legacy filename retained).
- `cancer_associated_ovarian_prediction.jsonl`: ovarian cancer-associated inflammation.
- `known_pmids.json`: derived union of valid PMIDs from both collections.

Visitors read these results; their temporary networks do not modify the collection files.
The private, explicitly enabled monthly updater appends completed new predictions to
persistent copies on Railway. Startup seeds only missing files and never replaces
existing volume copies with bundled data. See the project README and deployment guide.
