# Ovarian Network: public deployment and local analysis

This FastAPI application builds evidence-linked biological networks from two saved ovarian literature collections. Railway hosts the website and controller; the existing Modal T4 worker runs CellExLink inference when new papers are processed.

The package includes the full application source, both precomputed JSONL collections, reference resources, deployment files, and offline tests. Model weights remain in the existing Hugging Face/Modal model cache.

## Inputs

| Mode | Dropdown selection | Entered PMIDs |
| --- | --- | --- |
| Railway / production | Load the selected saved collection | Load matching saved papers from either collection |
| Local development | Load the selected saved collection | Retrieve and process any valid PubMed IDs |

Public PMID requests never fall back to fresh retrieval, GPU inference, or OpenAI calls. A mixed request uses available papers and reports missing IDs. A request with no saved papers returns a clear error. The local text-source selector supports abstracts or full text when available.

## Saved collections and PMID list

| Topic | File |
| --- | --- |
| Non-neoplastic ovarian inflammation | `neoplastic_ovarian_prediction.jsonl` |
| Ovarian cancer-associated inflammation | `cancer_associated_ovarian_prediction.jsonl` |

The existing non-neoplastic filename is retained for compatibility.

The supplied collections contain **2,697 distinct PMIDs**: 1,349 non-neoplastic and 1,348 cancer-associated papers. Their union is stored in:

```text
data/precomputed_corpora/known_pmids.json
```

The index is derived from actual paper IDs in both JSONL files and refreshes when either file changes. Rebuild it manually with:

```bash
python scripts/build_pmid_index.py
```

On Railway, the bundled files seed `/data/precomputed_corpora` once. Existing volume files are preserved. Monthly predictions append atomically, deduplicate PMIDs, and refresh the index and summaries. Temporary visitor runs never change these collections.

## Monthly update

Updates and the Modal schedule are **off by default**.

Each authorized update:

1. Searches both topic queries from the beginning and retrieves PubMed IDs.
2. Compares those IDs against the union of saved papers before retrieving metadata/full text or calling models.
3. Processes only absent papers through the existing retrieval, entity normalization, and relation pipeline.
4. Adds completed predictions to the applicable saved collection, including completed papers with no extracted relations.
5. Keeps deferred or failed papers pending and stores completed prediction checkpoints on the persistent volume.

Discovery handles PubMed's 10,000-result limit by splitting large searches into PMID ranges and checking completeness. It does not depend on a publication-date watermark. Here, a **new paper means a PMID absent from the saved resources**, so an initial update can include a historical backlog.

The default cap is **300 processing attempts per UTC calendar month across both topics**. A failed or interrupted attempt consumes a slot and is not submitted again that month. Repeated triggers share the same persistent budget. Remaining papers wait for a later month. An update with no absent IDs performs no GPU inference or OpenAI extraction.

Topic queries are defined in `backend/services/corpus_updates.py`; they retain the existing human ovarian/immune selection blocks and distinguish cancer-associated from non-neoplastic content.

## Switches

Edit `backend/deployment_flags.py` before deployment:

```python
PUBLIC_PRECOMPUTED_ONLY = None  # Automatic: production True; local False.
MONTHLY_UPDATES_ENABLED = False
MODAL_MONTHLY_SCHEDULE_ENABLED = False
MONTHLY_UPDATE_CRON = "0 3 1 * *"  # First of each month, 03:00 UTC.
MONTHLY_UPDATE_MAX_NEW_PAPERS = 300
```

Environment variables override these source-code defaults. Use `PUBLIC_PRECOMPUTED_ONLY=true` on Railway and `false` for unrestricted local analysis. Enabling the monthly workflow requires both the Railway update switch and the deployed Modal schedule switch.

Read [DEPLOYMENT_NEW_APP.md](DEPLOYMENT_NEW_APP.md) for complete deployment, preview, and monitoring commands.

## Ontology

Normalization uses the approved bundled **`cell_ontology_v2026-06-08.jsonl`**. Resource versions derive from a central constant, and embedding/artifact caches include the actual ontology file hash.

The separate hierarchy explorer still uses the accurately labeled **2025-12-17 hierarchy**. The June lexicon has identifiers, labels, and synonyms but no parent relationships, so it cannot supply a replacement hierarchy. Hierarchy responses report both dates.

## Local setup

Use **Python 3.11**. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements-local.txt
cp .env.example .env
```

Set your own `OPENAI_API_KEY` and `NCBI_EMAIL` in `.env`. Choose the relation model and reasoning effort there. The local example uses `CELL_ANNOTATION_BACKEND=local`, unrestricted PMID input, and disabled updates.

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. Local fresh inference and API use run with the local user's resources and credentials. Saved collection exploration does not require a GPU or an OpenAI key.

## Persistent and temporary data

```text
backend/                              Application, interface, and pipelines
modal_app.py                          GPU worker and optional monthly CPU trigger
data/precomputed_corpora/             Saved predictions and union PMID index
data/reference_data/                 HGNC and other reference resources
/data/corpus_updates/                Railway update state and checkpoints
/data/runs/                          Temporary visitor/update run files
```

Railway requires a volume mounted at `/data` and one web replica/worker. This preserves new predictions, pending IDs, and budget accounting across redeployments. Browser networks remain temporary SQLite files; starting a new browser run reads the current saved resources.

The update pipeline has a separate queue so waiting for monthly inference does not occupy the visitor stage queues. Trusted maintenance runs are hidden from visitor run/network endpoints.

Disabling the update switch prevents new maintenance work. Shutdown/cancellation requests stop subsequent stages, request cancellation of an active Modal input, and cancel local pending relation tasks. Work already accepted by a provider may still incur usage. Changing deployment settings takes effect after restart/redeploy.

## Network behavior

The graph preserves the original directed relation rules, normalized CL/HGNC/MeSH identifiers, evidence passages, support counts, relation filters, node/edge editing, neighbor expansion, and displayed-network export. A relation is counted once per supporting paper even when repeated across chunks. Existing predictions are preserved; changing the ontology does not automatically recompute historical papers.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use fixtures and mocked external computation. They verify public/local routing, saved-network construction and downloads, index recovery, atomic/deduplicated publication, monthly budgeting and checkpoints, authentication, disabled switches, cancellation, and ontology provenance. No real GPU or OpenAI requests are made by the tests.
