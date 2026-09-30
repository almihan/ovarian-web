# Ovarian Network: public deployment and local analysis

This FastAPI application builds evidence-linked biological networks from two saved ovarian literature collections. Railway hosts the website and controller; the existing Modal T4 worker runs CellExLink inference when new papers are processed.

The package includes the full application source, both precomputed JSONL collections, reference resources, deployment files, and offline tests. Model weights remain in the existing Hugging Face/Modal model cache.

## Inputs

| Mode | Dropdown selection | Entered PMIDs |
| --- | --- | --- |
| Railway / production | Load the selected saved collection | Load matching saved papers from saved collection |
| Local development | Load the selected saved collection | Retrieve and process any valid PubMed IDs |

Public PMID requests don't fall back to fresh retrieval, GPU inference, or OpenAI calls. A mixed request uses available papers and reports missing IDs. A request with no saved papers returns a clear error. The local text-source selector supports abstracts or full text when available.

## Saved collections and PMID list

| Topic | File |
| --- | --- |
| Non-neoplastic ovarian inflammation | `neoplastic_ovarian_prediction.jsonl` |
| Ovarian cancer-associated inflammation | `cancer_associated_ovarian_prediction.jsonl` |

The existing papers list:
```text
data/precomputed_corpora/known_pmids.json
```

The index is derived from actual paper IDs in both JSONL files and refreshes when either file changes. Rebuild it manually with:

```bash
python scripts/build_pmid_index.py
```

On Railway, the bundled files seed `/data/precomputed_corpora` once. Existing volume files are preserved. Monthly predictions append atomically, deduplicate PMIDs, and refresh the index and summaries. Temporary visitor runs never change these collections.

## Monthly update

Each authorized update:

1. Searches both topic queries from the beginning and retrieves PubMed IDs.
2. Compares those IDs against the union of saved papers before retrieving metadata/full text or calling models.
3. Processes only absent papers through the existing retrieval, entity normalization, and relation pipeline.
4. Adds completed predictions to the applicable saved collection.
5. Keeps deferred or failed papers pending and stores completed prediction checkpoints on the persistent volume.

The monthly update processing attempts per UTC calendar month across both topics**. An update with no absent IDs performs no GPU inference or OpenAI extraction.

Topic queries are defined in `backend/services/corpus_updates.py`; they retain the existing human ovarian/immune selection blocks and distinguish cancer-associated from non-neoplastic content.



## Ontology

Normalization uses the approved bundled **`cell_ontology_v2026-06-08.jsonl`**. Resource versions derive from a central constant, and embedding/artifact caches include the actual ontology file hash.


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

## Network behavior

The graph preserves the original directed relation rules, normalized CL/HGNC/MeSH identifiers, evidence passages, support counts, relation filters, node/edge editing, neighbor expansion, and displayed-network export. A relation is counted once per supporting paper even when repeated across chunks. Existing predictions are preserved.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use fixtures and mocked external computation. They verify public/local routing, saved-network construction and downloads, index recovery, atomic/deduplicated publication, monthly budgeting and checkpoints, authentication, cancellation, and ontology provenance. No real GPU or OpenAI requests are made by the tests.
