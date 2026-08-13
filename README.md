# Ovarian Network

A four-stage FastAPI application for ovarian-literature retrieval, entity extraction, relation extraction, and interaction-network exploration.

## State and cache design

The public application intentionally has **no persistent user job history**.

- A browser run receives a random 32-character run ID.
- The run ID exists only in the current page's JavaScript memory.
- The browser does not use cookies, `localStorage`, `sessionStorage`, or a recent-jobs endpoint.
- Reloading or reopening `/` shows a clean Stage 1 screen.
- Worker progress records are process-local and disappear when the web process restarts.

Only the fixed default corpus is reusable:

| Output | Storage policy |
|---|---|
| Default Stage 1 chunks | Persist and reuse across users |
| Default Stage 2 entity annotations | Persist and reuse across users |
| Default Stage 3 relations | Persist and reuse across users |
| Custom Stage 1–3 additions | Unique run namespace; never reused by another run |
| Stage 4 graph and entity index | Temporary local files only; never uploaded |

A custom search always starts from the shared default result. Stage 1 searches the augmented query, removes every paper already present in the default corpus, and processes only the added papers. Stages 2 and 3 then process only those added-paper artifacts and merge them logically with the shared default artifacts for the current run. Even two users entering the same custom terms receive different run namespaces, so the custom portion is recomputed.

The reusable default artifacts are invalidated automatically when the retrieval contract, model configuration, prompt/policy signature, or relevant bundled resources change.

## Pipeline

1. **Retrieve papers**
   - Reuses or builds the shared default `chunks.jsonl.gz`.
   - Retrieves and chunks only papers added by the current custom input.
   - Keeps paper text out of later-stage artifacts.

2. **Extract and normalize entities**
   - CellExLink recognition and normalization run sequentially on Modal T4 or in a short-lived local worker.
   - PubTator3 gene/protein and hormone processing runs on Railway/local CPU in parallel.
   - Branch artifacts are deleted after the final aligned annotation artifact is created.
   - Model progress bars, model predictions, and model-library informational output are suppressed from application logs.

3. **Extract relations**
   - Uses the existing OpenAI relation-extraction implementation in `backend/services/relation_executor.py` and `backend/pipeline/relation_extraction.py`.
   - Reuses only the shared default relation artifact.
   - Recomputes relations for every run-specific added-paper artifact.

4. **Build the network**
   - Combines the aligned default and custom Stage 1–3 inputs.
   - Builds a temporary SQLite graph and compressed entity index under `data/runs/<run-id>/stage4/`.
   - Does not publish or cache Stage 4 artifacts.

## Local setup

Create the environment file:

```bash
cp .env.example .env
```

Create and activate a virtual environment, then install the web/controller dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For local CellExLink execution, install the appropriate PyTorch build for the computer first, then install:

```bash
pip install -r requirements-local.txt
```

Set at least `NCBI_EMAIL`, `OPENAI_API_KEY`, `ARTIFACT_BACKEND=local`, and `CELL_ANNOTATION_BACKEND=local` in `.env`, then run:

```bash
uvicorn backend.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Local reusable default artifacts are stored under `data/artifacts/`. Temporary custom and Stage 4 files are stored under `data/runs/`. They are removed on application start/stop and after an inactive run exceeds `RUN_RETENTION_SECONDS` when the next run begins.

## Railway and Modal deployment

Use **one Railway replica**. Per-page run state is intentionally process-local, and one worker per stage also prevents accidental parallel GPU/OpenAI spending.

Railway should run the included Dockerfile. Configure:

- `APP_ENV=production`;
- a Railway Bucket or another S3-compatible bucket for `ARTIFACT_BACKEND=s3`;
- `PUBLIC_BASE_URL` with the public Railway URL;
- Modal workspace credentials in `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`;
- `CELL_ANNOTATION_BACKEND=modal`;
- `OPENAI_API_KEY` and the existing Stage 3 settings;
- `NCBI_EMAIL` and optionally `NCBI_API_KEY`.

Deploy the Modal worker from the same repository:

```bash
modal deploy modal_app.py
```

The Modal image uses one T4, `max_containers=1`, sequentially loads the recognition and normalization checkpoints, and uses a persistent model-cache volume. Railway keeps only the small FastAPI/controller dependencies; PyTorch and the CellExLink checkpoints are not installed in the Railway image.

## Main code layout

```text
backend/main.py                         FastAPI app and lifecycle
backend/api/runs.py                     Per-page pipeline API
backend/api/networks.py                 Temporary network explorer API
backend/runtime.py                      In-memory browser-run registry
backend/worker_state.py                 In-memory Stage 2/3 worker coordination
backend/orchestrator.py                 Four bounded stage queues
backend/default_cache.py                Shared default cache + custom delta build
backend/pipeline/retrieval.py            Retrieval and chunking
backend/services/railway_annotation_executor.py
backend/services/relation_executor.py    Existing Stage 3 OpenAI implementation
backend/pipeline/network_builder.py
backend/services/network_repository.py
modal_app.py                            Modal T4 deployment
```

## Public API flow

```text
POST /api/runs
GET  /api/runs/{run_id}
POST /api/runs/{run_id}/stages/2
POST /api/runs/{run_id}/stages/3
POST /api/runs/{run_id}/stages/4
```

There are no endpoints that list previous user jobs or select a “latest” job.
