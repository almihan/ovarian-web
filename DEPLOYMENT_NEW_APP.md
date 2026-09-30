# Deploy the updated application to Railway and Modal

Railway runs the FastAPI website, PMID discovery, reference handling, OpenAI relation requests, and CPU network construction. The existing Modal T4 function runs the trained CellExLink models. A separate optional Modal CPU function triggers the monthly updater; it does not allocate a GPU.

## 1. Deploy the existing GPU worker

Use Python 3.11, install the web requirements, and authenticate Modal with your own account:

```bash
python -m pip install -r requirements.txt
modal setup
modal deploy modal_app.py
```

Keep `MODAL_MONTHLY_SCHEDULE_ENABLED=false` for this first deployment. The app/function defaults remain:

```text
MODAL_APP_NAME=ovarian-cellexlink-hgnc-chebi
MODAL_FUNCTION_NAME=annotate_bundle
```

If your existing deployment uses different names, keep those names in both the deployment environment and Railway variables. The existing named model-cache volume is reused. Model warm-up is optional if already populated:

```bash
modal run modal_app.py::warm_model_cache
```

## 2. Configure Railway

Deploy this project with its Dockerfile and `railway.toml`. Attach a **persistent volume at /data** to the web service. Keep **one replica and one Uvicorn worker** because browser run state is process-local.

Use `.env.railway.example` as the variable checklist. The essential public-mode settings are:

```text
APP_ENV=production
PUBLIC_PRECOMPUTED_ONLY=true
APP_DATA_DIR=/data
PRECOMPUTED_CORPORA_DIR=/data/precomputed_corpora
CELL_ANNOTATION_BACKEND=modal
MONTHLY_UPDATES_ENABLED=false
MONTHLY_UPDATE_MAX_NEW_PAPERS=300
```

Supply your existing Modal credentials and S3-compatible artifact bucket credentials, plus `OPENAI_API_KEY`, `NCBI_EMAIL`, and optionally `NCBI_API_KEY`. The S3 bucket carries signed input/output bundles between Railway and Modal; it is separate from the Railway persistent corpus files.

Set `PUBLIC_BASE_URL` to the Railway HTTPS URL. The controller also detects `RAILWAY_PUBLIC_DOMAIN` for its URL when available. No secret values are included in this package.

The Docker image includes both saved collections and the reference files. Startup copies missing collections/reference files into their active persistent directories and preserves existing files. The union index rebuilds automatically.

Check:

```text
GET /health
GET /api/system/status
GET /api/corpora
```

Test one saved PMID from each collection and an unavailable PMID. Available saved input should build a network without submitting a GPU job or a new relation request. A blank PMID field should still load the selected dropdown collection.

## 3. Configure the private monthly update

Generate a private token:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Put the value into Railway's `MONTHLY_UPDATE_TOKEN`. Set `MONTHLY_UPDATES_ENABLED=true` only when the volume and inference/API credentials are ready.

Create a Modal secret named **ovarian-monthly-update** with these two values:

```text
CORPUS_UPDATE_BASE_URL=https://your-service.up.railway.app
MONTHLY_UPDATE_TOKEN=<the same private token>
```

The GPU worker does not need this update secret. The CPU scheduler uses it only to authenticate its Railway trigger.

Set these values in the shell/environment used to deploy Modal, or edit the corresponding source defaults in `backend/deployment_flags.py`:

```text
MODAL_MONTHLY_SCHEDULE_ENABLED=true
MODAL_MONTHLY_SECRET_NAME=ovarian-monthly-update
MODAL_MONTHLY_CRON=0 3 1 * *
```

Redeploy:

```bash
modal deploy modal_app.py
```

The default schedule triggers on the first day of every month at 03:00 UTC. Use Modal's scheduler; do not add a second Railway cron schedule for the same updater.

## 4. Preview before processing

Set `PUBLIC_BASE_URL` and `MONTHLY_UPDATE_TOKEN` in your local shell or private `.env`. The Railway switch must be enabled even for an authenticated preview.

```bash
python scripts/trigger_monthly_update.py --url https://your-service.up.railway.app
python scripts/trigger_monthly_update.py --url https://your-service.up.railway.app --status
```

The default request is a **dry run**: it searches IDs, reports known/new/deferred papers, and selects a preview queue without GPU inference, OpenAI extraction, corpus publication, or budget consumption.

To process the selected absent papers manually:

```bash
python scripts/trigger_monthly_update.py --url https://your-service.up.railway.app --commit
```

Triggers return promptly while Railway continues the update. Use `--status` and Railway/Modal logs to monitor it. The scheduled Modal function sends the same authenticated commit request.

## 5. Update behavior and recovery

Discovery searches both topic queries from the beginning. It checks the saved PMID union before full-text/entity/relation computation. A paper already saved in either topic is not inferred again; if necessary its saved prediction is copied into another matching topic.

The default 300-paper cap applies across both topics and persists per UTC calendar month. Failed/interrupted attempts consume a slot and wait until a later month. Duplicate triggers cannot reset the budget or repeat a completed month's processing. Deferred papers remain pending.

Completed predictions are checkpointed before publication. Atomic corpus writes and a derived index make duplicate publication safe after interruption. Completed annotation/relation artifacts use persistent content/model signatures and survive temporary visitor-run cleanup.

`MONTHLY_UPDATE_TIMEOUT_SECONDS` defaults to 86,400 seconds. It bounds the update batch and individual-paper polling. A timed-out/disabled run stops further submissions and requests cancellation of active work. Charges already incurred by a provider cannot be undone.

Persistent resources:

```text
/data/precomputed_corpora/*.jsonl
/data/precomputed_corpora/known_pmids.json
/data/corpus_updates/state.json
/data/corpus_updates/completed/
```

Do not mount these on ephemeral storage. Keep them when replacing application code.

## 6. Turn updates off

On Railway set:

```text
MONTHLY_UPDATES_ENABLED=false
```

Restart/redeploy the service. In the Modal deployment environment set:

```text
MODAL_MONTHLY_SCHEDULE_ENABLED=false
```

Then run `modal deploy modal_app.py` again to remove the schedule. The public website continues serving saved resources. Both source-code defaults are already False.

## 7. Local unrestricted use

Use `.env.example`, `APP_ENV=development`, `PUBLIC_PRECOMPUTED_ONLY=false`, and `CELL_ANNOTATION_BACKEND=local`. Install `requirements-local.txt` after PyTorch, supply the local user's API key, and run Uvicorn. Local PMID input does not require membership in the saved list. Details are in [README.md](README.md).

## Ontology versions

As confirmed, normalization uses the supplied `cell_ontology_v2026-06-08.jsonl`. The hierarchy explorer remains an explicitly separate `2025-12-17` hierarchy because the June lexicon has no parent links. Historical saved predictions are preserved.

## Validation boundary

Offline tests exercise the control flow, storage, auth, scheduling request, and network output. Deployment commands are provided for your accounts; no live Railway/Modal deployment, GPU inference, PubMed update, or OpenAI billing has been performed by this code-editing task.
