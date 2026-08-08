# Ovarian Network web starter

A runnable first-stage web application for literature mining and interactive
cell–gene–chemical network exploration.

## Included in this starter

- FastAPI service
- Jinja2 landing-page template
- ordinary JavaScript
- vis-network visualization
- SQLite job persistence
- progress polling and recent-job history
- Railway-ready Dockerfile and health check
- persistent `/data` layout for papers, results, model cache, and SQLite
- one-time FastAPI startup loading for `almire/CellExLink-bioformer16L`
- session-only bring-your-own OpenAI API-key interface

The processing workflow currently uses timed demonstration stages and a sample
network. Replace those stages with your existing retrieval, entity-extraction,
relation-extraction, and graph-building functions under `backend/pipeline/`.

## Project structure

```text
ovarian-network-starter/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   ├── jobs.py
│   │   ├── networks.py
│   │   └── papers.py
│   ├── database/
│   │   └── database.py
│   ├── models/
│   │   └── loaders.py
│   ├── pipeline/
│   │   ├── retrieval.py
│   │   ├── entity_extraction.py
│   │   ├── relation_extraction.py
│   │   └── graph_builder.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/styles.css
│       └── js/app.js
├── data/
│   ├── papers/
│   ├── results/
│   └── model_cache/
├── Dockerfile
├── railway.toml
├── requirements.txt
└── .env.example
```

## Run locally

Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`.

The example `.env` keeps model loading disabled so you can inspect the interface
immediately. To test the real startup loader, set:

```env
LOAD_CELL_MODEL=true
```

The model is loaded once and reused through `app.state.cell_model`.

## Persistent storage layout

Locally, data is written under `./data`. On Railway, attach one persistent
volume at `/data`. Railway exposes its mount path as
`RAILWAY_VOLUME_MOUNT_PATH`, which the application detects automatically.

```text
/data/database.sqlite
/data/papers/
/data/results/
/data/model_cache/
```

Do not run more than one replica while using SQLite and one attached volume.
The supplied Docker command also keeps Uvicorn at one worker so only one model
copy is held in memory.

## OpenAI API-key modes

### Recommended for a private lab application

Set `OPENAI_API_KEY` in Railway's **Variables** tab. The browser never receives
that key.

### Bring your own key — prototype mode

A user may enter a key in the **API Settings** modal. The starter:

- keeps it only in a JavaScript variable for the current tab;
- does not use local storage or session storage;
- sends it to FastAPI only with the submitted job;
- uses Pydantic `SecretStr` so representations are redacted;
- does not write it to SQLite or result files;
- discards the job-local reference after processing.

The current demonstration does not call OpenAI. Connect the actual API request
inside `backend/pipeline/relation_extraction.py` and pass the job-local key to
that function. Before enabling bring-your-own-key mode for public users, vendor
the vis-network JavaScript inside the application, add authentication and rate
limits, and complete a security review. For the first lab deployment, a restricted
project key in Railway is simpler and safer.

## Before deploying

Create these accounts:

1. a GitHub account and a new repository, such as `ovarian-network-web`;
2. a Railway account connected to that GitHub account;
3. an OpenAI API Platform project only when you are ready to connect real relation extraction. A ChatGPT subscription is not an API credential.

You can inspect the interface on Railway's trial with `LOAD_CELL_MODEL=false`. For the real model deployment, the Hobby plan is the practical starting point because the Free plan is limited to 0.5 GB RAM and 0.5 GB volume storage.

Push the project to a new GitHub repository:

```bash
git init
git add .
git commit -m "Initial FastAPI web interface"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ovarian-network-web.git
git push -u origin main
```

## Deploy on Railway

1. Create a GitHub repository and push this project to its root.
2. Create a Railway account and connect it to GitHub.
3. Create a new Railway project and select **Deploy from GitHub repo**.
4. Select your repository. Railway detects the root `Dockerfile`. Keep `LOAD_CELL_MODEL=false` for this first deployment.
5. After the first interface deployment, attach a volume to the FastAPI service
   and set its mount path to `/data`.
6. In the service **Variables** tab, add:

   ```env
   APP_ENV=production
   LOAD_CELL_MODEL=true
   CELL_MODEL_ID=almire/CellExLink-bioformer16L
   ```

   Optionally add a lab-owned `OPENAI_API_KEY` there. Do not put it in GitHub.
7. Redeploy. The first startup downloads the Hugging Face model into the
   persistent model cache. Later restarts reuse the cached files.
8. Open the service **Settings** tab, find **Networking**, and select
   **Generate Domain**.
9. Verify `/health`, the landing page, model status, job progress, recent jobs,
   and the interactive sample network.

## Next coding stage

Replace `process_demo_job()` in `backend/api/jobs.py` progressively:

1. call the real PMID/PMCID/keyword retrieval function;
2. store normalized paper content under `/data/papers`;
3. call the reusable CellExLink bundle from `app.state.cell_model`;
4. run gene and chemical extraction;
5. call OpenAI relation extraction with the job-local key;
6. write normalized nodes, edges, evidence, and provenance to SQLite/JSON;
7. return the resulting network through `/api/networks/{job_id}`.
