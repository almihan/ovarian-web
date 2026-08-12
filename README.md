# Ovarian Network

A staged FastAPI application for ovarian-literature retrieval, normalized entity
extraction, relation extraction, and interactive network exploration.

The active pipeline produces reusable outputs for four independent stages:

- **Stage 1:** ovarian literature chunks from PubMed and PubMed Central;
- **Stage 2:** Cell Ontology-normalized cell types, NCBI Gene-normalized human
  genes/proteins, and MeSH-normalized hormones;
- **Stage 3:** locally validated biological relations with explicit ovarian cell
  context for hormone-gene relations;
- **Stage 4:** a compact SQLite interaction graph, a compressed one-row-per-entity
  index, and a separate PyVis browser explorer.

The cost-aware design stores paper text only once, streams aligned artifacts,
skips relation-ineligible chunks before inference, holds at most 500 Stage 3 rows
in memory, and reuses deterministic completed outputs. CellExLink recognition and
normalization remain sequential on Modal, while Stage 3 uses bounded,
asynchronous online OpenAI Responses calls with `gpt-5.4-nano` by default.

## Architecture

```text
Browser
   |
   v
Railway FastAPI controller (one replica while SQLite is used)
   |-- Stage 1: PubMed/PMC retrieval and chunk generation (CPU)
   |-- SQLite job/checkpoint state on one persistent Railway volume
   |-- canonical compressed artifacts in local storage or one Railway Bucket
   |
   |-- Stage 2 coordinator
   |     |-- Railway CPU: PubTator3 genes + MeSH hormone filter/cache
   |     |-- Modal T4 or local CPU: CellExLink cells
   |     `-- Railway CPU: streaming merge and compact entity artifact
   |
   |-- Stage 3 coordinator
   |     |-- stream Stage 1 text + aligned Stage 2 offsets
   |     |-- tag C/G/H entities and skip impossible relation pairs
   |     |-- hold <=500 rows in a bounded local window
   |     |-- run a small number of /v1/responses calls concurrently
   |     |-- enforce strict schema, direction matrix, and ovarian cell context
   |     `-- checkpoint each completed response and publish a compact artifact
   |
   `-- Stage 4 coordinator
         |-- stream aligned Stage 3 relation rows + Stage 1 source passages
         |-- map chunk-local C/G/H tags to normalized global node identities
         |-- aggregate directed edges, evidence, and tagged-cell contexts in SQLite
         `-- serve a bounded PyVis view with lazy search, expansion, and evidence
```

## User workflow

No stage starts automatically.

1. **Start retrieval**
   - runs the built-in ovarian PubMed query plus optional user additions;
   - retrieves metadata and available PMC full text;
   - generates reusable paper chunks;
   - publishes one deterministic `chunks.jsonl.gz` artifact.
2. **Start entity extraction**
   - becomes available after a completed retrieval is selected;
   - runs CellExLink and PubTator3 branches;
   - publishes one aligned `entity_annotations.jsonl.gz` artifact;
   - reports cell, human-gene, hormone, identifier, and timing counts.
3. **Start relation extraction**
   - becomes available after a completed Stage 2 result is selected;
   - reads at most 500 rows per local window and sends only eligible tagged chunks;
   - runs online Responses calls asynchronously with bounded concurrency;
   - extracts activation, inhibition, proliferation, merged secretion/production,
     binding, upregulation, and downregulation;
   - records explicit cell context for hormone-gene relations;
   - publishes `relations.jsonl.gz` plus a token/cache/provenance summary.
4. **Build and explore the interaction network**
   - becomes available after a completed Stage 3 result is selected;
   - streams the relation artifact together with its aligned Stage 1 passages;
   - converts chunk-local `C#`, `G#`, and `H#` tags into deterministic global nodes;
   - aggregates directed relations, undirected binding edges, paper/chunk counts,
     evidence, and hormone-gene cell contexts into SQLite;
   - opens a separate explorer page immediately and shows Stage 4 build progress;
   - loads a bounded initial graph, then searches and adds selected node
     neighbourhoods without replacing the current browser graph;
   - publishes `interaction-network.sqlite`, `entity-relation-index.jsonl.gz`, and
     `summary.json` without rerunning Stages 1-3.

Stage 2 reuse is keyed by the Stage 1 SHA-256 and the complete annotation model
signature. Stage 3 reuse is keyed by both source artifact SHA-256 values and the
relation model/prompt/policy signature. Only one distinct Stage 2 job and one
distinct Stage 3 job may be active, preventing accidental queues of billable GPU
or OpenAI work. Stage 4 reuse is keyed by both source artifact fingerprints and
the network-contract signature; one Stage 4 SQLite writer may run at a time.

## Stage 2 data contract

Each Stage 2 line corresponds to one original Stage 1 chunk. The chunk text is
not copied into Stage 2.

```json
{
  "base": "f4d9...",
  "doc_key": "PMC1234567",
  "canonical_id": "pmcid:PMC1234567",
  "pmid": "12345678",
  "pmcid": "PMC1234567",
  "section_type": "ABSTRACT",
  "chunk_id": 2,
  "annotations": [
    {
      "obj": "cell",
      "start": 0,
      "end": 15,
      "mention": "granulosa cells",
      "concept_id": "CL:0000501",
      "preferred_label": "granulosa cell",
      "normalization_source": "CellExLink"
    },
    {
      "obj": "gene",
      "start": 31,
      "end": 34,
      "mention": "INS",
      "concept_id": "NCBIGene:3630",
      "gene_id": "3630",
      "preferred_label": "INS",
      "normalization_source": "PubTator3",
      "label_source": "NCBI Gene"
    },
    {
      "obj": "hormone",
      "source_entity_type": "Chemical",
      "start": 48,
      "end": 57,
      "mention": "estradiol",
      "concept_id": "MESH:D004958",
      "hormone_id": "D004958",
      "chemical_id": "D004958",
      "preferred_label": "Estradiol",
      "normalization_source": "PubTator3",
      "label_source": "MeSH"
    }
  ]
}
```

All `start` and `end` values are relative to the corresponding Stage 1 `chunk`
string. Stage 3 can therefore join the two artifacts by `base`/`chunk_id` or the
paper-level identifiers without storing text twice. The Stage 2 summary includes
an occurrence count for retained hormone mentions:

```json
{
  "hormone_count": 214
}
```

PubTator3 annotations are filtered before writing: disease, species, variant,
cell-line, relation, non-human genes, and non-hormone Chemical annotations are
discarded. Only PubTator3 entity types `Gene`, `Gene/Protein`, and `Protein` are
accepted as genes, and an annotation is retained only when its identifier has the
human form `9606:<GeneID>`. The stored `gene_id` is the numeric part after
`9606:`. Chemical annotations are kept only when their MeSH descriptor belongs
to the biological Hormones hierarchy `D06.472`, or a supplementary MeSH concept
maps to that hierarchy. Retained mentions are emitted as `obj: "hormone"` with
`hormone_id`; `chemical_id` is also kept for MeSH compatibility.

Gene preferred labels use only the NCBI Gene ESummary `name` field; when it is
unavailable, the extracted mention is used. Hormone preferred labels use only
the authoritative MeSH RDF label; when it is unavailable, the extracted mention
is used. PubTator3 label hints, other NCBI Gene fields, and concept identifiers
are not used as label fallbacks.


## Stage 3 relation contract

Each Stage 3 line remains aligned to one Stage 1/Stage 2 chunk, but does not copy
the chunk text. Entity IDs are local to the row: `C#` for cells, `G#` for
genes/proteins, and `H#` for hormones. Only entities referenced by a relation or
its context are retained.

| Predicate | Accepted directions |
|---|---|
| `activation` | `G -> C`, `H -> C`, `H -> G` |
| `inhibition` | `G -> C`, `H -> C`, `H -> G` |
| `proliferation` | `G -> C`, `H -> C` |
| `secreted` | `C -> G`; for `C -> H`, includes secretion, release, production, generation, and synthesis |
| `binding` | predicted as `H -> G`; displayed as undirected `H -- G` in the network |
| `upregulation` | `H -> G` |
| `downregulation` | `H -> G` |

`G -> G`, `C -> C`, self-relations, `C secreted C`, `H secreted C`,
`G secreted C`, reversed `G binding H`, and every other unlisted combination are
rejected locally. Hormone-gene relations include a `cell_context` array containing
only explicitly linked `C#` entities. The default keeps an otherwise explicit
hormone-gene relation with an empty array when no cell is explicit; strict mode
can discard it. Stage 4 removes the arrow from binding edges without changing the
Stage 3 relation record.

```json
{
  "base": "f4d9...",
  "doc_key": "PMC1234567",
  "canonical_id": "pmcid:PMC1234567",
  "pmid": "12345678",
  "pmcid": "PMC1234567",
  "journal": "Example Journal",
  "pub_year": 2026,
  "section_type": "ABSTRACT",
  "chunk_id": 2,
  "entities": [
    {"id": "C1", "obj": "cell", "concept_id": "CL:0000501", "preferred_label": "granulosa cell"},
    {"id": "G1", "obj": "gene", "gene_id": "2492", "preferred_label": "FSHR"},
    {"id": "H1", "obj": "hormone", "hormone_id": "D004958", "preferred_label": "Estradiol"}
  ],
  "relations": [
    {
      "subject": "H1",
      "predicate": "upregulation",
      "object": "G1",
      "cell_context": ["C1"]
    }
  ]
}
```

See `docs/RELATION_EXTRACTION.md` for the prompt/cache design, resume checkpoints,
configuration, output details, and validation behavior.

## Stage 4 network contract

Stage 3 entity tags are local to one chunk and cannot be used as global network
identifiers. Stage 4 therefore creates deterministic global IDs from the entity
type plus its normalized identifier. Repeated Cell Ontology, NCBI Gene, and MeSH
hormone identities merge across chunks and papers even when each chunk calls them
`C1`, `G1`, or `H1`.

The SQLite graph retains all supported entity types. The explorer has no entity
scope radio buttons and no top-relations selector: all cells, genes/proteins, and
hormones are eligible, and all edges among the initial nodes are returned. The
initial node count controls only the first/reset view. Searching a node and
choosing **Add node + neighbours** performs an incremental browser `DataSet.update`
so the existing graph and manually arranged positions are preserved.

For a cell normalized to a Cell Ontology identifier, **Show cell hierarchy**
adds its direct parents and children to that same browser graph. Existing
interaction cells retain their teal node style; ontology terms absent from the
extracted relations appear as light-blue boxes connected by dashed child-to-parent
`is_a` arrows. Expansion is one hop at a time, so selecting an added ontology
term allows the hierarchy to be extended without loading the complete ontology.
The compact hierarchy for Cell Ontology release `2025-12-17` is bundled with the
application and requires no runtime ontology download.

The global graph stores compact counts and evidence IDs, while source passage text
is included only in the edge-evidence table. Search, neighbourhood expansion,
selection details, and evidence are queried lazily from SQLite rather than loading
the complete network into browser memory. See `docs/NETWORK_EXPLORER.md`.

## Storage policy

Reuse requires retaining the final output of each stage, not every temporary
file.

### Railway volume

```text
/data/
├── database.sqlite
├── papers/
│   ├── corpus.jsonl
│   ├── fulltext_bioc/*.bioc.json.gz
│   ├── chunks_by_paper/<paper-key>/chunks.jsonl.gz
│   └── jobs/<retrieval-job-id>/summary.json
├── local_annotation_jobs/             # bounded active Stage 2 work
├── relation_jobs/<job-id>/             # current <=500-row Stage 3 checkpoint only
└── network_jobs/                       # bounded active Stage 4 work + graph cache
```

The per-paper cache prevents repeated metadata/full-text downloads and repeated
chunk construction.

### Railway Bucket

```text
ovarian-network/
├── retrieval/chunks/<sha-prefix>/<source-sha>.jsonl.gz
├── annotations/<source-sha>/<model-signature>/
│   ├── branches/                         # deleted after final merge
│   ├── entity_annotations.jsonl.gz       # retained
│   └── summary.json                      # retained
├── relations/<entity-sha>/<source-sha>/<model-signature>/
│   ├── relations.jsonl.gz                # retained; no duplicated chunk text
│   └── summary.json                      # retained token/cache/provenance metrics
└── networks/<relation-sha>/<chunks-sha>/<network-signature>/
    ├── interaction-network.sqlite        # retained lazy graph index
    ├── entity-relation-index.jsonl.gz    # retained one row per global entity
    └── summary.json                      # retained Stage 4 provenance and counts
```

The combined Stage 1 bundle is built in a temporary directory and removed after
upload. CellExLink and PubTator3 per-paper work files live on ephemeral disk.
Their compact branch objects are used only for coordination and are deleted from
the bucket after Railway publishes the final pair.

### Persistent caches

```text
/model-cache/                         # Modal Volume
├── huggingface/                      # NER and NEN snapshots
└── ontology-embeddings/              # Cell Ontology embedding cache

/data/model_cache/                    # Railway volume
└── pubtator3-preferred-labels.sqlite # small gene/hormone label and classification cache
```

CellExLink checkpoint snapshots are not deleted after every job. They are loaded
one at a time, so recognition and normalization never coexist in GPU memory.

## Stage 2 lifecycle

```text
Railway downloads chunks.jsonl.gz        Modal downloads chunks.jsonl.gz
              |                                           |
Split into per-paper files                   Split into per-paper files
              |                                           |
PubTator3 BioC requests                      Load CellExLink Bioformer
Keep Gene + MeSH-classified Hormone                         Recognize cell-type mentions
Map offsets to Stage 1 chunks                Release model + CUDA cache
Resolve/cache preferred labels               Load CellExLink SapBERT
              |                              Normalize to Cell Ontology
Publish PubTator3 branch                                  |
              |                              Publish cell branch and release T4
              +-------------------------+-----------------+
                                        |
                        Railway streams aligned branch rows
                        Merge and sort annotations per chunk
                                        |
                     Upload final artifact + summary; delete branches
```

Only one paper's chunk list and annotation sidecars are held during a merge.
The final artifact is written as a stream and never assembled fully in memory.

## Project structure

```text
.
├── backend/
│   ├── api/
│   │   ├── jobs.py
│   │   ├── papers.py
│   │   ├── annotations.py
│   │   ├── relations.py
│   │   └── networks.py
│   ├── database/database.py
│   ├── pipeline/
│   │   ├── retrieval.py
│   │   ├── entity_artifacts.py
│   │   ├── relation_extraction.py
│   │   ├── relation_contract.py
│   │   ├── network_builder.py
│   │   └── network_contract.py
│   ├── services/
│   │   ├── railway_annotation_executor.py
│   │   ├── relation_executor.py
│   │   ├── network_executor.py
│   │   └── network_repository.py
│   ├── static/
│   └── templates/
│       ├── index.html
│       └── network.html
├── docs/RELATION_EXTRACTION.md
├── docs/NETWORK_EXPLORER.md
├── tests/
│   ├── test_relation_extraction.py
│   ├── test_relation_executor.py
│   ├── test_network_builder.py
│   └── test_network_repository.py
├── modal_app.py
├── Dockerfile
├── requirements.txt
├── requirements-local.txt
└── requirements-modal.txt
```

## PubTator3 implementation

The PubTator3 parser is in
`backend/pipeline/pubtator3_annotation_worker.py`; Railway orchestration and the
final merge are in `backend/services/railway_annotation_executor.py`. Important
behavior:

- uses the PubTator3 BioC JSON export endpoints already compatible with Stage 1;
- requests PMC full text first and requests PubMed title/abstract only for
  uncovered title/abstract chunks;
- batches identifiers with `PUBTATOR3_BATCH_SIZE`;
- paces NCBI requests to at most three requests per second;
- retries transient HTTP failures with bounded exponential backoff;
- matches sanitized PubTator3 passages to the exact Stage 1 chunk;
- converts document-level BioC offsets to chunk-relative offsets;
- classifies normalized MeSH concepts against the biological hormone hierarchy
  `D06.472`, including mapped supplementary concepts;
- retains genes only when PubTator3 supplies a human `9606:<GeneID>` identifier;
- resolves gene labels from the NCBI Gene ESummary `name` field and hormone labels from MeSH RDF, otherwise using the mention;
- writes sparse per-paper gzip sidecars containing genes and hormones only;
- publishes a text-free PubTator3 branch that Railway streams together with the CellExLink branch;
- fails Stage 2 when PubTator3 is required but no requested document can be
  annotated, preventing a silently incomplete result.

## Local Stages 1-4 test

Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Apple Silicon Mac:
python -m pip install torch==2.4.1

# Intel Mac, use instead:
# python -m pip install torch==2.2.2

# Linux or Windows CPU, use instead:
# python -m pip install torch==2.4.1 \
#   --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements-local.txt
cp .env.example .env
```

Set at least these values in `.env`:

```env
APP_ENV=development
NCBI_EMAIL=your_real_email@example.edu
NCBI_TOOL=ovarian_network_web

ARTIFACT_BACKEND=local
CELL_ANNOTATION_BACKEND=local
CELL_LOCAL_DEVICE=cpu

PUBTATOR3_BATCH_SIZE=20
PUBTATOR3_REQUEST_TIMEOUT=120
PUBTATOR3_REQUIRED=true
PUBTATOR3_RESOLVE_PREFERRED_LABELS=true

# Keep a small first test.
RETRIEVAL_KEYWORD_LIMIT=5

# Stage 3; keep this key only in the server-side .env file.
OPENAI_API_KEY=your-server-side-key
OPENAI_RELATION_MODEL=gpt-5.4-nano
RELATION_WINDOW_SIZE=500
RELATION_CONCURRENCY=8
RELATION_REQUEST_TIMEOUT_SECONDS=180
RELATION_MAX_REQUEST_RETRIES=5
RELATION_RETRY_BASE_SECONDS=1
RELATION_PROGRESS_UPDATE_EVERY=5
RELATION_REASONING_EFFORT=none
RELATION_PROMPT_CACHE_KEY=ovarian-relations-v4
RELATION_PROMPT_CACHE_SHARDS=32
RELATION_REQUIRE_HORMONE_GENE_CELL_CONTEXT=false

# Stage 4 bounded browser/index settings.
NETWORK_INITIAL_NODES=120
NETWORK_MAX_INITIAL_NODES=1000
NETWORK_EXPANSION_LIMIT=150
NETWORK_SEARCH_LIMIT=30
NETWORK_EVIDENCE_LIMIT=100
NETWORK_HIERARCHY_LIMIT=120
```

Start the application:

```bash
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/`, complete Stages 1–3, select the completed
relation result in Stage 4, and click **Build and explore network**. The browser
moves to `/network/<job-id>` immediately, displays build progress, and opens the
PyVis graph when the SQLite index is ready. The local worker requires outbound
HTTPS access to NCBI and OpenAI for Stages 1–3; Stage 4 itself is local CPU and
does not make OpenAI requests. For the most stable long-running test, omit
`--reload`.

### Conda equivalent

```bash
conda create -n ovarian-network python=3.11 -y
conda activate ovarian-network
python -m pip install --upgrade pip
python -m pip install torch==2.4.1
python -m pip install -r requirements-local.txt
cp .env.example .env
uvicorn backend.main:app --reload
```

## Deploy the Modal worker

```bash
pip install 'modal>=1.5,<2'
modal setup
modal deploy modal_app.py
modal run modal_app.py::app.warm_model_cache
```

The warm function downloads or reuses the CellExLink snapshots before a T4 job.
PubTator3 no longer runs in Modal; its preferred-label cache is stored under
`CELL_MODEL_CACHE_DIR` on the Railway volume.

For reproducible CellExLink results, set immutable Hugging Face revisions:

```env
CELL_NER_REVISION=<recognition-model-commit-hash>
CELL_NEN_REVISION=<normalization-model-commit-hash>
```

## Deploy the Railway controller

1. Push the project to GitHub.
2. Create a Railway service from the repository.
3. Attach one persistent volume at `/data`.
4. Attach one Railway Bucket.
5. Generate the public domain.
6. Deploy `modal_app.py` from a trusted workstation or CI environment.
7. Set the variables below.

```env
APP_ENV=production
PUBLIC_BASE_URL=https://YOUR-SERVICE.up.railway.app
NCBI_EMAIL=your_real_email@example.edu
NCBI_TOOL=ovarian_network_web

ARTIFACT_BACKEND=s3
ARTIFACT_PREFIX=ovarian-network
ARTIFACT_PRESIGNED_TTL_SECONDS=86400

CELL_ANNOTATION_BACKEND=modal
MODAL_TOKEN_ID=your-token-id
MODAL_TOKEN_SECRET=your-token-secret
MODAL_APP_NAME=ovarian-cellexlink
MODAL_FUNCTION_NAME=annotate_bundle

CELL_NER_MODEL=almire/CellExLink-bioformer16L
CELL_NER_REVISION=
CELL_NEN_MODEL=almire/CellExLink-Sapbert
CELL_NEN_REVISION=
CELL_DISABLE_ABBREVIATIONS=false
CELL_CPU_THREADS=4
CELL_NER_TEXT_BATCH_SIZE=8
CELL_NER_WINDOW_BATCH_SIZE=4
CELL_NEN_BATCH_SIZE=64
CELL_NEN_REQUEST_BATCH_SIZE=128

PUBTATOR3_BATCH_SIZE=20
PUBTATOR3_REQUEST_TIMEOUT=120
PUBTATOR3_REQUIRED=true
PUBTATOR3_RESOLVE_PREFERRED_LABELS=true

OPENAI_API_KEY=your-private-service-key
OPENAI_RELATION_MODEL=gpt-5.4-nano
RELATION_WINDOW_SIZE=500
RELATION_CONCURRENCY=8
RELATION_REQUEST_TIMEOUT_SECONDS=180
RELATION_MAX_REQUEST_RETRIES=5
RELATION_RETRY_BASE_SECONDS=1
RELATION_PROGRESS_UPDATE_EVERY=5
RELATION_MAX_OUTPUT_TOKENS=1200
RELATION_REASONING_EFFORT=none
RELATION_PROMPT_CACHE_KEY=ovarian-relations-v4
RELATION_PROMPT_CACHE_SHARDS=32
RELATION_REQUIRE_HORMONE_GENE_CELL_CONTEXT=false

NETWORK_INITIAL_NODES=120
NETWORK_MAX_INITIAL_NODES=1000
NETWORK_EXPANSION_LIMIT=150
NETWORK_SEARCH_LIMIT=30
NETWORK_EVIDENCE_LIMIT=100
```

Stage 3 does not upload a Batch JSONL file and does not create or poll an OpenAI
Batch. Each eligible chunk uses the online Responses API. Durable local request
events under `/data/relation_jobs` allow an interrupted job to resume unfinished
chunks without resending responses already journaled successfully.

Railway supplies `BUCKET`, `ENDPOINT`, `ACCESS_KEY_ID`, `SECRET_ACCESS_KEY`, and
`REGION` when a Bucket is attached. Keep one Railway service replica while
SQLite is used. Keep `OPENAI_API_KEY` private; it is never rendered into the
browser. Active Stage 3 jobs resume from the `/data/relation_jobs` checkpoint.
Active Stage 4 jobs resume from `/data/network_jobs`; completed graph artifacts
are content-addressed and reusable.

This release does not add user authentication. Because `POST /api/relations`
starts billable API work, place a public deployment behind access control or
add application authentication before sharing it broadly. The one-active-job
policy limits concurrency; it is not an authorization boundary.

### PubTator3 fallback policy

`PUBTATOR3_REQUIRED=true` is recommended because Stage 2 is defined as a merged
cell/gene/hormone process. With this setting, complete loss of PubTator3 causes
the job to fail rather than publishing a cell-only artifact that looks complete.

Set `PUBTATOR3_REQUIRED=false` only when an explicitly documented cell-only
fallback is acceptable. Individual papers or passages that PubTator3 does not
cover are recorded in the summary statistics either way.

## Cost and resource controls

- one active Stage 2 job globally;
- one Modal T4 container maximum;
- short idle scale-down window;
- CellExLink checkpoints warmed on CPU and reused from a persistent Volume;
- recognition and normalization loaded sequentially;
- PubTator3 runs in one bounded Railway background thread concurrently with Modal;
- Modal returns after publishing the cell branch and does not wait for NCBI or merging;
- requests and identifiers are processed in bounded batches;
- preferred labels are cached across jobs;
- paper text exists only in Stage 1;
- per-paper Stage 2 work uses ephemeral disk, and temporary branch objects are deleted after merging;
- final artifacts are gzip-compressed and content-addressed for reuse;
- chunks with no possible allowed entity-direction pair make no OpenAI request;
- each Stage 3 local window contains at most 500 rows, while only `RELATION_CONCURRENCY` eligible online requests run at once;
- the static relation instructions/schema precede variable chunk text and use stable sharded prompt-cache keys;
- strict structured output is validated again in Python before storage;
- Stage 3 output omits chunk text and copies only relation-referenced entities;
- one active Stage 3 job, per-response durable journals, resumable checkpoints, and deterministic reuse limit duplicate spend and storage;
- Stage 4 streams aligned gzip inputs one row at a time into SQLite and commits in bounded intervals;
- chunk-local entity tags are resolved to deterministic normalized global nodes before aggregation;
- only a bounded initial graph is sent to the browser; search, neighbourhoods, details, and evidence are fetched on demand;
- one Stage 4 SQLite writer and deterministic artifact reuse avoid duplicate graph builds.

Prompt caching is best-effort rather than guaranteed. Stage 3 keeps a roughly
10,000-character static instruction prefix and a stable strict schema before the
changing chunk, records the returned cached-token count, and displays the
measured cache rate instead of assuming every repeated prefix was cached.

## API

### Retrieval

```text
POST /api/jobs
GET  /api/jobs?limit=10
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/summary
GET  /api/papers/jobs/{job_id}/download
```

### Entity extraction

```text
GET  /api/annotations/status
POST /api/annotations
GET  /api/annotations?limit=10
GET  /api/annotations/{job_id}
POST /api/annotations/{job_id}/callback
GET  /api/annotations/{job_id}/summary
GET  /api/annotations/{job_id}/download
```

The callback endpoint requires a different randomly generated token for every
job. Modal and bucket secrets are never exposed to browser JavaScript.

### Relation extraction

```text
GET  /api/relations/status
POST /api/relations
GET  /api/relations?limit=10
GET  /api/relations/{job_id}
GET  /api/relations/{job_id}/summary
GET  /api/relations/{job_id}/download
```

The OpenAI key is server-side only. Browser responses expose model/status and
usage counters, never credentials or raw remote file contents.

### Interaction network

```text
GET  /api/networks/status
POST /api/networks
GET  /api/networks?limit=10
GET  /api/networks/{job_id}
GET  /api/networks/{job_id}/summary
GET  /api/networks/{job_id}/download/entity-index
GET  /api/networks/{job_id}/graph?top_nodes=120
GET  /api/networks/{job_id}/nodes/search?q=granulosa
GET  /api/networks/{job_id}/nodes/{node_id}/neighborhood
GET  /api/networks/{job_id}/nodes/{node_id}
GET  /api/networks/{job_id}/edges/{edge_id}
GET  /api/networks/{job_id}/evidence/nodes/{node_id}
GET  /api/networks/{job_id}/evidence/edges/{edge_id}
```

The explorer page is `/network/{job_id}`. PyVis prepares vis-network node and
edge payloads on the server; the bundled browser assets are served from the
installed package rather than an external CDN.

## Validation

```bash
PYTHONPATH=. pytest -q
node --check backend/static/js/app.js
node --check backend/static/js/network.js
python -m compileall -q backend modal_app.py tests
```

Tests cover the exact Stage 3 relation matrix and prohibitions, hormone-gene
cell context, tag reuse, cache-sharded structured requests, aggregate token
accounting, skipped ineligible chunks, streamed output, restart-safe
checkpointing, bounded asynchronous concurrency, retry accounting, per-response
token usage, interrupted-window resumption, and prevention of duplicate completed
requests after restart. Stage 4 tests additionally cover global identity merging,
chunk-local tag isolation, hormone nodes, cell-context evidence, compressed entity
index output, lazy SQLite graph queries, and incremental browser insertion.

## PubTator3 citation and limitations

When reporting results produced by this integration, cite:

> Wei C-H, Allot A, Lai P-T, et al. PubTator 3.0: an AI-powered literature
> resource for unlocking biomedical knowledge. *Nucleic Acids Research*.
> 2024;52(W1):W540-W546. doi:10.1093/nar/gkae235.

PubTator3 annotations are automated predictions. They can miss entities or
assign an incorrect boundary or identifier, so downstream network results
should preserve provenance and support review of the original text. NCBI/NLM
services and literature data remain subject to their applicable policies and
the rights attached to the underlying articles; see `THIRD_PARTY_NOTICES.md`.

## License and attribution

This project is distributed under GNU GPL v3.0. The compact CellExLink-derived
runtime and bundled resources retain their applicable licenses and notices. See
`LICENSE.txt`, `THIRD_PARTY_NOTICES.md`, `backend/cellexlink_lite/NOTICE.md`, and
`backend/licenses/`.
