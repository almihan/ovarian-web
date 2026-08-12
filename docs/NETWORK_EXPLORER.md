# Stage 4 interaction-network explorer

Stage 4 consumes a completed Stage 3 relation artifact and the exact Stage 1
chunk artifact used by that relation job. It does not call OpenAI and does not
rerun retrieval or entity extraction.

## Why the entity index is rebuilt

Stage 3 tags such as `C1`, `G1`, and `H1` are native identifiers only inside one
chunk. A later chunk can reuse `G1` for a completely different gene. Stage 4
therefore never treats a tag as a global node ID.

For every relation row, the builder maps each local tag to a deterministic global
identity using this priority:

| Type | Normalized identity priority |
|---|---|
| Cell | `concept_id`, `cell_ontology_id`, `normalized_id`, then normalized label/mention |
| Gene/protein | `gene_id`, `concept_id`, `normalized_id`, then normalized label/mention |
| Hormone | `hormone_id`, `chemical_id`, `concept_id`, `normalized_id`, then normalized label/mention |

The global node ID is a short SHA-256-derived identifier prefixed with `C-`,
`G-`, or `H-`. The readable normalized identifier and preferred label remain in
the node record. This merges repeated normalized entities across chunks and
papers while preventing accidental merging of unrelated local tags.

## Build flow

```text
relations.jsonl.gz + aligned chunks.jsonl.gz
                     |
                     v
        stream one aligned row pair at a time
                     |
        map C/G/H tags to global normalized IDs
                     |
        upsert nodes, aliases, papers, chunks, edges,
        source evidence, and explicit cell contexts
                     |
      finalize counts and searchable SQLite indexes
                     |
        +------------+------------------+
        |                               |
interaction-network.sqlite   entity-relation-index.jsonl.gz
```

The builder commits every 500 rows by default and uses SQLite temporary storage
rather than accumulating the corpus in Python collections. Paper counts are
computed from SQLite tables instead of retaining a corpus-wide paper set in
memory.

## SQLite graph schema

The graph artifact contains these logical tables:

- `nodes`: global identity, type, label, and aggregate counts;
- `node_aliases`: preferred names, mentions, and normalized identifiers;
- `node_papers` and `node_chunks`: unique source membership;
- `edges`: one directed `(subject, predicate, object)` row;
- `edge_papers`: unique supporting papers;
- `edge_evidence`: aligned source passage and provenance;
- `edge_contexts`: explicit tagged-cell context for relation evidence;
- `meta`: schema, pipeline version, and summary statistics.

The graph stores all cell, gene/protein, and hormone types. No entity-type filter
is applied when ranking the initial view or searching the complete index.

## Browser behavior

Clicking **Build and explore network** creates or reuses a Stage 4 job and moves
the browser immediately to:

```text
/network/<job-id>
```

The explorer page displays build progress until the SQLite graph is ready. It
then loads a bounded initial graph ordered by relation count. The initial node
selector affects only the first view and explicit resets. There is no
“top relations” selector: every edge among the returned initial nodes is
included.

The sidebar has three tabs:

- **Explore**: initial node limit, normalized-node search, incremental insertion,
  visible counts, legend, and controls;
- **Details**: normalized node or directed-edge metadata and aggregate counts;
- **Evidence**: source passages and explicit cell contexts, loaded only for the
  current selection.

### Incremental node insertion

Searching selects from the complete SQLite node index, not only visible nodes.
Choosing **Add node + neighbours** requests a bounded one-hop neighbourhood and
calls `vis.DataSet.update()` for its nodes and edges. It does not clear or replace
the existing datasets, so previously displayed nodes and manually arranged
positions remain in the graph.

A double-click removes a node and its incident visible edges from only the
current browser view. The node remains in SQLite and can be added again through
search.

### Cell Ontology hierarchy expansion

When a selected cell has a `CL:` identifier, the Details panel and the selected
search-result card expose **Show cell hierarchy**. The request reads the compact
bundled Cell Ontology release and adds the selected term's direct `is_a` parents
and children with `vis.DataSet.update()`. The current interaction graph is not
cleared, and manually arranged positions are retained.

Hierarchy expansion is intentionally one hop per click. Selecting an added
ontology term and choosing **Show cell hierarchy** again moves another level up
or down without loading the complete ontology into browser memory. The response
is bounded by `NETWORK_HIERARCHY_LIMIT`; direct parents are retained first, then
children are added in label order until the limit is reached.

Cell terms already present in the extracted interaction graph reuse their normal
teal interaction node. Terms absent from extracted interactions are shown as
light-blue boxes. Dashed `is_a` arrows point from the more specific child term to
its direct parent. Hierarchy-only nodes and ontology edges do not claim paper
evidence; their Details view identifies them as ontology information rather than
relations extracted from literature.

The hierarchy resource is bundled as a gzip-compressed JSONL index derived from
Cell Ontology release `2025-12-17`. It stores term IDs, labels, synonyms, direct
parents, and enough metadata for incremental display. No external ontology API or
runtime ontology download is required locally or on Railway.

## Node and edge styling

| Entity | Default color |
|---|---|
| Cell / cell type | teal |
| Gene / protein | orange |
| Hormone | purple |
| Hierarchy-only Cell Ontology term | light-blue box |

Cell Ontology `is_a` edges are dashed blue-gray arrows from child to parent.
Directed biological edge width grows logarithmically with evidence count. Inhibition and
downregulation are red; binding is blue; secretion is teal; production and
biosynthesis are violet; other positive directed relations are green.

## API

```text
GET  /api/networks/status
POST /api/networks
GET  /api/networks?limit=10
GET  /api/networks/{job_id}
GET  /api/networks/{job_id}/summary
GET  /api/networks/{job_id}/download/entity-index
GET  /api/networks/{job_id}/graph?top_nodes=120
GET  /api/networks/{job_id}/nodes/search?q=estradiol
GET  /api/networks/{job_id}/nodes/{node_id}/neighborhood?limit=150
GET  /api/networks/{job_id}/nodes/{node_id}/hierarchy?limit=120
GET  /api/networks/{job_id}/nodes/{node_id}
GET  /api/networks/{job_id}/edges/{edge_id}
GET  /api/networks/{job_id}/evidence/nodes/{node_id}
GET  /api/networks/{job_id}/evidence/edges/{edge_id}
```

Example job submission:

```json
{
  "source_relation_job_id": "completed-stage-3-job-id"
}
```

## Configuration

```env
NETWORK_JOBS_DIR=./data/network_jobs
NETWORK_INITIAL_NODES=120
NETWORK_MAX_INITIAL_NODES=1000
NETWORK_EXPANSION_LIMIT=150
NETWORK_SEARCH_LIMIT=30
NETWORK_EVIDENCE_LIMIT=100
NETWORK_HIERARCHY_LIMIT=120
```

- `NETWORK_INITIAL_NODES` controls the default initial/reset view.
- `NETWORK_MAX_INITIAL_NODES` is the server-side safety ceiling.
- `NETWORK_EXPANSION_LIMIT` limits incident edges returned for one selected node.
- `NETWORK_SEARCH_LIMIT` bounds normalized-node search results.
- `NETWORK_EVIDENCE_LIMIT` bounds evidence loaded for one selection.
- `NETWORK_HIERARCHY_LIMIT` bounds the direct parent and child terms returned by one hierarchy expansion; the selected root term is returned in addition to that bounded neighbourhood.

## Local and Railway deployment

Stage 4 uses CPU and SQLite only. It runs locally and on Railway without a GPU.
For Railway, retain one application replica while SQLite coordinates jobs and
mount persistent storage at `/data`. With local artifact storage, the completed
SQLite graph is already available on the volume. With S3-compatible artifact
storage, the repository downloads the graph once into a SHA-verified local cache
and reuses that cached copy for later interactive queries.

The browser receives neither storage credentials nor local paths. PyVis prepares
the network payload server-side, and the vis-network JavaScript/CSS bundled with
the installed PyVis package are served from same-origin `/vendor` routes.
