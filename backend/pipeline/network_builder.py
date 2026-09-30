"""Streaming Stage 4 entity indexing and interaction-network construction.

The Stage 3 tags (C1/G1/H1) are local to one chunk.  This module maps each
local tag to a deterministic global node ID before merging evidence across
chunks, then persists the graph in a compact SQLite database for lazy browser
queries.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from backend.pipeline.entity_artifacts import iter_jsonl
from backend.pipeline.entity_overlap import prefer_longest_spans
from backend.pipeline.network_contract import (
    ENTITY_INDEX_SCHEMA,
    NETWORK_GRAPH_SCHEMA,
    NETWORK_PIPELINE_VERSION,
)
from backend.pipeline.identifier_identity import canonical_identifier
from backend.pipeline.relation_extraction import _annotation_span, relation_allowed

ProgressCallback = Callable[[int, str, Mapping[str, Any]], None]
_SPACE_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"^[CGH]\d+$")
_TYPE_PREFIX = {"cell": "C", "gene": "G", "protein": "G", "hormone": "H"}
_TYPE_ORDER = {"cell": 0, "gene": 1, "hormone": 2}
_ID_FIELDS = {
    "cell": ("concept_id", "cell_ontology_id", "normalized_id"),
    "gene": (
        "concept_id",
        "hgnc_id",
        "normalized_id",
        "ncbi_gene_id",
        # Rolling-deployment compatibility for older sidecars.
        "gene_id",
        "pubtator_gene_id",
    ),
    "hormone": (
        "concept_id",
        "mesh_id",
        "hormone_id",
        "normalized_id",
        "pubtator_mesh_id",
        "chemical_id",
        # Rolling-deployment compatibility for older hormone rows.
        "chebi_id",
        "hgnc_id",
    ),
}

@dataclass(frozen=True, slots=True)
class BuildResult:
    graph_path: Path
    entity_index_path: Path
    stats: dict[str, Any]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return _SPACE_RE.sub(" ", str(value)).strip()


def _norm(value: Any) -> str:
    return _text(value).casefold()


def _uniprot_text(entity: Mapping[str, Any]) -> str:
    values: list[str] = []
    raw = entity.get("uniprot_ids")
    if isinstance(raw, (list, tuple, set)):
        values.extend(_text(value).upper() for value in raw)
    elif raw not in (None, ""):
        values.extend(
            _text(value).upper()
            for value in re.split(r"\s*[;,|]\s*", str(raw))
        )
    existing = entity.get("UniProt")
    if existing not in (None, ""):
        values.extend(
            _text(value).upper()
            for value in re.split(r"\s*[;,|]\s*", str(existing))
        )
    return "; ".join(dict.fromkeys(value for value in values if value))


def _entity_type(entity: Mapping[str, Any]) -> str:
    value = _norm(entity.get("obj")).replace("_", " ")
    if value in {"protein", "gene/protein", "gene / protein", "gene-protein"}:
        return "gene"
    if value in {"cell type", "cell-type"}:
        return "cell"
    return value if value in {"cell", "gene", "hormone"} else ""


def _entity_identity(entity: Mapping[str, Any]) -> tuple[str, str, str]:
    entity_type = _entity_type(entity)
    if not entity_type:
        return "", "", ""
    for field in _ID_FIELDS[entity_type]:
        value = _text(entity.get(field))
        if value:
            namespace, canonical_value = canonical_identifier(
                entity_type,
                field,
                value,
            )
            return entity_type, namespace or field, canonical_value or value
    fallback = _norm(entity.get("preferred_label") or entity.get("mention"))
    if not fallback:
        return "", "", ""
    return entity_type, "mention", fallback


def global_entity_id(entity: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return (global ID, canonical type, normalized identity display value)."""

    entity_type, source_field, identity = _entity_identity(entity)
    if not entity_type:
        return "", "", ""
    digest = hashlib.sha256(
        f"{entity_type}\x1f{source_field}\x1f{identity.casefold()}".encode("utf-8")
    ).hexdigest()[:24]
    return f"{_TYPE_PREFIX[entity_type]}-{digest}", entity_type, identity


def _edge_id(subject_id: str, predicate: str, object_id: str) -> str:
    digest = hashlib.sha256(
        f"{subject_id}\x1f{predicate}\x1f{object_id}".encode("utf-8")
    ).hexdigest()[:24]
    return f"R-{digest}"


def _paper_id(row: Mapping[str, Any]) -> str:
    return (
        _text(row.get("canonical_id"))
        or _text(row.get("doc_key"))
        or (f"pmid:{_text(row.get('pmid'))}" if _text(row.get("pmid")) else "")
        or (f"pmcid:{_text(row.get('pmcid'))}" if _text(row.get("pmcid")) else "")
    )


def _aligned(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for field in ("base", "doc_key", "canonical_id", "pmid", "pmcid", "chunk_id"):
        a = _text(left.get(field))
        b = _text(right.get(field))
        if a and b and a != b:
            return False
    return True


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _source_text(value: Any) -> str:
    """Preserve source characters exactly because Stage 2 offsets refer to them."""

    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-32768;
        PRAGMA foreign_keys=ON;

        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            normalized_id TEXT NOT NULL,
            canonical_name TEXT NOT NULL DEFAULT '',
            canonical_id_type TEXT NOT NULL DEFAULT '',
            hgnc_id TEXT NOT NULL DEFAULT '',
            ncbi_gene_id TEXT NOT NULL DEFAULT '',
            chebi_id TEXT NOT NULL DEFAULT '',
            mesh_id TEXT NOT NULL DEFAULT '',
            uniprot_ids TEXT NOT NULL DEFAULT '',
            source_concept_id TEXT NOT NULL DEFAULT '',
            normalization_source TEXT NOT NULL DEFAULT '',
            normalization_status TEXT NOT NULL DEFAULT '',
            normalization_match_method TEXT NOT NULL DEFAULT '',
            chebi_release TEXT NOT NULL DEFAULT '',
            label TEXT NOT NULL DEFAULT '',
            label_norm TEXT NOT NULL DEFAULT '',
            occurrence_count INTEGER NOT NULL DEFAULT 0,
            paper_count INTEGER NOT NULL DEFAULT 0,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            relation_count INTEGER NOT NULL DEFAULT 0,
            incoming_count INTEGER NOT NULL DEFAULT 0,
            outgoing_count INTEGER NOT NULL DEFAULT 0,
            context_count INTEGER NOT NULL DEFAULT 0
        ) WITHOUT ROWID;

        CREATE TABLE node_aliases (
            node_id TEXT NOT NULL,
            alias TEXT NOT NULL,
            alias_norm TEXT NOT NULL,
            alias_kind INTEGER NOT NULL DEFAULT 0,
            count INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (node_id, alias_norm),
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        ) WITHOUT ROWID;

        CREATE TABLE node_papers (
            node_id TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            PRIMARY KEY (node_id, paper_id),
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        ) WITHOUT ROWID;

        CREATE TABLE node_chunks (
            node_id TEXT NOT NULL,
            base TEXT NOT NULL,
            PRIMARY KEY (node_id, base),
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        ) WITHOUT ROWID;

        CREATE TABLE edges (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_id TEXT NOT NULL,
            paper_count INTEGER NOT NULL DEFAULT 0,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            evidence_count INTEGER NOT NULL DEFAULT 0,
            context_evidence_count INTEGER NOT NULL DEFAULT 0,
            UNIQUE (subject_id, predicate, object_id),
            FOREIGN KEY (subject_id) REFERENCES nodes(id),
            FOREIGN KEY (object_id) REFERENCES nodes(id)
        ) WITHOUT ROWID;

        CREATE TABLE edge_papers (
            edge_id TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            PRIMARY KEY (edge_id, paper_id),
            FOREIGN KEY (edge_id) REFERENCES edges(id) ON DELETE CASCADE
        ) WITHOUT ROWID;

        CREATE TABLE edge_evidence (
            edge_id TEXT NOT NULL,
            base TEXT NOT NULL,
            doc_key TEXT NOT NULL DEFAULT '',
            canonical_id TEXT NOT NULL DEFAULT '',
            pmid TEXT NOT NULL DEFAULT '',
            pmcid TEXT NOT NULL DEFAULT '',
            journal TEXT NOT NULL DEFAULT '',
            pub_year TEXT NOT NULL DEFAULT '',
            section_type TEXT NOT NULL DEFAULT '',
            chunk_id TEXT NOT NULL DEFAULT '',
            chunk_text TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (edge_id, base),
            FOREIGN KEY (edge_id) REFERENCES edges(id) ON DELETE CASCADE
        ) WITHOUT ROWID;

        CREATE TABLE edge_contexts (
            edge_id TEXT NOT NULL,
            base TEXT NOT NULL,
            cell_node_id TEXT NOT NULL,
            PRIMARY KEY (edge_id, base, cell_node_id),
            FOREIGN KEY (edge_id, base) REFERENCES edge_evidence(edge_id, base)
                ON DELETE CASCADE,
            FOREIGN KEY (cell_node_id) REFERENCES nodes(id)
        ) WITHOUT ROWID;

        CREATE TABLE evidence_entities (
            base TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            entity_type TEXT NOT NULL,
            normalized_id TEXT NOT NULL DEFAULT '',
            node_id TEXT NOT NULL DEFAULT '',
            mention TEXT NOT NULL DEFAULT '',
            preferred_label TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (
                base, start_offset, end_offset, entity_type, normalized_id, mention
            )
        ) WITHOUT ROWID;
        """
    )


def _upsert_node(
    connection: sqlite3.Connection,
    entity: Mapping[str, Any],
    *,
    paper_id: str,
    base: str,
) -> str:
    node_id, entity_type, normalized_id = global_entity_id(entity)
    if not node_id:
        return ""

    metadata = {
        "canonical_name": _text(entity.get("canonical_name")),
        "canonical_id_type": _text(entity.get("canonical_id_type")),
        "hgnc_id": _text(entity.get("hgnc_id")),
        "ncbi_gene_id": _text(entity.get("ncbi_gene_id")),
        "chebi_id": _text(entity.get("chebi_id")),
        "mesh_id": _text(entity.get("mesh_id")),
        "uniprot_ids": _uniprot_text(entity),
        "source_concept_id": _text(entity.get("source_concept_id")),
        "normalization_source": _text(entity.get("normalization_source")),
        "normalization_status": _text(entity.get("normalization_status")),
        "normalization_match_method": _text(
            entity.get("normalization_match_method")
        ),
        "chebi_release": _text(entity.get("chebi_release")),
    }
    connection.execute(
        """
        INSERT INTO nodes (
            id, entity_type, normalized_id, canonical_name, canonical_id_type,
            hgnc_id, ncbi_gene_id, chebi_id, mesh_id, uniprot_ids,
            source_concept_id, normalization_source, normalization_status,
            normalization_match_method, chebi_release, occurrence_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(id) DO UPDATE SET
            occurrence_count = occurrence_count + 1,
            canonical_name = CASE WHEN nodes.canonical_name = ''
                THEN excluded.canonical_name ELSE nodes.canonical_name END,
            canonical_id_type = CASE WHEN nodes.canonical_id_type = ''
                THEN excluded.canonical_id_type ELSE nodes.canonical_id_type END,
            hgnc_id = CASE WHEN nodes.hgnc_id = ''
                THEN excluded.hgnc_id ELSE nodes.hgnc_id END,
            ncbi_gene_id = CASE WHEN nodes.ncbi_gene_id = ''
                THEN excluded.ncbi_gene_id ELSE nodes.ncbi_gene_id END,
            chebi_id = CASE WHEN nodes.chebi_id = ''
                THEN excluded.chebi_id ELSE nodes.chebi_id END,
            mesh_id = CASE WHEN nodes.mesh_id = ''
                THEN excluded.mesh_id ELSE nodes.mesh_id END,
            uniprot_ids = CASE WHEN nodes.uniprot_ids = ''
                THEN excluded.uniprot_ids ELSE nodes.uniprot_ids END,
            source_concept_id = CASE WHEN nodes.source_concept_id = ''
                THEN excluded.source_concept_id ELSE nodes.source_concept_id END,
            normalization_source = CASE WHEN nodes.normalization_source = ''
                THEN excluded.normalization_source ELSE nodes.normalization_source END,
            normalization_status = CASE WHEN nodes.normalization_status = ''
                THEN excluded.normalization_status ELSE nodes.normalization_status END,
            normalization_match_method = CASE
                WHEN nodes.normalization_match_method = ''
                THEN excluded.normalization_match_method
                ELSE nodes.normalization_match_method END,
            chebi_release = CASE WHEN nodes.chebi_release = ''
                THEN excluded.chebi_release ELSE nodes.chebi_release END
        """,
        (
            node_id,
            entity_type,
            normalized_id,
            metadata["canonical_name"],
            metadata["canonical_id_type"],
            metadata["hgnc_id"],
            metadata["ncbi_gene_id"],
            metadata["chebi_id"],
            metadata["mesh_id"],
            metadata["uniprot_ids"],
            metadata["source_concept_id"],
            metadata["normalization_source"],
            metadata["normalization_status"],
            metadata["normalization_match_method"],
            metadata["chebi_release"],
        ),
    )
    preferred = _text(entity.get("preferred_label"))
    mention = _text(entity.get("mention"))
    identifier_aliases = (
        normalized_id,
        metadata["hgnc_id"],
        metadata["ncbi_gene_id"],
        metadata["chebi_id"],
        metadata["mesh_id"],
        metadata["source_concept_id"],
        *tuple(
            value for value in metadata["uniprot_ids"].split("; ") if value
        ),
    )
    aliases = [
        (preferred, 3),
        (metadata["canonical_name"], 2),
        (mention, 1),
        *((value, 0) for value in identifier_aliases),
    ]
    for alias, kind in aliases:
        alias_norm = _norm(alias)
        if not alias_norm:
            continue
        connection.execute(
            """
            INSERT INTO node_aliases (node_id, alias, alias_norm, alias_kind, count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(node_id, alias_norm) DO UPDATE SET
                count = count + 1,
                alias_kind = MAX(alias_kind, excluded.alias_kind),
                alias = CASE
                    WHEN excluded.alias_kind > alias_kind THEN excluded.alias
                    ELSE alias
                END
            """,
            (node_id, alias, alias_norm, kind),
        )
    if paper_id:
        connection.execute(
            "INSERT OR IGNORE INTO node_papers (node_id, paper_id) VALUES (?, ?)",
            (node_id, paper_id),
        )
    if base:
        connection.execute(
            "INSERT OR IGNORE INTO node_chunks (node_id, base) VALUES (?, ?)",
            (node_id, base),
        )
    return node_id


def _finalize_graph(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        UPDATE nodes
        SET label = COALESCE((
                SELECT alias FROM node_aliases a
                WHERE a.node_id = nodes.id
                ORDER BY a.alias_kind DESC, a.count DESC,
                         LENGTH(a.alias) ASC, a.alias COLLATE NOCASE ASC
                LIMIT 1
            ), normalized_id),
            label_norm = COALESCE((
                SELECT alias_norm FROM node_aliases a
                WHERE a.node_id = nodes.id
                ORDER BY a.alias_kind DESC, a.count DESC,
                         LENGTH(a.alias) ASC, a.alias COLLATE NOCASE ASC
                LIMIT 1
            ), LOWER(normalized_id)),
            paper_count = (SELECT COUNT(*) FROM node_papers p WHERE p.node_id = nodes.id),
            chunk_count = (SELECT COUNT(*) FROM node_chunks c WHERE c.node_id = nodes.id),
            outgoing_count = (
                SELECT COUNT(*) FROM edges e
                WHERE e.subject_id = nodes.id
            ),
            incoming_count = (
                SELECT COUNT(*) FROM edges e
                WHERE e.object_id = nodes.id
            ),
            relation_count =
                (SELECT COUNT(*) FROM edges e WHERE e.subject_id = nodes.id)
                + (SELECT COUNT(*) FROM edges e WHERE e.object_id = nodes.id),
            context_count = (SELECT COUNT(*) FROM edge_contexts x WHERE x.cell_node_id = nodes.id);

        UPDATE edges
        SET paper_count = (SELECT COUNT(*) FROM edge_papers p WHERE p.edge_id = edges.id),
            chunk_count = (SELECT COUNT(*) FROM edge_evidence v WHERE v.edge_id = edges.id),
            evidence_count = (SELECT COUNT(*) FROM edge_evidence v WHERE v.edge_id = edges.id),
            context_evidence_count = (
                SELECT COUNT(DISTINCT x.base) FROM edge_contexts x WHERE x.edge_id = edges.id
            );

        CREATE INDEX idx_nodes_rank
            ON nodes(paper_count DESC, relation_count DESC, chunk_count DESC);
        CREATE INDEX idx_nodes_type ON nodes(entity_type, label_norm);
        CREATE INDEX idx_nodes_label ON nodes(label_norm);
        CREATE INDEX idx_alias_norm ON node_aliases(alias_norm);
        CREATE INDEX idx_edges_subject
            ON edges(subject_id, paper_count DESC, evidence_count DESC);
        CREATE INDEX idx_edges_object
            ON edges(object_id, paper_count DESC, evidence_count DESC);
        CREATE INDEX idx_edges_rank
            ON edges(paper_count DESC, evidence_count DESC);
        CREATE INDEX idx_edges_predicate_support
            ON edges(predicate, paper_count DESC, evidence_count DESC);
        CREATE INDEX idx_evidence_edge ON edge_evidence(edge_id, base);
        CREATE INDEX idx_context_cell ON edge_contexts(cell_node_id, edge_id);
        CREATE INDEX idx_evidence_entities_base
            ON evidence_entities(base, start_offset, end_offset);
        ANALYZE;
        """
    )


def _write_entity_index(connection: sqlite3.Connection, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            fileobj=raw,
            mode="wb",
            compresslevel=6,
            mtime=0,
        ) as zipped:
            nodes = connection.execute(
                "SELECT * FROM nodes ORDER BY entity_type, label_norm, id"
            )
            for node in nodes:
                aliases = [
                    row["alias"]
                    for row in connection.execute(
                        """
                        SELECT alias FROM node_aliases
                        WHERE node_id = ?
                        ORDER BY alias_kind DESC, count DESC, alias COLLATE NOCASE
                        """,
                        (node["id"],),
                    )
                ]
                relations: list[dict[str, Any]] = []
                for edge in connection.execute(
                    """
                    SELECT e.*, s.label AS subject_label, o.label AS object_label
                    FROM edges e
                    JOIN nodes s ON s.id = e.subject_id
                    JOIN nodes o ON o.id = e.object_id
                    WHERE e.subject_id = ? OR e.object_id = ?
                    ORDER BY e.evidence_count DESC, e.predicate, e.id
                    """,
                    (node["id"], node["id"]),
                ):
                    evidence = [
                        {
                            "base": row["base"],
                            "canonical_id": row["canonical_id"],
                            "pmid": row["pmid"],
                            "pmcid": row["pmcid"],
                        }
                        for row in connection.execute(
                            """
                            SELECT base, canonical_id, pmid, pmcid
                            FROM edge_evidence WHERE edge_id = ? ORDER BY base
                            """,
                            (edge["id"],),
                        )
                    ]
                    predicate = str(edge["predicate"])
                    relations.append(
                        {
                            "id": edge["id"],
                            "subject": edge["subject_id"],
                            "subject_label": edge["subject_label"],
                            "predicate": predicate,
                            "object": edge["object_id"],
                            "object_label": edge["object_label"],
                            "direction": (
                                "outgoing"
                                if edge["subject_id"] == node["id"]
                                else "incoming"
                            ),
                            "stats": {
                                "paper_count": edge["paper_count"],
                                "chunk_count": edge["chunk_count"],
                                "evidence_count": edge["evidence_count"],
                                "cell_context_evidence_count": edge[
                                    "context_evidence_count"
                                ],
                            },
                            "evidence": evidence,
                        }
                    )
                undirected_count = 0
                row = {
                    "schema": ENTITY_INDEX_SCHEMA,
                    "id": node["id"],
                    "normalized_id": node["normalized_id"],
                    "obj": node["entity_type"],
                    "label": node["label"],
                    "canonical_name": node["canonical_name"],
                    "canonical_id_type": node["canonical_id_type"],
                    "hgnc_id": node["hgnc_id"],
                    "ncbi_gene_id": node["ncbi_gene_id"],
                    "chebi_id": node["chebi_id"],
                    "mesh_id": node["mesh_id"],
                    "uniprot_ids": node["uniprot_ids"],
                    "source_concept_id": node["source_concept_id"],
                    "normalization_source": node["normalization_source"],
                    "normalization_status": node["normalization_status"],
                    "normalization_match_method": node[
                        "normalization_match_method"
                    ],
                    "chebi_release": node["chebi_release"],
                    "mentions": aliases,
                    "stats": {
                        "paper_count": node["paper_count"],
                        "chunk_count": node["chunk_count"],
                        "relation_count": node["relation_count"],
                        "incoming_count": node["incoming_count"],
                        "outgoing_count": node["outgoing_count"],
                        "undirected_count": undirected_count,
                        "cell_context_count": node["context_count"],
                    },
                    "relations": relations,
                }
                zipped.write((_json(row) + "\n").encode("utf-8"))
                count += 1
        raw.flush()
        os.fsync(raw.fileno())
    return count


def build_interaction_network(
    *,
    relation_path: Path,
    chunks_path: Path,
    annotations_path: Path,
    graph_path: Path,
    entity_index_path: Path,
    progress: ProgressCallback | None = None,
    commit_every: int = 500,
) -> BuildResult:
    """Build a globally normalized graph while holding only one chunk in memory."""

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.unlink(missing_ok=True)
    entity_index_path.unlink(missing_ok=True)

    stats: dict[str, Any] = {
        "pipeline_version": NETWORK_PIPELINE_VERSION,
        "graph_schema": NETWORK_GRAPH_SCHEMA,
        "entity_index_schema": ENTITY_INDEX_SCHEMA,
        "chunk_count": 0,
        "paper_count": 0,
        "relation_evidence_count": 0,
        "invalid_relation_count": 0,
        "unresolved_entity_count": 0,
        "cell_context_link_count": 0,
        "evidence_entity_span_count": 0,
        "invalid_evidence_span_count": 0,
    }
    connection = sqlite3.connect(graph_path)
    connection.row_factory = sqlite3.Row
    try:
        _create_schema(connection)
        connection.execute("BEGIN")
        rows = zip_longest(
            iter_jsonl(relation_path),
            iter_jsonl(chunks_path),
            iter_jsonl(annotations_path),
        )
        for row_number, row_group in enumerate(rows, start=1):
            relation_row, source_row, annotation_row = row_group
            if relation_row is None or source_row is None or annotation_row is None:
                raise ValueError(
                    "Stage 3 relations, Stage 1 chunks, and Stage 2 annotations "
                    "have different row counts."
                )
            if not _aligned(relation_row, source_row) or not _aligned(
                relation_row, annotation_row
            ):
                raise ValueError(
                    "Stage 3 relations, Stage 1 chunks, and Stage 2 annotations "
                    f"are misaligned at row {row_number}."
                )

            stats["chunk_count"] = row_number
            base = _text(relation_row.get("base")) or f"row-{row_number:010d}"
            paper_id = _paper_id(relation_row)
            chunk_text = _source_text(source_row.get("chunk"))

            annotation_values = annotation_row.get("annotations")
            annotations = prefer_longest_spans(
                annotation_values if isinstance(annotation_values, list) else []
            )

            entities_raw = relation_row.get("entities")
            relations_raw = relation_row.get("relations")
            entities = entities_raw if isinstance(entities_raw, list) else []
            relations = relations_raw if isinstance(relations_raw, list) else []
            tag_to_node: dict[str, str] = {}
            for raw_entity in entities:
                if not isinstance(raw_entity, Mapping):
                    continue
                tag = _text(raw_entity.get("id"))
                entity_type = _entity_type(raw_entity)
                expected_prefix = _TYPE_PREFIX.get(entity_type, "")
                if (
                    _TAG_RE.fullmatch(tag) is None
                    or not expected_prefix
                    or not tag.startswith(expected_prefix)
                ):
                    stats["unresolved_entity_count"] += 1
                    continue
                node_id = _upsert_node(
                    connection,
                    raw_entity,
                    paper_id=paper_id,
                    base=base,
                )
                if node_id:
                    tag_to_node[tag] = node_id
                else:
                    stats["unresolved_entity_count"] += 1

            has_valid_relation = False
            for raw_relation in relations:
                if not isinstance(raw_relation, Mapping):
                    stats["invalid_relation_count"] += 1
                    continue
                subject_tag = _text(raw_relation.get("subject"))
                object_tag = _text(raw_relation.get("object"))
                predicate = _norm(raw_relation.get("predicate"))
                subject_id = tag_to_node.get(subject_tag, "")
                object_id = tag_to_node.get(object_tag, "")
                if (
                    not subject_id
                    or not object_id
                    or subject_id == object_id
                    or not relation_allowed(
                        subject_tag,
                        predicate,
                        object_tag,
                    )
                ):
                    stats["invalid_relation_count"] += 1
                    continue

                edge_id = _edge_id(subject_id, predicate, object_id)
                has_valid_relation = True
                connection.execute(
                    """
                    INSERT OR IGNORE INTO edges
                        (id, subject_id, predicate, object_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (edge_id, subject_id, predicate, object_id),
                )
                if paper_id:
                    connection.execute(
                        "INSERT OR IGNORE INTO edge_papers (edge_id, paper_id) VALUES (?, ?)",
                        (edge_id, paper_id),
                    )
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO edge_evidence (
                        edge_id, base, doc_key, canonical_id, pmid, pmcid,
                        journal, pub_year, section_type, chunk_id, chunk_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge_id,
                        base,
                        _text(relation_row.get("doc_key")),
                        _text(relation_row.get("canonical_id")),
                        _text(relation_row.get("pmid")),
                        _text(relation_row.get("pmcid")),
                        _text(relation_row.get("journal")),
                        _text(relation_row.get("pub_year")),
                        _text(relation_row.get("section_type")),
                        _text(relation_row.get("chunk_id")),
                        chunk_text,
                    ),
                )
                if cursor.rowcount:
                    stats["relation_evidence_count"] += 1

                raw_context = raw_relation.get("cell_context")
                context_tags = raw_context if isinstance(raw_context, list) else []
                for context_tag in context_tags:
                    cell_id = tag_to_node.get(_text(context_tag), "")
                    if not cell_id:
                        continue
                    cell_type = connection.execute(
                        "SELECT entity_type FROM nodes WHERE id = ?", (cell_id,)
                    ).fetchone()
                    if cell_type is None or cell_type["entity_type"] != "cell":
                        continue
                    inserted = connection.execute(
                        """
                        INSERT OR IGNORE INTO edge_contexts
                            (edge_id, base, cell_node_id)
                        VALUES (?, ?, ?)
                        """,
                        (edge_id, base, cell_id),
                    )
                    if inserted.rowcount:
                        stats["cell_context_link_count"] += 1

            # Evidence pages can only show chunks that contain at least one
            # validated relation.  Store all Stage 2 spans for those chunks, but
            # avoid retaining annotations from relation-free chunks.  This keeps
            # the SQLite artifact bounded without weakening evidence highlighting.
            if has_valid_relation:
                for raw_annotation in annotations:
                    if not isinstance(raw_annotation, Mapping):
                        continue
                    span = _annotation_span(raw_annotation, chunk_text)
                    node_id, entity_type, normalized_id = global_entity_id(
                        raw_annotation
                    )
                    if span is None or not entity_type:
                        stats["invalid_evidence_span_count"] += 1
                        continue
                    start_offset, end_offset = span
                    mention = _source_text(raw_annotation.get("mention"))
                    if not mention:
                        mention = chunk_text[start_offset:end_offset]
                    inserted_span = connection.execute(
                        """
                        INSERT OR IGNORE INTO evidence_entities (
                            base, start_offset, end_offset, entity_type,
                            normalized_id, node_id, mention, preferred_label
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            base,
                            start_offset,
                            end_offset,
                            entity_type,
                            normalized_id,
                            node_id,
                            mention,
                            _text(raw_annotation.get("preferred_label")),
                        ),
                    )
                    if inserted_span.rowcount:
                        stats["evidence_entity_span_count"] += 1

            if row_number % max(1, commit_every) == 0:
                connection.commit()
                connection.execute("BEGIN")
                if progress:
                    progress(
                        row_number,
                        f"Indexed {row_number:,} relation rows.",
                        dict(stats),
                    )
        connection.commit()

        if progress:
            progress(
                int(stats["chunk_count"]),
                "Finalizing global node and edge statistics.",
                dict(stats),
            )
        _finalize_graph(connection)
        stats["paper_count"] = int(
            connection.execute(
                "SELECT COUNT(DISTINCT paper_id) FROM edge_papers"
            ).fetchone()[0]
        )
        stats["node_count"] = int(
            connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        )
        stats["edge_count"] = int(
            connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        )
        stats["evidence_count"] = int(
            connection.execute("SELECT COUNT(*) FROM edge_evidence").fetchone()[0]
        )
        stats["entity_counts"] = {
            row["entity_type"]: int(row["count"])
            for row in connection.execute(
                "SELECT entity_type, COUNT(*) AS count FROM nodes GROUP BY entity_type"
            )
        }
        stats["predicate_counts"] = {
            row["predicate"]: int(row["count"])
            for row in connection.execute(
                "SELECT predicate, COUNT(*) AS count FROM edges GROUP BY predicate"
            )
        }
        direction_counts: dict[str, int] = {}
        for row in connection.execute(
            """
            SELECT e.predicate, s.entity_type AS subject_type,
                   o.entity_type AS object_type, COUNT(*) AS count
            FROM edges e
            JOIN nodes s ON s.id = e.subject_id
            JOIN nodes o ON o.id = e.object_id
            GROUP BY e.predicate, s.entity_type, o.entity_type
            """
        ):
            key = f"{row['subject_type']}->{row['object_type']}"
            direction_counts[key] = direction_counts.get(key, 0) + int(row["count"])
        stats["direction_counts"] = direction_counts
        for key, value in {
            "schema": NETWORK_GRAPH_SCHEMA,
            "pipeline_version": NETWORK_PIPELINE_VERSION,
            "stats": stats,
        }.items():
            connection.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, _json(value)),
            )
        connection.commit()

        if progress:
            progress(
                int(stats["chunk_count"]),
                "Writing the compressed one-row-per-entity index.",
                dict(stats),
            )
        stats["entity_index_row_count"] = _write_entity_index(
            connection, entity_index_path
        )
        connection.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('stats', ?)",
            (_json(stats),),
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
    finally:
        connection.close()

    return BuildResult(
        graph_path=graph_path,
        entity_index_path=entity_index_path,
        stats=stats,
    )


_ENDPOINT_RE = re.compile(r"^(.*?)\s*\(([^()]+)\)\s*$")


def _endpoint_entity(value: Any) -> tuple[dict[str, Any], str]:
    """Build a normalized entity from ``label (CURIE)`` relation text."""

    raw = _text(value)
    match = _ENDPOINT_RE.fullmatch(raw)
    if match is None:
        return {}, ""
    label = _text(match.group(1))
    identifier = _text(match.group(2))
    prefix = identifier.split(":", 1)[0].casefold() if ":" in identifier else ""
    if prefix == "cl":
        entity_type, field = "cell", "concept_id"
    elif prefix == "hgnc":
        entity_type, field = "gene", "hgnc_id"
    elif prefix in {"ncbigene", "geneid", "gene"}:
        entity_type, field = "gene", "ncbi_gene_id"
    elif prefix == "chebi":
        entity_type, field = "hormone", "chebi_id"
    elif prefix == "mesh":
        entity_type, field = "hormone", "mesh_id"
    else:
        return {}, ""
    _namespace, canonical_value = canonical_identifier(entity_type, field, identifier)
    canonical_value = canonical_value or identifier
    entity = {
        "obj": entity_type,
        field: canonical_value,
        "normalized_id": canonical_value,
        "mention": label or canonical_value,
        "preferred_label": label or canonical_value,
    }
    return entity, canonical_value


def _annotation_identity(annotation: Mapping[str, Any]) -> str:
    _entity_type_value, _source, identity = _entity_identity(annotation)
    return identity.casefold() if identity else ""


def _merge_endpoint_annotation(
    endpoint: Mapping[str, Any],
    annotation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Prefer complete Stage 2 metadata while forcing the relation's explicit ID."""

    result = dict(annotation or {})
    result.update(
        {
            key: value
            for key, value in endpoint.items()
            if key in {"obj", "concept_id", "hgnc_id", "ncbi_gene_id", "chebi_id", "mesh_id", "normalized_id"}
            and value not in (None, "")
        }
    )
    if not _text(result.get("mention")):
        result["mention"] = endpoint.get("mention")
    if not _text(result.get("preferred_label")):
        result["preferred_label"] = endpoint.get("preferred_label")
    return result


def _finalize_nested_build(
    connection: sqlite3.Connection,
    stats: dict[str, Any],
    entity_index_path: Path,
    progress: ProgressCallback | None,
) -> None:
    if progress:
        progress(
            int(stats.get("chunk_count") or 0),
            "Finalizing global node and edge statistics.",
            dict(stats),
        )
    _finalize_graph(connection)
    stats["paper_count"] = int(
        connection.execute(
            "SELECT COUNT(DISTINCT paper_id) FROM edge_papers"
        ).fetchone()[0]
    )
    stats["node_count"] = int(
        connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    )
    stats["edge_count"] = int(
        connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    )
    stats["evidence_count"] = int(
        connection.execute("SELECT COUNT(*) FROM edge_evidence").fetchone()[0]
    )
    stats["entity_counts"] = {
        row["entity_type"]: int(row["count"])
        for row in connection.execute(
            "SELECT entity_type, COUNT(*) AS count FROM nodes GROUP BY entity_type"
        )
    }
    stats["predicate_counts"] = {
        row["predicate"]: int(row["count"])
        for row in connection.execute(
            "SELECT predicate, COUNT(*) AS count FROM edges GROUP BY predicate"
        )
    }
    direction_counts: dict[str, int] = {}
    for row in connection.execute(
        """
        SELECT e.predicate, s.entity_type AS subject_type,
               o.entity_type AS object_type, COUNT(*) AS count
        FROM edges e
        JOIN nodes s ON s.id = e.subject_id
        JOIN nodes o ON o.id = e.object_id
        GROUP BY e.predicate, s.entity_type, o.entity_type
        """
    ):
        key = f"{row['subject_type']}->{row['object_type']}"
        direction_counts[key] = direction_counts.get(key, 0) + int(row["count"])
    stats["direction_counts"] = direction_counts
    for key, value in {
        "schema": NETWORK_GRAPH_SCHEMA,
        "pipeline_version": NETWORK_PIPELINE_VERSION,
        "stats": stats,
    }.items():
        connection.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, _json(value)),
        )
    connection.commit()

    if progress:
        progress(
            int(stats.get("chunk_count") or 0),
            "Writing the compressed one-row-per-entity index.",
            dict(stats),
        )
    stats["entity_index_row_count"] = _write_entity_index(
        connection, entity_index_path
    )
    connection.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('stats', ?)",
        (_json(stats),),
    )
    connection.commit()
    connection.execute("PRAGMA optimize")


def build_interaction_network_from_final_annotations(
    *,
    prediction_paths: Sequence[Path],
    graph_path: Path,
    entity_index_path: Path,
    progress: ProgressCallback | None = None,
    commit_every: int = 500,
) -> BuildResult:
    """Build a graph directly from one or more final one-row-per-paper files.

    Only relation endpoints are inserted as graph nodes. The same normalized
    ``(subject, predicate, object)`` is processed once per paper, so a paper can
    support an edge only once even when the relation is repeated across chunks.
    """

    sources = [Path(value).expanduser().resolve() for value in prediction_paths]
    if not sources:
        raise ValueError("At least one final prediction file is required.")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.unlink(missing_ok=True)
    entity_index_path.unlink(missing_ok=True)
    stats: dict[str, Any] = {
        "pipeline_version": NETWORK_PIPELINE_VERSION,
        "graph_schema": NETWORK_GRAPH_SCHEMA,
        "entity_index_schema": ENTITY_INDEX_SCHEMA,
        "source_file_count": len(sources),
        "source_paper_count": 0,
        "chunk_count": 0,
        "paper_count": 0,
        "unique_paper_relation_count": 0,
        "duplicate_paper_relation_count": 0,
        "relation_evidence_count": 0,
        "invalid_relation_count": 0,
        "unresolved_entity_count": 0,
        "cell_context_link_count": 0,
        "evidence_entity_span_count": 0,
        "invalid_evidence_span_count": 0,
    }

    connection = sqlite3.connect(graph_path)
    connection.row_factory = sqlite3.Row
    try:
        _create_schema(connection)
        connection.execute("BEGIN")
        for source_index, source in enumerate(sources, start=1):
            for paper_number, paper_row in enumerate(iter_jsonl(source), start=1):
                stats["source_paper_count"] += 1
                paper_id = _paper_id(paper_row) or f"source-{source_index}:paper-{paper_number}"
                raw_chunks = paper_row.get("chunks")
                chunks = raw_chunks if isinstance(raw_chunks, list) else []
                paper_annotations: dict[str, Mapping[str, Any]] = {}
                for raw_chunk in chunks:
                    if not isinstance(raw_chunk, Mapping):
                        continue
                    raw_annotations = raw_chunk.get("annotations")
                    annotations = prefer_longest_spans(
                        raw_annotations if isinstance(raw_annotations, list) else []
                    )
                    for annotation in annotations:
                        if not isinstance(annotation, Mapping):
                            continue
                        identity = _annotation_identity(annotation)
                        if identity:
                            paper_annotations.setdefault(identity, annotation)

                seen_paper_relations: set[tuple[str, str, str]] = set()
                for chunk_index, raw_chunk in enumerate(chunks, start=1):
                    if not isinstance(raw_chunk, Mapping):
                        continue
                    stats["chunk_count"] += 1
                    base = (
                        _text(raw_chunk.get("base"))
                        or f"{paper_id}:chunk-{_text(raw_chunk.get('chunk_id')) or chunk_index}"
                    )
                    chunk_text = _source_text(
                        raw_chunk.get("text")
                        if raw_chunk.get("text") is not None
                        else raw_chunk.get("chunk")
                    )
                    raw_annotations = raw_chunk.get("annotations")
                    annotations = prefer_longest_spans(
                        raw_annotations if isinstance(raw_annotations, list) else []
                    )
                    local_annotations: dict[str, Mapping[str, Any]] = {}
                    for annotation in annotations:
                        if not isinstance(annotation, Mapping):
                            continue
                        identity = _annotation_identity(annotation)
                        if identity:
                            local_annotations.setdefault(identity, annotation)

                    raw_relations = raw_chunk.get("relations")
                    relations = raw_relations if isinstance(raw_relations, list) else []
                    has_valid_relation = False
                    for raw_relation in relations:
                        if not isinstance(raw_relation, Mapping):
                            stats["invalid_relation_count"] += 1
                            continue
                        subject_endpoint, subject_identity = _endpoint_entity(
                            raw_relation.get("subject")
                        )
                        object_endpoint, object_identity = _endpoint_entity(
                            raw_relation.get("object")
                        )
                        predicate = _norm(raw_relation.get("predicate"))
                        if not subject_endpoint or not object_endpoint or not predicate:
                            stats["invalid_relation_count"] += 1
                            continue
                        subject_annotation = local_annotations.get(
                            subject_identity.casefold()
                        ) or paper_annotations.get(subject_identity.casefold())
                        object_annotation = local_annotations.get(
                            object_identity.casefold()
                        ) or paper_annotations.get(object_identity.casefold())
                        subject_entity = _merge_endpoint_annotation(
                            subject_endpoint, subject_annotation
                        )
                        object_entity = _merge_endpoint_annotation(
                            object_endpoint, object_annotation
                        )
                        subject_global, subject_type, _subject_value = global_entity_id(
                            subject_entity
                        )
                        object_global, object_type, _object_value = global_entity_id(
                            object_entity
                        )
                        subject_tag = f"{_TYPE_PREFIX.get(subject_type, '')}1"
                        object_tag = f"{_TYPE_PREFIX.get(object_type, '')}2"
                        if (
                            not subject_global
                            or not object_global
                            or subject_global == object_global
                            or not relation_allowed(subject_tag, predicate, object_tag)
                        ):
                            stats["invalid_relation_count"] += 1
                            continue

                        paper_relation_key = (
                            subject_global,
                            predicate,
                            object_global,
                        )
                        if paper_relation_key in seen_paper_relations:
                            stats["duplicate_paper_relation_count"] += 1
                            continue
                        seen_paper_relations.add(paper_relation_key)

                        subject_id = _upsert_node(
                            connection,
                            subject_entity,
                            paper_id=paper_id,
                            base=base,
                        )
                        object_id = _upsert_node(
                            connection,
                            object_entity,
                            paper_id=paper_id,
                            base=base,
                        )
                        if not subject_id or not object_id:
                            stats["unresolved_entity_count"] += 1
                            continue

                        edge_id = _edge_id(subject_id, predicate, object_id)
                        connection.execute(
                            """
                            INSERT OR IGNORE INTO edges
                                (id, subject_id, predicate, object_id)
                            VALUES (?, ?, ?, ?)
                            """,
                            (edge_id, subject_id, predicate, object_id),
                        )
                        connection.execute(
                            "INSERT OR IGNORE INTO edge_papers (edge_id, paper_id) VALUES (?, ?)",
                            (edge_id, paper_id),
                        )
                        cursor = connection.execute(
                            """
                            INSERT OR IGNORE INTO edge_evidence (
                                edge_id, base, doc_key, canonical_id, pmid, pmcid,
                                journal, pub_year, section_type, chunk_id, chunk_text
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                edge_id,
                                base,
                                _text(paper_row.get("doc_key")),
                                _text(paper_row.get("canonical_id")),
                                _text(paper_row.get("pmid")),
                                _text(paper_row.get("pmcid")),
                                _text(paper_row.get("journal")),
                                _text(paper_row.get("pub_year")),
                                _text(raw_chunk.get("section_type")),
                                _text(raw_chunk.get("chunk_id")),
                                chunk_text,
                            ),
                        )
                        if cursor.rowcount:
                            stats["relation_evidence_count"] += 1
                        stats["unique_paper_relation_count"] += 1
                        has_valid_relation = True

                    if has_valid_relation:
                        for raw_annotation in annotations:
                            if not isinstance(raw_annotation, Mapping):
                                continue
                            span = _annotation_span(raw_annotation, chunk_text)
                            node_id, entity_type, normalized_id = global_entity_id(
                                raw_annotation
                            )
                            if span is None or not entity_type:
                                stats["invalid_evidence_span_count"] += 1
                                continue
                            start_offset, end_offset = span
                            mention = _source_text(raw_annotation.get("mention"))
                            if not mention:
                                mention = chunk_text[start_offset:end_offset]
                            inserted_span = connection.execute(
                                """
                                INSERT OR IGNORE INTO evidence_entities (
                                    base, start_offset, end_offset, entity_type,
                                    normalized_id, node_id, mention, preferred_label
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    base,
                                    start_offset,
                                    end_offset,
                                    entity_type,
                                    normalized_id,
                                    node_id,
                                    mention,
                                    _text(raw_annotation.get("preferred_label")),
                                ),
                            )
                            if inserted_span.rowcount:
                                stats["evidence_entity_span_count"] += 1

                    if int(stats["chunk_count"]) % max(1, commit_every) == 0:
                        connection.commit()
                        connection.execute("BEGIN")
                        if progress:
                            progress(
                                int(stats["chunk_count"]),
                                f"Indexed {int(stats['chunk_count']):,} final-annotation chunks.",
                                dict(stats),
                            )
        connection.commit()
        _finalize_nested_build(connection, stats, entity_index_path, progress)
    finally:
        connection.close()

    return BuildResult(
        graph_path=graph_path,
        entity_index_path=entity_index_path,
        stats=stats,
    )


__all__ = [
    "BuildResult",
    "build_interaction_network",
    "build_interaction_network_from_final_annotations",
    "global_entity_id",
]
