"""Lazy SQLite graph queries and PyVis-compatible browser payloads."""

from __future__ import annotations

import html
import json
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


from backend.config import settings
from backend.runtime import run_registry
from backend.services.cell_hierarchy import (
    CellHierarchyError,
    CellHierarchyTermNotFound,
    cell_hierarchy_service,
    normalize_cl_id,
    ontology_node_id,
)

_NODE_COLORS = {
    "cell": {
        "background": "#2F9D78",
        "border": "#2F9D78",
        "highlight": {"background": "#52B996", "border": "#52B996"},
        "hover": {"background": "#45AE8B", "border": "#45AE8B"},
    },
    "gene": {
        "background": "#E99A2E",
        "border": "#E99A2E",
        "highlight": {"background": "#F5B85E", "border": "#F5B85E"},
        "hover": {"background": "#EFA847", "border": "#EFA847"},
    },
    "hormone": {
        "background": "#9B6AD6",
        "border": "#9B6AD6",
        "highlight": {"background": "#B891E5", "border": "#B891E5"},
        "hover": {"background": "#AA7FDD", "border": "#AA7FDD"},
    },
}
_NEGATIVE = {"inhibition", "downregulation"}
_EDGE_COLORS = {
    "binding": "#4F72B8",
    "biosynthesis": "#7A63B8",
    "secreted": "#2D8C87",
}
_ONE_MIB = 1024 * 1024
_HIERARCHY_NODE_COLOR = {
    "background": "#E8F7F2",
    "border": "#24866A",
    "highlight": {"background": "#D6F1E8", "border": "#176B53"},
    "hover": {"background": "#DFF4ED", "border": "#176B53"},
}


class _FallbackNetwork:
    """Small test/development fallback; deployments install the PyVis dependency."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.options: dict[str, Any] = {}

    def add_node(self, node_id: str, **options: Any) -> None:
        self.nodes.append({"id": node_id, **options})

    def add_edge(self, source: str, target: str, **options: Any) -> None:
        self.edges.append({"from": source, "to": target, **options})

    def set_options(self, options: str) -> None:
        self.options = json.loads(options)


def _network_class() -> tuple[type[Any], str]:
    try:
        from pyvis.network import Network

        return Network, "pyvis"
    except ImportError:  # pragma: no cover - exercised only in stripped test images
        return _FallbackNetwork, "pyvis-compatible-fallback"


def pyvis_asset_path(filename: str) -> Path:
    """Locate the vis-network assets bundled with PyVis 0.3.2."""

    if filename not in {"vis-network.min.js", "vis-network.css"}:
        raise FileNotFoundError(filename)
    try:
        import pyvis
    except ImportError as exc:
        raise RuntimeError("PyVis is not installed. Install requirements.txt.") from exc
    root = Path(pyvis.__file__).resolve().parent
    candidates = [
        root / "lib" / "vis-9.1.2" / filename,
        root / "templates" / "lib" / "vis-9.1.2" / filename,
        root / "lib" / "vis-9.0.4" / filename,
        root / "templates" / "lib" / "vis-9.0.4" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"The PyVis asset {filename} was not found.")


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _node_title(node: Mapping[str, Any]) -> str:
    kind = str(node.get("entity_type") or "entity").replace("gene", "gene/protein")
    return (
        f"<strong>{html.escape(str(node.get('label') or node.get('id') or ''))}</strong><br>"
        f"{html.escape(kind.title())}<br>"
        f"ID: {html.escape(str(node.get('normalized_id') or 'unresolved'))}<br>"
        f"Relations: {int(node.get('relation_count') or 0):,}<br>"
        f"Papers: {int(node.get('paper_count') or 0):,}<br>"
        f"Evidence chunks: {int(node.get('chunk_count') or 0):,}"
    )


def _edge_title(edge: Mapping[str, Any]) -> str:
    predicate = str(edge.get("predicate") or "relation")
    separator = " — " if predicate == "binding" else " → "
    return (
        f"<strong>{html.escape(predicate)}</strong><br>"
        f"{html.escape(str(edge.get('subject_label') or edge.get('subject_id') or ''))}"
        f"{separator}"
        f"{html.escape(str(edge.get('object_label') or edge.get('object_id') or ''))}<br>"
        f"Papers: {int(edge.get('paper_count') or 0):,}<br>"
        f"Evidence chunks: {int(edge.get('evidence_count') or 0):,}<br>"
        f"Cell-context evidence: {int(edge.get('context_evidence_count') or 0):,}"
    )


def _edge_color(predicate: str) -> str:
    if predicate in _NEGATIVE:
        return "#C64B4B"
    return _EDGE_COLORS.get(predicate, "#2F8B68")


def _node_payload(node: Mapping[str, Any]) -> dict[str, Any]:
    relation_count = int(node.get("relation_count") or 0)
    paper_count = int(node.get("paper_count") or 0)
    return {
        "id": str(node["id"]),
        "label": str(node.get("label") or node["id"]),
        "title": _node_title(node),
        "group": str(node.get("entity_type") or "entity"),
        "node_kind": "interaction",
        "ontology_only": False,
        "entity_type": str(node.get("entity_type") or ""),
        "normalized_id": str(node.get("normalized_id") or ""),
        "cl_id": normalize_cl_id(node.get("normalized_id")),
        "paper_count": paper_count,
        "chunk_count": int(node.get("chunk_count") or 0),
        "relation_count": relation_count,
        "incoming_count": int(node.get("incoming_count") or 0),
        "outgoing_count": int(node.get("outgoing_count") or 0),
        "context_count": int(node.get("context_count") or 0),
        "shape": "dot",
        "size": round(12 + min(28, 5.5 * math.log1p(paper_count)), 2),
        "mass": round(1 + min(5, math.log1p(paper_count)), 2),
        "color": _NODE_COLORS.get(str(node.get("entity_type")), "#78909C"),
        "font": {"face": "Inter, Arial, sans-serif", "size": 14, "color": "#17324D"},
        # Interaction nodes are filled circles without an outline.  Keep both
        # normal and selected widths at zero so hover/selection does not add a ring.
        "borderWidth": 0,
        "borderWidthSelected": 0,
    }


def _edge_payload(edge: Mapping[str, Any]) -> dict[str, Any]:
    evidence_count = int(edge.get("evidence_count") or 0)
    paper_count = int(edge.get("paper_count") or 0)
    predicate = str(edge.get("predicate") or "relation")
    directed = predicate != "binding"
    color = _edge_color(predicate)
    return {
        "id": str(edge["id"]),
        "from": str(edge["subject_id"]),
        "to": str(edge["object_id"]),
        # Show the biological predicate directly on the interaction edge.
        "label": predicate,
        "predicate": predicate,
        "edge_kind": "interaction",
        "title": _edge_title(edge),
        "paper_count": paper_count,
        "chunk_count": int(edge.get("chunk_count") or 0),
        "evidence_count": evidence_count,
        "context_evidence_count": int(edge.get("context_evidence_count") or 0),
        "arrows": {"to": {"enabled": directed, "scaleFactor": 0.75}},
        "color": {"color": color, "highlight": color, "hover": color, "opacity": 0.86},
        "width": round(1.5 + min(7, 1.6 * math.log1p(paper_count)), 2),
        "smooth": {"enabled": True, "type": "dynamic", "roundness": 0.22},
        "font": {
            "size": 11,
            "face": "Inter, Arial, sans-serif",
            "color": "#29445C",
            "align": "middle",
            "strokeWidth": 3,
            "strokeColor": "#FFFFFF",
        },
    }



def _hierarchy_node_title(term: Mapping[str, Any], *, interaction_node: bool) -> str:
    label = html.escape(str(term.get("label") or term.get("concept_id") or "Cell Ontology term"))
    concept_id = html.escape(str(term.get("concept_id") or ""))
    role = str(term.get("role") or "term").replace("_", " ").title()
    source = "Extracted interaction node" if interaction_node else "Ontology-only hierarchy term"
    definition = str(term.get("definition") or "").strip()
    definition_line = f"<br>{html.escape(definition[:280])}" if definition else ""
    return (
        f"<strong>{label}</strong><br>"
        f"Cell Ontology · {concept_id}<br>"
        f"{html.escape(role)} · {html.escape(source)}"
        f"{definition_line}"
    )


def _hierarchy_only_node_payload(term: Mapping[str, Any]) -> dict[str, Any]:
    concept_id = normalize_cl_id(term.get("concept_id"))
    node_id = ontology_node_id(concept_id)
    synonyms = [
        str(value)
        for value in term.get("synonyms", [])
        if isinstance(value, str) and value.strip()
    ]
    return {
        "id": node_id,
        "label": str(term.get("label") or concept_id),
        "title": _hierarchy_node_title(term, interaction_node=False),
        "group": "cell_hierarchy",
        "node_kind": "ontology",
        "ontology_only": True,
        "entity_type": "cell",
        "normalized_id": concept_id,
        "cl_id": concept_id,
        "hierarchy_role": str(term.get("role") or "term"),
        "definition": str(term.get("definition") or ""),
        "synonyms": synonyms,
        "parent_count": int(term.get("parent_count") or 0),
        "child_count": int(term.get("child_count") or 0),
        "paper_count": 0,
        "chunk_count": 0,
        "relation_count": 0,
        "incoming_count": 0,
        "outgoing_count": 0,
        "context_count": 0,
        "shape": "box",
        "margin": {"top": 9, "right": 12, "bottom": 9, "left": 12},
        "color": _HIERARCHY_NODE_COLOR,
        "font": {
            "face": "Inter, Arial, sans-serif",
            "size": 13,
            "color": "#17324D",
            "multi": False,
        },
        "borderWidth": 2,
        "borderWidthSelected": 3,
        "mass": 1.25,
    }



_GRAPH_OPTIONS: dict[str, Any] = {
    "autoResize": True,
    "interaction": {
        "hover": True,
        "hoverConnectedEdges": True,
        "keyboard": {"enabled": True, "bindToWindow": False},
        "multiselect": True,
        "navigationButtons": True,
        "tooltipDelay": 180,
    },
    "layout": {"improvedLayout": True, "randomSeed": 17},
    "physics": {
        "enabled": True,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -58,
            "centralGravity": 0.012,
            "springLength": 125,
            "springConstant": 0.055,
            "damping": 0.5,
            "avoidOverlap": 0.55,
        },
        "stabilization": {"enabled": True, "iterations": 650, "updateInterval": 25},
        "minVelocity": 0.75,
    },
    "nodes": {"chosen": True, "borderWidth": 0, "borderWidthSelected": 0},
    "edges": {"chosen": True, "selectionWidth": 2},
}


class NetworkRepository:
    def _graph_path(self, run_id: str) -> Path:
        # Stage 4 is intentionally ephemeral. The graph exists only inside this
        # process's private run directory and is never uploaded to shared storage.
        return run_registry.graph_path(run_id)

    @contextmanager
    def connection(self, job_id: str) -> Iterator[sqlite3.Connection]:
        path = self._graph_path(job_id)
        # The artifact path has already been resolved and verified. Opening the
        # concrete path and enabling query_only is portable across local Windows,
        # macOS/Linux, and Railway volume mounts.
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=15000")
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _nodes_by_ids(
        connection: sqlite3.Connection, node_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        placeholders = ",".join("?" for _ in node_ids)
        rows = connection.execute(
            f"SELECT * FROM nodes WHERE id IN ({placeholders})",  # noqa: S608
            tuple(node_ids),
        ).fetchall()
        order = {value: index for index, value in enumerate(node_ids)}
        result = [_row_dict(row) for row in rows]
        result.sort(key=lambda row: order.get(str(row["id"]), len(order)))
        return result

    @staticmethod
    def _edges_between(
        connection: sqlite3.Connection,
        node_ids: Sequence[str],
        *,
        relation_support_min: int = 1,
    ) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        values = ",".join("(?)" for _ in node_ids)
        rows = connection.execute(
            f"""
            WITH selected(id) AS (VALUES {values})
            SELECT e.*, s.label AS subject_label, o.label AS object_label
            FROM edges e
            JOIN selected a ON a.id = e.subject_id
            JOIN selected b ON b.id = e.object_id
            JOIN nodes s ON s.id = e.subject_id
            JOIN nodes o ON o.id = e.object_id
            WHERE e.paper_count >= ?
            ORDER BY e.paper_count DESC, e.evidence_count DESC, e.id
            """,  # noqa: S608
            (*tuple(node_ids), max(0, int(relation_support_min))),
        ).fetchall()
        return [_row_dict(row) for row in rows]

    @staticmethod
    def _pyvis_payload(
        nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        Network, renderer = _network_class()
        net = Network(
            height="100%",
            width="100%",
            directed=True,
            bgcolor="#F7FAFC",
            font_color="#17324D",
            cdn_resources="local",
        )
        for node in nodes:
            payload = _node_payload(node)
            node_id = str(payload.pop("id"))
            net.add_node(node_id, **payload)
        for edge in edges:
            payload = _edge_payload(edge)
            source = str(payload.pop("from"))
            target = str(payload.pop("to"))
            net.add_edge(source, target, **payload)
        net.set_options(json.dumps(_GRAPH_OPTIONS, separators=(",", ":")))
        return {
            "nodes": list(net.nodes),
            "edges": list(net.edges),
            "options": _GRAPH_OPTIONS,
            "renderer": renderer,
        }

    def initial_graph(
        self,
        job_id: str,
        *,
        top_nodes: int,
        relation_support_min: int = 1,
    ) -> dict[str, Any]:
        safe_limit = max(1, min(settings.network_max_initial_nodes, int(top_nodes)))
        support_min = max(0, int(relation_support_min))
        with self.connection(job_id) as connection:
            candidate_limit = min(
                max(settings.network_max_initial_nodes, safe_limit),
                max(safe_limit * 5, safe_limit + 250),
            )
            ranked_nodes = [
                _row_dict(row)
                for row in connection.execute(
                    """
                    SELECT n.* FROM nodes n
                    WHERE EXISTS (
                        SELECT 1 FROM edges e
                        WHERE e.paper_count >= ?
                          AND (e.subject_id = n.id OR e.object_id = n.id)
                    )
                    ORDER BY paper_count DESC, relation_count DESC,
                             chunk_count DESC, label_norm, id
                    LIMIT ?
                    """,
                    (support_min, candidate_limit),
                ).fetchall()
            ]
            selected_ids = [str(node["id"]) for node in ranked_nodes[:safe_limit]]
            next_rank = len(selected_ids)
            edges: list[dict[str, Any]] = []
            # Remove non-connected choices and refill from the paper-support
            # ranking. The loop is bounded so large graphs stay predictable.
            for _ in range(12):
                edges = self._edges_between(
                    connection,
                    selected_ids,
                    relation_support_min=support_min,
                )
                if safe_limit <= 1:
                    break
                connected = {
                    str(endpoint)
                    for edge in edges
                    for endpoint in (edge["subject_id"], edge["object_id"])
                }
                retained = [node_id for node_id in selected_ids if node_id in connected]
                changed = retained != selected_ids
                selected_ids = retained
                while len(selected_ids) < safe_limit and next_rank < len(ranked_nodes):
                    selected_ids.append(str(ranked_nodes[next_rank]["id"]))
                    next_rank += 1
                    changed = True
                if not changed or next_rank >= len(ranked_nodes):
                    edges = self._edges_between(
                        connection,
                        selected_ids,
                        relation_support_min=support_min,
                    )
                    break
            nodes = self._nodes_by_ids(connection, selected_ids)
            meta_row = connection.execute(
                "SELECT value FROM meta WHERE key = 'stats'"
            ).fetchone()
            stats = json.loads(meta_row["value"]) if meta_row else {}
        payload = self._pyvis_payload(nodes, edges)
        payload.update(
            {
                "mode": "initial",
                "requested_top_nodes": safe_limit,
                "relation_support_min": support_min,
                "ranking_metric": "paper_count",
                "returned_node_count": len(nodes),
                "returned_edge_count": len(edges),
                "network_stats": stats,
            }
        )
        return payload

    def relation_types(
        self,
        job_id: str,
        *,
        relation_support_min: int = 1,
    ) -> list[dict[str, Any]]:
        """Return every stored predicate with counts at the active support threshold."""

        support_min = max(0, int(relation_support_min))
        with self.connection(job_id) as connection:
            rows = connection.execute(
                """
                SELECT predicate,
                       COUNT(*) AS total_edge_count,
                       SUM(CASE WHEN paper_count >= ? THEN 1 ELSE 0 END)
                           AS edge_count,
                       SUM(CASE WHEN paper_count >= ? THEN paper_count ELSE 0 END)
                           AS paper_support,
                       SUM(CASE WHEN paper_count >= ? THEN evidence_count ELSE 0 END)
                           AS evidence_count
                FROM edges
                GROUP BY predicate
                ORDER BY predicate COLLATE NOCASE
                """,
                (support_min, support_min, support_min),
            ).fetchall()
            endpoint_rows = connection.execute(
                """
                WITH endpoints(predicate, node_id) AS (
                    SELECT predicate, subject_id FROM edges WHERE paper_count >= ?
                    UNION
                    SELECT predicate, object_id FROM edges WHERE paper_count >= ?
                )
                SELECT predicate, COUNT(*) AS node_count
                FROM endpoints
                GROUP BY predicate
                """,
                (support_min, support_min),
            ).fetchall()
        node_counts = {str(row["predicate"]): int(row["node_count"] or 0) for row in endpoint_rows}
        return [
            {
                "predicate": str(row["predicate"]),
                "edge_count": int(row["edge_count"] or 0),
                "total_edge_count": int(row["total_edge_count"] or 0),
                "node_count": node_counts.get(str(row["predicate"]), 0),
                "paper_support": int(row["paper_support"] or 0),
                "evidence_count": int(row["evidence_count"] or 0),
                "relation_support_min": support_min,
            }
            for row in rows
        ]

    def graph_for_relations(
        self,
        job_id: str,
        predicates: Sequence[str],
        *,
        relation_support_min: int = 1,
    ) -> dict[str, Any]:
        """Return all edges and endpoint nodes for one or more predicates."""

        requested = list(
            dict.fromkeys(
                str(predicate or "").strip().lower()
                for predicate in predicates
                if str(predicate or "").strip()
            )
        )
        if not requested:
            raise ValueError("Select at least one relation type.")
        if len(requested) > 50:
            raise ValueError("Too many relation types were selected.")

        support_min = max(0, int(relation_support_min))
        with self.connection(job_id) as connection:
            available = {
                str(row["predicate"]).lower(): str(row["predicate"])
                for row in connection.execute(
                    "SELECT DISTINCT predicate FROM edges ORDER BY predicate"
                ).fetchall()
            }
            selected = [available[value] for value in requested if value in available]
            if not selected:
                raise ValueError("None of the selected relation types exists in this network.")

            placeholders = ",".join("?" for _ in selected)
            parameters = (*selected, support_min)
            edges = [
                _row_dict(row)
                for row in connection.execute(
                    f"""
                    SELECT e.*, s.label AS subject_label, o.label AS object_label
                    FROM edges e
                    JOIN nodes s ON s.id = e.subject_id
                    JOIN nodes o ON o.id = e.object_id
                    WHERE e.predicate IN ({placeholders})
                      AND e.paper_count >= ?
                    ORDER BY e.paper_count DESC, e.evidence_count DESC,
                             e.predicate COLLATE NOCASE, e.id
                    """,  # noqa: S608 -- placeholders are generated, values are bound
                    parameters,
                ).fetchall()
            ]
            nodes = [
                _row_dict(row)
                for row in connection.execute(
                    f"""
                    WITH matching_edges(subject_id, object_id) AS (
                        SELECT subject_id, object_id
                        FROM edges
                        WHERE predicate IN ({placeholders})
                          AND paper_count >= ?
                    ),
                    endpoints(id) AS (
                        SELECT subject_id FROM matching_edges
                        UNION
                        SELECT object_id FROM matching_edges
                    )
                    SELECT n.*
                    FROM endpoints p
                    JOIN nodes n ON n.id = p.id
                    ORDER BY n.paper_count DESC, n.relation_count DESC,
                             n.chunk_count DESC, n.label_norm, n.id
                    """,  # noqa: S608 -- placeholders are generated, values are bound
                    parameters,
                ).fetchall()
            ]
            meta_row = connection.execute(
                "SELECT value FROM meta WHERE key = 'stats'"
            ).fetchone()
            stats = json.loads(meta_row["value"]) if meta_row else {}

        payload = self._pyvis_payload(nodes, edges)
        payload.update(
            {
                "mode": "relation_types",
                "selected_predicates": selected,
                "relation_support_min": support_min,
                "returned_node_count": len(nodes),
                "returned_edge_count": len(edges),
                "network_stats": stats,
            }
        )
        return payload

    def neighborhood(
        self,
        job_id: str,
        node_id: str,
        *,
        limit: int,
        relation_support_min: int = 1,
    ) -> dict[str, Any]:
        safe_limit = max(1, min(settings.network_expansion_limit, int(limit)))
        support_min = max(0, int(relation_support_min))
        with self.connection(job_id) as connection:
            root = connection.execute(
                "SELECT * FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if root is None:
                raise KeyError(node_id)
            edge_rows = connection.execute(
                """
                SELECT e.*, s.label AS subject_label, o.label AS object_label
                FROM edges e
                JOIN nodes s ON s.id = e.subject_id
                JOIN nodes o ON o.id = e.object_id
                WHERE (e.subject_id = ? OR e.object_id = ?)
                  AND e.paper_count >= ?
                ORDER BY e.paper_count DESC, e.evidence_count DESC, e.id
                LIMIT ?
                """,
                (node_id, node_id, support_min, safe_limit),
            ).fetchall()
            edges = [_row_dict(row) for row in edge_rows]
            node_ids: list[str] = [node_id]
            seen = {node_id}
            for edge in edges:
                for candidate in (str(edge["subject_id"]), str(edge["object_id"])):
                    if candidate not in seen:
                        seen.add(candidate)
                        node_ids.append(candidate)
            nodes = self._nodes_by_ids(connection, node_ids)
        payload = self._pyvis_payload(nodes, edges)
        payload.update(
            {
                "mode": "incremental_neighborhood",
                "root_node_id": node_id,
                "relation_support_min": support_min,
                "returned_node_count": len(nodes),
                "returned_edge_count": len(edges),
            }
        )
        return payload

    @staticmethod
    def _interaction_cells_by_cl(
        connection: sqlite3.Connection,
    ) -> dict[str, dict[str, Any]]:
        """Map canonical CL IDs to the strongest extracted interaction node."""

        interaction_by_cl: dict[str, dict[str, Any]] = {}
        rows = connection.execute(
            """
            SELECT * FROM nodes
            WHERE entity_type = 'cell'
            ORDER BY paper_count DESC, relation_count DESC, chunk_count DESC, id
            """
        ).fetchall()
        for row in rows:
            node = _row_dict(row)
            resolved = cell_hierarchy_service.resolve_id(node.get("normalized_id"))
            if resolved:
                interaction_by_cl.setdefault(resolved, node)
        return interaction_by_cl

    def cell_hierarchy(
        self,
        job_id: str,
        concept_ids: Sequence[str] | str,
        *,
        max_paths: int = 3,
    ) -> dict[str, Any]:
        """Return one merged root-to-term hierarchy for visible cell nodes.

        The browser supplies the Cell Ontology identifiers currently displayed
        in the interaction graph.  Each identifier can have several ``is_a``
        parent paths because Cell Ontology is a DAG; the hierarchy service
        merges those paths into one safe tree-like HTML representation.
        """

        requested_ids = (
            [concept_ids]
            if isinstance(concept_ids, str)
            else list(concept_ids)
        )
        with self.connection(job_id) as connection:
            interaction_by_cl = self._interaction_cells_by_cl(connection)
        graph_node_ids = {
            cl_id: str(node["id"]) for cl_id, node in interaction_by_cl.items()
        }
        # A hierarchy-only node has a deterministic browser ID, so it remains
        # addressable if the user has already inserted it into the current view.
        for requested in requested_ids:
            resolved = cell_hierarchy_service.resolve_id(requested)
            if resolved:
                graph_node_ids.setdefault(resolved, ontology_node_id(resolved))

        view = cell_hierarchy_service.hierarchy_view_for_terms(
            requested_ids,
            graph_node_ids=graph_node_ids,
            max_paths=max(1, min(10, int(max_paths))),
        )
        return {
            "mode": "visible_cell_ontology_root_paths",
            "root_concept_id": "CL:0000000",
            "visible_cell_count": int(view["term_count"]),
            "visible_concept_ids": view["resolved_cl_ids"],
            "path_count": int(view["path_count"]),
            "records": view["records"],
            "terms": view["terms"],
            "html": view["html"],
            "ontology": view["source"],
        }

    def cell_term_neighborhood(
        self,
        job_id: str,
        concept_id: str,
        *,
        limit: int,
        relation_support_min: int = 1,
    ) -> dict[str, Any]:
        """Add a hierarchy-selected CL term to the current interaction graph."""

        term = cell_hierarchy_service.term_detail(concept_id)
        canonical = str(term["concept_id"])
        with self.connection(job_id) as connection:
            interaction_node = self._interaction_cells_by_cl(connection).get(canonical)

        if interaction_node is not None:
            payload = self.neighborhood(
                job_id,
                str(interaction_node["id"]),
                limit=limit,
                relation_support_min=relation_support_min,
            )
            payload.update(
                {
                    "mode": "incremental_hierarchy_term_neighborhood",
                    "root_concept_id": canonical,
                    "root_node_id": str(interaction_node["id"]),
                    "ontology_only": False,
                    "ontology_term": term,
                }
            )
            return payload

        ontology_node = _hierarchy_only_node_payload({**term, "role": "selected"})
        return {
            "mode": "incremental_hierarchy_term_neighborhood",
            "root_concept_id": canonical,
            "root_node_id": str(ontology_node["id"]),
            "ontology_only": True,
            "ontology_term": term,
            "nodes": [ontology_node],
            "edges": [],
            "options": _GRAPH_OPTIONS,
            "renderer": "pyvis-compatible-hierarchy",
            "returned_node_count": 1,
            "returned_edge_count": 0,
            "relation_support_min": max(0, int(relation_support_min)),
        }

    def search_nodes(self, job_id: str, query: str, *, limit: int) -> list[dict[str, Any]]:
        cleaned = " ".join(query.split()).casefold()
        if not cleaned:
            return []
        safe_limit = max(1, min(settings.network_search_limit, int(limit)))
        prefix = cleaned + "%"
        contains = "%" + cleaned + "%"
        with self.connection(job_id) as connection:
            rows = connection.execute(
                """
                WITH matched AS (
                    SELECT id AS node_id,
                           CASE WHEN label_norm = ? THEN 0
                                WHEN label_norm LIKE ? THEN 1 ELSE 3 END AS rank
                    FROM nodes
                    WHERE label_norm = ? OR label_norm LIKE ? OR label_norm LIKE ?
                    UNION ALL
                    SELECT node_id,
                           CASE WHEN alias_norm = ? THEN 0
                                WHEN alias_norm LIKE ? THEN 2 ELSE 4 END AS rank
                    FROM node_aliases
                    WHERE alias_norm = ? OR alias_norm LIKE ? OR alias_norm LIKE ?
                ), best AS (
                    SELECT node_id, MIN(rank) AS rank FROM matched GROUP BY node_id
                )
                SELECT n.*, best.rank
                FROM best JOIN nodes n ON n.id = best.node_id
                ORDER BY best.rank, n.paper_count DESC, n.relation_count DESC,
                         n.label_norm, n.id
                LIMIT ?
                """,
                (
                    cleaned, prefix, cleaned, prefix, contains,
                    cleaned, prefix, cleaned, prefix, contains,
                    safe_limit,
                ),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "label": row["label"],
                "entity_type": row["entity_type"],
                "normalized_id": row["normalized_id"],
                "relation_count": row["relation_count"],
                "paper_count": row["paper_count"],
            }
            for row in rows
        ]

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    @classmethod
    def _attach_evidence_entities(
        cls,
        connection: sqlite3.Connection,
        evidence: list[dict[str, Any]],
    ) -> None:
        """Attach all Stage 2 entity spans for every returned source passage."""

        if not evidence:
            return
        for item in evidence:
            item["entities"] = []
        if not cls._table_exists(connection, "evidence_entities"):
            return
        bases = list(
            dict.fromkeys(str(item.get("base") or "") for item in evidence)
        )
        bases = [base for base in bases if base]
        if not bases:
            return
        placeholders = ",".join("?" for _ in bases)
        rows = connection.execute(
            f"""
            SELECT base, start_offset AS start, end_offset AS end,
                   entity_type, normalized_id, node_id, mention, preferred_label
            FROM evidence_entities
            WHERE base IN ({placeholders})
            ORDER BY base, start_offset, end_offset DESC, entity_type, normalized_id
            """,  # noqa: S608
            tuple(bases),
        ).fetchall()
        by_base: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_base.setdefault(str(row["base"]), []).append(_row_dict(row))
        for item in evidence:
            item["entities"] = by_base.get(str(item.get("base") or ""), [])

    def node_detail(self, job_id: str, node_id: str) -> dict[str, Any]:
        with self.connection(job_id) as connection:
            node = connection.execute(
                """
                SELECT n.*,
                       (
                           SELECT COUNT(*) FROM edges e
                           WHERE e.predicate = 'binding'
                             AND (e.subject_id = n.id OR e.object_id = n.id)
                       ) AS undirected_count
                FROM nodes n WHERE n.id = ?
                """,
                (node_id,),
            ).fetchone()
            if node is None:
                raise KeyError(node_id)
            aliases = [
                {"text": row["alias"], "count": row["count"]}
                for row in connection.execute(
                    """
                    SELECT alias, count FROM node_aliases WHERE node_id = ?
                    ORDER BY alias_kind DESC, count DESC, alias COLLATE NOCASE
                    """,
                    (node_id,),
                ).fetchall()
            ]
            predicates = [
                _row_dict(row)
                for row in connection.execute(
                    """
                    SELECT predicate,
                           SUM(
                               CASE WHEN predicate <> 'binding' AND subject_id = ?
                                    THEN 1 ELSE 0 END
                           ) AS outgoing,
                           SUM(
                               CASE WHEN predicate <> 'binding' AND object_id = ?
                                    THEN 1 ELSE 0 END
                           ) AS incoming,
                           SUM(
                               CASE WHEN predicate = 'binding' THEN 1 ELSE 0 END
                           ) AS undirected,
                           SUM(evidence_count) AS evidence_count
                    FROM edges WHERE subject_id = ? OR object_id = ?
                    GROUP BY predicate ORDER BY evidence_count DESC, predicate
                    """,
                    (node_id, node_id, node_id, node_id),
                ).fetchall()
            ]
        return {**_row_dict(node), "aliases": aliases, "predicates": predicates}

    def edge_detail(self, job_id: str, edge_id: str) -> dict[str, Any]:
        with self.connection(job_id) as connection:
            edge = connection.execute(
                """
                SELECT e.*, s.label AS subject_label, s.entity_type AS subject_type,
                       o.label AS object_label, o.entity_type AS object_type
                FROM edges e
                JOIN nodes s ON s.id = e.subject_id
                JOIN nodes o ON o.id = e.object_id
                WHERE e.id = ?
                """,
                (edge_id,),
            ).fetchone()
            if edge is None:
                raise KeyError(edge_id)
            contexts = [
                _row_dict(row)
                for row in connection.execute(
                    """
                    SELECT c.cell_node_id AS id, n.label, COUNT(*) AS evidence_count
                    FROM edge_contexts c JOIN nodes n ON n.id = c.cell_node_id
                    WHERE c.edge_id = ?
                    GROUP BY c.cell_node_id, n.label
                    ORDER BY evidence_count DESC, n.label
                    """,
                    (edge_id,),
                ).fetchall()
            ]
        payload = {**_row_dict(edge), "cell_contexts": contexts}
        payload["directed"] = str(payload.get("predicate") or "") != "binding"
        return payload

    def edge_evidence(self, job_id: str, edge_id: str, *, limit: int) -> dict[str, Any]:
        safe_limit = max(1, min(settings.network_evidence_limit, int(limit)))
        detail = self.edge_detail(job_id, edge_id)
        with self.connection(job_id) as connection:
            rows = connection.execute(
                """
                SELECT * FROM edge_evidence WHERE edge_id = ?
                ORDER BY pub_year DESC, canonical_id, base LIMIT ?
                """,
                (edge_id, safe_limit),
            ).fetchall()
            evidence: list[dict[str, Any]] = []
            for row in rows:
                contexts = [
                    _row_dict(context)
                    for context in connection.execute(
                        """
                        SELECT n.id, n.label FROM edge_contexts c
                        JOIN nodes n ON n.id = c.cell_node_id
                        WHERE c.edge_id = ? AND c.base = ?
                        ORDER BY n.label
                        """,
                        (edge_id, row["base"]),
                    ).fetchall()
                ]
                evidence.append({**_row_dict(row), "cell_contexts": contexts})
            self._attach_evidence_entities(connection, evidence)
        return {"edge": detail, "evidence": evidence, "limit": safe_limit}

    def node_evidence(self, job_id: str, node_id: str, *, limit: int) -> dict[str, Any]:
        safe_limit = max(1, min(settings.network_evidence_limit, int(limit)))
        detail = self.node_detail(job_id, node_id)
        with self.connection(job_id) as connection:
            rows = connection.execute(
                """
                WITH relevant(edge_id) AS (
                    SELECT id FROM edges WHERE subject_id = ? OR object_id = ?
                    UNION
                    SELECT edge_id FROM edge_contexts WHERE cell_node_id = ?
                )
                SELECT v.*, e.predicate, e.subject_id, s.label AS subject_label,
                       e.object_id, o.label AS object_label
                FROM relevant r
                JOIN edge_evidence v ON v.edge_id = r.edge_id
                JOIN edges e ON e.id = v.edge_id
                JOIN nodes s ON s.id = e.subject_id
                JOIN nodes o ON o.id = e.object_id
                ORDER BY v.pub_year DESC, v.canonical_id, v.base
                LIMIT ?
                """,
                (node_id, node_id, node_id, safe_limit),
            ).fetchall()
            evidence: list[dict[str, Any]] = []
            for row in rows:
                contexts = [
                    _row_dict(context)
                    for context in connection.execute(
                        """
                        SELECT n.id, n.label FROM edge_contexts c
                        JOIN nodes n ON n.id = c.cell_node_id
                        WHERE c.edge_id = ? AND c.base = ?
                        ORDER BY n.label
                        """,
                        (row["edge_id"], row["base"]),
                    ).fetchall()
                ]
                evidence.append({**_row_dict(row), "cell_contexts": contexts})
            self._attach_evidence_entities(connection, evidence)
        return {"node": detail, "evidence": evidence, "limit": safe_limit}


network_repository = NetworkRepository()

__all__ = [
    "CellHierarchyError",
    "CellHierarchyTermNotFound",
    "NetworkRepository",
    "network_repository",
    "pyvis_asset_path",
]
