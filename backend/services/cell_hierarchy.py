"""Lazy local Cell Ontology ``is_a`` paths for the Stage 4 explorer."""

from __future__ import annotations

import gzip
import json
import re
import threading
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.cellexlink_lite.resources import (
    CELL_ONTOLOGY_RELEASE,
    DEFAULT_HIERARCHY_PATH,
)

_CL_ID_RE = re.compile(r"(?<![A-Za-z0-9])CL[_:](\d{7})(?!\d)", re.IGNORECASE)
CELL_ROOT = "CL:0000000"


class CellHierarchyError(RuntimeError):
    """Raised when the bundled Cell Ontology hierarchy cannot be read."""


class CellHierarchyTermNotFound(CellHierarchyError):
    """Raised when an identifier is not present in the bundled CL release."""


def normalize_cl_id(value: Any) -> str:
    """Return a canonical ``CL:0000000`` identifier or an empty string."""

    text = str(value or "").strip()
    if not text:
        return ""
    match = _CL_ID_RE.search(text)
    return f"CL:{match.group(1)}" if match else ""


def ontology_node_id(value: Any) -> str:
    """Return the stable browser ID used for hierarchy-only CL nodes."""

    cl_id = normalize_cl_id(value)
    return f"CLH-{cl_id.split(':', 1)[1]}" if cl_id else ""


def ontology_node_concept_id(value: Any) -> str:
    """Recover a CL identifier from a hierarchy-only browser node ID."""

    text = str(value or "").strip()
    match = re.fullmatch(r"CLH-(\d{7})", text, flags=re.IGNORECASE)
    return f"CL:{match.group(1)}" if match else ""


def _text_list(value: Any, *, limit: int = 50) -> tuple[str, ...]:
    raw_values = value if isinstance(value, list) else [value]
    output: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        text = " ".join(str(raw or "").split())
        folded = text.casefold()
        if text and folded not in seen:
            seen.add(folded)
            output.append(text)
        if len(output) >= limit:
            break
    return tuple(output)


class CellHierarchyIndex:
    """Load the compact hierarchy once and expose root-to-term CL paths."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or DEFAULT_HIERARCHY_PATH).expanduser().resolve()
        self._lock = threading.Lock()
        self._loaded = False
        self._terms: dict[str, dict[str, Any]] = {}
        self._children: dict[str, tuple[str, ...]] = {}
        self._alias_to_id: dict[str, str] = {}
        self._metadata: dict[str, Any] = {
            "ontology": "Cell Ontology",
            "release": CELL_ONTOLOGY_RELEASE,
            "relation": "is_a",
        }

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if not self.path.is_file():
                raise CellHierarchyError(
                    f"The bundled Cell Ontology hierarchy is missing: {self.path.name}"
                )

            terms: dict[str, dict[str, Any]] = {}
            alias_to_id: dict[str, str] = {}
            children: dict[str, list[str]] = defaultdict(list)
            metadata = dict(self._metadata)
            try:
                with gzip.open(self.path, "rt", encoding="utf-8") as source:
                    for line in source:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        row = json.loads(stripped)
                        if not isinstance(row, Mapping):
                            continue
                        row_metadata = row.get("_meta")
                        if isinstance(row_metadata, Mapping):
                            metadata.update(
                                {
                                    str(key): value
                                    for key, value in row_metadata.items()
                                    if value is not None
                                }
                            )
                            continue

                        cl_id = normalize_cl_id(row.get("id"))
                        if not cl_id:
                            continue
                        parents = tuple(
                            dict.fromkeys(
                                parent
                                for parent in (
                                    normalize_cl_id(value)
                                    for value in row.get("parents", [])
                                )
                                if parent and parent != cl_id
                            )
                        )
                        aliases = tuple(
                            dict.fromkeys(
                                alias
                                for alias in (
                                    normalize_cl_id(value)
                                    for value in row.get("aliases", [])
                                )
                                if alias and alias != cl_id
                            )
                        )
                        terms[cl_id] = {
                            "cl_id": cl_id,
                            "label": " ".join(
                                str(row.get("label") or cl_id).split()
                            )
                            or cl_id,
                            "definition": " ".join(
                                str(row.get("definition") or "").split()
                            ),
                            "synonyms": _text_list(row.get("synonyms")),
                            "parents": parents,
                            "aliases": aliases,
                        }
                        alias_to_id[cl_id] = cl_id
                        for alias in aliases:
                            alias_to_id[alias] = cl_id
                        for parent in parents:
                            children[parent].append(cl_id)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise CellHierarchyError(
                    "The bundled Cell Ontology hierarchy could not be read."
                ) from exc

            def sort_ids(values: Iterable[str]) -> tuple[str, ...]:
                return tuple(
                    sorted(
                        set(values),
                        key=lambda value: (
                            str(terms.get(value, {}).get("label") or value).casefold(),
                            value,
                        ),
                    )
                )

            self._terms = terms
            self._children = {
                parent: sort_ids(child_ids) for parent, child_ids in children.items()
            }
            self._alias_to_id = alias_to_id
            self._metadata = metadata
            self._loaded = True

    def resolve_id(self, value: Any) -> str:
        self._load()
        cl_id = normalize_cl_id(value)
        if not cl_id:
            return ""
        return self._alias_to_id.get(cl_id, cl_id if cl_id in self._terms else "")

    def get_label(self, value: Any) -> str:
        self._load()
        cl_id = self.resolve_id(value)
        return str(self._terms.get(cl_id, {}).get("label") or cl_id or value or "")

    def _public_term(self, cl_id: str, *, role: str = "term") -> dict[str, Any]:
        term = self._terms[cl_id]
        return {
            "cl_id": cl_id,
            "concept_id": cl_id,
            "label": term["label"],
            "definition": term["definition"],
            "synonyms": list(term["synonyms"]),
            "aliases": list(term["aliases"]),
            "role": role,
            "parent_count": len(term["parents"]),
            "child_count": len(self._children.get(cl_id, ())),
            "has_children": bool(self._children.get(cl_id)),
        }

    def source(self) -> dict[str, Any]:
        self._load()
        return {"provider": "Bundled OBO Cell Ontology snapshot", **self._metadata}

    def term_detail(self, value: Any) -> dict[str, Any]:
        self._load()
        requested = normalize_cl_id(value)
        if not requested:
            raise ValueError(
                "A valid Cell Ontology identifier such as CL:0000501 is required."
            )
        cl_id = self.resolve_id(requested)
        if not cl_id:
            raise CellHierarchyTermNotFound(
                f"Cell Ontology term {requested} is not in the bundled release."
            )
        term = self._public_term(cl_id, role="selected")
        term["ontology"] = self.source()
        return term

    def immediate_subclasses(self, value: Any) -> list[dict[str, Any]]:
        self._load()
        cl_id = self.resolve_id(value)
        if not cl_id:
            return []
        return [
            self._public_term(child_id, role="subclass")
            for child_id in self._children.get(cl_id, ())
            if child_id in self._terms
        ]

    def get_hierarchy_paths(
        self,
        value: Any,
        *,
        root_id: str = CELL_ROOT,
        max_paths: int | None = 3,
        max_depth: int = 100,
    ) -> list[list[str]]:
        """Return deterministic root-to-term ``is_a`` paths in the CL DAG."""

        self._load()
        selected_id = self.resolve_id(value)
        canonical_root = self.resolve_id(root_id)
        if not selected_id or not canonical_root:
            return []
        if selected_id == canonical_root:
            return [[canonical_root]]

        paths: list[list[str]] = []

        def visit(current: str, path_to_current: list[str], seen: set[str]) -> None:
            if max_paths is not None and len(paths) >= max_paths:
                return
            if len(path_to_current) > max_depth:
                return
            if current == canonical_root:
                paths.append(list(reversed(path_to_current)))
                return
            parents = [
                parent
                for parent in self._terms.get(current, {}).get("parents", ())
                if parent in self._terms and parent not in seen
            ]
            parents.sort(key=lambda term_id: (self.get_label(term_id).casefold(), term_id))
            for parent in parents:
                visit(parent, [*path_to_current, parent], seen | {parent})
                if max_paths is not None and len(paths) >= max_paths:
                    break

        visit(selected_id, [selected_id], {selected_id})
        return paths

    def make_hierarchy_records(
        self,
        cl_ids: Iterable[Any],
        *,
        root_id: str = CELL_ROOT,
        max_paths: int | None = 3,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for original in cl_ids:
            cl_id = self.resolve_id(original)
            if not cl_id:
                continue
            records.append(
                {
                    "input_cl_id": cl_id,
                    "input_label": self.get_label(cl_id),
                    "paths": self.get_hierarchy_paths(
                        cl_id, root_id=root_id, max_paths=max_paths
                    ),
                }
            )
        return records

    def render_hierarchy_html(
        self,
        cl_ids: Sequence[Any],
        *,
        highlighted_ids: Iterable[Any] | None = None,
        graph_node_ids: Mapping[str, str] | None = None,
        root_id: str = CELL_ROOT,
        max_paths: int | None = 3,
    ) -> str:
        """Render requested CL paths as one merged root-to-term tree-like list."""

        highlighted = {
            resolved
            for value in (highlighted_ids or [])
            if (resolved := self.resolve_id(value))
        }
        graph_node_by_cl_id = {
            resolved: str(node_id)
            for raw_id, node_id in (graph_node_ids or {}).items()
            if (resolved := self.resolve_id(raw_id)) and str(node_id)
        }
        records = self.make_hierarchy_records(
            cl_ids, root_id=root_id, max_paths=max_paths
        )
        if not records:
            return '<p class="hierarchy-empty">No valid Cell Ontology ID was found.</p>'

        tree: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for record in records:
            paths = record["paths"]
            if not paths:
                missing.append(f"{record['input_label']} ({record['input_cl_id']})")
                continue
            for path in paths:
                branch = tree
                for ancestor_id in path:
                    node = branch.setdefault(ancestor_id, {"children": {}})
                    branch = node["children"]

        pieces = ['<ol class="hierarchy-path">']

        def descendant_graph_node_ids(
            node_id: str, branch_node: Mapping[str, Any]
        ) -> list[str]:
            values: list[str] = []
            graph_id = graph_node_by_cl_id.get(node_id)
            if graph_id:
                values.append(graph_id)
            children = branch_node.get("children", {})
            if isinstance(children, Mapping):
                for child_id, child_node in children.items():
                    if isinstance(child_node, Mapping):
                        values.extend(descendant_graph_node_ids(str(child_id), child_node))
            return list(dict.fromkeys(values))

        def render_branch(branch: Mapping[str, Mapping[str, Any]], depth: int) -> None:
            sorted_ids = sorted(
                branch,
                key=lambda term_id: (self.get_label(term_id).casefold(), term_id),
            )
            for term_id in sorted_ids:
                label = self.get_label(term_id)
                is_hit = term_id in highlighted
                subclasses = self.immediate_subclasses(term_id) if is_hit else []
                class_names = [name for name, enabled in (
                    ("highlighted", is_hit),
                    ("has-subclasses", bool(subclasses)),
                ) if enabled]
                class_attr = (
                    f' class="{escape(" ".join(class_names), quote=True)}"'
                    if class_names
                    else ""
                )
                toggle_attr = ' data-subclass-toggle="1"' if subclasses else ""
                graph_id = graph_node_by_cl_id.get(term_id, "")
                connect_ids = descendant_graph_node_ids(term_id, branch[term_id])
                pieces.append(
                    f'<li{class_attr}{toggle_attr} data-term-id="{escape(term_id, quote=True)}" '
                    f'data-term-label="{escape(label, quote=True)}" '
                    f'data-node-id="{escape(graph_id, quote=True)}" '
                    f'data-connect-ids="{escape(",".join(connect_ids), quote=True)}" '
                    f'data-depth="{depth}" style="--depth:{depth}">'
                    f'<span class="tree-prefix">{escape("" if depth == 0 else "└── ")}</span>'
                    f'<span class="term-label">{escape(label)}</span> '
                    f'<span class="term-id">({escape(term_id)})</span>'
                    + (
                        f'<span class="subclass-count"> {len(subclasses)} direct subclasses</span>'
                        if subclasses
                        else ""
                    )
                    + "</li>"
                )
                if subclasses:
                    pieces.append(
                        f'<li class="subclass-panel" data-depth="{depth + 1}" '
                        f'style="--depth:{depth + 1}" hidden>'
                        '<div class="subclass-box"><div class="subclass-title">'
                        'Direct subclasses</div><ul>'
                    )
                    for subclass in subclasses:
                        child_id = str(subclass["concept_id"])
                        child_label = str(subclass["label"])
                        pieces.append(
                            f'<li data-term-id="{escape(child_id, quote=True)}" '
                            f'data-term-label="{escape(child_label, quote=True)}" '
                            f'data-node-id="{escape(graph_node_by_cl_id.get(child_id, ""), quote=True)}">'
                            f'{escape(child_label)} '
                            f'<span class="term-id">({escape(child_id)})</span></li>'
                        )
                    pieces.append("</ul></div></li>")
                child_branch = branch[term_id].get("children", {})
                if isinstance(child_branch, Mapping):
                    render_branch(child_branch, depth + 1)

        render_branch(tree, 0)
        pieces.append("</ol>")
        if missing:
            pieces.append(
                '<p class="hierarchy-empty">No root path found for: '
                + escape(", ".join(missing))
                + "</p>"
            )
        return "\n".join(pieces)

    def hierarchy_view(
        self,
        value: Any,
        *,
        graph_node_ids: Mapping[str, str] | None = None,
        max_paths: int = 3,
    ) -> dict[str, Any]:
        """Return one selected term using the same merged-tree representation."""

        view = self.hierarchy_view_for_terms(
            [value],
            graph_node_ids=graph_node_ids,
            max_paths=max_paths,
        )
        record = view["records"][0]
        term = view["terms"][0]
        return {
            "requested_cl_id": normalize_cl_id(value),
            "resolved_cl_id": record["input_cl_id"],
            "term": term,
            "path_count": len(record["paths"]),
            "paths": record["paths"],
            "html": view["html"],
            "source": view["source"],
        }

    def hierarchy_view_for_terms(
        self,
        values: Sequence[Any],
        *,
        graph_node_ids: Mapping[str, str] | None = None,
        max_paths: int = 3,
    ) -> dict[str, Any]:
        """Return a merged hierarchy for one or more visible Cell Ontology terms.

        Cell Ontology is a directed acyclic graph, so each visible cell can have
        multiple root-to-term ``is_a`` paths.  The paths are kept as records for
        programmatic use and merged into one tree-like HTML view for the browser.
        """

        self._load()
        requested = list(values)
        resolved_ids = list(
            dict.fromkeys(
                resolved
                for value in requested
                if (resolved := self.resolve_id(value))
            )
        )
        if not resolved_ids:
            normalized = [normalize_cl_id(value) for value in requested]
            valid_requested = [value for value in normalized if value]
            if valid_requested:
                raise CellHierarchyTermNotFound(
                    f"Cell Ontology term {valid_requested[0]} is not in the bundled release."
                )
            raise ValueError(
                "At least one valid Cell Ontology identifier such as CL:0000501 is required."
            )

        safe_max_paths = max(1, min(10, int(max_paths)))
        records = self.make_hierarchy_records(
            resolved_ids,
            max_paths=safe_max_paths,
        )
        terms = [
            self._public_term(cl_id, role="selected") for cl_id in resolved_ids
        ]
        return {
            "requested_cl_ids": [normalize_cl_id(value) for value in requested],
            "resolved_cl_ids": resolved_ids,
            "term_count": len(resolved_ids),
            "path_count": sum(len(record["paths"]) for record in records),
            "records": records,
            "terms": terms,
            "html": self.render_hierarchy_html(
                resolved_ids,
                highlighted_ids=resolved_ids,
                graph_node_ids=graph_node_ids,
                max_paths=safe_max_paths,
            ),
            "source": self.source(),
        }


cell_hierarchy_index = CellHierarchyIndex()
cell_hierarchy_service = cell_hierarchy_index

__all__ = [
    "CELL_ROOT",
    "CellHierarchyError",
    "CellHierarchyTermNotFound",
    "CellHierarchyIndex",
    "cell_hierarchy_index",
    "cell_hierarchy_service",
    "normalize_cl_id",
    "ontology_node_concept_id",
    "ontology_node_id",
]
