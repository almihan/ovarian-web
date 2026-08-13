"""Ephemeral Stage 4 exploration API.

Stage 4 graph files are deliberately local to one in-memory browser run. They
are never uploaded to shared storage and cannot be recovered after a reload or
application restart.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.runtime import run_registry
from backend.services.network_repository import (
    CellHierarchyError,
    CellHierarchyTermNotFound,
    network_repository,
)

router = APIRouter(prefix="/api/networks", tags=["networks"])


class CellHierarchyRequest(BaseModel):
    concept_ids: list[str] = Field(min_length=1, max_length=5000)
    max_paths: int = Field(default=3, ge=1, le=10)

    @field_validator("concept_ids")
    @classmethod
    def clean_concept_ids(cls, values: list[str]) -> list[str]:
        cleaned = list(
            dict.fromkeys(
                " ".join(str(value or "").split())
                for value in values
                if " ".join(str(value or "").split())
            )
        )
        if not cleaned:
            raise ValueError("At least one Cell Ontology identifier is required.")
        return cleaned


def _public_job(run_id: str) -> dict[str, Any]:
    try:
        run = run_registry.public(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    stage = dict(run["stages"]["network"])
    stats = dict(stage.get("stats") or {})
    return {
        "id": run_id,
        **stage,
        "paper_count": int(stats.get("paper_count") or 0),
        "node_count": int(stats.get("node_count") or 0),
        "edge_count": int(stats.get("edge_count") or 0),
        "evidence_count": int(stats.get("evidence_count") or 0),
        "explore_url": stage.get("open_url") or f"/network/{run_id}",
        "downloads": (
            {"entity_index": f"/api/networks/{run_id}/download/entity-index"}
            if stage.get("status") == "completed"
            else {}
        ),
    }


def _ready_job(run_id: str) -> dict[str, Any]:
    job = _public_job(run_id)
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="The network is not ready yet.")
    try:
        run_registry.graph_path(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=410,
            detail="This temporary network is no longer available. Start a new run.",
        ) from exc
    return job


@router.get("/status")
def network_pipeline_status() -> dict[str, Any]:
    return {
        "status": "connected",
        "executor": "ephemeral_sqlite_pyvis",
        "compute": "Railway/local CPU · temporary SQLite · PyVis",
        "persistent_artifacts": False,
        "initial_nodes": settings.network_initial_nodes,
        "max_initial_nodes": settings.network_max_initial_nodes,
        "expansion_limit": settings.network_expansion_limit,
        "search_limit": settings.network_search_limit,
        "hierarchy_max_paths": settings.network_hierarchy_max_paths,
        "entity_types": ["cell", "gene", "hormone"],
        "message": "Stage 4 exploration is available for the current page run.",
    }


@router.get("/{job_id}")
def network_job(job_id: str) -> dict[str, Any]:
    return _public_job(job_id)


@router.get("/{job_id}/summary")
def network_summary(job_id: str) -> dict[str, Any]:
    job = _ready_job(job_id)
    return {
        "run_id": job_id,
        "stats": dict(job.get("stats") or {}),
        "message": job.get("message"),
        "completed_at": job.get("completed_at"),
        "job": job,
    }


@router.get("/{job_id}/download/entity-index")
def download_entity_index(job_id: str):
    _ready_job(job_id)
    try:
        raw_path = run_registry.get_private(job_id, "entity_index_path")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    if not raw_path:
        raise HTTPException(status_code=410, detail="The temporary index is unavailable.")
    from pathlib import Path

    path = Path(str(raw_path)).expanduser().resolve()
    root = (settings.data_dir / "runs").expanduser().resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=410, detail="The temporary index is unavailable.")
    return FileResponse(
        path,
        media_type="application/gzip",
        filename=f"ovarian-entity-relation-index-{job_id}.jsonl.gz",
    )


@router.get("/{job_id}/graph")
def initial_graph(
    job_id: str,
    top_nodes: int = Query(default=settings.network_initial_nodes, ge=1),
    relation_support_min: int = Query(default=1, ge=0),
) -> dict[str, Any]:
    _ready_job(job_id)
    try:
        return network_repository.initial_graph(
            job_id,
            top_nodes=top_nodes,
            relation_support_min=relation_support_min,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{job_id}/relation-types")
def relation_types(
    job_id: str,
    relation_support_min: int = Query(default=1, ge=0),
) -> dict[str, Any]:
    """List available predicates and their supported edge/node counts."""

    _ready_job(job_id)
    try:
        relations = network_repository.relation_types(
            job_id, relation_support_min=relation_support_min
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "relation_types": relations,
        "relation_support_min": relation_support_min,
    }


@router.get("/{job_id}/graph/relations")
def relation_type_graph(
    job_id: str,
    predicates: str = Query(min_length=1, max_length=1000),
    relation_support_min: int = Query(default=1, ge=0),
) -> dict[str, Any]:
    """Return every supported edge and endpoint for the selected predicates."""

    _ready_job(job_id)
    selected = [value.strip() for value in predicates.split(",") if value.strip()]
    try:
        return network_repository.graph_for_relations(
            job_id,
            selected,
            relation_support_min=relation_support_min,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{job_id}/cell-hierarchy")
def cell_hierarchy_single(
    job_id: str,
    concept_id: str = Query(min_length=3, max_length=160),
    max_paths: int = Query(default=3, ge=1, le=10),
) -> dict[str, Any]:
    _ready_job(job_id)
    try:
        return network_repository.cell_hierarchy(
            job_id,
            [concept_id],
            max_paths=max_paths,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CellHierarchyTermNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CellHierarchyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/{job_id}/cell-hierarchy")
def visible_cell_hierarchy(
    job_id: str,
    payload: CellHierarchyRequest,
) -> dict[str, Any]:
    """Render merged root paths for the cells visible in the browser graph."""

    _ready_job(job_id)
    try:
        return network_repository.cell_hierarchy(
            job_id,
            payload.concept_ids,
            max_paths=payload.max_paths,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CellHierarchyTermNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CellHierarchyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/{job_id}/cell-hierarchy/term-neighborhood")
def cell_hierarchy_term_neighborhood(
    job_id: str,
    concept_id: str = Query(min_length=3, max_length=160),
    limit: int = Query(default=settings.network_expansion_limit, ge=1, le=1000),
    relation_support_min: int = Query(default=1, ge=0),
) -> dict[str, Any]:
    _ready_job(job_id)
    try:
        return network_repository.cell_term_neighborhood(
            job_id,
            concept_id,
            limit=limit,
            relation_support_min=relation_support_min,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CellHierarchyTermNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CellHierarchyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/{job_id}/nodes/search")
def search_nodes(
    job_id: str,
    q: str = Query(min_length=1, max_length=160),
    limit: int = Query(default=settings.network_search_limit, ge=1, le=100),
) -> dict[str, Any]:
    _ready_job(job_id)
    return {"nodes": network_repository.search_nodes(job_id, q, limit=limit)}


@router.get("/{job_id}/nodes/{node_id}/neighborhood")
def node_neighborhood(
    job_id: str,
    node_id: str,
    limit: int = Query(default=settings.network_expansion_limit, ge=1, le=1000),
    relation_support_min: int = Query(default=1, ge=0),
) -> dict[str, Any]:
    _ready_job(job_id)
    try:
        return network_repository.neighborhood(
            job_id,
            node_id,
            limit=limit,
            relation_support_min=relation_support_min,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Network node not found.") from exc


@router.get("/{job_id}/nodes/{node_id}")
def node_detail(job_id: str, node_id: str) -> dict[str, Any]:
    _ready_job(job_id)
    try:
        return network_repository.node_detail(job_id, node_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Network node not found.") from exc


@router.get("/{job_id}/edges/{edge_id}")
def edge_detail(job_id: str, edge_id: str) -> dict[str, Any]:
    _ready_job(job_id)
    try:
        return network_repository.edge_detail(job_id, edge_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Network edge not found.") from exc


@router.get("/{job_id}/evidence/nodes/{node_id}")
def node_evidence(
    job_id: str,
    node_id: str,
    limit: int = Query(default=settings.network_evidence_limit, ge=1, le=500),
) -> dict[str, Any]:
    _ready_job(job_id)
    try:
        return network_repository.node_evidence(job_id, node_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Network node not found.") from exc


@router.get("/{job_id}/evidence/edges/{edge_id}")
def edge_evidence(
    job_id: str,
    edge_id: str,
    limit: int = Query(default=settings.network_evidence_limit, ge=1, le=500),
) -> dict[str, Any]:
    _ready_job(job_id)
    try:
        return network_repository.edge_evidence(job_id, edge_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Network edge not found.") from exc
