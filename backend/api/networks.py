"""Stage 4 API for interaction-network construction and exploration."""

from __future__ import annotations

import uuid
from typing import Any, Mapping

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.database.database import (
    create_network_job,
    find_active_network,
    find_any_active_network,
    find_reusable_network,
    get_network_job,
    get_relation_job,
    list_network_jobs,
    update_network_job,
    utc_now,
)
from backend.pipeline.network_contract import (
    network_artifact_keys,
    network_signature,
)
from backend.services.network_executor import network_executor
from backend.services.network_repository import (
    CellHierarchyError,
    CellHierarchyTermNotFound,
    network_repository,
)
from backend.storage.artifacts import get_artifact_store

router = APIRouter(prefix="/api/networks", tags=["networks"])


class NetworkCreate(BaseModel):
    source_relation_job_id: str = Field(min_length=1, max_length=64)

    @field_validator("source_relation_job_id")
    @classmethod
    def clean_source_relation_job_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(
            not (character.isalnum() or character in "-_")
            for character in cleaned
        ):
            raise ValueError("source_relation_job_id contains unsupported characters")
        return cleaned


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


def _source_payload(relation_job_id: str) -> dict[str, Any] | None:
    source = get_relation_job(relation_job_id)
    if source is None:
        return None
    return {
        "id": source["id"],
        "status": source.get("status"),
        "source_annotation_job_id": source.get("source_annotation_job_id"),
        "paper_count": int(source.get("paper_count") or 0),
        "chunk_count": int(source.get("chunk_count") or 0),
        "relation_count": int(source.get("relation_count") or 0),
        "created_at": source.get("created_at"),
        "completed_at": source.get("completed_at"),
    }


def _public_job(job: Mapping[str, Any], *, reused: bool = False) -> dict[str, Any]:
    payload = dict(job)
    payload["source_relation_job"] = _source_payload(
        str(job["source_relation_job_id"])
    )
    payload["reused"] = reused or bool(job.get("reused_from_job_id"))
    payload["explore_url"] = f"/network/{job['id']}"
    if job.get("status") == "completed":
        payload["downloads"] = {
            "entity_index": f"/api/networks/{job['id']}/download/entity-index"
        }
    return payload


def _artifacts_exist(job: Mapping[str, Any]) -> bool:
    store = get_artifact_store()
    return bool(
        job.get("graph_artifact_key")
        and job.get("entity_index_artifact_key")
        and job.get("summary_artifact_key")
        and store.head(str(job["graph_artifact_key"])) is not None
        and store.head(str(job["entity_index_artifact_key"])) is not None
        and store.head(str(job["summary_artifact_key"])) is not None
    )


def _recover_completed(job: Mapping[str, Any]) -> dict[str, Any] | None:
    if not _artifacts_exist(job):
        return None
    summary = get_artifact_store().read_json(str(job["summary_artifact_key"]))
    stats_raw = summary.get("stats")
    stats = dict(stats_raw) if isinstance(stats_raw, Mapping) else {}
    return update_network_job(
        str(job["id"]),
        status="completed",
        stage="completed",
        progress=100,
        message=str(
            summary.get("message")
            or f"Finished: {int(stats.get('node_count') or 0):,} nodes are ready."
        ),
        paper_count=int(stats.get("paper_count") or job.get("paper_count") or 0),
        node_count=int(stats.get("node_count") or job.get("node_count") or 0),
        edge_count=int(stats.get("edge_count") or job.get("edge_count") or 0),
        evidence_count=int(
            stats.get("evidence_count") or job.get("evidence_count") or 0
        ),
        stats=stats,
        elapsed_seconds=float(stats.get("elapsed_seconds") or 0.0),
        completed_at=str(summary.get("completed_at") or utc_now()),
        error=None,
    )


def _ready_job(job_id: str, *, artifact_field: str | None = None) -> dict[str, Any]:
    """Return a completed job without issuing unnecessary object-store HEAD calls.

    Interactive graph requests are validated by NetworkRepository when it opens
    the SQLite artifact. Endpoints that need a different artifact can request a
    targeted existence check through ``artifact_field``.
    """

    job = get_network_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Network job not found.")
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="The network is not ready yet.")
    if artifact_field:
        key = str(job.get(artifact_field) or "")
        if not key or get_artifact_store().head(key) is None:
            raise HTTPException(
                status_code=410,
                detail="The requested network artifact is no longer available. Rebuild Stage 4.",
            )
    return job


@router.get("/status")
def network_pipeline_status() -> dict[str, Any]:
    return {
        "status": "connected",
        "executor": "railway_sqlite_pyvis",
        "compute": "Railway/local CPU · SQLite · PyVis",
        "initial_nodes": settings.network_initial_nodes,
        "max_initial_nodes": settings.network_max_initial_nodes,
        "expansion_limit": settings.network_expansion_limit,
        "search_limit": settings.network_search_limit,
        "hierarchy_max_paths": settings.network_hierarchy_max_paths,
        "cell_hierarchy": "bundled_cell_ontology_is_a",
        "entity_scope": "all",
        "entity_types": ["cell", "gene", "hormone"],
        "message": "Stage 4 is ready.",
    }


@router.post("", status_code=202)
def submit_network(payload: NetworkCreate) -> dict[str, Any]:
    relation = get_relation_job(payload.source_relation_job_id)
    if relation is None:
        raise HTTPException(status_code=404, detail="Relation-extraction job not found.")
    if relation.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail="Stage 3 must complete before network generation starts.",
        )

    store = get_artifact_store()
    relation_key = str(relation.get("output_artifact_key") or "")
    chunks_key = str(relation.get("source_chunks_artifact_key") or "")
    annotations_key = str(relation.get("source_annotation_artifact_key") or "")
    relation_ref = store.head(relation_key) if relation_key else None
    chunks_ref = store.head(chunks_key) if chunks_key else None
    annotations_ref = store.head(annotations_key) if annotations_key else None
    if relation_ref is None:
        raise HTTPException(
            status_code=409,
            detail="The Stage 3 relation artifact is missing. Run Stage 3 again.",
        )
    if chunks_ref is None:
        raise HTTPException(
            status_code=409,
            detail="The aligned Stage 1 chunk artifact is missing. Run retrieval again.",
        )
    if annotations_ref is None:
        raise HTTPException(
            status_code=409,
            detail="The aligned Stage 2 entity-annotation artifact is missing. Run Stage 2 again.",
        )
    expected_chunks_sha = str(relation.get("source_chunks_artifact_sha256") or "")
    if expected_chunks_sha and chunks_ref.sha256 != expected_chunks_sha:
        raise HTTPException(
            status_code=409,
            detail="The Stage 1 chunk fingerprint no longer matches Stage 3.",
        )
    expected_annotations_sha = str(
        relation.get("source_annotation_artifact_sha256") or ""
    )
    if expected_annotations_sha and annotations_ref.sha256 != expected_annotations_sha:
        raise HTTPException(
            status_code=409,
            detail="The Stage 2 annotation fingerprint no longer matches Stage 3.",
        )

    signature = network_signature()
    keys = network_artifact_keys(
        source_relation_sha256=relation_ref.sha256,
        source_chunks_sha256=chunks_ref.sha256,
        source_annotations_sha256=annotations_ref.sha256,
        signature=signature,
    )
    reusable = find_reusable_network(
        source_relation_artifact_sha256=relation_ref.sha256,
        source_chunks_artifact_sha256=chunks_ref.sha256,
        source_annotation_artifact_sha256=annotations_ref.sha256,
        network_signature=signature,
    )
    if reusable is not None:
        recovered = _recover_completed(reusable)
        if recovered is not None:
            return _public_job(recovered, reused=True)

    active = find_active_network(
        source_relation_artifact_sha256=relation_ref.sha256,
        source_chunks_artifact_sha256=chunks_ref.sha256,
        source_annotation_artifact_sha256=annotations_ref.sha256,
        network_signature=signature,
    )
    if active is not None:
        network_executor.submit(str(active["id"]))
        return _public_job(active, reused=True)

    other_active = find_any_active_network()
    if other_active is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Another network build is already active "
                f"({other_active['id']}). Stage 4 is limited to one SQLite writer."
            ),
        )

    job_id = uuid.uuid4().hex[:12]
    job = create_network_job(
        job_id=job_id,
        source_relation_job_id=payload.source_relation_job_id,
        network_signature=signature,
        source_relation_artifact_key=relation_key,
        source_relation_artifact_sha256=relation_ref.sha256,
        source_chunks_artifact_key=chunks_key,
        source_chunks_artifact_sha256=chunks_ref.sha256,
        source_annotation_artifact_key=annotations_key,
        source_annotation_artifact_sha256=annotations_ref.sha256,
        graph_artifact_key=keys.graph,
        entity_index_artifact_key=keys.entity_index,
        summary_artifact_key=keys.summary,
    )
    job = update_network_job(
        job_id,
        paper_count=int(relation.get("paper_count") or 0),
        stats={
            "source_chunk_count": int(relation.get("chunk_count") or 0),
            "source_relation_count": int(relation.get("relation_count") or 0),
            "entity_scope": "all",
            "entity_types": ["cell", "gene", "hormone"],
        },
    ) or job
    network_executor.submit(job_id)
    return _public_job(job)


@router.get("")
def network_jobs(limit: int = Query(default=10, ge=1, le=50)) -> dict[str, Any]:
    return {"jobs": [_public_job(job) for job in list_network_jobs(limit)]}


@router.get("/{job_id}")
def network_job(job_id: str) -> dict[str, Any]:
    job = get_network_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Network job not found.")
    return _public_job(job)


@router.get("/{job_id}/summary")
def network_summary(job_id: str) -> dict[str, Any]:
    job = _ready_job(job_id, artifact_field="summary_artifact_key")
    summary = get_artifact_store().read_json(str(job["summary_artifact_key"]))
    return {**summary, "job": _public_job(job)}


@router.get("/{job_id}/download/entity-index")
def download_entity_index(job_id: str):
    job = _ready_job(job_id, artifact_field="entity_index_artifact_key")
    key = str(job["entity_index_artifact_key"])
    store = get_artifact_store()
    local = store.local_path(key)
    filename = f"ovarian-entity-relation-index-{job_id}.jsonl.gz"
    if local is not None:
        return FileResponse(
            local,
            media_type="application/gzip",
            filename=filename,
        )
    return RedirectResponse(
        store.presign_get(key, download_name=filename),
        status_code=307,
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
