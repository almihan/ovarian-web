"""Public per-page pipeline API with no job-history endpoints."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal, Mapping

import requests
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.orchestrator import RetrievalError, pipeline_orchestrator
from backend.pipeline.annotation_download import sanitize_jsonl_files
from backend.pipeline.entity_overlap import ENTITY_OVERLAP_POLICY
from backend.pipeline.precomputed_corpora import (
    DEFAULT_CORPUS_ID,
    CorpusNotFoundError,
    corpus_definition,
    corpus_root,
)
from backend.runtime import run_registry
from backend.storage.artifacts import ArtifactRef, get_artifact_store

router = APIRouter(prefix="/api", tags=["pipeline runs"])
_ONE_MIB = 1024 * 1024


class RunCreate(BaseModel):
    query: str = Field(
        default="",
        max_length=4000,
        description=(
            "Comma-separated PMIDs. The public deployment loads saved results "
            "only; local installations can analyze arbitrary PMIDs. Leave blank "
            "to load the selected saved corpus."
        ),
    )

    corpus_id: Literal[
        "non_neoplastic_inflammatory",
        "cancer_associated_inflammatory",
    ] = Field(
        default=DEFAULT_CORPUS_ID,
        description="Permanent saved corpus used when the PMID field is blank.",
    )

    text_mode: Literal["abstract", "fulltext"] = Field(
        default="fulltext",
        description=(
            "For PMID runs, use PubMed abstracts only or PMC full text when available."
        ),
    )

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        return " ".join(str(value or "").split())


class AnnotationCallback(BaseModel):
    status: str = Field(default="processing", max_length=32)
    stage: str = Field(default="processing", max_length=80)
    progress: int = Field(default=0, ge=0, le=100)
    message: str = Field(default="", max_length=2000)
    stats: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(default=None, max_length=8000)
    output_sha256: str | None = Field(default=None, max_length=128)
    summary_sha256: str | None = Field(default=None, max_length=128)


def _clean_run_id(run_id: str) -> str:
    value = str(run_id or "").strip().lower()
    if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
        raise HTTPException(status_code=404, detail="Run not found.")
    return value


def _run(run_id: str) -> dict[str, Any]:
    cleaned = _clean_run_id(run_id)
    try:
        if run_registry.get_private(cleaned, "trusted_update", False):
            raise HTTPException(status_code=404, detail="Run not found.")
        return run_registry.public(cleaned)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                "This temporary run is not available. Reloading the page starts "
                "a new run by design."
            ),
        ) from exc


@router.post("/runs", status_code=202)
def create_run(payload: RunCreate, request: Request) -> dict[str, Any]:
    try:
        return pipeline_orchestrator.create_run(
            payload.query,
            text_mode=payload.text_mode,
            corpus_id=payload.corpus_id,
            callback_base_url=str(request.base_url).rstrip("/"),
        )
    except (RetrievalError, CorpusNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/runs/{run_id}")
def run_status(run_id: str) -> dict[str, Any]:
    return _run(run_id)


@router.post("/runs/{run_id}/stages/{stage_number}", status_code=202)
def start_stage(
    run_id: str,
    stage_number: int,
    request: Request,
) -> dict[str, Any]:
    cleaned = _clean_run_id(run_id)
    _run(cleaned)
    stage_name = {2: "annotation", 3: "relation", 4: "network"}.get(stage_number)
    if stage_name is None:
        raise HTTPException(status_code=404, detail="Pipeline stage not found.")
    try:
        return pipeline_orchestrator.start_stage(
            cleaned,
            stage_name,
            callback_base_url=str(request.base_url).rstrip("/"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/internal/annotations/{worker_job_id}/callback", include_in_schema=False)
def annotation_callback(
    worker_job_id: str,
    payload: AnnotationCallback,
    x_annotation_token: str | None = Header(default=None, alias="X-Annotation-Token"),
) -> dict[str, str]:
    try:
        pipeline_orchestrator.handle_annotation_callback(
            worker_job_id,
            token=x_annotation_token or "",
            payload=payload.model_dump(),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Annotation worker not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "accepted"}


def _download_source_path(
    ref: ArtifactRef,
    *,
    destination: Path,
) -> Path:
    """Return a local readable copy of an artifact."""

    store = get_artifact_store()
    local = store.local_path(ref.key)
    if local is not None:
        return local
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = store.presign_get(
        ref.key,
        expires_seconds=settings.artifact_presigned_ttl_seconds,
    )
    with requests.get(url, stream=True, timeout=(20, 900)) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for block in response.iter_content(chunk_size=_ONE_MIB):
                if block:
                    output.write(block)
    return destination


def _sanitized_download_root(run_id: str) -> Path:
    root = (settings.data_dir / "runs" / run_id / "downloads").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


@router.get("/runs/{run_id}/download/{stage_name}")
def download_stage(run_id: str, stage_name: str):
    cleaned = _clean_run_id(run_id)
    run = _run(cleaned)
    if stage_name != "stage3":
        raise HTTPException(status_code=404, detail="Download not found.")
    if run["stages"]["relation"].get("status") != "completed":
        raise HTTPException(status_code=409, detail="Stage 3 is not complete.")
    download_root = _sanitized_download_root(cleaned)
    policy_headers = {
        "Cache-Control": "no-store",
        "X-Entity-Overlap-Policy": ENTITY_OVERLAP_POLICY,
    }

    if str(run.get("input_mode") or "") == "precomputed":
        raw_path = run_registry.get_private(cleaned, "precomputed_prediction_path")
        if not raw_path:
            raise HTTPException(status_code=404, detail="Saved corpus file not found.")
        path = Path(str(raw_path)).expanduser().resolve()
        subset = bool(run_registry.get_private(cleaned, "precomputed_pmid_selection", False))
        root = (
            (settings.data_dir / "runs" / cleaned).expanduser().resolve()
            if subset else corpus_root()
        )
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(status_code=404, detail="Saved corpus file not found.")
        definition = corpus_definition(str(run.get("corpus_id") or DEFAULT_CORPUS_ID))
        filename = "precomputed-pmid-results.jsonl" if subset else definition["filename"]
        destination = download_root / filename
        sanitize_jsonl_files([path], destination)
        return FileResponse(
            destination,
            media_type="application/x-ndjson",
            filename=filename,
            headers=policy_headers,
        )

    refs = pipeline_orchestrator.artifacts_for_download(cleaned, stage_name)
    if not refs:
        raise HTTPException(
            status_code=404,
            detail="The final annotated paper artifact was not found.",
        )

    sources: list[Path] = []
    source_root = download_root / "sources"
    shutil.rmtree(source_root, ignore_errors=True)
    source_root.mkdir(parents=True, exist_ok=True)
    for index, raw in enumerate(refs, start=1):
        ref = ArtifactRef.from_dict(raw)
        sources.append(
            _download_source_path(
                ref,
                destination=source_root / f"source-{index}.jsonl.gz",
            )
        )
    destination = download_root / "ovarian-metadata-and-annotations.jsonl.gz"
    sanitize_jsonl_files(sources, destination)
    return FileResponse(
        destination,
        media_type="application/gzip",
        filename="ovarian-metadata-and-annotations.jsonl.gz",
        headers={**policy_headers, "Content-Encoding": "identity"},
    )


@router.get("/runs/{run_id}/download/entity-index")
def download_entity_index(run_id: str):
    cleaned = _clean_run_id(run_id)
    run = _run(cleaned)
    if run["stages"]["network"].get("status") != "completed":
        raise HTTPException(status_code=409, detail="Stage 4 is not complete.")
    raw_path = run_registry.get_private(cleaned, "entity_index_path")
    if not raw_path:
        raise HTTPException(status_code=404, detail="Entity index not found.")
    path = Path(str(raw_path)).expanduser().resolve()
    root = (settings.data_dir / "runs").expanduser().resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=404, detail="Entity index not found.")
    return FileResponse(
        path,
        media_type="application/gzip",
        filename="entity-relation-index.jsonl.gz",
        headers={"Cache-Control": "no-store"},
    )
