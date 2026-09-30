"""Read-only API for the two permanent precomputed ovarian corpora."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.pipeline.precomputed_corpora import (
    CorpusNotFoundError,
    corpus_summary,
    list_corpora,
)

router = APIRouter(prefix="/api/corpora", tags=["saved corpora"])


@router.get("")
def saved_corpora() -> dict[str, Any]:
    return {"corpora": list_corpora()}


@router.get("/{corpus_id}")
def saved_corpus_summary(corpus_id: str) -> dict[str, Any]:
    try:
        return corpus_summary(corpus_id)
    except CorpusNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
