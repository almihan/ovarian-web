"""Authenticated scheduler/admin entry point; never exposed as public PMID input."""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from backend.config import settings
from backend.services.corpus_updates import corpus_update_service

router = APIRouter(prefix="/api/internal", tags=["private corpus maintenance"])


class UpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dry_run: bool = True


def _authorize(token: str | None) -> None:
    expected = str(getattr(settings, "monthly_update_token", "") or "")
    if not expected:
        raise HTTPException(status_code=503, detail="Corpus updates are not configured.")
    if not secrets.compare_digest((token or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid corpus update token.")


@router.post("/corpus-updates", status_code=202, include_in_schema=False)
def trigger_update(payload: UpdateRequest, x_corpus_update_token: str | None = Header(default=None, alias="X-Corpus-Update-Token")) -> dict[str, Any]:
    _authorize(x_corpus_update_token)
    if not getattr(settings, "monthly_updates_enabled", False):
        raise HTTPException(status_code=409, detail="Monthly corpus updates are disabled.")
    try:
        return corpus_update_service.start(dry_run=payload.dry_run, callback_base_url=settings.public_base_url)
    except InterruptedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/corpus-updates", include_in_schema=False)
def update_status(x_corpus_update_token: str | None = Header(default=None, alias="X-Corpus-Update-Token")) -> dict[str, Any]:
    _authorize(x_corpus_update_token)
    return corpus_update_service.status()
