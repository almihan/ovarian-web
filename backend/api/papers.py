"""Placeholder paper endpoints for the next implementation stage."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/papers", tags=["papers"])


@router.get("/status")
def paper_pipeline_status() -> dict:
    return {
        "status": "not_connected",
        "message": "Paper retrieval will be connected after the landing page is deployed.",
    }
