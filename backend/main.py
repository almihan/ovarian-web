"""FastAPI entry point for Ovarian Network."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.gzip import GZipMiddleware

from backend.api.annotations import router as annotations_router
from backend.api.jobs import router as jobs_router
from backend.api.papers import router as papers_router
from backend.api.relations import router as relations_router
from backend.api.networks import router as networks_router
from backend.config import settings
from backend.database.database import get_network_job, init_database
from backend.services.railway_annotation_executor import railway_annotation_executor
from backend.services.relation_executor import relation_executor
from backend.services.network_executor import network_executor
from backend.services.network_repository import pyvis_asset_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

VIS_NETWORK_VERSION = "9.1.2"
VIS_NETWORK_CDN_JS = (
    f"https://cdnjs.cloudflare.com/ajax/libs/vis-network/{VIS_NETWORK_VERSION}/"
    "dist/vis-network.min.js"
)
VIS_NETWORK_CDN_CSS = (
    f"https://cdnjs.cloudflare.com/ajax/libs/vis-network/{VIS_NETWORK_VERSION}/"
    "dist/dist/vis-network.min.css"
)
_VENDOR_CACHE_HEADERS = {
    "Cache-Control": "public, max-age=31536000, immutable",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    init_database()
    try:
        railway_annotation_executor.resume_active_jobs()
    except Exception:
        logger.exception("Could not resume Railway Stage 2 background work")
    try:
        relation_executor.resume_active_jobs()
    except Exception:
        logger.exception("Could not resume Railway Stage 3 background work")
    try:
        network_executor.resume_active_jobs()
    except Exception:
        logger.exception("Could not resume Railway Stage 4 background work")
    try:
        yield
    finally:
        network_executor.shutdown()
        relation_executor.shutdown()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)
# Relation-type views can include the full matching subgraph. Compress JSON and
# static responses so local and Railway transfers remain bounded.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.include_router(jobs_router)
app.include_router(papers_router)
app.include_router(annotations_router)
app.include_router(relations_router)
app.include_router(networks_router)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Apply conservative response headers for the browser interface."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), payment=()"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "font-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'none'; "
        "form-action 'self'"
    )
    if settings.environment.lower() == "production":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    return response


@app.get("/", response_class=HTMLResponse)
def landing_page(request: Request):
    controller_compute = (
        "Local CPU" if settings.environment.lower() == "development" else "Railway CPU"
    )
    annotation_compute = (
        "Local CPU"
        if settings.cell_annotation_backend == "local"
        else "Modal T4 GPU"
        if settings.cell_annotation_backend == "modal"
        else "Not configured"
    )
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "environment": settings.environment,
            "controller_compute": controller_compute,
            "annotation_backend": settings.cell_annotation_backend,
            "annotation_compute": annotation_compute,
            "relation_model": settings.relation_model,
            "relation_configured": settings.relation_configured,
        },
    )


@app.get("/network/{job_id}", response_class=HTMLResponse)
def network_explorer(request: Request, job_id: str):
    job = get_network_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Network job not found.")
    return templates.TemplateResponse(
        request=request,
        name="network.html",
        context={
            "app_name": settings.app_name,
            "network_job_id": job_id,
            "initial_nodes": settings.network_initial_nodes,
            "max_initial_nodes": settings.network_max_initial_nodes,
            "expansion_limit": settings.network_expansion_limit,
            "hierarchy_max_paths": settings.network_hierarchy_max_paths,
        },
    )


@app.get("/vendor/vis-network.min.js", include_in_schema=False)
def vis_network_javascript() -> Response:
    """Serve the PyVis-bundled browser library, with a pinned CDN fallback."""

    try:
        path = pyvis_asset_path("vis-network.min.js")
    except (RuntimeError, FileNotFoundError) as exc:
        logger.warning(
            "Local PyVis JavaScript asset is unavailable (%s); using the pinned CDN fallback.",
            exc,
        )
        return RedirectResponse(
            VIS_NETWORK_CDN_JS,
            status_code=307,
            headers={"Cache-Control": "no-store"},
        )
    return FileResponse(
        path,
        media_type="application/javascript",
        headers=_VENDOR_CACHE_HEADERS,
    )


@app.get("/vendor/vis-network.css", include_in_schema=False)
def vis_network_stylesheet() -> Response:
    """Serve the PyVis-bundled stylesheet, with a pinned CDN fallback."""

    try:
        path = pyvis_asset_path("vis-network.css")
    except (RuntimeError, FileNotFoundError) as exc:
        logger.warning(
            "Local PyVis stylesheet is unavailable (%s); using the pinned CDN fallback.",
            exc,
        )
        return RedirectResponse(
            VIS_NETWORK_CDN_CSS,
            status_code=307,
            headers={"Cache-Control": "no-store"},
        )
    return FileResponse(
        path,
        media_type="text/css",
        headers=_VENDOR_CACHE_HEADERS,
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "images" / "favicon.ico")


@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def apple_touch_icon() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "images" / "apple-touch-icon.png")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/system/status")
def system_status() -> dict:
    return {
        "application": "ready",
        "active_stages": [
            "paper_retrieval",
            "entity_extraction",
            "relation_extraction",
            "network_generation",
        ],
        "planned_stages": [],
        "cell_annotation_executor": settings.cell_annotation_backend,
        "cell_annotation_compute": (
            "Modal T4"
            if settings.cell_annotation_backend == "modal"
            else "local CPU"
            if settings.cell_annotation_backend == "local"
            else "disabled"
        ),
        "pubtator3_compute": "Railway CPU",
        "final_entity_merge_compute": "Railway CPU",
        "relation_extraction_compute": (
            f"OpenAI {settings.relation_model} Responses API"
            if settings.relation_configured
            else "not configured"
        ),
        "relation_window_size": settings.relation_window_size,
        "relation_concurrency": settings.relation_concurrency,
        "network_generation_compute": "Railway/local CPU · SQLite · PyVis",
        "network_initial_nodes": settings.network_initial_nodes,
        "network_expansion_limit": settings.network_expansion_limit,
        "network_hierarchy_max_paths": settings.network_hierarchy_max_paths,
        "entity_branches_run_concurrently": True,
        "cell_models_loaded_in_web_process": False,
        "artifact_backend": settings.artifact_backend,
        "data_directory": str(settings.data_dir),
    }
