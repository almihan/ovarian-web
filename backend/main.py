"""FastAPI entry point for the public Ovarian Network application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.gzip import GZipMiddleware

from backend.api.networks import router as networks_router
from backend.api.runs import router as runs_router
from backend.config import settings
from backend.orchestrator import pipeline_orchestrator
from backend.runtime import run_registry
from backend.services.network_repository import pyvis_asset_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
# Keep application lifecycle and failures visible, but silence model/library
# progress chatter and request-by-request access logs.
for noisy_logger in (
    "transformers",
    "huggingface_hub",
    "sentence_transformers",
    "urllib3",
    "httpx",
    "openai",
    "uvicorn.access",
):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
logging.getLogger("backend.pipeline.pubtator3_annotation_worker").setLevel(logging.ERROR)

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
_VENDOR_CACHE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    pipeline_orchestrator.initialize()
    try:
        yield
    finally:
        pipeline_orchestrator.shutdown()


app = FastAPI(title=settings.app_name, version="2.0.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.include_router(runs_router)
app.include_router(networks_router)


@app.middleware("http")
async def browser_headers(request: Request, call_next):
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
        "font-src 'self'; img-src 'self' data:; connect-src 'self'; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'none'; "
        "form-action 'self'"
    )
    if request.url.path in {"/", "/api/runs"} or request.url.path.startswith("/api/runs/"):
        response.headers["Cache-Control"] = "no-store"
    if settings.environment.casefold() == "production":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    return response


@app.get("/", response_class=HTMLResponse)
def landing_page(request: Request):
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
            "controller_compute": (
                "Local CPU"
                if settings.environment.casefold() == "development"
                else "Railway CPU"
            ),
            "annotation_backend": settings.cell_annotation_backend,
            "annotation_compute": annotation_compute,
            "relation_model": settings.relation_model,
            "relation_configured": settings.relation_configured,
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/network/{run_id}", response_class=HTMLResponse)
def network_explorer(request: Request, run_id: str):
    try:
        run = run_registry.public(run_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="This temporary network run is no longer available.",
        ) from exc
    if run["stages"]["network"].get("status") not in {
        "queued",
        "processing",
        "completed",
    }:
        raise HTTPException(status_code=409, detail="Stage 4 has not been started.")
    return templates.TemplateResponse(
        request=request,
        name="network.html",
        context={
            "app_name": settings.app_name,
            "network_job_id": run_id,
            "initial_nodes": settings.network_initial_nodes,
            "max_initial_nodes": settings.network_max_initial_nodes,
            "expansion_limit": settings.network_expansion_limit,
            "hierarchy_max_paths": settings.network_hierarchy_max_paths,
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/vendor/vis-network.min.js", include_in_schema=False)
def vis_network_javascript() -> Response:
    try:
        path = pyvis_asset_path("vis-network.min.js")
    except (RuntimeError, FileNotFoundError):
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
    try:
        path = pyvis_asset_path("vis-network.css")
    except (RuntimeError, FileNotFoundError):
        return RedirectResponse(
            VIS_NETWORK_CDN_CSS,
            status_code=307,
            headers={"Cache-Control": "no-store"},
        )
    return FileResponse(path, media_type="text/css", headers=_VENDOR_CACHE_HEADERS)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "images" / "favicon.ico")


@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def apple_touch_icon() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "images" / "apple-touch-icon.png")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/system/status")
def system_status() -> dict[str, object]:
    return {
        "application": "ready",
        "state_model": "ephemeral_page_runs",
        "persistent_user_jobs": False,
        "shared_cache": ["default_stage1", "default_stage2", "default_stage3"],
        "custom_result_reuse": False,
        "stage4_persistent": False,
        "cell_annotation_executor": settings.cell_annotation_backend,
        "cell_annotation_compute": (
            "Modal T4"
            if settings.cell_annotation_backend == "modal"
            else "local CPU"
            if settings.cell_annotation_backend == "local"
            else "disabled"
        ),
        "pubtator3_compute": "Railway/local CPU",
        "relation_extraction_compute": (
            f"OpenAI {settings.relation_model} Responses API"
            if settings.relation_configured
            else "not configured"
        ),
        "network_generation_compute": "Railway/local CPU · temporary SQLite",
        "artifact_backend": settings.artifact_backend,
    }
