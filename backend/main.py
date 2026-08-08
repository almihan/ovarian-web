"""FastAPI entry point for Ovarian Network."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.api.jobs import router as jobs_router
from backend.api.networks import router as networks_router
from backend.api.papers import router as papers_router
from backend.config import settings
from backend.database.database import init_database
from backend.models.loaders import load_cell_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


async def _load_model_once(app: FastAPI) -> None:
    app.state.model_status = {
        "state": "loading",
        "model_id": settings.cell_model_id,
        "message": "Downloading or loading the CellExLink model...",
    }
    try:
        bundle = await asyncio.to_thread(
            load_cell_model,
            settings.cell_model_id,
            settings.model_cache_dir,
        )
        app.state.cell_model = bundle
        app.state.model_status = {
            "state": "ready",
            "model_id": bundle.model_id,
            "device": bundle.device,
            "message": "Cell-type model is ready.",
        }
    except Exception as exc:  # Keep the interface available if model loading fails.
        logger.exception("Cell model failed to load")
        app.state.cell_model = None
        app.state.model_status = {
            "state": "error",
            "model_id": settings.cell_model_id,
            "message": str(exc),
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    init_database()
    app.state.cell_model = None

    model_task: asyncio.Task | None = None
    if settings.load_cell_model:
        # The application becomes reachable immediately while the single model
        # copy is initialized in the background.
        model_task = asyncio.create_task(_load_model_once(app))
    else:
        app.state.model_status = {
            "state": "disabled",
            "model_id": settings.cell_model_id,
            "message": "Model loading is disabled by configuration.",
        }

    yield

    if model_task and not model_task.done():
        model_task.cancel()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.include_router(jobs_router)
app.include_router(networks_router)
app.include_router(papers_router)


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
        "script-src 'self' 'unsafe-inline' https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://unpkg.com; "
        "font-src 'self' https://fonts.gstatic.com; "
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
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": settings.app_name,
            "model_id": settings.cell_model_id,
            "environment": settings.environment,
        },
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
def system_status(request: Request) -> dict:
    return {
        "application": "ready",
        "model": request.app.state.model_status,
        "server_openai_key_configured": bool(settings.openai_api_key),
        "data_directory": str(settings.data_dir),
    }
