"""Deploy the CellExLink T4 worker with ``modal deploy modal_app.py``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal
from dotenv import load_dotenv

from backend import deployment_flags

# Loading the local .env is convenient for ``modal deploy``.  The file is
# ignored by Git and is never added to the Modal image.
load_dotenv()

APP_NAME = (
    os.getenv("MODAL_APP_NAME") or "ovarian-cellexlink-hgnc-chebi"
).strip()
MODEL_VOLUME_NAME = (
    os.getenv("MODAL_MODEL_VOLUME_NAME")
    or "ovarian-cellexlink-hgnc-chebi-model-cache"
).strip()


def _enabled(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


MONTHLY_SCHEDULE_ENABLED = _enabled(
    os.getenv("MODAL_MONTHLY_SCHEDULE_ENABLED"),
    deployment_flags.MODAL_MONTHLY_SCHEDULE_ENABLED,
)
MONTHLY_CRON = os.getenv("MODAL_MONTHLY_CRON") or deployment_flags.MONTHLY_UPDATE_CRON
MONTHLY_SECRET_NAME = os.getenv("MODAL_MONTHLY_SECRET_NAME") or "ovarian-monthly-update"

app = modal.App(APP_NAME)
model_cache_volume = modal.Volume.from_name(
    MODEL_VOLUME_NAME,
    create_if_missing=True,
)

# This scheduled function only sends an authenticated trigger to Railway.
# It has no GPU. Railway performs ID discovery and submits the existing GPU
# worker only after it has identified genuinely new eligible papers.
schedule_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("requests>=2.32,<3")
    .env({"MODAL_MONTHLY_SCHEDULE_ENABLED": "true" if MONTHLY_SCHEDULE_ENABLED else "false"})
    .add_local_file("backend/__init__.py", remote_path="/root/backend/__init__.py")
    .add_local_file("backend/deployment_flags.py", remote_path="/root/backend/deployment_flags.py")
)


@app.function(
    image=schedule_image,
    schedule=modal.Cron(MONTHLY_CRON, timezone="UTC") if MONTHLY_SCHEDULE_ENABLED else None,
    secrets=[modal.Secret.from_name(MONTHLY_SECRET_NAME)] if MONTHLY_SCHEDULE_ENABLED else [],
    cpu=0.25,
    memory=256,
    timeout=120,
    max_containers=1,
    scaledown_window=5,
)
def monthly_update_trigger() -> dict[str, Any]:
    """Trigger one budget-limited monthly update; disabled by default."""
    if os.getenv("MODAL_MONTHLY_SCHEDULE_ENABLED", "false").casefold() != "true":
        return {"status": "disabled"}
    import requests
    from urllib.parse import urlsplit

    base_url = (os.getenv("CORPUS_UPDATE_BASE_URL") or "").strip().rstrip("/")
    token = (os.getenv("MONTHLY_UPDATE_TOKEN") or "").strip()
    if urlsplit(base_url).scheme != "https" or not urlsplit(base_url).netloc:
        raise RuntimeError("CORPUS_UPDATE_BASE_URL must be your HTTPS Railway URL.")
    if not token:
        raise RuntimeError("MONTHLY_UPDATE_TOKEN is missing from the Modal update secret.")
    response = requests.post(
        f"{base_url}/api/internal/corpus-updates",
        headers={"X-Corpus-Update-Token": token},
        json={"dry_run": False},
        timeout=(10, 60),
    )
    response.raise_for_status()
    result = response.json()
    return result if isinstance(result, dict) else {"status": "accepted"}

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface-hub>=0.24,<1")
    .env(
        {
            "HF_HOME": "/model-cache/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_VERBOSITY": "error",
            "TRANSFORMERS_VERBOSITY": "error",
            "TQDM_DISABLE": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_file("backend/__init__.py", remote_path="/root/backend/__init__.py")
    .add_local_file("backend/deployment_flags.py", remote_path="/root/backend/deployment_flags.py")
)

worker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.4.1",
        "transformers==4.45.2",
        "safetensors>=0.5,<1",
        "numpy>=1.26,<3",
        "pyab3p>=0.1.1,<1",
        "huggingface-hub>=0.24,<1",
        "requests>=2.32,<3",
        "python-dotenv>=1.1,<2",
    )
    .env(
        {
            "HF_HOME": "/model-cache/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_VERBOSITY": "error",
            "TRANSFORMERS_VERBOSITY": "error",
            "TQDM_DISABLE": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        "backend",
        remote_path="/root/backend",
        # add_local_dir intentionally includes the CellExLink ontology JSONL and
        # abbreviation TSV; the Modal worker does not need the web UI assets.
        ignore=[
            "**/__pycache__/**",
            "**/*.pyc",
            "static/**",
            "templates/**",
        ],
    )
)


@app.function(
    image=download_image,
    volumes={"/model-cache": model_cache_volume},
    cpu=1,
    memory=2048,
    timeout=3600,
    max_containers=1,
    scaledown_window=5,
)
def warm_model_cache(
    ner_model: str = "almire/CellExLink-bioformer16L",
    ner_revision: str | None = None,
    nen_model: str = "almire/CellExLink-Sapbert",
    nen_revision: str | None = None,
) -> dict[str, Any]:
    """Download both checkpoints without allocating a GPU."""

    from huggingface_hub import snapshot_download

    model_cache_volume.reload()
    cache_root = Path("/model-cache/huggingface")
    cache_root.mkdir(parents=True, exist_ok=True)

    def ensure(repo_id: str, revision: str | None) -> tuple[str, bool]:
        kwargs = {
            "repo_id": repo_id,
            "revision": revision,
            "cache_dir": str(cache_root),
        }
        try:
            return str(snapshot_download(local_files_only=True, **kwargs)), False
        except Exception:
            return str(snapshot_download(local_files_only=False, **kwargs)), True

    ner_path, ner_downloaded = ensure(ner_model, ner_revision)
    nen_path, nen_downloaded = ensure(nen_model, nen_revision)
    if ner_downloaded or nen_downloaded:
        model_cache_volume.commit()
    return {
        "ner_snapshot": ner_path,
        "nen_snapshot": nen_path,
        "ner_downloaded": ner_downloaded,
        "nen_downloaded": nen_downloaded,
    }


@app.function(
    image=worker_image,
    gpu="T4",
    volumes={"/model-cache": model_cache_volume},
    cpu=2,
    memory=12288,
    timeout=21600,
    max_containers=1,
    scaledown_window=10,
    retries=modal.Retries(
        max_retries=1,
        backoff_coefficient=2.0,
        initial_delay=5.0,
    ),
)
def annotate_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    """Run and publish only the sequential CellExLink branch on one T4."""

    from backend.modal_worker.pipeline import run_annotation_bundle

    # A reused container may have been started before the CPU warm-cache
    # function committed new snapshots. Reload before resolving checkpoints.
    model_cache_volume.reload()
    return run_annotation_bundle(
        payload,
        model_cache_root=Path("/model-cache"),
        commit_callback=model_cache_volume.commit,
    )
