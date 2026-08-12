"""Deploy the CellExLink T4 worker with ``modal deploy modal_app.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

APP_NAME = "ovarian-cellexlink"
MODEL_VOLUME_NAME = "ovarian-cellexlink-model-cache"

app = modal.App(APP_NAME)
model_cache_volume = modal.Volume.from_name(
    MODEL_VOLUME_NAME,
    create_if_missing=True,
)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface-hub>=0.24,<1")
    .env(
        {
            "HF_HOME": "/model-cache/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
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
    ephemeral_disk=20480,
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
