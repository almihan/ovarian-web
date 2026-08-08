"""Load reusable deep-learning resources once per FastAPI process."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CellModelBundle:
    model_id: str
    tokenizer: Any
    model: Any
    device: str


def load_cell_model(model_id: str, cache_dir: Path) -> CellModelBundle:
    """Download/cache and load CellExLink for token classification.

    The returned bundle should be stored on ``app.state`` and reused by every
    request. Running one Uvicorn worker ensures only one model copy is loaded.
    """

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading cell-type model %s", model_id)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    logger.info("Cell-type model loaded on %s", device)
    return CellModelBundle(
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
