"""Process-local lock for stages sharing the paper cache and model memory."""

from __future__ import annotations

import threading

# Retrieval writes the shared per-paper chunk cache. Cell annotation reads that
# cache and launches memory-intensive model workers. One heavy pipeline at a
# time avoids partial reads and prevents a small Railway service from combining
# retrieval traffic with model inference.
PIPELINE_LOCK = threading.Lock()

__all__ = ["PIPELINE_LOCK"]
