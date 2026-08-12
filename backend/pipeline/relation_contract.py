"""Stable cache keys and fingerprints for Stage 3 relation extraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from backend.config import settings
from backend.pipeline.relation_extraction import (
    PROMPT_VERSION,
    RELATION_PIPELINE_VERSION,
    SYSTEM_INSTRUCTIONS,
)
from backend.storage.artifacts import prefixed_key


@dataclass(frozen=True, slots=True)
class RelationArtifactKeys:
    relations: str
    summary: str


def relation_model_signature() -> str:
    payload = {
        "pipeline_version": RELATION_PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest(),
        "model": settings.relation_model,
        "max_output_tokens": settings.relation_max_output_tokens,
        "reasoning_effort": settings.relation_reasoning_effort,
        "enable_biosynthesis": settings.relation_enable_biosynthesis,
        "require_hormone_gene_cell_context": (
            settings.relation_require_hormone_gene_cell_context
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def relation_artifact_keys(
    *,
    source_annotation_sha256: str,
    source_chunks_sha256: str,
    model_signature: str,
) -> RelationArtifactKeys:
    root = (
        f"relations/{source_annotation_sha256[:2]}/{source_annotation_sha256}/"
        f"{source_chunks_sha256[:16]}/{model_signature[:20]}"
    )
    return RelationArtifactKeys(
        relations=prefixed_key(f"{root}/relations.jsonl.gz"),
        summary=prefixed_key(f"{root}/summary.json"),
    )


__all__ = [
    "RelationArtifactKeys",
    "relation_artifact_keys",
    "relation_model_signature",
]
