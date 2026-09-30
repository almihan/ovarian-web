"""Stable cache keys and fingerprints for Stage 3 relation extraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from backend.config import settings
from backend.pipeline.final_annotations import FINAL_ANNOTATION_EXPORT_VERSION
from backend.pipeline.relation_extraction import (
    PROMPT_VERSION,
    RELATION_PIPELINE_VERSION,
    SYSTEM,
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
        "prompt_sha256": hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest(),
        "model": settings.relation_model,
        "max_output_tokens": settings.relation_max_output_tokens,
        "max_output_tokens_ceiling": (
            settings.relation_max_output_tokens_ceiling
        ),
        "reasoning_effort": settings.relation_reasoning_effort,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def relation_artifact_keys(
    *,
    source_annotation_sha256: str,
    source_chunks_sha256: str,
    model_signature: str,
) -> RelationArtifactKeys:
    signature_parts = model_signature.split("-", 2)
    if (
        len(signature_parts) == 3
        and signature_parts[0] == "run"
        and len(signature_parts[1]) == 32
        and all(character in "0123456789abcdef" for character in signature_parts[1])
    ):
        root = f"runs/{signature_parts[1]}/stage3/{signature_parts[2]}"
    else:
        root = (
            f"relations/{source_annotation_sha256[:2]}/{source_annotation_sha256}/"
            f"{source_chunks_sha256[:16]}/{model_signature[:20]}"
        )
    return RelationArtifactKeys(
        relations=prefixed_key(f"{root}/relations.jsonl.gz"),
        summary=prefixed_key(f"{root}/summary.json"),
    )


def final_annotation_artifact_key(
    *,
    source_annotation_sha256: str,
    source_chunks_sha256: str,
    relation_sha256: str,
    model_signature: str,
) -> str:
    """Return a content/version-specific key for the user-facing Stage 3 export."""

    signature_parts = model_signature.split("-", 2)
    if (
        len(signature_parts) == 3
        and signature_parts[0] == "run"
        and len(signature_parts[1]) == 32
        and all(character in "0123456789abcdef" for character in signature_parts[1])
    ):
        root = f"runs/{signature_parts[1]}/stage3/{signature_parts[2]}"
    else:
        root = (
            f"relations/{source_annotation_sha256[:2]}/{source_annotation_sha256}/"
            f"{source_chunks_sha256[:16]}/{model_signature[:20]}"
        )
    payload = {
        "export_version": FINAL_ANNOTATION_EXPORT_VERSION,
        "source_annotation_sha256": source_annotation_sha256,
        "source_chunks_sha256": source_chunks_sha256,
        "relation_sha256": relation_sha256,
        "model_signature": model_signature,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return prefixed_key(f"{root}/exports/{digest}.jsonl.gz")


__all__ = [
    "RelationArtifactKeys",
    "final_annotation_artifact_key",
    "relation_artifact_keys",
    "relation_model_signature",
]
