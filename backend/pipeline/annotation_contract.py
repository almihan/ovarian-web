"""Shared contract for Railway orchestration and Modal cell annotation."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from backend.cellexlink_lite.resources import (
    DEFAULT_ABBREVIATIONS_PATH,
    DEFAULT_ONTOLOGY_PATH,
)
from backend.config import settings
from backend.pipeline.pubtator3_annotation_worker import PUBTATOR3_PIPELINE_VERSION
from backend.storage.artifacts import ArtifactRef, prefixed_key

ANNOTATION_PIPELINE_VERSION = "entity-extraction-parallel-v6-ncbi-human-genes"


@dataclass(frozen=True, slots=True)
class AnnotationArtifactKeys:
    final_annotations: str
    final_summary: str
    cell_annotations: str
    cell_summary: str
    pubtator_annotations: str
    pubtator_summary: str


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotation_model_signature() -> str:
    payload = {
        "pipeline_version": ANNOTATION_PIPELINE_VERSION,
        "execution_layout": "modal-cellexlink-plus-railway-pubtator3",
        "ner_model": settings.cell_ner_model,
        "ner_revision": settings.cell_ner_revision,
        "nen_model": settings.cell_nen_model,
        "nen_revision": settings.cell_nen_revision,
        "ontology_sha256": _sha256_path(DEFAULT_ONTOLOGY_PATH),
        "abbreviations_sha256": _sha256_path(DEFAULT_ABBREVIATIONS_PATH),
        "abbreviations_enabled": not settings.cell_disable_abbreviations,
        "pubtator3_pipeline_version": PUBTATOR3_PIPELINE_VERSION,
        "pubtator3_required": settings.pubtator3_required,
        "pubtator3_resolve_preferred_labels": (
            settings.pubtator3_resolve_preferred_labels
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def annotation_artifact_keys(
    *, source_sha256: str, model_signature: str
) -> AnnotationArtifactKeys:
    root = (
        f"annotations/{source_sha256[:2]}/{source_sha256}/"
        f"{model_signature[:20]}"
    )
    return AnnotationArtifactKeys(
        final_annotations=prefixed_key(f"{root}/entity_annotations.jsonl.gz"),
        final_summary=prefixed_key(f"{root}/summary.json"),
        cell_annotations=prefixed_key(f"{root}/branches/cell_annotations.jsonl.gz"),
        cell_summary=prefixed_key(f"{root}/branches/cell_summary.json"),
        pubtator_annotations=prefixed_key(
            f"{root}/branches/pubtator3_annotations.jsonl.gz"
        ),
        pubtator_summary=prefixed_key(f"{root}/branches/pubtator3_summary.json"),
    )


def annotation_output_keys(
    *, source_sha256: str, model_signature: str
) -> tuple[str, str]:
    """Return the stable final pair retained after Stage 2 completes."""

    keys = annotation_artifact_keys(
        source_sha256=source_sha256,
        model_signature=model_signature,
    )
    return keys.final_annotations, keys.final_summary


def callback_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def callback_token_matches(token: str, expected_hash: str | None) -> bool:
    if not token or not expected_hash:
        return False
    return hmac.compare_digest(callback_token_hash(token), expected_hash)


def source_artifact_from_summary(summary: Mapping[str, Any]) -> ArtifactRef:
    files = summary.get("files") or {}
    if not isinstance(files, Mapping):
        raise ValueError("The retrieval summary has no files object.")
    raw_artifact = files.get("artifact")
    if not isinstance(raw_artifact, Mapping):
        raise ValueError(
            "This retrieval predates object publishing. Run Stage 1 again before "
            "starting GPU annotation."
        )
    artifact = ArtifactRef.from_dict(raw_artifact)
    if not artifact.sha256:
        raise ValueError("The retrieval artifact has no SHA-256 fingerprint.")
    return artifact


__all__ = [
    "ANNOTATION_PIPELINE_VERSION",
    "AnnotationArtifactKeys",
    "annotation_artifact_keys",
    "annotation_model_signature",
    "annotation_output_keys",
    "callback_token_hash",
    "callback_token_matches",
    "source_artifact_from_summary",
]
