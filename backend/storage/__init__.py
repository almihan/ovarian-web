"""Artifact storage adapters used by the Railway controller."""

from backend.storage.artifacts import (
    ArtifactRef,
    ArtifactStore,
    LocalArtifactStore,
    S3ArtifactStore,
    get_artifact_store,
)

__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "LocalArtifactStore",
    "S3ArtifactStore",
    "get_artifact_store",
]
