"""Deterministic Stage 4 artifact names and cache signatures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

NETWORK_PIPELINE_VERSION = "network-v1"
NETWORK_GRAPH_SCHEMA = "ovarian-interaction-network-sqlite-v1"
ENTITY_INDEX_SCHEMA = "ovarian-entity-relation-index-v1"
PYVIS_VERSION = "0.3.2"
VIS_NETWORK_VERSION = "9.1.2"


@dataclass(frozen=True, slots=True)
class NetworkArtifactKeys:
    graph: str
    entity_index: str
    summary: str


def network_signature() -> str:
    payload = {
        "pipeline": NETWORK_PIPELINE_VERSION,
        "graph_schema": NETWORK_GRAPH_SCHEMA,
        "entity_index_schema": ENTITY_INDEX_SCHEMA,
        "pyvis": PYVIS_VERSION,
        "vis_network": VIS_NETWORK_VERSION,
        "entity_types": ["cell", "gene", "hormone"],
        "global_identity": "normalized-id-or-normalized-label-sha256",
        "evidence": "aligned-stage1-chunk-plus-stage2-entity-spans",
        "indexes": "predicate-support-v1",
        "binding_semantics": "undirected-canonical-hormone-gene",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def network_artifact_keys(
    *,
    source_relation_sha256: str,
    source_chunks_sha256: str,
    source_annotations_sha256: str,
    signature: str,
) -> NetworkArtifactKeys:
    digest = hashlib.sha256(
        (
            f"{source_relation_sha256}:{source_chunks_sha256}:"
            f"{source_annotations_sha256}:{signature}"
        ).encode("utf-8")
    ).hexdigest()
    prefix = f"network-generation/{digest[:2]}/{digest}"
    return NetworkArtifactKeys(
        graph=f"{prefix}/interaction-network.sqlite",
        entity_index=f"{prefix}/entity-relation-index.jsonl.gz",
        summary=f"{prefix}/summary.json",
    )


__all__ = [
    "ENTITY_INDEX_SCHEMA",
    "NETWORK_GRAPH_SCHEMA",
    "NETWORK_PIPELINE_VERSION",
    "NetworkArtifactKeys",
    "network_artifact_keys",
    "network_signature",
]
