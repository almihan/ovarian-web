"""Network construction utilities.

The demo builder makes the first interface fully testable. Replace this with
construction from normalized entities and extracted relations when the real
pipeline is connected.
"""

from __future__ import annotations

from typing import Any


def build_demo_network(query: str) -> dict[str, Any]:
    nodes = [
        {
            "id": "CL:0000235",
            "label": "macrophage",
            "group": "cell",
            "count": 18,
            "description": "A phagocytic innate immune cell.",
        },
        {
            "id": "CL:0000084",
            "label": "T cell",
            "group": "cell",
            "count": 14,
            "description": "A lymphocyte involved in adaptive immunity.",
        },
        {
            "id": "CL:0000057",
            "label": "fibroblast",
            "group": "cell",
            "count": 9,
            "description": "A stromal cell that produces extracellular matrix.",
        },
        {
            "id": "HGNC:6018",
            "label": "IL6",
            "group": "gene",
            "count": 16,
            "description": "Interleukin 6.",
        },
        {
            "id": "HGNC:11998",
            "label": "TP53",
            "group": "gene",
            "count": 12,
            "description": "Tumor protein p53.",
        },
        {
            "id": "HGNC:1100",
            "label": "BRCA1",
            "group": "gene",
            "count": 10,
            "description": "BRCA1 DNA repair associated.",
        },
        {
            "id": "HGNC:10672",
            "label": "CXCL12",
            "group": "gene",
            "count": 8,
            "description": "C-X-C motif chemokine ligand 12.",
        },
        {
            "id": "CHEBI:27899",
            "label": "cisplatin",
            "group": "chemical",
            "count": 11,
            "description": "A platinum-containing antineoplastic agent.",
        },
        {
            "id": "CHEBI:45863",
            "label": "paclitaxel",
            "group": "chemical",
            "count": 7,
            "description": "A taxane antineoplastic agent.",
        },
    ]

    edges = [
        {
            "id": "e1",
            "from": "CL:0000235",
            "to": "HGNC:6018",
            "label": "expresses",
            "confidence": 0.94,
            "pmid": "34567890",
            "evidence": "Tumor-associated macrophages showed increased IL6 expression in ovarian cancer tissue.",
        },
        {
            "id": "e2",
            "from": "CL:0000084",
            "to": "HGNC:6018",
            "label": "responds to",
            "confidence": 0.86,
            "pmid": "35678901",
            "evidence": "IL6-associated signaling was linked to altered T-cell activity in the tumor microenvironment.",
        },
        {
            "id": "e3",
            "from": "CL:0000057",
            "to": "HGNC:10672",
            "label": "secretes",
            "confidence": 0.91,
            "pmid": "36789012",
            "evidence": "Cancer-associated fibroblasts were reported to secrete CXCL12.",
        },
        {
            "id": "e4",
            "from": "CHEBI:27899",
            "to": "HGNC:11998",
            "label": "modulates",
            "confidence": 0.82,
            "pmid": "37890123",
            "evidence": "Cisplatin treatment was associated with TP53-dependent cellular responses.",
        },
        {
            "id": "e5",
            "from": "CHEBI:45863",
            "to": "CL:0000235",
            "label": "alters activity",
            "confidence": 0.79,
            "pmid": "38901234",
            "evidence": "Paclitaxel exposure altered macrophage-associated inflammatory activity.",
        },
        {
            "id": "e6",
            "from": "HGNC:1100",
            "to": "CHEBI:27899",
            "label": "associated response",
            "confidence": 0.88,
            "pmid": "39012345",
            "evidence": "BRCA1 status was associated with response patterns to platinum treatment.",
        },
        {
            "id": "e7",
            "from": "HGNC:10672",
            "to": "CL:0000084",
            "label": "recruits",
            "confidence": 0.84,
            "pmid": "40123456",
            "evidence": "CXCL12-related signaling was connected to lymphocyte recruitment.",
        },
    ]

    return {
        "query": query,
        "summary": {
            "papers": 24,
            "entities": len(nodes),
            "relations": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
        "demo": True,
    }
