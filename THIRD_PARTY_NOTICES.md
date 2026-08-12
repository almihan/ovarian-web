# Third-party notices

## CellExLink

The cell-type recognition and Cell Ontology normalization implementation in
`backend/cellexlink_lite/` is adapted from **CellExLink**, authored by Alimire
Nabijiang and the Shahriyari Lab. The upstream project is distributed under the
GNU General Public License v3.0 only.

This integration retains only the components required to annotate the
application's `chunks.jsonl` and `chunks.jsonl.gz` records:

- cell-type named-entity recognition;
- Cell Ontology normalization;
- Cell Ontology and abbreviation resources; and
- abbreviation-aware disambiguation.

The upstream plain-text, BioC, PubTator3, PMID/PMCID retrieval, command-line,
notebook-display, and general file-routing workflows are intentionally not
included. A copy of the upstream license is retained in
`backend/cellexlink_lite/LICENSE.txt` and
`backend/licenses/CellExLink-LICENSE.txt`.

Upstream project: https://github.com/ShahriyariLab/CellExLink

## NCBI and NLM services and data

This application accesses NCBI/NLM services, including PubMed, PubMed Central,
PubTator3, NCBI Gene, MeSH, and the Entrez Programming Utilities (E-utilities).
Those services and their data are not bundled with or licensed by this project.
Their use remains subject to the applicable NCBI/NLM policies, disclaimers, and
the rights attached to the underlying literature. In particular, PubMed
abstracts and PMC articles may contain copyrighted material.

PubTator3 entity annotations are produced automatically and can contain missed
or incorrect mentions, identifiers, or boundaries. The output should be
reviewed before use in clinical, regulatory, or other high-stakes decisions.

PubTator3 reference: Chih-Hsuan Wei et al., “PubTator 3.0: an AI-powered
literature resource for unlocking biomedical knowledge,” *Nucleic Acids
Research* 52(W1), 2024, W540–W546. DOI: 10.1093/nar/gkae235.

## Cell Ontology hierarchy resource

The Stage 4 explorer bundles a compact transformation of Cell Ontology release
`2025-12-17`. The retained fields are ontology identifiers, labels, synonyms,
definitions, alternate identifiers, and direct `is_a` parent relationships. The
compressed resource is used only for on-demand hierarchy display and does not
replace the original ontology. Cell Ontology is distributed under the Creative
Commons Attribution 4.0 International license (CC BY 4.0).

Source project: https://github.com/obophenotype/cell-ontology

## PyVis and vis-network

Stage 4 uses PyVis 0.3.2 to construct browser-compatible network node, edge, and
option payloads. PyVis is distributed under the BSD 3-Clause License.

The interactive browser rendering is performed by vis-network assets bundled in
the installed PyVis package. vis-network is available under the Apache License
2.0 and the MIT License. The application serves those assets from same-origin
`/vendor` routes and does not copy them into this repository.

Project names and licenses remain the property of their respective authors.
