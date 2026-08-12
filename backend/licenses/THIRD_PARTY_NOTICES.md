# Third-party notices

## CellExLink

The cell-type recognition and Cell Ontology normalization stage is adapted from
CellExLink by Alimire Nabijiang and the Shahriyari Lab. The original project is
licensed under the GNU General Public License v3.0 only. The retained license is
in `CellExLink-LICENSE.txt`.

Only the model inference, Cell Ontology linking, plural normalization, and
abbreviation-resource logic needed for `chunks.jsonl` processing are included.
The original plaintext, BioC, PubTator3, PMID/PMCID retrieval, CLI, notebook,
and display adapters are not copied into this web application.

## NCBI and NLM services

The web application calls NCBI/NLM services including PubMed, PubMed Central,
PubTator3, NCBI Gene, MeSH, and E-utilities. Their services and data are not
bundled with this project and remain subject to NCBI/NLM policies and the rights
attached to the underlying literature. Automated PubTator3 annotations can be
incomplete or incorrect and should be reviewed for high-stakes uses.

## Cell Ontology hierarchy resource

The Stage 4 explorer bundles a compact transformation of Cell Ontology release
`2025-12-17`. The retained fields are ontology identifiers, labels, synonyms,
definitions, alternate identifiers, and direct `is_a` parent relationships. The
compressed resource is used only for on-demand hierarchy display and does not
replace the original ontology. Cell Ontology is distributed under the Creative
Commons Attribution 4.0 International license (CC BY 4.0).

Source project: https://github.com/obophenotype/cell-ontology

## PyVis and vis-network

The Stage 4 explorer uses PyVis 0.3.2 (BSD 3-Clause) to construct network
payloads and serves the vis-network JavaScript/CSS bundled with PyVis from
same-origin routes. vis-network is dual-licensed under Apache-2.0 and MIT. These
third-party assets are installed dependencies and are not copied into the
application source tree.
