"""Human (NCBI taxon 9606) evidence for PubTator Gene IDs.

An exact numeric Gene ID in an approved HGNC human record is usable local
identity evidence when the independent NCBI reference is unavailable. This is
an ID-to-ID lookup, never a name-based conversion of an animal Gene ID.
Explicit opposing taxa always take precedence at the calling annotation site.
Verified human NCBI records still do not require HGNC membership.
"""
from __future__ import annotations

import csv
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.pipeline.reference_normalization import (
    CachedFileStatus, HgncRecord, ensure_cached_file,
)

HUMAN_TAXONOMY_POLICY_VERSION = "human-id-evidence-before-span-veto-v2"
NCBI_HUMAN_TAXONOMY_SOURCE = "NCBI Homo_sapiens.gene_info taxon 9606"
HGNC_HUMAN_TAXONOMY_SOURCE = "HGNC approved human record: exact NCBI Gene ID cross-reference"

HUMAN_GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
)


@dataclass(frozen=True, slots=True)
class HumanGeneRecord:
    gene_id: str
    symbol: str
    name: str
    tax_id: str = "9606"
    taxonomy_source: str = NCBI_HUMAN_TAXONOMY_SOURCE


def annotation_taxa(infons: Mapping[str, Any]) -> set[str]:
    taxa: set[str] = set()
    for key, value in infons.items():
        if re.sub(r"[^a-z]", "", str(key).casefold()) not in {
            "species", "speciesid", "taxid", "taxids", "taxon", "taxonid", "taxonids", "taxonomyid", "organism"
        }:
            continue
        raw = str(value or "").strip()
        if raw.casefold() in {"human", "homo sapiens"}:
            taxa.add("9606")
        elif raw.casefold() in {"mouse", "mus musculus"}:
            taxa.add("10090")
        elif raw.casefold() in {"rat", "rattus norvegicus"}:
            taxa.add("10116")
        else:
            taxa.update(re.findall(r"\d+", raw))
    return taxa


def ensure_human_gene_reference(*, cache_dir: Path, session: Any,
                                request_timeout: int, max_attempts: int) -> CachedFileStatus:
    return ensure_cached_file(session=session, url=HUMAN_GENE_INFO_URL,
        path=cache_dir / "Homo_sapiens.gene_info.gz", max_age_seconds=7*24*60*60,
        request_timeout=request_timeout, max_attempts=max_attempts, minimum_bytes=1_000_000)


def load_human_gene_records(path: Path) -> dict[str, HumanGeneRecord]:
    opener = gzip.open if path.suffix == ".gz" else open
    records = {}
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"GeneID", "Symbol"}.issubset(reader.fieldnames):
            raise ValueError(f"Invalid NCBI gene_info header: {path}")
        for row in reader:
            tax_id = row.get("#tax_id", row.get("tax_id", ""))
            if tax_id != "9606":
                continue
            identifier = row.get("GeneID", "")
            if identifier.isdigit():
                records[identifier] = HumanGeneRecord(identifier, row.get("Symbol", ""),
                    row.get("description", ""))
    if not records:
        raise ValueError(f"NCBI reference contains no taxon 9606 records: {path}")
    return records


def numeric_ncbi_gene_id(value: Any) -> str:
    """Read one numeric NCBI Gene ID without conflating an HGNC accession.

    Multi-ID strings and explicitly nonhuman prefixes are deliberately not
    converted here. This helper must never map an animal ID by surface name.
    """
    raw = str(value or "").strip()
    raw = re.sub(r"(?i)^(?:NCBIGene|NCBI\s*Gene|GeneID|Gene)\s*:\s*", "", raw)
    match = re.fullmatch(r"(?:9606\s*:\s*)?(\d+)", raw)
    if match is None:
        return ""
    identifier = match.group(1).lstrip("0")
    return identifier if identifier else ""


def human_record_for_gene_id(
    gene_id: Any,
    *,
    human_gene_metadata: Mapping[str, HumanGeneRecord] | None = None,
    hgnc_gene_metadata: Mapping[str, HgncRecord] | None = None,
) -> HumanGeneRecord | None:
    """Resolve positive human identity evidence, using numeric IDs only.

    Annotation-level species evidence is checked by the caller before this
    lookup. A matching alias/name alone is never sufficient for this fallback.
    """
    identifier = numeric_ncbi_gene_id(gene_id)
    if not identifier:
        return None
    human = (human_gene_metadata or {}).get(identifier)
    if (human is not None and str(human.tax_id) == "9606"
            and numeric_ncbi_gene_id(human.gene_id) == identifier):
        return human
    hgnc = (hgnc_gene_metadata or {}).get(identifier)
    if (hgnc is None or str(hgnc.status).strip().casefold() != "approved"
            or numeric_ncbi_gene_id(hgnc.entrez_id) != identifier
            or re.fullmatch(r"HGNC:\d+", str(hgnc.hgnc_id)) is None):
        return None
    return HumanGeneRecord(
        gene_id=identifier,
        symbol=hgnc.symbol,
        name=hgnc.name,
        taxonomy_source=HGNC_HUMAN_TAXONOMY_SOURCE,
    )


def reconcile_taxonomy_exclusion_masks(
    masks: Sequence[Mapping[str, Any]],
    *,
    hgnc_gene_metadata: Mapping[str, HgncRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recheck only unverified masks carrying a verifiable human Gene ID.

    Older branches used the same exclusion field for a nonhuman annotation
    and for a failed reference download. An old unverified mask may be lifted
    only when its OWN numeric Gene ID is in an approved human HGNC record.
    Explicit nonhuman/conflicting taxa, missing IDs, and unknown IDs stay
    excluded. No name-based reassignment or cross-species remapping occurs.
    """
    retained: list[dict[str, Any]] = []
    repaired: list[dict[str, Any]] = []
    for raw in masks:
        mask = dict(raw)
        taxa = annotation_taxa(mask)
        explicit_nonhuman = (
            bool(taxa - {"9606"})
            or mask.get("exclusion_kind") == "explicit_nonhuman"
            or mask.get("normalization_status") == "excluded_nonhuman"
        )
        identifier = numeric_ncbi_gene_id(mask.get("gene_id"))
        human = None if explicit_nonhuman else human_record_for_gene_id(
            identifier, hgnc_gene_metadata=hgnc_gene_metadata)
        if human is None:
            retained.append(mask)
            continue
        hgnc = hgnc_gene_metadata[identifier]
        repaired.append({
            "start": mask.get("start"),
            "end": mask.get("end"),
            "gene_id": identifier,
            "hgnc_id": hgnc.hgnc_id,
            "previous_reason": mask.get("exclusion_reason"),
            "resolution": "unverified_veto_replaced_by_exact_human_id_evidence",
            "tax_id": "9606",
            "taxonomy_source": human.taxonomy_source,
        })
    return retained, repaired
