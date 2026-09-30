"""Document-level abbreviation and exact-entity recovery before CellExLink NER.

Ab3P runs once per document. Complete long forms are resolved against Cell
Ontology, MeSH hormone terms and human HGNC naming fields. Resource-validated
parentheses supplement non-initialisms such as P4. Every span and abbreviation
constraint retains the original source offsets and document context.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

from backend.cellexlink_lite.normalization import (
    Ab3PDefinition,
    DocumentAbbreviationContext,
    DocumentText,
    build_document_text,
    ensure_ab3p_healthy,
    run_ab3p_for_document,
)
from backend.pipeline.entity_lexicons import (
    DEFAULT_HORMONE_LEXICON_PATH,
    ExactTargetIndex,
    MESH_HORMONE_RESOURCE_VERSION,
    build_cell_hormone_index,
    ensure_hormone_lexicon,
)
from backend.pipeline.entity_text_normalization import (
    canonical_short_form_key,
    normalize_unicode,
    short_form_key_candidates,
    spans_overlap,
)

from backend.pipeline.document_entity_recovery import (
    RecoveryResources, abbreviation_pair_conflicts, enrich_gene_row, load_local_hgnc,
    recover_document_entities, supplement_parenthetical_pairs, abbreviation_surfaces, document_abbreviation_constraints,
)
from backend.cellexlink_lite.resources import DEFAULT_ABBREVIATIONS_PATH
from backend.pipeline.entity_span_rules import prune_cell_fragments, apply_document_constraints, sanitize_source_annotations

logger = logging.getLogger(__name__)

ABBREVIATION_CONTEXT_FILENAME = "abbreviation_context.json"
ABBREVIATION_ANNOTATIONS_FILENAME = "abbreviation_annotations.jsonl.gz"
ABBREVIATION_CONTEXT_SCHEMA = "document-ab3p-individual-gene-marker-safe-v5"
ABBREVIATION_PREPASS_VERSION = "document-abbreviation-cell-boundaries-coordination-v7"

_DASHES = "-−‐‑‒–—﹘﹣－"
_SOURCE_FIELDS = (
    "base",
    "doc_key",
    "canonical_id",
    "pmid",
    "pmcid",
    "journal",
    "pub_year",
    "section_type",
    "chunk_id",
)


class ProgressSink(Protocol):
    def emit(
        self,
        *,
        stage: str,
        percent: float,
        message: str,
        stats: Mapping[str, Any],
        force: bool = False,
    ) -> None: ...


def _source_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in _SOURCE_FIELDS
        if record.get(key) is not None
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as raw:
            temporary = Path(raw.name)
            count = 0
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as out:
                for row in rows:
                    out.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode(
                            "utf-8"
                        )
                    )
                    out.write(b"\n")
                    count += 1
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
        return count
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _open_jsonl(path: Path):
    if path.suffix.casefold() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_jsonl(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_no}")
            yield row


def _flexible_literal(value: str) -> str:
    pieces: list[str] = []
    for character in normalize_unicode(value):
        if character.isspace() or character in "._":
            pieces.append(r"[\s._]*")
        elif character == "-":
            pieces.append(f"[{re.escape(_DASHES)}]")
        else:
            pieces.append(re.escape(character))
    return "".join(pieces)


def _definition_patterns(short_form: str, long_form: str) -> tuple[re.Pattern[str], ...]:
    sf = _flexible_literal(short_form)
    lf = _flexible_literal(long_form)
    flags = re.IGNORECASE
    return (
        re.compile(
            rf"(?P<lf>{lf})\s*[\(\[]\s*(?P<sf>{sf})\s*[\)\]]",
            flags,
        ),
        re.compile(
            rf"(?P<sf>{sf})\s*[\(\[]\s*(?P<lf>{lf})\s*[\)\]]",
            flags,
        ),
    )


def _located_definitions(
    document: DocumentText,
    *,
    short_form: str,
    long_form: str,
) -> list[tuple[re.Match[str], Any]]:
    located: list[tuple[re.Match[str], Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for pattern in _definition_patterns(short_form, long_form):
        for match in pattern.finditer(document.text):
            sf_start, sf_end = match.span("sf")
            lf_start, lf_end = match.span("lf")
            chunk = document.chunk_for_span(match.start(), match.end())
            if chunk is None:
                # Definitions are not allowed to cross reconstructed chunk boundaries.
                continue
            signature = (sf_start, sf_end, lf_start, lf_end)
            if signature in seen:
                continue
            seen.add(signature)
            located.append((match, chunk))
    located.sort(key=lambda item: (item[0].start(), item[0].end()))
    return located


def _definition_from_match(
    *,
    short_form: str,
    long_form: str,
    key: str,
    stable_index: int,
    occurrence_index: int,
    match: re.Match[str] | None,
    chunk: Any | None,
    target_index: ExactTargetIndex,
) -> Ab3PDefinition:
    resolution = target_index.resolve(long_form)
    candidate = resolution.candidate
    definition_id = f"abbr-def-{stable_index:06d}-{occurrence_index:03d}"
    definition = Ab3PDefinition(
        short_form=short_form,
        long_form=long_form,
        key=key,
        stable_index=stable_index,
        definition_id=definition_id,
        location_status="located" if match is not None else "unlocated",
        resolution_status=resolution.status,
        entity_type=candidate.entity_type if candidate else None,
        concept_id=candidate.concept_id if candidate else None,
        preferred_label=candidate.preferred_label if candidate else None,
        matched_term=candidate.matched_term if candidate else None,
        term_kind=candidate.term_kind if candidate else None,
        resource_version=candidate.resource_version if candidate else None,
        match_method=resolution.match_method,
    )
    if match is not None and chunk is not None:
        sf_start, sf_end = match.span("sf")
        lf_start, lf_end = match.span("lf")
        definition.definition_start = match.start()
        definition.definition_end = match.end()
        definition.document_start = sf_start
        definition.document_end = sf_end
        definition.long_form_document_start = lf_start
        definition.long_form_document_end = lf_end
        definition.chunk_id = chunk.chunk_id
        definition.section = chunk.section
        definition.chunk_start = sf_start - chunk.document_start
        definition.long_form_chunk_start = lf_start - chunk.document_start
        if definition.entity_type == "cell" and definition.concept_id:
            # Compatibility object used by CellOntologyNormalizer.
            from backend.cellexlink_lite.normalization import OntologyMatch

            definition.ontology_match = OntologyMatch(
                identifier=definition.concept_id,
                preferred_label=str(definition.preferred_label or ""),
                matched_alias=str(definition.matched_term or long_form),
                raw_cosine=1.0,
                exact_unique_alias=True,
                final_score=1.0,
            )
    return definition


def build_document_context(
    document: DocumentText,
    pairs: Sequence[tuple[str, str]],
    *,
    target_index: ExactTargetIndex,
) -> DocumentAbbreviationContext:
    definitions: list[Ab3PDefinition] = []
    seen_locations: set[tuple[int, int, int, int, str]] = set()
    stable_definition_index = 0

    for pair_index, (short_form, long_form) in enumerate(pairs):
        # Also validate direct callers/native Ab3P output, not only supplements.
        # Discarding the pair is essential: an unresolved STAT1 definition would
        # otherwise mask the valid STAT1 gene throughout the paper at merge.
        if abbreviation_pair_conflicts(short_form, long_form, target_index):
            continue
        key = canonical_short_form_key(short_form)
        if not key:
            continue
        located = _located_definitions(
            document,
            short_form=short_form,
            long_form=long_form,
        )
        if located:
            for occurrence_index, (match, chunk) in enumerate(located):
                sf_start, sf_end = match.span("sf")
                lf_start, lf_end = match.span("lf")
                signature = (sf_start, sf_end, lf_start, lf_end, key)
                if signature in seen_locations:
                    continue
                seen_locations.add(signature)
                definitions.append(
                    _definition_from_match(
                        short_form=short_form,
                        long_form=long_form,
                        key=key,
                        stable_index=stable_definition_index,
                        occurrence_index=occurrence_index,
                        match=match,
                        chunk=chunk,
                        target_index=target_index,
                    )
                )
                stable_definition_index += 1
            # An identical pair can be returned more than once by a detector.
            # All defensible textual definition locations have already been
            # retained above; do not invent an additional unlocated definition.
            continue

        # Ab3P returned a pair that cannot be defensibly aligned to the source.
        # Preserve it as explicitly unlocated rather than assigning an arbitrary
        # short-form use as the definition site.
        definitions.append(
            _definition_from_match(
                short_form=short_form,
                long_form=long_form,
                key=key,
                stable_index=stable_definition_index,
                occurrence_index=pair_index,
                match=None,
                chunk=None,
                target_index=target_index,
            )
        )
        stable_definition_index += 1

    by_key: dict[str, list[Ab3PDefinition]] = defaultdict(list)
    all_keys: list[str] = []
    key_to_stable_index: dict[str, int] = {}
    for definition in definitions:
        # Store controlled lowercase plural aliases as lookup-only keys.  This
        # makes a definition such as ``TAMs`` available to an NER occurrence of
        # ``TAM`` without changing the original cached short form or its key.
        for lookup_key in short_form_key_candidates(definition.short_form):
            by_key[lookup_key].append(definition)
        if definition.key not in key_to_stable_index:
            key_to_stable_index[definition.key] = len(all_keys)
            all_keys.append(definition.key)

    return DocumentAbbreviationContext(
        document_key=document.document_key,
        ab3p_status="definitions_found" if definitions else "no_definitions",
        definitions=definitions,
        definitions_by_key=dict(by_key),
        all_keys=all_keys,
        key_to_stable_index=key_to_stable_index,
    )


def _definition_rank_core(
    definition: Ab3PDefinition,
    *,
    occurrence_document_start: int,
    chunk_id: str,
    section: str,
) -> tuple[int, int, int]:
    start = definition.definition_start
    if definition.chunk_id and definition.chunk_id == chunk_id:
        if start is None:
            return (0, 2, 10**12)
        return (
            0,
            0 if start <= occurrence_document_start else 1,
            abs(occurrence_document_start - start),
        )
    if start is not None and start <= occurrence_document_start:
        return (1, 0, occurrence_document_start - start)
    if definition.section and definition.section == section:
        return (
            2,
            0,
            abs(occurrence_document_start - start) if start is not None else 10**12,
        )
    if start is not None:
        return (3, 0, abs(occurrence_document_start - start))
    return (4, 0, 10**12)


def select_contextual_definition(
    definitions: Sequence[Ab3PDefinition],
    *,
    occurrence_document_start: int,
    chunk_id: str,
    section: str,
) -> Ab3PDefinition | None:
    if not definitions:
        return None
    ranked = sorted(
        definitions,
        key=lambda item: (
            _definition_rank_core(
                item,
                occurrence_document_start=occurrence_document_start,
                chunk_id=chunk_id,
                section=section,
            ),
            item.stable_index,
        ),
    )
    best_core = _definition_rank_core(
        ranked[0],
        occurrence_document_start=occurrence_document_start,
        chunk_id=chunk_id,
        section=section,
    )
    tied = [
        item
        for item in ranked
        if _definition_rank_core(
            item,
            occurrence_document_start=occurrence_document_start,
            chunk_id=chunk_id,
            section=section,
        )
        == best_core
    ]
    identities = {
        (item.resolution_status, item.entity_type, item.concept_id) for item in tied
    }
    if len(identities) > 1:
        return None
    return tied[0]


def _annotation_from_definition_span(
    definition: Ab3PDefinition,
    *,
    document: DocumentText,
    span_kind: str,
) -> dict[str, Any] | None:
    if not definition.resolved or definition.chunk_id is None:
        return None
    chunk = document.chunk_by_id(definition.chunk_id)
    if chunk is None:
        return None
    if span_kind == "long_form":
        document_start = definition.long_form_document_start
        document_end = definition.long_form_document_end
        recognition_source = "ab3p_definition_long_form"
    else:
        document_start = definition.document_start
        document_end = definition.document_end
        recognition_source = "ab3p_definition_short_form"
    if document_start is None or document_end is None:
        return None
    start = document_start - chunk.document_start
    end = document_end - chunk.document_start
    mention = str(chunk.source.get("chunk") or "")[start:end]
    return _annotation_row(
        chunk=chunk,
        start=start,
        end=end,
        document_start=document_start,
        document_end=document_end,
        mention=mention,
        definition=definition,
        recognition_source=recognition_source,
    )


def _annotation_row(
    *,
    chunk: Any,
    start: int,
    end: int,
    document_start: int,
    document_end: int,
    mention: str,
    definition: Ab3PDefinition,
    recognition_source: str,
) -> dict[str, Any]:
    row = _source_projection(chunk.source)
    row.update(
        {
            "mention_id": (
                f"{chunk.source.get('base') or chunk.source.get('doc_key') or 'chunk'}:"
                f"ab3p:{start}:{end}:{definition.entity_type}"
            ),
            "mention": mention,
            "entity_type": definition.entity_type,
            "concept_id": definition.concept_id,
            "preferred_label": definition.preferred_label,
            "matched_term": definition.matched_term,
            "term_kind": definition.term_kind,
            "resource_version": definition.resource_version,
            "start": start,
            "end": end,
            "document_start": document_start,
            "document_end": document_end,
            "offset_scope": "chunk",
            "normalization_status": "normalized",
            "normalization_source": recognition_source,
            "recognition_source": (recognition_source.replace("ab3p_", "validated_parenthetical_")
                if definition.definition_detector != "ab3p" else recognition_source),
            "definition_id": definition.definition_id,
            "definition_detector": definition.definition_detector,
            "expanded_long_form": definition.long_form,
            "matched_abbreviation_key": definition.key,
            "ab3p_match_method": definition.match_method,
            "locked": True,
        }
    )
    if definition.entity_type == "cell":
        row["cell_ontology_id"] = definition.concept_id
        row["cell_ontology_label"] = definition.preferred_label
        row["normalization_system"] = "Cell Ontology"
    elif definition.entity_type == "hormone":
        row["hormone_id"] = definition.concept_id
        row["mesh_id"] = definition.concept_id
        row["normalization_system"] = "MeSH"
        row["source_entity_type"] = "Chemical"
    return {key: value for key, value in row.items() if value is not None}


def annotations_from_context(
    document: DocumentText,
    context: DocumentAbbreviationContext,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str, str]] = set()

    # Definition long-form and short-form spans are direct exact annotations.
    for definition in context.definitions:
        for span_kind in ("long_form", "short_form"):
            row = _annotation_from_definition_span(
                definition,
                document=document,
                span_kind=span_kind,
            )
            if row is None:
                continue
            signature = (
                str(row.get("chunk_id") or ""),
                int(row["start"]),
                int(row["end"]),
                str(row.get("entity_type") or ""),
                str(row.get("concept_id") or ""),
            )
            if signature not in seen:
                seen.add(signature)
                output.append(row)

    # Every known short form is then found throughout the same document. The
    # locally applicable definition is selected per occurrence, so repeated
    # definitions are not collapsed to one document-wide meaning.
    for key in context.all_keys:
        definitions = context.definitions_by_key.get(key, [])
        surfaces = abbreviation_surfaces(definitions)
        patterns = [
            re.compile(
                rf"(?<!\w){_flexible_literal(surface)}(?!\w)",
                re.IGNORECASE,
            )
            for surface in surfaces
            if surface
        ]
        occurrences: dict[tuple[int, int], re.Match[str]] = {}
        for pattern in patterns:
            for match in pattern.finditer(document.text):
                occurrences.setdefault(match.span(), match)
        for (document_start, document_end), match in sorted(occurrences.items()):
            chunk = document.chunk_for_span(document_start, document_end)
            if chunk is None:
                continue
            start = document_start - chunk.document_start
            end = document_end - chunk.document_start
            existing_overlap = any(
                str(row.get("chunk_id") or "") == chunk.chunk_id
                and spans_overlap(
                    (start, end),
                    (int(row.get("start") or 0), int(row.get("end") or 0)),
                )
                and str(row.get("entity_type") or "")
                in {
                    str(definition.entity_type or "") for definition in definitions
                }
                for row in output
            )
            if existing_overlap:
                continue
            selected = select_contextual_definition(
                definitions,
                occurrence_document_start=document_start,
                chunk_id=chunk.chunk_id,
                section=chunk.section,
            )
            if selected is None or not selected.resolved:
                continue
            row = _annotation_row(
                chunk=chunk,
                start=start,
                end=end,
                document_start=document_start,
                document_end=document_end,
                mention=match.group(0),
                definition=selected,
                recognition_source="ab3p_document_short_form",
            )
            signature = (
                chunk.chunk_id,
                start,
                end,
                str(row.get("entity_type") or ""),
                str(row.get("concept_id") or ""),
            )
            if signature not in seen:
                seen.add(signature)
                output.append(row)

    output.sort(
        key=lambda row: (
            document.chunk_by_id(row.get("chunk_id")).chunk_order
            if document.chunk_by_id(row.get("chunk_id")) is not None
            else 10**9,
            int(row.get("start") or 0),
            int(row.get("end") or 0),
            str(row.get("entity_type") or ""),
        )
    )
    return output


def definition_to_dict(definition: Ab3PDefinition) -> dict[str, Any]:
    keys = (
        "short_form",
        "long_form",
        "key",
        "stable_index",
        "definition_id",
        "definition_detector",
        "location_status",
        "resolution_status",
        "definition_start",
        "definition_end",
        "document_start",
        "document_end",
        "long_form_document_start",
        "long_form_document_end",
        "chunk_id",
        "section",
        "chunk_start",
        "long_form_chunk_start",
        "entity_type",
        "concept_id",
        "preferred_label",
        "matched_term",
        "term_kind",
        "resource_version",
        "match_method",
    )
    return {
        key: getattr(definition, key)
        for key in keys
        if getattr(definition, key) is not None
    }


def definition_from_dict(row: Mapping[str, Any]) -> Ab3PDefinition:
    definition = Ab3PDefinition(
        short_form=str(row.get("short_form") or ""),
        long_form=str(row.get("long_form") or ""),
        key=str(row.get("key") or ""),
        stable_index=int(row.get("stable_index") or 0),
        definition_id=str(row.get("definition_id") or ""),
        definition_detector=str(row.get("definition_detector") or "ab3p"),
        location_status=str(row.get("location_status") or "unlocated"),
        resolution_status=str(row.get("resolution_status") or "unmatched"),
        definition_start=_optional_int(row.get("definition_start")),
        definition_end=_optional_int(row.get("definition_end")),
        document_start=_optional_int(row.get("document_start")),
        document_end=_optional_int(row.get("document_end")),
        long_form_document_start=_optional_int(row.get("long_form_document_start")),
        long_form_document_end=_optional_int(row.get("long_form_document_end")),
        chunk_id=str(row.get("chunk_id")) if row.get("chunk_id") is not None else None,
        section=str(row.get("section")) if row.get("section") is not None else None,
        chunk_start=_optional_int(row.get("chunk_start")),
        long_form_chunk_start=_optional_int(row.get("long_form_chunk_start")),
        entity_type=str(row.get("entity_type")) if row.get("entity_type") else None,
        concept_id=str(row.get("concept_id")) if row.get("concept_id") else None,
        preferred_label=(
            str(row.get("preferred_label")) if row.get("preferred_label") else None
        ),
        matched_term=str(row.get("matched_term")) if row.get("matched_term") else None,
        term_kind=str(row.get("term_kind")) if row.get("term_kind") else None,
        resource_version=(
            str(row.get("resource_version")) if row.get("resource_version") else None
        ),
        match_method=str(row.get("match_method")) if row.get("match_method") else None,
    )
    if definition.entity_type == "cell" and definition.concept_id:
        from backend.cellexlink_lite.normalization import OntologyMatch

        definition.ontology_match = OntologyMatch(
            identifier=definition.concept_id,
            preferred_label=str(definition.preferred_label or ""),
            matched_alias=str(definition.matched_term or definition.long_form),
            raw_cosine=1.0,
            exact_unique_alias=True,
            final_score=1.0,
        )
    return definition


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def context_to_dict(
    context: DocumentAbbreviationContext,
    *,
    hormone_lexicon_source: str,
) -> dict[str, Any]:
    return {
        "schema": ABBREVIATION_CONTEXT_SCHEMA,
        "pipeline_version": ABBREVIATION_PREPASS_VERSION,
        "document_key": context.document_key,
        "ab3p_status": context.ab3p_status,
        "hormone_lexicon_source": hormone_lexicon_source,
        "hormone_resource_version": MESH_HORMONE_RESOURCE_VERSION,
        "definitions": [definition_to_dict(item) for item in context.definitions],
    }


def context_from_dict(payload: Mapping[str, Any]) -> DocumentAbbreviationContext:
    raw_definitions = payload.get("definitions") or []
    definitions = [
        definition_from_dict(item)
        for item in raw_definitions
        if isinstance(item, Mapping)
    ]
    by_key: dict[str, list[Ab3PDefinition]] = defaultdict(list)
    all_keys: list[str] = []
    stable: dict[str, int] = {}
    for definition in definitions:
        for lookup_key in short_form_key_candidates(definition.short_form):
            by_key[lookup_key].append(definition)
        if definition.key not in stable:
            stable[definition.key] = len(all_keys)
            all_keys.append(definition.key)
    return DocumentAbbreviationContext(
        document_key=str(payload.get("document_key") or ""),
        ab3p_status=str(payload.get("ab3p_status") or "not_run"),
        definitions=definitions,
        definitions_by_key=dict(by_key),
        all_keys=all_keys,
        key_to_stable_index=stable,
    )


def load_document_context(path: str | Path) -> DocumentAbbreviationContext:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid abbreviation context: {path}")
    return context_from_dict(payload)


def run_abbreviation_prepass(
    args: Any,
    manifest: dict[str, Any],
    progress: ProgressSink,
) -> dict[str, Any]:
    """Run document abbreviation and exact-entity recovery before CellExLink NER."""

    entries = [dict(entry) for entry in manifest.get("entries") or []]
    total = len(entries)
    statuses: Counter[str] = Counter()
    stats: dict[str, Any] = {
        "papers_total": total,
        "papers_processed": 0,
        "ab3p_calls": 0,
        "ab3p_health_check": "disabled" if args.disable_abbreviations else "pending",
        "ab3p_definitions": 0,
        "located_definitions": 0,
        "unlocated_definitions": 0,
        "resolved_cell_definitions": 0,
        "resolved_hormone_definitions": 0,
        "unmatched_definitions": 0,
        "ambiguous_definitions": 0,
        "ab3p_cell_annotations": 0,
        "ab3p_hormone_annotations": 0,
        "hormone_lexicon_source": "disabled",
        "hormone_lexicon_entries": 0,
        "ab3p_document_statuses": {},
    }

    if not entries:
        progress.emit(
            stage="abbreviation_prepass",
            percent=100,
            message="No papers required abbreviation preprocessing.",
            stats=stats,
            force=True,
        )
        return stats

    progress.emit(
        stage="abbreviation_prepass",
        percent=1,
        message="Loading Cell Ontology, MeSH hormones and human HGNC nomenclature...",
        stats=stats,
        force=True,
    )
    if not args.disable_abbreviations:
        ensure_ab3p_healthy()
        stats["ab3p_health_check"] = "passed"
    hormone_entries, hormone_source = ensure_hormone_lexicon(
        getattr(args, "hormone_lexicon_path", None) or DEFAULT_HORMONE_LEXICON_PATH
    )
    stats["hormone_lexicon_source"] = hormone_source
    stats["hormone_lexicon_entries"] = len(hormone_entries)
    target_index = RecoveryResources(
        ontology_path=Path(args.ontology_path),
        ontology_version=manifest.get("ontology_version"),
        hormone_entries=hormone_entries,
        abbreviations_path=Path(getattr(args, "abbreviations_path", None) or DEFAULT_ABBREVIATIONS_PATH),
        gene_matcher=load_local_hgnc(),
    )

    for index, entry in enumerate(entries, start=1):
        chunk_records = list(_iter_jsonl(Path(entry["chunk_path"])))
        document_key = str(
            entry.get("paper_identity")
            or (chunk_records[0].get("doc_key") if chunk_records else "")
            or f"paper-{index}"
        )
        document = build_document_text(chunk_records, document_key=document_key)
        pairs = []
        supplements = set()
        if not args.disable_abbreviations:
            pairs = run_ab3p_for_document(document.text, document_key=document_key)
            stats["ab3p_calls"] += 1
            pairs, supplements = supplement_parenthetical_pairs(document, pairs, target_index)
        context = build_document_context(
            document,
            pairs,
            target_index=target_index,
        )
        for definition in context.definitions:
            if (definition.short_form, definition.long_form) in supplements:
                definition.definition_detector = "resource_validated_parenthetical"
        annotations = annotations_from_context(document, context)
        annotations.extend(recover_document_entities(document, context, target_index))
        annotations = [enrich_gene_row(row, target_index.genes) for row in annotations]
        from backend.pipeline.receptor_annotations import reconcile_receptor_annotations
        annotations = [row for chunk in document.chunks
            for row in reconcile_receptor_annotations(str(chunk.source.get("chunk") or ""),
                [item for item in annotations if str(item.get("chunk_id")) == str(chunk.chunk_id)],
                matcher=target_index.genes)]
        annotations = apply_document_constraints(annotations,
            document_abbreviation_constraints(document, context, target_index.genes))
        annotations = [row for chunk in document.chunks for row in sanitize_source_annotations(
            str(chunk.source.get("chunk") or ""),
            [value for value in annotations if str(value.get("chunk_id")) == str(chunk.chunk_id)],
            cell_surface_allowed=target_index.cell_surface_allowed)]
        annotations = prune_cell_fragments(annotations)
        stats["supplemental_parenthetical_definitions"] = int(stats.get("supplemental_parenthetical_definitions", 0)) + len(supplements)
        stats["gene_recovery_annotations"] = int(stats.get("gene_recovery_annotations", 0)) + sum(row.get("entity_type") == "gene" for row in annotations)
        _atomic_write_json(
            Path(entry["abbreviation_context_path"]),
            context_to_dict(context, hormone_lexicon_source=hormone_source),
        )
        _atomic_write_gzip_jsonl(
            Path(entry["abbreviation_annotations_path"]),
            annotations,
        )

        statuses[context.ab3p_status] += 1
        stats["ab3p_definitions"] += len(context.definitions)
        stats["located_definitions"] += sum(
            item.location_status == "located" for item in context.definitions
        )
        stats["unlocated_definitions"] += sum(
            item.location_status != "located" for item in context.definitions
        )
        stats["resolved_cell_definitions"] += context.cell_definition_count
        stats["resolved_hormone_definitions"] += context.hormone_definition_count
        stats["unmatched_definitions"] += sum(
            item.resolution_status == "unmatched" for item in context.definitions
        )
        stats["ambiguous_definitions"] += sum(
            item.resolution_status.startswith("ambiguous")
            for item in context.definitions
        )
        stats["ab3p_cell_annotations"] += sum(
            row.get("entity_type") == "cell" for row in annotations
        )
        stats["ab3p_hormone_annotations"] += sum(
            row.get("entity_type") == "hormone" for row in annotations
        )
        stats["papers_processed"] = index
        stats["ab3p_document_statuses"] = dict(statuses)
        progress.emit(
            stage="abbreviation_prepass",
            percent=5 + 95 * index / max(1, total),
            message=(
                f"Processed abbreviations in {index} of {total} papers; "
                f"created {stats['ab3p_cell_annotations']:,} cell and "
                f"{stats['ab3p_hormone_annotations']:,} hormone annotations."
            ),
            stats=stats,
            force=True,
        )

    return stats


__all__ = [
    "ABBREVIATION_ANNOTATIONS_FILENAME",
    "ABBREVIATION_CONTEXT_FILENAME",
    "ABBREVIATION_CONTEXT_SCHEMA",
    "ABBREVIATION_PREPASS_VERSION",
    "annotations_from_context",
    "build_document_context",
    "context_from_dict",
    "context_to_dict",
    "definition_from_dict",
    "definition_to_dict",
    "load_document_context",
    "run_abbreviation_prepass",
    "select_contextual_definition",
]
