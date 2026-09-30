"""Cost-conscious relation extraction helpers for tagged ovarian-literature chunks.

This module is deliberately independent of FastAPI and the OpenAI client.  It
prepares one compact request per eligible chunk, validates every returned
relation locally, and emits small text-free rows that can later feed network
generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from backend.pipeline.identifier_identity import canonical_identifier

RELATION_PIPELINE_VERSION = "ovarian-openai-online-four-relations-cell-context-v3"
RELATION_OUTPUT_SCHEMA = "chunk-biological-relations-four-relations-direct-proximal"
PROMPT_VERSION = "ovarian-relations-four-relations-direct-proximal"

ALLOWED_PREDICATES: tuple[str, ...] = (
    "activation",
    "inhibition",
    "proliferation",
    "secreted",
)
CACHE_TARGET_REQUESTS_PER_SHARD = 15

ENTITY_PREFIX = {"cell": "C", "gene": "G", "protein": "G", "hormone": "H"}
ENTITY_PRIORITY = {"cell": 0, "hormone": 1, "gene": 2, "protein": 2}
_ID_RE = re.compile(r"^[CGH]\d+$")

# Static instructions are intentionally placed before the changing chunk.  The
# request schema is also static, so repeated requests share a long exact prefix
# that is eligible for automatic prompt caching.
SYSTEM = """
Extract only biological relations that are explicitly supported by the text
between tagged entities.

Tags:
- [C1]...[/C1] = cell type
- [G1]...[/G1] = gene/protein
- [H1]...[/H1] = hormone

Trust the entity type assigned by each tag.

Return exactly one JSON object and no other text:
{"triples":[{"subject":"G1","predicate":"activation","object":"C1","conditions":""}]}

Allowed predicates and directions:

activation:
Use for both functional activation and upregulation.
The subject explicitly increases the activity or signaling of a gene/protein; increases its expression, transcription, translation, mRNA level, protein level, or abundance; or induces a cell to enter an activated, effector, differentiated, or polarized state. Do not infer activation from correlation, binding, secretion, migration,
invasion, EMT, survival, or proliferation alone.
  
- inhibition:
Use for both functional inhibition and downregulation.
The subject explicitly decreases the activity, signaling, expression, abundance, stability, survival, viability, proliferation, or another clearly defined biological function of the object. For a gene/protein object, this includes reduced molecular activity or signaling; decreased transcription, mRNA level, protein level, or abundance; and increased degradation or inactivation. For a hormone object, this includes degradation or inactivation that reduces
its abundance, stability, bioavailability, or biological activity. For a cell object, this includes suppressed activation or function, reduced proliferation or viability, apoptosis, cytotoxicity, cell killing, or another form of cell death.
  
- proliferation: C->C, G->C, H->C
  The subject increases proliferation, cell division, mitosis, or proliferative or clonal expansion of the object cell. Include increased cell numbers in an expansion context.

- secreted: C->G, C->H
  The subject cell secretes, releases, sheds, produce or exports the object protein or hormone.

Rules:
- Use only tagged IDs present in the text and output them without brackets.
- Both the subject and object must be tagged entities.
- Preserve biological direction, including relations written in passive voice.
- Exact predicate words are not required; use the explicit biological meaning.
-A cell merely expressing or being positive for a marker is not a causal C->G relation.
- Do not infer indirect, transitive, or pathway-mediated relations. Keep direct, explicit, proximal, and causally supported relations.
- Use cross-sentence evidence only when the reference and direction are unambiguous.
- Exclude study aims, unsupported hypotheses, explicit null findings, correlations, co-occurrence, binding alone, directionless associations, unsupported directions, and self-relations.
- If no valid relation exists, return {"triples":[]}.
- "conditions" must be one concise string containing explicitly stated qualifications, such as co-treatment, dose, timing, pretreatment, absence, or cellular context. Use "" when no condition is stated.
-When A modifies C with B, keep A as subject and B as a condition. Example: "A acts synergistically with B or D to increase proliferation of C" gives A -> proliferation -> C, conditions = "in synergy with B or D".
""".strip()


def allowed_predicates() -> tuple[str, ...]:
    return ALLOWED_PREDICATES


def response_schema() -> dict[str, Any]:
    """Return one invariant schema for every request in a deployment."""

    return {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "predicate": {
                            "type": "string",
                            "enum": list(allowed_predicates()),
                        },
                        "object": {"type": "string"},
                    },
                    "required": ["subject", "predicate", "object"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["triples"],
        "additionalProperties": False,
    }


def relation_allowed(
    subject: str,
    predicate: str,
    object_: str,
) -> bool:
    """Validate one relation against the four-predicate direction rules.

    Activation and inhibition have no entity-type pairing restriction: any C,
    G, or H tag may be the subject or object. Proliferation and secretion keep
    their biologically constrained directions.
    """

    if subject == object_:
        return False
    if _ID_RE.fullmatch(subject) is None or _ID_RE.fullmatch(object_) is None:
        return False

    normalized_predicate = str(predicate or "").strip().casefold()
    if normalized_predicate not in ALLOWED_PREDICATES:
        return False

    if normalized_predicate in {"activation", "inhibition"}:
        return True
    if normalized_predicate == "proliferation":
        return object_.startswith("C")
    if normalized_predicate == "secreted":
        return subject.startswith("C") and object_[0] in {"G", "H"}
    return False


def has_possible_allowed_pair(entity_ids: Iterable[str]) -> bool:
    ids = list(entity_ids)
    for subject in ids:
        for object_ in ids:
            if subject == object_:
                continue
            for predicate in allowed_predicates():
                if relation_allowed(subject, predicate, object_):
                    return True
    return False


def _as_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _annotation_span(annotation: Mapping[str, Any], text: str) -> tuple[int, int] | None:
    if "offset" in annotation and "length" in annotation:
        start = _as_int(annotation.get("offset"))
        length = _as_int(annotation.get("length"))
        if start is not None and length is not None and start >= 0 and length > 0:
            end = start + length
        else:
            return None
    else:
        start = _as_int(annotation.get("start"))
        end = _as_int(annotation.get("end"))
        if start is None or end is None:
            return None
    if start < 0 or end <= start or end > len(text):
        return None
    return start, end


def _normalized_entity_key(
    annotation: Mapping[str, Any], start: int, end: int
) -> str:
    entity_type = str(annotation.get("obj") or "").casefold()
    # Stage 2 uses the approved HGNC ID as the canonical gene/protein identity.
    # Gene and protein source labels both map to G tags and reuse one ID.
    key_type = "gene" if entity_type == "protein" else entity_type
    candidates: tuple[str, ...]
    if entity_type == "cell":
        candidates = ("concept_id", "cell_ontology_id", "normalized_id")
    elif entity_type in {"gene", "protein"}:
        candidates = (
            "concept_id",
            "hgnc_id",
            "normalized_id",
            "ncbi_gene_id",
            "gene_id",
            "pubtator_gene_id",
        )
    else:
        candidates = (
            "concept_id",
            "mesh_id",
            "hormone_id",
            "normalized_id",
            "pubtator_mesh_id",
            "chemical_id",
            # Rolling-deployment compatibility for older hormone rows.
            "chebi_id",
            "hgnc_id",
        )
    for field in candidates:
        value = annotation.get(field)
        if value is not None and str(value).strip():
            namespace, canonical_value = canonical_identifier(
                key_type,
                field,
                value,
            )
            return f"{key_type}:{namespace or field}:{canonical_value or str(value).strip()}"
    mention = str(annotation.get("mention") or "").strip().casefold()
    if mention:
        return f"{key_type}:mention:{mention}"
    return f"{key_type}:span:{start}:{end}"


@dataclass(slots=True)
class _Span:
    start: int
    end: int
    key: str
    prefix: str
    entity_type: str
    annotation: dict[str, Any]
    tag: str = ""


@dataclass(slots=True)
class PreparedChunk:
    custom_id: str
    identity: dict[str, Any]
    tagged_text: str
    entities: dict[str, dict[str, Any]]
    eligible: bool
    valid_annotation_count: int
    dropped_overlap_count: int


def _crosses(left: _Span, right: _Span) -> bool:
    return (
        left.start < right.start < left.end < right.end
        or right.start < left.start < right.end < left.end
    )


def _compact_entity(tag: str, span: _Span) -> dict[str, Any]:
    annotation = span.annotation
    row: dict[str, Any] = {
        "id": tag,
        "obj": "gene" if span.entity_type == "protein" else span.entity_type,
        "mention": str(annotation.get("mention") or ""),
        "concept_id": annotation.get("concept_id"),
        "normalized_id": annotation.get("normalized_id")
        or annotation.get("concept_id"),
        "preferred_label": annotation.get("preferred_label"),
    }

    shared_fields = (
        "canonical_id_type",
        "canonical_name",
        "normalization_source",
        "normalization_status",
        "source_concept_id",
        "label_source",
        "identified_source",
        "recognition_source",
        "seed_evidence",
    )
    gene_fields = (
        "hgnc_id",
        "hgnc_group_id",
        "entity_granularity",
        "entity_role",
        "tax_id",
        "tax_name",
        "taxonomy_source",
        "expanded_long_form",
        "ncbi_gene_id",
        "uniprot_ids",
    )
    hormone_fields = (
        "hormone_id",
        "mesh_id",
        "pubtator_mesh_id",
        "chemical_id",
        "hormone_classification_source",
    )
    entity_fields = (
        gene_fields
        if span.prefix == "G"
        else hormone_fields
        if span.prefix == "H"
        else ()
    )
    fields = shared_fields + entity_fields
    for field in fields:
        value = annotation.get(field)
        if value not in (None, "", [], ()):
            row[field] = value

    return {key: value for key, value in row.items() if value not in (None, "", [], ())}


def _select_non_crossing_spans(spans: list[_Span]) -> tuple[list[_Span], int]:
    # Exact-span cell > hormone > gene; longer spans still win across types.
    type_order = {"cell": 0, "hormone": 1, "gene": 2, "protein": 2}
    span_type = {}
    for item in spans:
        pair = (item.start, item.end)
        span_type[pair] = min(type_order.get(item.entity_type, 99), span_type.get(pair, 99))
    eligible = [item for item in spans if type_order.get(item.entity_type, 99)
                == span_type[(item.start, item.end)]]
    ranked = sorted(eligible, key=lambda item: (
        -(item.end - item.start), item.start, item.end,
        type_order.get(item.entity_type, 99), item.key,
    ))
    selected: list[_Span] = []
    dropped = len(spans) - len(eligible)
    for candidate in ranked:
        if any(candidate.start < kept.end and candidate.end > kept.start
               for kept in selected):
            dropped += 1
            continue
        selected.append(candidate)
    selected.sort(
        key=lambda item: (
            item.start,
            -item.end,
            ENTITY_PRIORITY.get(item.entity_type, 99),
            item.key,
        )
    )
    return selected, dropped


def _render_tags(text: str, spans: Sequence[_Span]) -> str:
    starts: dict[int, list[_Span]] = {}
    ends: dict[int, list[_Span]] = {}
    for span in spans:
        starts.setdefault(span.start, []).append(span)
        ends.setdefault(span.end, []).append(span)

    pieces: list[str] = []
    cursor = 0
    for position in sorted(set(starts) | set(ends)):
        pieces.append(text[cursor:position])
        if position in ends:
            # Inner spans close first. Exact spans close in reverse opening order.
            closing = sorted(
                ends[position],
                key=lambda item: (
                    -item.start,
                    -ENTITY_PRIORITY.get(item.entity_type, 99),
                    item.tag,
                ),
            )
            pieces.extend(f"[/{span.tag}]" for span in closing)
        if position in starts:
            # Outer spans open first so contained tags remain well formed.
            opening = sorted(
                starts[position],
                key=lambda item: (
                    -item.end,
                    ENTITY_PRIORITY.get(item.entity_type, 99),
                    item.tag,
                ),
            )
            pieces.extend(f"[{span.tag}]" for span in opening)
        cursor = position
    pieces.append(text[cursor:])
    return "".join(pieces)


def prepare_chunk(
    *,
    row_index: int,
    source_row: Mapping[str, Any],
    annotation_row: Mapping[str, Any],
) -> PreparedChunk:
    """Join one Stage 1 text row to its aligned Stage 2 annotation row."""

    identity_fields = (
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
    identity = {field: annotation_row.get(field) for field in identity_fields}
    custom_id = f"r-{row_index:010d}"
    text = source_row.get("chunk")
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    raw_annotations = annotation_row.get("annotations")
    if not isinstance(raw_annotations, list):
        raw_annotations = []
    valid_annotation_count = sum(
        1 for annotation in raw_annotations if isinstance(annotation, Mapping)
    )
    from backend.pipeline.entity_artifacts import _resolve_annotation_conflicts
    raw_annotations = _resolve_annotation_conflicts(raw_annotations, text=text)
    cell_gene_overlap_drops = valid_annotation_count - len(raw_annotations)

    hormone_spans: set[tuple[int, int]] = set()
    for raw in raw_annotations:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("obj") or "").casefold() != "hormone":
            continue
        span = _annotation_span(raw, text)
        if span is not None:
            hormone_spans.add(span)

    spans: list[_Span] = []
    seen: set[tuple[Any, ...]] = set()
    same_span_gene_drops = 0
    for raw in raw_annotations:
        if not isinstance(raw, Mapping):
            continue
        entity_type = str(raw.get("obj") or "").casefold()
        prefix = ENTITY_PREFIX.get(entity_type)
        if prefix is None:
            continue
        span = _annotation_span(raw, text)
        if span is None:
            continue
        start, end = span
        if entity_type in {"gene", "protein"} and (start, end) in hormone_spans:
            same_span_gene_drops += 1
            continue
        key = _normalized_entity_key(raw, start, end)
        signature = (start, end, key, prefix)
        if signature in seen:
            continue
        seen.add(signature)
        spans.append(
            _Span(
                start=start,
                end=end,
                key=key,
                prefix=prefix,
                entity_type=entity_type,
                annotation=dict(raw),
            )
        )

    selected, dropped = _select_non_crossing_spans(spans)
    dropped += same_span_gene_drops + cell_gene_overlap_drops
    key_to_tag: dict[tuple[str, str], str] = {}
    next_index = {"C": 1, "G": 1, "H": 1}
    entities: dict[str, dict[str, Any]] = {}
    for span in selected:
        keyed = (span.prefix, span.key)
        tag = key_to_tag.get(keyed)
        if tag is None:
            tag = f"{span.prefix}{next_index[span.prefix]}"
            next_index[span.prefix] += 1
            key_to_tag[keyed] = tag
            entities[tag] = _compact_entity(tag, span)
        span.tag = tag

    tagged = _render_tags(text, selected) if selected else text
    eligible = len(entities) >= 2 and has_possible_allowed_pair(entities)
    return PreparedChunk(
        custom_id=custom_id,
        identity=identity,
        tagged_text=tagged,
        entities=entities,
        eligible=eligible,
        valid_annotation_count=len(selected),
        dropped_overlap_count=dropped,
    )


def minimal_user_input(tagged_text: str) -> str:
    # The tags already encode entity IDs and types, so repeating an entity table
    # would waste input tokens.
    return "Tagged ovarian-literature chunk:\n" + tagged_text


def prompt_cache_key_for_request(
    base_key: str,
    *,
    custom_id: str,
    shard_count: int,
) -> str:
    """Return a stable bounded cache-routing key for one request.

    Online workers may execute many requests in a burst. Stable sharding keeps
    each cache key below a high request rate while preserving repeated prefixes
    across windows and jobs. It has no effect on extraction semantics.
    """

    safe_shards = max(1, int(shard_count))
    digest = hashlib.blake2s(custom_id.encode("utf-8"), digest_size=4).digest()
    shard = int.from_bytes(digest, "big") % safe_shards
    width = max(1, len(str(safe_shards - 1)))
    suffix = f":{shard:0{width}d}"
    prefix = (base_key.strip() or "ovarian-relations-four-predicates")[: 64 - len(suffix)]
    return prefix + suffix


def effective_prompt_cache_shards(
    request_count: int,
    *,
    maximum_shards: int,
    target_requests_per_shard: int = CACHE_TARGET_REQUESTS_PER_SHARD,
) -> int:
    """Choose enough stable cache keys for a burst without wasting cache hits.

    Small or retry windows stay on fewer routing keys, while a 500-request
    window can spread across the configured maximum. The value is persisted
    in the pending-window state so retries reuse the same keys.
    """

    safe_count = max(0, int(request_count))
    safe_maximum = max(1, int(maximum_shards))
    safe_target = max(1, int(target_requests_per_shard))
    needed = max(1, math.ceil(safe_count / safe_target))
    return min(safe_maximum, needed)


def request_body(
    *,
    tagged_text: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str,
    cache_key: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM,
        "input": minimal_user_input(tagged_text),
        "max_output_tokens": max_output_tokens,
        "store": False,
        "prompt_cache_key": cache_key,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ovarian_relation_extraction",
                "strict": True,
                "schema": response_schema(),
            }
        },
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def extract_response_text(body: Mapping[str, Any]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    parts: list[str] = []
    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") in {"output_text", "text"}:
                    value = part.get("text")
                    if isinstance(value, str):
                        parts.append(value)
    return "".join(parts)


def sanitize_triples(
    parsed: Any,
    *,
    entities: Mapping[str, Mapping[str, Any]],
    max_triples: int | None = None,
) -> list[dict[str, str]]:
    """Validate, deduplicate, and sort every returned triple without truncation.

    There is no count limit by default. An explicit ``max_triples`` is a guard
    on valid unique triples, not a slice: overflow raises instead of silently
    choosing a subset based on the model's output order.
    """

    if max_triples is not None and (
        isinstance(max_triples, bool)
        or not isinstance(max_triples, int)
        or max_triples < 0
    ):
        raise ValueError("max_triples must be None or a non-negative integer.")

    if not isinstance(parsed, Mapping):
        raise ValueError("The model response is not a JSON object.")
    raw_triples = parsed.get("triples")
    if not isinstance(raw_triples, list):
        raise ValueError("The model response has no triples array.")

    allowed_ids = set(entities)
    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_triples:
        if not isinstance(raw, Mapping):
            continue
        subject = str(raw.get("subject") or "").strip()
        predicate = str(raw.get("predicate") or "").strip().casefold()
        object_ = str(raw.get("object") or "").strip()
        if subject not in allowed_ids or object_ not in allowed_ids:
            continue
        if not relation_allowed(subject, predicate, object_):
            continue

        key = (subject, predicate, object_)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": object_,
            }
        )

    cleaned.sort(
        key=lambda row: (
            row["subject"][0],
            int(row["subject"][1:]),
            row["predicate"],
            row["object"][0],
            int(row["object"][1:]),
        )
    )
    if max_triples is not None and len(cleaned) > max_triples:
        raise ValueError(
            f"The model response contains {len(cleaned)} valid unique triples, "
            f"exceeding max_triples={max_triples}. "
            "Refusing to discard relations silently."
        )
    return cleaned


def output_row(
    prepared: PreparedChunk,
    triples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    referenced: set[str] = set()
    for triple in triples:
        referenced.add(str(triple.get("subject") or ""))
        referenced.add(str(triple.get("object") or ""))
    entities = [
        prepared.entities[entity_id]
        for entity_id in sorted(
            referenced & set(prepared.entities),
            key=lambda value: (value[0], int(value[1:])),
        )
    ]
    row = dict(prepared.identity)
    row["entities"] = entities
    row["relations"] = [dict(triple) for triple in triples]
    return row


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "ALLOWED_PREDICATES",
    "PROMPT_VERSION",
    "PreparedChunk",
    "RELATION_OUTPUT_SCHEMA",
    "RELATION_PIPELINE_VERSION",
    "SYSTEM",
    "allowed_predicates",
    "compact_json",
    "effective_prompt_cache_shards",
    "extract_response_text",
    "has_possible_allowed_pair",
    "output_row",
    "prepare_chunk",
    "prompt_cache_key_for_request",
    "relation_allowed",
    "request_body",
    "response_schema",
    "sanitize_triples",
]
