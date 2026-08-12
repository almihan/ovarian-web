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

RELATION_PIPELINE_VERSION = "ovarian-openai-online-relations-v1"
RELATION_OUTPUT_SCHEMA = "chunk-biological-relations-v1"
PROMPT_VERSION = "ovarian-relations-prompt-v1"

BASE_PREDICATES: tuple[str, ...] = (
    "activation",
    "inhibition",
    "proliferation",
    "secreted",
    "binding",
    "upregulation",
    "downregulation",
)
CACHE_TARGET_REQUESTS_PER_SHARD = 15

ENTITY_PREFIX = {"cell": "C", "gene": "G", "protein": "G", "hormone": "H"}
PREFIX_TYPE = {"C": "cell", "G": "gene", "H": "hormone"}
ENTITY_PRIORITY = {"cell": 0, "hormone": 1, "gene": 2, "protein": 2}
_ID_RE = re.compile(r"^[CGH]\d+$")
_CELL_ID_RE = re.compile(r"^C\d+$")

# Static instructions are intentionally placed before the changing chunk.  The
# request schema is also static, so repeated requests share a long exact prefix
# that is eligible for automatic prompt caching.
SYSTEM_INSTRUCTIONS = """
You extract only explicit biological relations between tagged entities in ovarian biomedical text.

Tags:
- [C1]...[/C1]: cell or cell type
- [G1]...[/G1]: gene or protein
- [H1]...[/H1]: hormone
IDs are local to the supplied chunk. Use only visible IDs.

Return one JSON object with the key "triples". Each triple contains exactly:
- subject: visible entity ID
- predicate: allowed predicate
- object: visible entity ID
- cell_context: array of visible C IDs

Extract only relations explicitly asserted in the chunk. Do not infer from background knowledge, typical biology, co-occurrence, correlation, an experimental aim, or a cited result not stated in the supplied text. Respect negation, uncertainty, comparison, attribution, passive voice, and cross-sentence references. Prefer no relation over a weak inference.

Allowed predicates and directions:
1. activation: G -> C, H -> C, or H -> G. Use for explicit activation, stimulation, induction, triggering, or promotion of cell behavior or gene/protein function. For H -> G expression or abundance changes, use upregulation instead.
2. inhibition: G -> C, H -> C, or H -> G. Use for explicit inhibition, blockade, suppression, prevention, attenuation, or impairment. For H -> G expression or abundance changes, use downregulation instead.
3. proliferation: G -> C or H -> C. Use only when the tagged cell population explicitly proliferates, divides, expands in cell number, or undergoes mitosis.
4. secreted: C -> G or C -> H. For C -> G, require explicit secretion, release, or export. For C -> H, use secreted when the cell explicitly secretes, releases, produces, generates, synthesizes, or is identified as the cellular source of the hormone.
5. binding: H -> G. Use only for explicit binding, receptor engagement, ligand-receptor association, or direct physical interaction.
6. upregulation: H -> G. Use only when the hormone explicitly increases expression, transcription, translation, or abundance of the gene/protein.
7. downregulation: H -> G. Use only when the hormone explicitly decreases expression, transcription, translation, or abundance of the gene/protein.

Validation rules:
- Never output G -> G, C -> C, self-relations, or a predicate-direction combination outside the matrix.
- Never output C secreted C, H secreted C, or G secreted C.
- A measured change in concentration, staining, density, expression, or abundance is not secretion unless the tagged cell is explicitly identified as the source.
- Shared pathway membership, treatment response, or statistical association is not binding.
- A change in activation, differentiation, viability, migration, morphology, follicle size, tumor mass, or marker intensity is not automatically proliferation.
- Emit each semantic relation once and return at most 50 unique triples.

Cell context:
- For H -> G relations, include every tagged C entity explicitly identified as the cell in which the relation occurs.
- Do not use a nearby cell mention unless the wording links it to that hormone-gene relation.
- If the relation is explicit but no tagged cell context is linked, return an empty array.
- For all other relations, return an empty cell_context array.

Examples:
- "[G1]KITLG[/G1] activated [C1]oocytes[/C1]" -> G1 activation C1.
- "[C1]Oocytes[/C1] were activated by [G1]KITLG[/G1]" -> G1 activation C1.
- "[H1]FSH[/H1] increased the number of [C1]granulosa cells[/C1]" -> H1 proliferation C1.
- "[H1]Estradiol[/H1] increased [G1]FSHR[/G1] expression in [C1]granulosa cells[/C1]" -> H1 upregulation G1 with cell_context [C1].
- "[H1]Estradiol[/H1] activated [G1]ESR1[/G1] signaling" -> H1 activation G1.
- "[C1]Granulosa cells[/C1] released [G1]VEGFA[/G1]" -> C1 secreted G1.
- "[C1]Granulosa cells[/C1] synthesized [H1]estradiol[/H1]" -> C1 secreted H1.
- "[H1]Estradiol[/H1] bound [G1]ESR1[/G1] in [C1]granulosa cells[/C1]" -> H1 binding G1 with cell_context [C1].
- "Neither [H1]estradiol[/H1] nor vehicle altered [G1]FSHR[/G1]" -> no relation.
- "We tested whether [H1]estradiol[/H1] increases [G1]FSHR[/G1]" -> no relation unless the result is also asserted.

If no valid relation is explicit, return {"triples":[]}.
""".strip()


def allowed_predicates() -> tuple[str, ...]:
    return BASE_PREDICATES


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
                        "cell_context": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "subject",
                        "predicate",
                        "object",
                        "cell_context",
                    ],
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
    """Enforce the exact entity-direction matrix requested by the project."""

    if subject == object_ or _ID_RE.fullmatch(subject) is None:
        return False
    if _ID_RE.fullmatch(object_) is None:
        return False
    source_type = PREFIX_TYPE.get(subject[0])
    target_type = PREFIX_TYPE.get(object_[0])
    pair = (source_type, target_type)

    matrix: dict[str, set[tuple[str, str]]] = {
        "activation": {("gene", "cell"), ("hormone", "cell"), ("hormone", "gene")},
        "inhibition": {("gene", "cell"), ("hormone", "cell"), ("hormone", "gene")},
        "proliferation": {("gene", "cell"), ("hormone", "cell")},
        "secreted": {("cell", "gene"), ("cell", "hormone")},
        "binding": {("hormone", "gene")},
        "upregulation": {("hormone", "gene")},
        "downregulation": {("hormone", "gene")},
    }
    return pair in matrix.get(predicate, set())


def is_hormone_gene_relation(subject: str, object_: str) -> bool:
    return {subject[:1], object_[:1]} == {"G", "H"}


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
        mention = annotation.get("mention")
        # Stage 2 stores exclusive ends. This compatibility branch only adjusts
        # an inclusive end when the mention length proves that convention.
        if (
            isinstance(mention, str)
            and end - start + 1 == len(mention)
            and text[start : end + 1] == mention
        ):
            end += 1
    if start < 0 or end <= start or end > len(text):
        return None
    return start, end


def _normalized_entity_key(
    annotation: Mapping[str, Any], start: int, end: int
) -> str:
    entity_type = str(annotation.get("obj") or "").casefold()
    # Stage 2 can label the same normalized HGNC/NCBI entity as either gene or
    # protein. Both map to G tags, so normalize the key as well and reuse one ID.
    key_type = "gene" if entity_type == "protein" else entity_type
    candidates: tuple[str, ...]
    if entity_type == "cell":
        candidates = ("concept_id", "cell_ontology_id", "normalized_id")
    elif entity_type in {"gene", "protein"}:
        candidates = ("gene_id", "concept_id", "normalized_id")
    else:
        candidates = (
            "hormone_id",
            "chemical_id",
            "concept_id",
            "normalized_id",
        )
    for field in candidates:
        value = annotation.get(field)
        if value is not None and str(value).strip():
            return f"{key_type}:{field}:{str(value).strip()}"
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
        "preferred_label": annotation.get("preferred_label"),
    }
    if span.prefix == "G":
        row["gene_id"] = annotation.get("gene_id")
        for field in ("tax_id", "tax_name", "taxonomy_source"):
            value = annotation.get(field)
            if value not in (None, ""):
                row[field] = value
    elif span.prefix == "H":
        row["hormone_id"] = annotation.get("hormone_id") or annotation.get(
            "chemical_id"
        )
    return {key: value for key, value in row.items() if value not in (None, "")}


def _select_non_crossing_spans(spans: list[_Span]) -> tuple[list[_Span], int]:
    # Long, normalized spans win only when two recognizers produce a crossing
    # overlap. Contained and exact spans remain representable as nested tags.
    ranked = sorted(
        spans,
        key=lambda item: (
            -(item.end - item.start),
            ENTITY_PRIORITY.get(item.entity_type, 99),
            item.start,
            item.end,
            item.key,
        ),
    )
    selected: list[_Span] = []
    dropped = 0
    for candidate in ranked:
        if any(_crosses(candidate, kept) for kept in selected):
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

    spans: list[_Span] = []
    seen: set[tuple[Any, ...]] = set()
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
    prefix = (base_key.strip() or "ovarian-relations-v4")[: 64 - len(suffix)]
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
        "instructions": SYSTEM_INSTRUCTIONS,
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
    require_hormone_gene_cell_context: bool,
    max_triples: int = 50,
) -> list[dict[str, Any]]:
    if not isinstance(parsed, Mapping):
        raise ValueError("The model response is not a JSON object.")
    raw_triples = parsed.get("triples")
    if not isinstance(raw_triples, list):
        raise ValueError("The model response has no triples array.")

    allowed_ids = set(entities)
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for raw in raw_triples[:max_triples]:
        if not isinstance(raw, Mapping):
            continue
        subject = str(raw.get("subject") or "").strip()
        predicate = str(raw.get("predicate") or "").strip().casefold()
        object_ = str(raw.get("object") or "").strip()
        if subject not in allowed_ids or object_ not in allowed_ids:
            continue
        if not relation_allowed(subject, predicate, object_):
            continue

        context: list[str] = []
        if is_hormone_gene_relation(subject, object_):
            raw_context = raw.get("cell_context")
            if isinstance(raw_context, list):
                context = sorted(
                    {
                        str(value).strip()
                        for value in raw_context
                        if isinstance(value, str)
                        and _CELL_ID_RE.fullmatch(value.strip()) is not None
                        and value.strip() in allowed_ids
                    },
                    key=lambda value: int(value[1:]),
                )
            if require_hormone_gene_cell_context and not context:
                continue
        # Context on non-H/G relations is prohibited rather than trusted.
        key = (subject, predicate, object_, tuple(context))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "cell_context": context,
            }
        )

    cleaned.sort(
        key=lambda row: (
            row["subject"][0],
            int(row["subject"][1:]),
            row["predicate"],
            row["object"][0],
            int(row["object"][1:]),
            tuple(row["cell_context"]),
        )
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
        context = triple.get("cell_context")
        if isinstance(context, list):
            referenced.update(str(value) for value in context)
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
    "BASE_PREDICATES",
    "PROMPT_VERSION",
    "PreparedChunk",
    "RELATION_OUTPUT_SCHEMA",
    "RELATION_PIPELINE_VERSION",
    "SYSTEM_INSTRUCTIONS",
    "allowed_predicates",
    "compact_json",
    "effective_prompt_cache_shards",
    "extract_response_text",
    "has_possible_allowed_pair",
    "is_hormone_gene_relation",
    "output_row",
    "prepare_chunk",
    "prompt_cache_key_for_request",
    "relation_allowed",
    "request_body",
    "response_schema",
    "sanitize_triples",
]
