"""Build the user-facing final paper annotation export.



Stage 4 still consumes the compact tag-based Stage 3 relation artifact. This

module creates a separate download artifact after Stage 3 by joining the

aligned Stage 1, Stage 2, and Stage 3 streams. The download keeps one JSONL row

per paper. Relations are retained only inside the chunk where they were

extracted; no duplicate paper-level relation list is emitted.



Each exported relation retains a plain-string conditions field, including an

empty string when no condition was supplied. Different condition strings are

kept as separate assertions. Older three-field input records receive an empty

string for compatibility; this does not recover previously omitted context.

"""



from __future__ import annotations



import gzip

import json

import os

import re

from itertools import zip_longest

from pathlib import Path

from typing import Any, Iterator, Mapping



from backend.pipeline.entity_artifacts import iter_jsonl, _resolve_annotation_conflicts

from backend.pipeline.identifier_identity import canonical_identifier

from backend.pipeline.relation_extraction import relation_allowed



FINAL_ANNOTATION_EXPORT_VERSION = "final-annotations-v6-cell-context-conditions"



_SPACE_RE = re.compile(r"\s+")

_EXCLUDED_ANNOTATION_FIELDS = frozenset(

    {

        "hgnc_record_status",

        "normalization_match_method",

        "tax_id",

        "tax_name",

        "taxonomy_source",

        "pubtator_tax_id",

    }

)

_ID_FIELDS: dict[str, tuple[str, ...]] = {

    "cell": ("concept_id", "cell_ontology_id", "normalized_id"),

    "gene": (

        "concept_id",

        "hgnc_id",

        "normalized_id",

        "ncbi_gene_id",

        # Rolling-deployment compatibility for older sidecars.

        "gene_id",

        "pubtator_gene_id",

    ),

    "hormone": (

        "concept_id",

        "mesh_id",

        "hormone_id",

        "normalized_id",

        "pubtator_mesh_id",

        "chemical_id",

        # Rolling-deployment compatibility for older hormone rows.

        "chebi_id",

        "hgnc_id",

        "ncbi_gene_id",

    ),

}





def _text(value: Any) -> str:

    if value is None:

        return ""

    return _SPACE_RE.sub(" ", str(value)).strip()





def _entity_type(entity: Mapping[str, Any]) -> str:

    value = _text(entity.get("obj")).casefold()

    if value == "protein":

        return "gene"

    return value if value in _ID_FIELDS else ""





def _entity_identifier(entity: Mapping[str, Any]) -> str:

    entity_type = _entity_type(entity)

    if not entity_type:

        return ""

    for field in _ID_FIELDS[entity_type]:

        value = entity.get(field)

        if not _text(value):

            continue

        _namespace, canonical_value = canonical_identifier(

            entity_type,

            field,

            value,

        )

        if canonical_value:

            return canonical_value

    return ""





def _entity_display(entity: Mapping[str, Any]) -> str:

    label = (

        _text(entity.get("mention"))

        or _text(entity.get("preferred_label"))

        or _text(entity.get("canonical_name"))

    )

    identifier = _entity_identifier(entity)

    if label and identifier:

        return f"{label} ({identifier})"

    return label or identifier





def _aligned(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:

    for field in ("base", "doc_key", "canonical_id", "pmid", "pmcid", "chunk_id"):

        a = _text(left.get(field))

        b = _text(right.get(field))

        if a and b and a != b:

            return False

    return True





def _paper_id(row: Mapping[str, Any]) -> str:

    return (

        _text(row.get("canonical_id"))

        or _text(row.get("doc_key"))

        or (f"pmid:{_text(row.get('pmid'))}" if _text(row.get("pmid")) else "")

        or (f"pmcid:{_text(row.get('pmcid'))}" if _text(row.get("pmcid")) else "")

    )





def _metadata_from_source(source_row: Mapping[str, Any]) -> dict[str, Any]:

    raw = source_row.get("paper_metadata")

    metadata = dict(raw) if isinstance(raw, Mapping) else {}

    for field in (

        "doc_key",

        "canonical_id",

        "pmid",

        "pmcid",

        "journal",

        "pub_year",

        "text_mode",

    ):

        if field not in metadata and source_row.get(field) not in (None, "", []):

            metadata[field] = source_row.get(field)

    if "selected_text_source" not in metadata and source_row.get("text_source"):

        metadata["selected_text_source"] = source_row.get("text_source")



    # The title is paper metadata and appears once at the paper level. The

    # abstract appears once as its annotatable ABSTRACT chunk. Internal schema

    # markers are not part of the user-facing download.

    metadata.pop("abstract", None)

    metadata.pop("schema", None)

    return {

        key: value

        for key, value in metadata.items()

        if value not in (None, "", [], {})

    }





def _export_annotation(annotation: Mapping[str, Any]) -> dict[str, Any]:

    """Remove internal normalization diagnostics from the public download."""



    return {

        key: value

        for key, value in annotation.items()

        if key not in _EXCLUDED_ANNOTATION_FIELDS

        and value not in (None, "", [], {})

    }





def _export_annotations(raw_annotations: Any, *, text: str | None = None) -> list[dict[str, Any]]:

    """Export annotations after applying all final cross-type precedence rules."""



    if not isinstance(raw_annotations, list):

        raw_annotations = []

    annotations = _resolve_annotation_conflicts(raw_annotations, text=text)

    return [_export_annotation(item) for item in annotations]





def _condition_text(relation: Mapping[str, Any]) -> str:

    """Preserve the extraction-stage condition without inferring new context.



    Missing fields are supported for legacy rows. Explicit non-string values

    raise instead of silently discarding a possibly meaningful qualification.

    Only outer whitespace is removed; interior wording is unchanged.

    """

    value = relation.get("conditions", "")

    if not isinstance(value, str):

        raise ValueError(

            "Relation 'conditions' must be a string; use an empty string "

            "when no condition is stated."

        )

    return value.strip()





def _resolved_relations(

    relation_row: Mapping[str, Any],

) -> tuple[list[dict[str, str]], int]:

    raw_entities = relation_row.get("entities")

    entities = raw_entities if isinstance(raw_entities, list) else []

    tag_map: dict[str, Mapping[str, Any]] = {}

    for entity in entities:

        if not isinstance(entity, Mapping):

            continue

        tag = _text(entity.get("id"))

        if tag:

            tag_map[tag] = entity



    raw_relations = relation_row.get("relations")

    relations = raw_relations if isinstance(raw_relations, list) else []

    output: list[dict[str, str]] = []

    unresolved = 0

    seen: set[tuple[str, str, str, str]] = set()

    for relation in relations:

        if not isinstance(relation, Mapping):

            unresolved += 1

            continue

        subject_tag = _text(relation.get("subject"))

        object_tag = _text(relation.get("object"))

        predicate = _text(relation.get("predicate")).casefold()

        subject = _entity_display(tag_map.get(subject_tag, {}))

        object_ = _entity_display(tag_map.get(object_tag, {}))

        if (

            not subject

            or not object_

            or not predicate

            or not relation_allowed(subject_tag, predicate, object_tag)

        ):

            unresolved += 1

            continue



        conditions = _condition_text(relation)

        key = (subject, predicate, object_, conditions)

        if key in seen:

            continue

        seen.add(key)

        output.append(

            {

                "subject": subject,

                "predicate": predicate,

                "object": object_,

                "conditions": conditions,

            }

        )

    return output, unresolved





def _write_row(handle: gzip.GzipFile, row: Mapping[str, Any]) -> None:

    handle.write(

        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    )

    handle.write(b"\n")





def build_final_annotation_artifact(

    *,

    chunks_path: Path,

    annotations_path: Path,

    relations_path: Path,

    output_path: Path,

) -> dict[str, int]:

    """Join the three aligned artifacts into one self-contained row per paper."""



    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")

    stats = {

        "paper_count": 0,

        "chunk_count": 0,

        "entity_annotation_count": 0,

        "relation_count": 0,

        "chunks_with_relations": 0,

        "unresolved_relation_reference_count": 0,

        "fulltext_chunk_count": 0,

        "abstract_fallback_chunk_count": 0,

        "abstract_chunk_count": 0,

        "metadata_only_chunk_count": 0,

    }

    current_id = ""

    current_metadata: dict[str, Any] = {}

    current_chunks: list[dict[str, Any]] = []



    def reset_paper(paper_id: str) -> None:

        nonlocal current_id, current_metadata

        current_id = paper_id

        current_metadata = {}

        current_chunks.clear()



    def write_paper(handle: gzip.GzipFile) -> None:

        if not current_id:

            return

        row: dict[str, Any] = {

            **current_metadata,

            "chunks": list(current_chunks),

        }

        _write_row(handle, row)

        stats["paper_count"] += 1



    try:

        with temporary.open("wb") as raw_output:

            with gzip.GzipFile(

                filename="",

                mode="wb",

                fileobj=raw_output,

                compresslevel=6,

                mtime=0,

            ) as compressed_output:

                rows: Iterator[tuple[Any, Any, Any]] = zip_longest(

                    iter_jsonl(chunks_path),

                    iter_jsonl(annotations_path),

                    iter_jsonl(relations_path),

                )

                for row_number, row_group in enumerate(rows, start=1):

                    source_row, annotation_row, relation_row = row_group

                    if (

                        source_row is None

                        or annotation_row is None

                        or relation_row is None

                    ):

                        raise ValueError(

                            "Stage 1 chunks, Stage 2 annotations, and Stage 3 "

                            "relations have different row counts."

                        )

                    if not _aligned(source_row, annotation_row) or not _aligned(

                        source_row, relation_row

                    ):

                        raise ValueError(

                            "Stage 1 chunks, Stage 2 annotations, and Stage 3 "

                            f"relations are misaligned at row {row_number}."

                        )



                    paper_id = _paper_id(source_row) or f"row:{row_number}"

                    if current_id and paper_id != current_id:

                        write_paper(compressed_output)

                        reset_paper(paper_id)

                    elif not current_id:

                        reset_paper(paper_id)



                    metadata = _metadata_from_source(source_row)

                    if metadata:

                        current_metadata.update(metadata)



                    # TITLE is retained only in paper metadata. Ignore any stale

                    # TITLE row from an older Stage 1 cache rather than exposing

                    # or annotating it in the final artifact.

                    section_type = _text(source_row.get("section_type")).upper()

                    if section_type == "TITLE":

                        continue



                    annotations = _export_annotations(

                        annotation_row.get("annotations"),

                        text=source_row.get("chunk") if isinstance(source_row.get("chunk"), str)

                            and section_type != "METADATA" else None,

                    )

                    relations, unresolved = _resolved_relations(relation_row)

                    text_source = _text(source_row.get("text_source"))

                    if text_source == "fulltext":

                        stats["fulltext_chunk_count"] += 1

                    elif "fallback" in text_source:

                        stats["abstract_fallback_chunk_count"] += 1

                    elif text_source == "abstract":

                        stats["abstract_chunk_count"] += 1

                    elif text_source.startswith("metadata_only"):

                        stats["metadata_only_chunk_count"] += 1



                    chunk: dict[str, Any] = {

                        "base": source_row.get("base"),

                        "section_type": source_row.get("section_type"),

                        "chunk_id": source_row.get("chunk_id"),

                        "text_source": source_row.get("text_source")

                        or source_row.get("text_mode"),

                        "text": source_row.get("chunk") or "",

                        "annotations": annotations,

                        "relations": relations,

                    }

                    # Keep the Stage 2 evidence in the actual downloadable

                    # artifact. Previously it existed only in an internal sidecar,

                    # so a null comparison entry could not be traced to its veto.

                    recovery_audit = annotation_row.get("entity_recovery_audit")

                    if isinstance(recovery_audit, Mapping):

                        chunk["entity_recovery_audit"] = dict(recovery_audit)

                    filter_context = {

                        field: annotation_row[field]

                        for field in (

                            "document_abbreviation_constraints",

                            "entity_span_exclusions",

                        )

                        if isinstance(annotation_row.get(field), list)

                        and annotation_row[field]

                    }

                    if filter_context:

                        chunk["entity_filter_context"] = filter_context

                    raw_annotations = annotation_row.get("annotations")

                    chunk["annotation_export_audit"] = {

                        "version": FINAL_ANNOTATION_EXPORT_VERSION,

                        "recovery_audit_present": isinstance(recovery_audit, Mapping),

                        "input_annotation_count": (

                            len(raw_annotations) if isinstance(raw_annotations, list) else 0

                        ),

                        "exported_annotation_count": len(annotations),

                    }

                    current_chunks.append(

                        {

                            key: value

                            for key, value in chunk.items()

                            if value not in (None, "")

                        }

                    )



                    stats["chunk_count"] += 1

                    stats["entity_annotation_count"] += len(annotations)

                    stats["relation_count"] += len(relations)

                    if relations:

                        stats["chunks_with_relations"] += 1

                    stats["unresolved_relation_reference_count"] += unresolved



                write_paper(compressed_output)

            raw_output.flush()

            os.fsync(raw_output.fileno())

        os.replace(temporary, output_path)

    except Exception:

        temporary.unlink(missing_ok=True)

        raise



    return stats





__all__ = [

    "FINAL_ANNOTATION_EXPORT_VERSION",

    "build_final_annotation_artifact",

]


