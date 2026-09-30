"""Abbreviation-aware CellExLink annotation worker.

The prepass supplies locked cell, hormone and gene annotations. NER runs on
unchanged text, followed by boundary repair and fragment rejection. Cell NEN
uses document definitions, static abbreviations, exact ontology aliases and
thresholded vector top-1. Weak/unresolved candidates cannot seed propagation.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

from backend.cellexlink_lite.normalization import (
    NORMALIZATION_METHODS,
    CellOntologyNormalizer,
    NormalizationDecision,
    RescuedMention,
    build_document_text,
    canonical_abbreviation_key,
)
from backend.pipeline.abbreviation_prepass import (
    load_document_context,
    run_abbreviation_prepass,
)
from backend.pipeline.entity_text_normalization import spans_overlap
from backend.cellexlink_lite.recognition import ChunkNER, EntitySpan
from backend.pipeline.entity_span_rules import repair_cell_spans, prune_cell_fragments, sanitize_source_annotations
from backend.pipeline.document_entity_recovery import default_cell_span_resources

logger = logging.getLogger(__name__)

SOURCE_FIELDS = (
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


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as raw_handle:
            temp_path = Path(raw_handle.name)
            count = 0
            with gzip.GzipFile(
                fileobj=raw_handle,
                mode="wb",
                compresslevel=6,
                mtime=0,
            ) as gzip_handle:
                for row in rows:
                    payload = json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    gzip_handle.write(payload)
                    gzip_handle.write(b"\n")
                    count += 1
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temp_path, path)
        return count
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _open_jsonl(path: Path):
    if path.suffix.casefold() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_jsonl(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON in {path} line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} line {line_no}")
            yield row


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
        for key in SOURCE_FIELDS
        if record.get(key) is not None
    }


def _mention_id(record: Mapping[str, Any], start: int, end: int) -> str:
    base = str(record.get("base") or record.get("doc_key") or "chunk")
    return f"{base}:cell:{start}:{end}"


def _canonical_entity_type(row: Mapping[str, Any]) -> str:
    value = str(row.get("entity_type") or row.get("obj") or "").casefold()
    if value in {"cell", "cell_type", "cell type"}:
        return "cell"
    if value in {"hormone", "chemical"}:
        return "hormone"
    return value


def _locked_spans(entry: Mapping[str, Any]) -> dict[str, list[tuple[int, int, str]]]:
    output: dict[str, list[tuple[int, int, str]]] = {}
    path_value = entry.get("abbreviation_annotations_path")
    if not path_value:
        return output
    path = Path(str(path_value))
    if not path.is_file():
        return output
    for row in _iter_jsonl(path):
        if not bool(row.get("locked")):
            continue
        try:
            start = int(row.get("start"))
            end = int(row.get("end"))
        except (TypeError, ValueError):
            continue
        output.setdefault(str(row.get("chunk_id") or ""), []).append(
            (start, end, _canonical_entity_type(row))
        )
    return output


def _resolve_local_annotation_conflicts(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the same longest-span policy independently to each chunk."""
    from backend.pipeline.entity_overlap import prefer_longest_spans
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in prune_cell_fragments(rows):
        grouped.setdefault(str(row.get("chunk_id") or ""), []).append(row)
    return [annotation for chunk_rows in grouped.values()
            for annotation in prefer_longest_spans(chunk_rows)]


def _process_ner_group(
    *,
    ner: ChunkNER,
    group: Sequence[tuple[dict[str, Any], list[dict[str, Any]]]],
    text_batch_size: int,
    model_name: str,
    pipeline_version: str,
) -> tuple[int, int, int]:
    all_records: list[dict[str, Any]] = []
    ranges: list[tuple[int, int]] = []
    for _entry, records in group:
        start = len(all_records)
        all_records.extend(records)
        ranges.append((start, len(all_records)))

    predictions = ner.predict_records(
        all_records,
        text_key="chunk",
        text_batch_size=text_batch_size,
    )
    total_mentions = 0
    blocked_mentions = 0
    total_chunks = len(all_records)

    for (entry, records), (start, end) in zip(group, ranges):
        mention_rows: list[dict[str, Any]] = []
        protected = _locked_spans(entry)
        for record, spans in zip(records, predictions[start:end]):
            text = str(record.get("chunk") or "")
            resources = default_cell_span_resources()
            repaired = repair_cell_spans(text, [(span.start, span.end) for span in spans],
                known_cell=lambda value: resources.resolve_cell(value).status == "resolved_target",
                known_gene=lambda value: bool(resources.genes and resources.genes.resolve(value)),
                cell_surface_allowed=resources.cell_surface_allowed)
            spans = [EntitySpan(text[a:b], a, b, "cell_type") for a,b in repaired]
            seen_spans: set[tuple[int, int, str]] = set()
            for span in spans:
                # A locked cell blocks only an equal or longer competitor.
                # Larger complete ontology-backed phrases remain eligible.
                if any(
                    locked_type == "cell"
                    and locked_end - locked_start >= span.end - span.start
                    and spans_overlap(
                        (span.start, span.end), (locked_start, locked_end)
                    )
                    for locked_start, locked_end, locked_type in protected.get(
                        str(record.get("chunk_id") or ""), []
                    )
                ):
                    blocked_mentions += 1
                    continue
                key = (span.start, span.end, span.text)
                if key in seen_spans:
                    continue
                seen_spans.add(key)
                row = _source_projection(record)
                row.update(
                    {
                        "mention_id": _mention_id(record, span.start, span.end),
                        "mention": span.text,
                        "start": span.start,
                        "end": span.end,
                        "offset_scope": "chunk",
                        "entity_type": "cell_type",
                        "ner_label": span.label,
                        "ner_model": model_name,
                        "recognition_source": "cell_ner",
                    }
                )
                mention_rows.append(row)

        mentions_path = Path(entry["mentions_path"])
        mentions_meta_path = Path(entry["mentions_meta_path"])
        row_count = _atomic_write_gzip_jsonl(mentions_path, mention_rows)
        _atomic_write_json(
            mentions_meta_path,
            {
                "status": "complete",
                "pipeline_version": pipeline_version,
                "ner_model": model_name,
                "source_fingerprint": entry["source_fingerprint"],
                "source_chunk_path": entry["chunk_path"],
                "chunk_count": len(records),
                "mention_count": row_count,
                "completed_at": utc_now(),
            },
        )
        total_mentions += row_count

    return total_chunks, total_mentions, blocked_mentions


def run_ner(args: Any, manifest: dict[str, Any], progress: ProgressSink) -> dict[str, Any]:
    entries = [dict(entry) for entry in manifest["entries"]]
    total_papers = len(entries)
    stats: dict[str, Any] = {
        "papers_total": total_papers,
        "papers_processed": 0,
        "chunks_processed": 0,
        "mentions_detected": 0,
        "mentions_blocked_by_ab3p": 0,
        "model_loaded": False,
        "recognition_requested_device": str(
            getattr(args, "device", "auto") or "auto"
        ),
        "recognition_compute_device": "not loaded",
    }
    if not entries:
        progress.emit(
            stage="recognition",
            percent=100,
            message="All paper-level recognition results were already cached.",
            stats=stats,
            force=True,
        )
        return stats

    progress.emit(
        stage="recognition",
        percent=1,
        message="Loading the CellExLink recognition model...",
        stats=stats,
        force=True,
    )

    ner: ChunkNER | None = None
    try:
        ner = ChunkNER(
            model_name_or_path=args.model,
            cache_dir=args.model_cache_dir,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            window_batch_size=args.window_batch_size,
            cpu_threads=args.cpu_threads,
            device=str(getattr(args, "device", "auto") or "auto"),
        )
        stats["model_loaded"] = True
        stats["recognition_compute_device"] = ner.compute_device
        progress.emit(
            stage="recognition",
            percent=5,
            message="Recognition model loaded. Detecting cell-type mentions...",
            stats=stats,
            force=True,
        )

        group: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        group_record_count = 0
        group_limit = max(args.text_batch_size * 4, args.text_batch_size)

        def flush_group() -> None:
            nonlocal group, group_record_count
            if not group:
                return
            chunk_count, mention_count, blocked_count = _process_ner_group(
                ner=ner,  # type: ignore[arg-type]
                group=group,
                text_batch_size=args.text_batch_size,
                model_name=getattr(args, "model_label", args.model),
                pipeline_version=manifest["pipeline_version"],
            )
            stats["papers_processed"] += len(group)
            stats["chunks_processed"] += chunk_count
            stats["mentions_detected"] += mention_count
            stats["mentions_blocked_by_ab3p"] += blocked_count
            percent = 5 + 95 * stats["papers_processed"] / max(1, total_papers)
            progress.emit(
                stage="recognition",
                percent=percent,
                message=(
                    f"Recognized {stats['papers_processed']} of {total_papers} papers; "
                    f"found {stats['mentions_detected']:,} cell-type mentions."
                ),
                stats=stats,
                force=True,
            )
            group = []
            group_record_count = 0

        for entry in entries:
            chunk_path = Path(entry["chunk_path"])
            records = list(_iter_jsonl(chunk_path))
            if group and group_record_count + len(records) > group_limit:
                flush_group()
            group.append((entry, records))
            group_record_count += len(records)
            if group_record_count >= group_limit:
                flush_group()
        flush_group()
    finally:
        if ner is not None:
            stats["recognition_compute_device"] = ner.compute_device
            ner.close()

    progress.emit(
        stage="recognition",
        percent=100,
        message=(
            f"Recognition complete: {stats['mentions_detected']:,} mentions "
            f"in {stats['chunks_processed']:,} chunks."
        ),
        stats=stats,
        force=True,
    )
    return stats


def _annotation_row_from_decision(
    mention: Mapping[str, Any],
    decision: NormalizationDecision,
    *,
    nen_model: str,
) -> dict[str, Any]:
    row = dict(mention)
    row["nen_model"] = nen_model
    row.update(decision.to_annotation_fields())
    return row


def _rescued_annotation_row(
    rescued: RescuedMention,
    *,
    ner_model: str,
    nen_model: str,
) -> dict[str, Any]:
    row = _source_projection(rescued.chunk.source)
    row.update(
        {
            "mention_id": _mention_id(
                rescued.chunk.source,
                rescued.start,
                rescued.end,
            ),
            "mention": rescued.mention,
            "start": rescued.start,
            "end": rescued.end,
            "offset_scope": "chunk",
            "entity_type": "cell_type",
            "ner_label": "DOCUMENT_RESCUE",
            "ner_model": ner_model,
            "nen_model": nen_model,
            "recognition_source": rescued.decision.normalization_source,
        }
    )
    row.update(rescued.decision.to_annotation_fields())
    return row


def _debug_mention_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "mention": row.get("mention"),
        "chunk_id": row.get("chunk_id"),
        "start": row.get("start"),
        "end": row.get("end"),
        "normalized_id": row.get("cell_ontology_id"),
        "preferred_label": row.get("cell_ontology_label"),
    }
    for key in (
        "matched_abbreviation_key",
        "matched_static_abbreviation_key",
        "expanded_long_form",
        "abbreviation_key_cosine",
        "ab3p_key_cosine",
        "ab3p_match_method",
        "ontology_raw_cosine",
        "matched_ontology_alias",
    ):
        value = row.get(key)
        if value is not None:
            output[key] = value
    return output


def _log_document_methods(
    *,
    document_key: str,
    ab3p_status: str,
    output_rows: Sequence[Mapping[str, Any]],
    definition_count: int,
    validated_definition_count: int,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {
        method: [] for method in NORMALIZATION_METHODS
    }
    for row in output_rows:
        source = str(row.get("normalization_source") or "unresolved")
        grouped.setdefault(source, []).append(_debug_mention_payload(row))

    counts = {method: len(grouped.get(method, [])) for method in NORMALIZATION_METHODS}
    logger.info(
        "[CELL_NORM_DEBUG] document=%s ab3p_status=%s ab3p_definitions=%d "
        "validated_ab3p_definitions=%d counts=%s",
        document_key,
        ab3p_status,
        definition_count,
        validated_definition_count,
        json.dumps(counts, ensure_ascii=False, separators=(",", ":")),
    )
    for method in NORMALIZATION_METHODS:
        mentions = grouped.get(method, [])
        if not mentions:
            continue
        logger.info(
            "[CELL_NORM_DEBUG] document=%s method=%s count=%d mentions=%s",
            document_key,
            method,
            len(mentions),
            json.dumps(mentions, ensure_ascii=False, separators=(",", ":")),
        )


def _row_sort_key(
    row: Mapping[str, Any],
    *,
    chunk_order: Mapping[str, int],
) -> tuple[int, int, int, str, str]:
    try:
        start = int(row.get("start"))
    except (TypeError, ValueError):
        start = 0
    try:
        end = int(row.get("end"))
    except (TypeError, ValueError):
        end = start
    return (
        chunk_order.get(str(row.get("chunk_id") or ""), 10**9),
        start,
        end,
        str(row.get("mention") or "").casefold(),
        str(row.get("normalization_source") or ""),
    )


def run_nen(args: Any, manifest: dict[str, Any], progress: ProgressSink) -> dict[str, Any]:
    """Normalize NER spans using the saved pre-NER abbreviation context."""

    entries = [dict(entry) for entry in manifest["entries"]]
    total_papers = len(entries)
    aggregate_methods: Counter[str] = Counter()
    unique_mentions: set[tuple[str, str]] = set()
    stats: dict[str, Any] = {
        "papers_total": total_papers,
        "papers_processed": 0,
        "ner_mention_occurrences": 0,
        "mention_occurrences": 0,
        "cell_occurrences": 0,
        "hormone_occurrences": 0,
        "unique_mentions": 0,
        "normalized_occurrences": 0,
        "unresolved_occurrences": 0,
        "documents_with_abbreviation_context": 0,
        "ab3p_health_check": "completed_in_prepass",
        "ab3p_document_statuses": {
            "definitions_found": 0,
            "no_definitions": 0,
            "disabled": 0,
        },
        "ab3p_definitions": 0,
        "validated_ab3p_definitions": 0,
        "ab3p_cell_annotations": 0,
        "ab3p_hormone_annotations": 0,
        "rescued_occurrences": 0,
        "normalization_methods": {method: 0 for method in NORMALIZATION_METHODS},
        "normalization_method_log": bool(
            getattr(args, "normalization_method_log", False)
        ),
        "model_loaded": False,
        "normalization_requested_device": str(
            getattr(args, "device", "auto") or "auto"
        ),
        "normalization_compute_device": "not loaded",
        "ontology_embedding_cache_reused": None,
    }
    if not entries:
        progress.emit(
            stage="normalization",
            percent=100,
            message="All paper-level normalization results were already cached.",
            stats=stats,
            force=True,
        )
        return stats

    progress.emit(
        stage="normalization",
        percent=1,
        message="Preparing the local CellExLink normalization resources...",
        stats=stats,
        force=True,
    )

    normalizer: CellOntologyNormalizer | None = None
    try:
        def report_normalizer_progress(
            stage: str,
            percent: float,
            message: str,
            detail_stats: Mapping[str, Any],
        ) -> None:
            merged_stats = {**stats, **dict(detail_stats)}
            progress.emit(
                stage=stage,
                percent=percent,
                message=message,
                stats=merged_stats,
                force=False,
            )

        normalizer = CellOntologyNormalizer(
            model_name_or_path=args.model,
            model_cache_dir=args.model_cache_dir,
            embedding_cache_dir=args.embedding_cache_dir,
            ontology_path=args.ontology_path,
            abbreviations_path=args.abbreviations_path,
            disable_abbreviations=args.disable_abbreviations,
            batch_size=args.batch_size,
            cpu_threads=args.cpu_threads,
            device=str(getattr(args, "device", "auto") or "auto"),
            model_identity=(
                str(getattr(args, "model_identity", "") or "") or None
            ),
            progress_callback=report_normalizer_progress,
        )

        for paper_index, entry in enumerate(entries, start=1):
            chunk_records = list(_iter_jsonl(Path(entry["chunk_path"])))
            document_key = str(
                entry.get("paper_identity")
                or (chunk_records[0].get("doc_key") if chunk_records else "")
                or (chunk_records[0].get("canonical_id") if chunk_records else "")
                or f"paper-{paper_index}"
            )
            document = build_document_text(chunk_records, document_key=document_key)
            mentions = list(_iter_jsonl(Path(entry["mentions_path"])))
            prepass_rows = list(
                _iter_jsonl(Path(entry["abbreviation_annotations_path"]))
            )
            context = load_document_context(entry["abbreviation_context_path"])
            stats["ner_mention_occurrences"] += len(mentions)

            statuses = stats["ab3p_document_statuses"]
            statuses[context.ab3p_status] = int(
                statuses.get(context.ab3p_status, 0)
            ) + 1
            if context.definitions:
                stats["documents_with_abbreviation_context"] += 1
            stats["ab3p_definitions"] += len(context.definitions)
            stats["validated_ab3p_definitions"] += context.validated_definition_count
            stats["ab3p_cell_annotations"] += sum(
                _canonical_entity_type(row) == "cell" for row in prepass_rows
            )
            stats["ab3p_hormone_annotations"] += sum(
                _canonical_entity_type(row) == "hormone" for row in prepass_rows
            )

            decisions = normalizer.normalize_document_mentions(
                document=document,
                mentions=mentions,
                context=context,
            )
            ner_rows = [
                _annotation_row_from_decision(
                    mention,
                    decision,
                    nen_model=getattr(args, "model_label", args.model),
                )
                for mention, decision in zip(mentions, decisions)
                if decision.normalized
            ]
            stats["rejected_cell_occurrences"] = int(stats.get("rejected_cell_occurrences", 0)) + sum(not decision.normalized for decision in decisions)
            output_rows: list[dict[str, Any]] = [
                *prepass_rows,
                *ner_rows,
            ]

            resources = default_cell_span_resources()
            def validate_source_rows(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
                grouped: dict[str, list[Mapping[str, Any]]] = {}
                for value in values:
                    grouped.setdefault(str(value.get("chunk_id")), []).append(value)
                from backend.pipeline.receptor_annotations import reconcile_receptor_annotations
                return [row for chunk in document.chunks for row in sanitize_source_annotations(
                    str(chunk.source.get("chunk") or ""),
                    reconcile_receptor_annotations(str(chunk.source.get("chunk") or ""),
                        grouped.get(str(chunk.chunk_id), []), matcher=resources.genes),
                    known_cell=lambda value: resources.resolve_cell(value).status == "resolved_target",
                    known_gene=lambda value: bool(resources.genes and resources.genes.resolve(value)),
                    cell_surface_allowed=resources.cell_surface_allowed)]

            # Validate before surface recovery so a false cytokine-as-cell or
            # receptor fragment can neither hide a gene nor seed propagation.
            before_validation = len(output_rows)
            output_rows = validate_source_rows(output_rows)
            stats["source_span_rejections"] = int(stats.get("source_span_rejections", 0)) + before_validation - len(output_rows)
            rescued = normalizer.rescue_document_cell_mentions(
                document=document,
                accepted_rows=output_rows,
                context=context,
            )
            output_rows.extend(
                _rescued_annotation_row(
                    item,
                    ner_model=manifest["ner_model"],
                    nen_model=getattr(args, "model_label", args.model),
                )
                for item in rescued
            )
            before_validation = len(output_rows)
            output_rows = validate_source_rows(output_rows)
            stats["source_span_rejections"] += before_validation - len(output_rows)
            output_rows = _resolve_local_annotation_conflicts(output_rows)

            chunk_order = {
                chunk.chunk_id: chunk.chunk_order for chunk in document.chunks
            }
            output_rows.sort(
                key=lambda row: _row_sort_key(row, chunk_order=chunk_order)
            )

            paper_methods: Counter[str] = Counter(
                str(row.get("normalization_source") or "unresolved")
                for row in output_rows
            )
            aggregate_methods.update(paper_methods)
            paper_cell_rows = [
                row for row in output_rows if _canonical_entity_type(row) == "cell"
            ]
            paper_hormone_rows = [
                row for row in output_rows if _canonical_entity_type(row) == "hormone"
            ]
            paper_gene_rows = [row for row in output_rows if _canonical_entity_type(row) == "gene"]
            stats["gene_occurrences"] = int(stats.get("gene_occurrences", 0)) + len(paper_gene_rows)
            paper_normalized = sum(
                1
                for row in paper_cell_rows
                if row.get("normalization_status") == "normalized"
            )
            paper_unresolved = len(paper_cell_rows) - paper_normalized
            stats["rescued_occurrences"] += len(rescued)
            stats["mention_occurrences"] += len(output_rows)
            stats["cell_occurrences"] += len(paper_cell_rows)
            stats["hormone_occurrences"] += len(paper_hormone_rows)
            stats["normalized_occurrences"] += paper_normalized
            stats["unresolved_occurrences"] += paper_unresolved

            for row in output_rows:
                mention_key = canonical_abbreviation_key(row.get("mention") or "")
                if mention_key:
                    unique_mentions.add(
                        (document_key, f"{_canonical_entity_type(row)}:{mention_key}")
                    )

            annotations_path = Path(entry["annotations_path"])
            annotations_meta_path = Path(entry["annotations_meta_path"])
            row_count = _atomic_write_gzip_jsonl(annotations_path, output_rows)
            _atomic_write_json(
                annotations_meta_path,
                {
                    "status": "complete",
                    "pipeline_version": manifest["pipeline_version"],
                    "ner_model": manifest["ner_model"],
                    "nen_model": getattr(args, "model_label", args.model),
                    "ontology_version": manifest["ontology_version"],
                    "abbreviation_version": manifest["abbreviation_version"],
                    "hormone_resource_version": manifest.get(
                        "hormone_resource_version"
                    ),
                    "abbreviations_enabled": bool(
                        manifest.get("abbreviations_enabled", True)
                    ),
                    "normalization_method_log": bool(
                        getattr(args, "normalization_method_log", False)
                    ),
                    "source_fingerprint": entry["source_fingerprint"],
                    "source_chunk_path": entry["chunk_path"],
                    "ner_mention_count": len(mentions),
                    "ab3p_annotation_count": len(prepass_rows),
                    "rescued_mention_count": len(rescued),
                    "mention_count": row_count,
                    "cell_count": len(paper_cell_rows),
                    "hormone_count": len(paper_hormone_rows),
                    "gene_count": len(paper_gene_rows),
                    "normalized_count": paper_normalized,
                    "unresolved_count": paper_unresolved,
                    "ab3p_status": context.ab3p_status,
                    "ab3p_definition_count": len(context.definitions),
                    "validated_ab3p_definition_count": (
                        context.validated_definition_count
                    ),
                    "normalization_methods": dict(paper_methods),
                    "completed_at": utc_now(),
                },
            )

            if getattr(args, "normalization_method_log", False):
                _log_document_methods(
                    document_key=document_key,
                    ab3p_status=context.ab3p_status,
                    output_rows=output_rows,
                    definition_count=len(context.definitions),
                    validated_definition_count=context.validated_definition_count,
                )

            if not args.keep_ner_intermediates:
                Path(entry["mentions_path"]).unlink(missing_ok=True)
                Path(entry["mentions_meta_path"]).unlink(missing_ok=True)

            stats["papers_processed"] = paper_index
            stats["unique_mentions"] = len(unique_mentions)
            stats["normalization_methods"] = dict(aggregate_methods)
            stats["model_loaded"] = bool(normalizer.model_loaded)
            stats["normalization_compute_device"] = getattr(normalizer, "compute_device", "not loaded")
            stats["ontology_embedding_cache_reused"] = (
                getattr(normalizer, "dictionary_embedding_cache_reused", None)
            )
            percent = 70 + 30 * paper_index / max(1, total_papers)
            progress.emit(
                stage="normalization",
                percent=percent,
                message=(
                    f"Normalized {paper_index} of {total_papers} papers; "
                    f"{stats['normalized_occurrences']:,} cell occurrences linked, "
                    f"{stats['hormone_occurrences']:,} hormone occurrences retained, "
                    f"and {stats['rescued_occurrences']:,} cell spans propagated."
                ),
                stats=stats,
                force=True,
            )

        stats["model_loaded"] = bool(normalizer.model_loaded)
        stats["normalization_compute_device"] = getattr(normalizer, "compute_device", "not loaded")
        stats["ontology_embedding_cache_reused"] = (
            getattr(normalizer, "dictionary_embedding_cache_reused", None)
        )
        progress.emit(
            stage="normalization",
            percent=100,
            message=(
                f"Normalization complete: {stats['normalized_occurrences']:,} "
                "cell occurrences processed with thresholded vector fallback."
            ),
            stats=stats,
            force=True,
        )
        return stats
    finally:
        if normalizer is not None:
            stats["model_loaded"] = bool(normalizer.model_loaded)
            stats["normalization_compute_device"] = getattr(normalizer, "compute_device", "not loaded")
            stats["ontology_embedding_cache_reused"] = (
                getattr(normalizer, "dictionary_embedding_cache_reused", None)
            )
            normalizer.close()


__all__ = ["run_abbreviation_prepass", "run_ner", "run_nen", "utc_now"]
