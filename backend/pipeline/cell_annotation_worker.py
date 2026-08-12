"""Lean chunk-level CellExLink recognition and normalization routines.

The Railway FastAPI process never imports this module. A Modal T4 worker calls
``run_ner`` and then ``run_nen`` sequentially. Each routine explicitly closes
its model before the next checkpoint is loaded, while all per-job intermediates
remain on ephemeral disk.
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

from backend.cellexlink_lite.normalization import (
    CellOntologyNormalizer,
    NormalizationRequest,
    extract_document_abbreviations,
    is_abbreviation_like,
    plural_normalize_text,
)
from backend.cellexlink_lite.recognition import ChunkNER
from backend.cellexlink_lite.resources import (
    DEFAULT_ABBREVIATIONS_PATH,
    DEFAULT_ONTOLOGY_PATH,
)

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
            with gzip.GzipFile(fileobj=raw_handle, mode="wb", compresslevel=6) as gzip_handle:
                for row in rows:
                    payload = json.dumps(
                        row, ensure_ascii=False, separators=(",", ":")
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


def _process_ner_group(
    *,
    ner: ChunkNER,
    group: Sequence[tuple[dict[str, Any], list[dict[str, Any]]]],
    text_batch_size: int,
    model_name: str,
    pipeline_version: str,
) -> tuple[int, int]:
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
    total_chunks = len(all_records)

    for (entry, records), (start, end) in zip(group, ranges):
        mention_rows: list[dict[str, Any]] = []
        for record, spans in zip(records, predictions[start:end]):
            seen_spans: set[tuple[int, int, str]] = set()
            for span in spans:
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

    return total_chunks, total_mentions


def run_ner(args: Any, manifest: dict[str, Any], progress: ProgressSink) -> dict[str, Any]:
    entries = [dict(entry) for entry in manifest["entries"]]
    total_papers = len(entries)
    stats: dict[str, Any] = {
        "papers_total": total_papers,
        "papers_processed": 0,
        "chunks_processed": 0,
        "mentions_detected": 0,
        "model_loaded": False,
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
        )
        stats["model_loaded"] = True
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
            chunk_count, mention_count = _process_ner_group(
                ner=ner,  # type: ignore[arg-type]
                group=group,
                text_batch_size=args.text_batch_size,
                model_name=getattr(args, "model_label", args.model),
                pipeline_version=manifest["pipeline_version"],
            )
            stats["papers_processed"] += len(group)
            stats["chunks_processed"] += chunk_count
            stats["mentions_detected"] += mention_count
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


def _connect_work_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            document_key TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            mention_text TEXT NOT NULL,
            PRIMARY KEY (document_key, normalized_text)
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            document_key TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            result_json TEXT NOT NULL,
            PRIMARY KEY (document_key, normalized_text)
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS document_abbreviations (
            document_key TEXT PRIMARY KEY,
            lookup_json TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    return connection


def _document_text_by_key(chunk_path: Path) -> dict[str, str]:
    parts: dict[str, list[str]] = defaultdict(list)
    for row in _iter_jsonl(chunk_path):
        text = str(row.get("chunk") or "").strip()
        if text:
            parts[str(row.get("doc_key") or "")].append(text)
    return {key: "\n".join(values) for key, values in parts.items()}


def _load_context_for_documents(
    connection: sqlite3.Connection,
    document_keys: set[str],
) -> dict[str, dict[str, str]]:
    keys = sorted(key for key in document_keys if key)
    if not keys:
        return {}
    context: dict[str, dict[str, str]] = {}
    for start in range(0, len(keys), 400):
        batch = keys[start : start + 400]
        placeholders = ",".join("?" for _ in batch)
        rows = connection.execute(
            f"SELECT document_key, lookup_json FROM document_abbreviations "
            f"WHERE document_key IN ({placeholders})",  # noqa: S608
            batch,
        ).fetchall()
        for document_key, raw_lookup in rows:
            try:
                lookup = json.loads(raw_lookup)
            except json.JSONDecodeError:
                lookup = {}
            if isinstance(lookup, dict):
                context[str(document_key)] = {
                    str(key): str(value) for key, value in lookup.items()
                }
    return context


def _best_candidate_payload(result_payload: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates = result_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    best = candidates[0]
    return dict(best) if isinstance(best, dict) else None


def _final_annotation_row(
    mention: Mapping[str, Any],
    result_payload: Mapping[str, Any] | None,
    *,
    nen_model: str,
) -> dict[str, Any]:
    row = dict(mention)
    row["nen_model"] = nen_model
    best = _best_candidate_payload(result_payload or {})
    if best is None:
        row["normalization_status"] = "unresolved"
        row["cell_ontology_id"] = None
        row["cell_ontology_label"] = None
        return row

    row.update(
        {
            "normalization_status": "normalized",
            "cell_ontology_id": best.get("identifier"),
            "cell_ontology_label": best.get("preferred_label"),
            "normalization_score": best.get("final_score"),
            "normalization_embedding_score": best.get("embedding_score"),
            "normalization_source": best.get("source"),
        }
    )
    for key in (
        "abbreviation_method",
        "expanded_long_form",
        "ab3p_method",
        "ab3p_matched_key",
        "ab3p_match_score",
    ):
        if best.get(key) is not None:
            row[key] = best[key]
    return row


def run_nen(args: Any, manifest: dict[str, Any], progress: ProgressSink) -> dict[str, Any]:
    entries = [dict(entry) for entry in manifest["entries"]]
    total_papers = len(entries)
    stats: dict[str, Any] = {
        "papers_total": total_papers,
        "papers_processed": 0,
        "mention_occurrences": 0,
        "unique_mentions": 0,
        "normalized_occurrences": 0,
        "unresolved_occurrences": 0,
        "documents_with_abbreviation_context": 0,
        "model_loaded": False,
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

    work_database_path = Path(args.work_database)
    work_database_path.unlink(missing_ok=True)
    connection = _connect_work_database(work_database_path)
    normalizer: CellOntologyNormalizer | None = None

    try:
        # Loading ontology and abbreviation TSV resources does not load a model.
        normalizer = CellOntologyNormalizer(
            model_name_or_path=args.model,
            model_cache_dir=args.model_cache_dir,
            embedding_cache_dir=args.embedding_cache_dir,
            ontology_path=args.ontology_path,
            abbreviations_path=args.abbreviations_path,
            disable_abbreviations=args.disable_abbreviations,
            batch_size=args.batch_size,
            cpu_threads=args.cpu_threads,
        )

        progress.emit(
            stage="normalization",
            percent=1,
            message="Indexing unique cell-type mentions without loading the model...",
            stats=stats,
            force=True,
        )

        for paper_index, entry in enumerate(entries, start=1):
            mentions_path = Path(entry["mentions_path"])
            document_keys_needing_context: set[str] = set()
            rows_to_insert: list[tuple[str, str, str]] = []
            for mention in _iter_jsonl(mentions_path):
                mention_text = str(mention.get("mention") or "").strip()
                document_key = str(mention.get("doc_key") or "")
                if not mention_text:
                    continue
                normalized_text = plural_normalize_text(mention_text)
                rows_to_insert.append((document_key, normalized_text, mention_text))
                stats["mention_occurrences"] += 1
                if (
                    not args.disable_abbreviations
                    and is_abbreviation_like(mention_text)
                ):
                    document_keys_needing_context.add(document_key)

            connection.executemany(
                "INSERT OR IGNORE INTO requests "
                "(document_key, normalized_text, mention_text) VALUES (?, ?, ?)",
                rows_to_insert,
            )

            if document_keys_needing_context:
                document_texts = _document_text_by_key(Path(entry["chunk_path"]))
                context_rows: list[tuple[str, str]] = []
                for document_key in document_keys_needing_context:
                    lookup = extract_document_abbreviations(
                        document_texts.get(document_key, "")
                    )
                    if lookup:
                        context_rows.append(
                            (document_key, json.dumps(lookup, ensure_ascii=False))
                        )
                if context_rows:
                    connection.executemany(
                        "INSERT OR REPLACE INTO document_abbreviations "
                        "(document_key, lookup_json) VALUES (?, ?)",
                        context_rows,
                    )

            if paper_index % 25 == 0 or paper_index == total_papers:
                connection.commit()
                percent = 1 + 14 * paper_index / max(1, total_papers)
                progress.emit(
                    stage="normalization",
                    percent=percent,
                    message=(
                        f"Indexed mentions from {paper_index} of {total_papers} papers; "
                        f"{stats['mention_occurrences']:,} occurrences found."
                    ),
                    stats=stats,
                    force=True,
                )

        stats["unique_mentions"] = int(
            connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        )
        stats["documents_with_abbreviation_context"] = int(
            connection.execute(
                "SELECT COUNT(*) FROM document_abbreviations"
            ).fetchone()[0]
        )

        if stats["unique_mentions"]:
            progress.emit(
                stage="normalization",
                percent=16,
                message=(
                    "Loading the CellExLink normalization model and the compact "
                    "Cell Ontology embedding index..."
                ),
                stats=stats,
                force=True,
            )

            processed_unique = 0
            last_document_key: str | None = None
            last_normalized_text: str | None = None
            while True:
                if last_document_key is None:
                    rows = connection.execute(
                        "SELECT document_key, normalized_text, mention_text "
                        "FROM requests ORDER BY document_key, normalized_text LIMIT ?",
                        (args.request_batch_size,),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT document_key, normalized_text, mention_text FROM requests "
                        "WHERE document_key > ? "
                        "OR (document_key = ? AND normalized_text > ?) "
                        "ORDER BY document_key, normalized_text LIMIT ?",
                        (
                            last_document_key,
                            last_document_key,
                            last_normalized_text or "",
                            args.request_batch_size,
                        ),
                    ).fetchall()
                if not rows:
                    break
                requests = [
                    NormalizationRequest(
                        mention_text=str(row[2]), document_key=str(row[0])
                    )
                    for row in rows
                ]
                context = _load_context_for_documents(
                    connection, {request.document_key for request in requests}
                )
                linked = normalizer.normalize_batch(
                    requests,
                    document_abbreviations=context,
                )
                connection.executemany(
                    "INSERT OR REPLACE INTO results "
                    "(document_key, normalized_text, result_json) VALUES (?, ?, ?)",
                    [
                        (
                            result.document_key,
                            result.normalized_text,
                            json.dumps(
                                result.to_dict(),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        )
                        for result in linked
                    ],
                )
                connection.commit()
                processed_unique += len(rows)
                last_document_key = str(rows[-1][0])
                last_normalized_text = str(rows[-1][1])
                percent = 16 + 62 * processed_unique / max(
                    1, stats["unique_mentions"]
                )
                progress.emit(
                    stage="normalization",
                    percent=percent,
                    message=(
                        f"Normalized {processed_unique:,} of "
                        f"{stats['unique_mentions']:,} unique mentions."
                    ),
                    stats=stats,
                    force=True,
                )
        else:
            progress.emit(
                stage="normalization",
                percent=78,
                message="No recognized cell-type mentions require normalization.",
                stats=stats,
                force=True,
            )

        # Record whether the encoder was needed, then drop it before the final
        # disk-only join. Empty mention sets never load the NEN checkpoint.
        stats["model_loaded"] = bool(getattr(normalizer, "model_loaded", False))
        normalizer.close()
        normalizer = None

        for paper_index, entry in enumerate(entries, start=1):
            mentions = list(_iter_jsonl(Path(entry["mentions_path"])))
            document_keys = sorted(
                {str(mention.get("doc_key") or "") for mention in mentions}
            )
            result_map: dict[tuple[str, str], dict[str, Any]] = {}
            for start in range(0, len(document_keys), 400):
                batch = document_keys[start : start + 400]
                if not batch:
                    continue
                placeholders = ",".join("?" for _ in batch)
                result_rows = connection.execute(
                    f"SELECT document_key, normalized_text, result_json FROM results "
                    f"WHERE document_key IN ({placeholders})",  # noqa: S608
                    batch,
                ).fetchall()
                for document_key, normalized_text, raw_result in result_rows:
                    try:
                        payload = json.loads(raw_result)
                    except json.JSONDecodeError:
                        payload = {}
                    result_map[(str(document_key), str(normalized_text))] = payload

            output_rows: list[dict[str, Any]] = []
            paper_normalized = 0
            paper_unresolved = 0
            for mention in mentions:
                document_key = str(mention.get("doc_key") or "")
                normalized_text = plural_normalize_text(mention.get("mention") or "")
                output_row = _final_annotation_row(
                    mention,
                    result_map.get((document_key, normalized_text)),
                    nen_model=getattr(args, "model_label", args.model),
                )
                if output_row["normalization_status"] == "normalized":
                    paper_normalized += 1
                else:
                    paper_unresolved += 1
                output_rows.append(output_row)

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
                    "abbreviations_enabled": bool(
                        manifest.get("abbreviations_enabled", True)
                    ),
                    "source_fingerprint": entry["source_fingerprint"],
                    "source_chunk_path": entry["chunk_path"],
                    "mention_count": row_count,
                    "normalized_count": paper_normalized,
                    "unresolved_count": paper_unresolved,
                    "completed_at": utc_now(),
                },
            )
            stats["papers_processed"] += 1
            stats["normalized_occurrences"] += paper_normalized
            stats["unresolved_occurrences"] += paper_unresolved

            if not args.keep_ner_intermediates:
                Path(entry["mentions_path"]).unlink(missing_ok=True)
                Path(entry["mentions_meta_path"]).unlink(missing_ok=True)

            percent = 78 + 22 * paper_index / max(1, total_papers)
            progress.emit(
                stage="normalization",
                percent=percent,
                message=(
                    f"Wrote sparse annotations for {paper_index} of {total_papers} papers; "
                    f"{stats['normalized_occurrences']:,} occurrences normalized."
                ),
                stats=stats,
                force=True,
            )

        progress.emit(
            stage="normalization",
            percent=100,
            message=(
                f"Normalization complete: {stats['normalized_occurrences']:,} "
                "cell-type occurrences linked to Cell Ontology."
            ),
            stats=stats,
            force=True,
        )
        return stats
    finally:
        if normalizer is not None:
            normalizer.close()
        connection.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(work_database_path) + suffix).unlink(missing_ok=True)



__all__ = ["run_ner", "run_nen", "utc_now"]
