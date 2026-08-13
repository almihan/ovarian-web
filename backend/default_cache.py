"""Shared default Stage 1 cache and run-scoped Stage 1 additions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

from backend.config import settings
from backend.pipeline.retrieval import (
    DEFAULT_PMIDS,
    DEFAULT_PMCIDS,
    DEFAULT_PUBMED_QUERY,
    FULLTEXT_RETRIEVER_VERSION,
    INPUT_PARSER_VERSION,
    RetrievalResult,
    run_paper_retrieval,
)
from backend.storage.artifacts import (
    ArtifactRef,
    ArtifactStore,
    get_artifact_store,
    prefixed_key,
    sha256_file,
)
from backend.storage.bundles import build_deterministic_gzip_bundle

Progress = Callable[[str, int, str, dict[str, Any]], None]
_ONE_MIB = 1024 * 1024
_DEFAULT_STAGE1_LOCK = threading.Lock()
_DEFAULT_STAGE1_KEY = prefixed_key("shared-default/stage1/chunks.jsonl.gz")
_DEFAULT_STAGE1_SUMMARY_KEY = prefixed_key("shared-default/stage1/summary.json")


def default_stage1_signature() -> str:
    payload = {
        "cache_schema": "shared-default-stage1-v2",
        "input_parser": INPUT_PARSER_VERSION,
        "fulltext_retriever": FULLTEXT_RETRIEVER_VERSION,
        "query": DEFAULT_PUBMED_QUERY,
        "pmids": DEFAULT_PMIDS,
        "pmcids": DEFAULT_PMCIDS,
        "keyword_limit": settings.retrieval_keyword_limit,
        "batch_size": settings.retrieval_batch_size,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_scoped_signature(run_id: str, base_signature: str) -> str:
    return f"run-{run_id}-{base_signature[:24]}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _paper_index(corpus_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if not corpus_path.is_file():
        return records
    with corpus_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            canonical_id = str(row.get("canonical_id") or "").strip()
            pmid = str(row.get("pmid") or "").strip()
            pmcid = str(row.get("pmcid") or "").strip()
            if canonical_id or pmid or pmcid:
                records.append(
                    {
                        "canonical_id": canonical_id,
                        "pmid": pmid,
                        "pmcid": pmcid,
                    }
                )
    return records


def _cached_default_stage1(store: ArtifactStore) -> dict[str, Any] | None:
    summary_ref = store.head(_DEFAULT_STAGE1_SUMMARY_KEY)
    artifact_ref = store.head(_DEFAULT_STAGE1_KEY)
    if summary_ref is None or artifact_ref is None:
        return None
    try:
        summary = store.read_json(_DEFAULT_STAGE1_SUMMARY_KEY)
    except Exception:
        return None
    if summary.get("cache_signature") != default_stage1_signature():
        return None
    files = summary.get("files")
    files = files if isinstance(files, Mapping) else {}
    raw_artifact = files.get("artifact")
    raw_artifact = raw_artifact if isinstance(raw_artifact, Mapping) else {}
    if str(raw_artifact.get("sha256") or "") != artifact_ref.sha256:
        return None
    paper_index = summary.get("paper_index")
    if not isinstance(paper_index, list):
        return None
    return {
        "artifact": artifact_ref.to_dict(),
        "summary": summary,
        "stats": dict(summary.get("stats") or {}),
        "paper_index": paper_index,
        "reused": True,
    }


def get_or_build_default_stage1(progress: Progress | None = None) -> dict[str, Any]:
    store = get_artifact_store()
    cached = _cached_default_stage1(store)
    if cached is not None:
        return cached

    with _DEFAULT_STAGE1_LOCK:
        cached = _cached_default_stage1(store)
        if cached is not None:
            return cached

        work_root = settings.data_dir / "work" / f"default-stage1-{uuid.uuid4().hex}"
        papers_root = work_root / "papers"
        bundle_path = work_root / "chunks.jsonl.gz"
        summary_path = work_root / "summary.json"
        work_root.mkdir(parents=True, exist_ok=True)
        try:
            result = run_paper_retrieval(
                job_id="shared-default",
                input_type="keywords",
                user_input="",
                papers_root=papers_root,
                ncbi_email=settings.ncbi_email,
                ncbi_tool=settings.ncbi_tool,
                ncbi_api_key=settings.ncbi_api_key,
                keyword_limit=settings.retrieval_keyword_limit,
                batch_size=settings.retrieval_batch_size,
                request_timeout=settings.retrieval_request_timeout,
                progress_callback=progress,
            )
            line_count = build_deterministic_gzip_bundle(
                result.chunk_paths,
                bundle_path,
            )
            artifact_ref, _ = store.put_file(
                bundle_path,
                key=_DEFAULT_STAGE1_KEY,
                content_type="application/gzip",
                sha256=sha256_file(bundle_path),
            )
            source_summary = json.loads(
                result.summary_path.read_text(encoding="utf-8")
            )
            if not isinstance(source_summary, dict):
                source_summary = {}
            paper_index = _paper_index(papers_root / "corpus.jsonl")
            summary = {
                "status": "completed",
                "cache_scope": "shared_default_only",
                "cache_signature": default_stage1_signature(),
                "stats": dict(result.stats),
                "paper_index": paper_index,
                "input": source_summary.get("input", {}),
                "files": {
                    "artifact": {
                        **artifact_ref.to_dict(),
                        "record_count": line_count,
                    }
                },
            }
            _write_json(summary_path, summary)
            store.put_file(
                summary_path,
                key=_DEFAULT_STAGE1_SUMMARY_KEY,
                content_type="application/json",
                sha256=sha256_file(summary_path),
            )
            return {
                "artifact": artifact_ref.to_dict(),
                "summary": summary,
                "stats": dict(result.stats),
                "paper_index": paper_index,
                "reused": False,
            }
        finally:
            shutil.rmtree(work_root, ignore_errors=True)


def build_custom_stage1(
    *,
    run_id: str,
    query: str,
    baseline: Mapping[str, Any],
    progress: Progress | None = None,
) -> dict[str, Any] | None:
    paper_index = baseline.get("paper_index")
    paper_index = paper_index if isinstance(paper_index, list) else []
    exclude_pmids = [
        str(item.get("pmid") or "")
        for item in paper_index
        if isinstance(item, Mapping) and item.get("pmid")
    ]
    exclude_pmcids = [
        str(item.get("pmcid") or "")
        for item in paper_index
        if isinstance(item, Mapping) and item.get("pmcid")
    ]
    exclude_canonical = [
        str(item.get("canonical_id") or "")
        for item in paper_index
        if isinstance(item, Mapping) and item.get("canonical_id")
    ]

    work_root = settings.data_dir / "work" / f"run-{run_id}-stage1"
    papers_root = work_root / "papers"
    bundle_path = work_root / "chunks.jsonl.gz"
    shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        result: RetrievalResult = run_paper_retrieval(
            job_id=run_id,
            input_type="keywords",
            user_input=query,
            papers_root=papers_root,
            ncbi_email=settings.ncbi_email,
            ncbi_tool=settings.ncbi_tool,
            ncbi_api_key=settings.ncbi_api_key,
            keyword_limit=settings.retrieval_keyword_limit,
            batch_size=settings.retrieval_batch_size,
            request_timeout=settings.retrieval_request_timeout,
            progress_callback=progress,
            exclude_pmids=exclude_pmids,
            exclude_pmcids=exclude_pmcids,
            exclude_canonical_ids=exclude_canonical,
        )
        if not result.chunk_paths or int(result.stats.get("paper_count") or 0) == 0:
            return None
        line_count = build_deterministic_gzip_bundle(result.chunk_paths, bundle_path)
        key = prefixed_key(f"runs/{run_id}/stage1/chunks.jsonl.gz")
        artifact_ref, _ = get_artifact_store().put_file(
            bundle_path,
            key=key,
            content_type="application/gzip",
            sha256=sha256_file(bundle_path),
        )
        return {
            "artifact": artifact_ref.to_dict(),
            "stats": dict(result.stats),
            "record_count": line_count,
            "reused": False,
        }
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def materialize_artifact(
    artifact: Mapping[str, Any] | ArtifactRef,
    destination: Path,
    *,
    store: ArtifactStore | None = None,
) -> Path:
    selected_store = store or get_artifact_store()
    ref = artifact if isinstance(artifact, ArtifactRef) else ArtifactRef.from_dict(artifact)
    destination.parent.mkdir(parents=True, exist_ok=True)
    local = selected_store.local_path(ref.key)
    if local is not None:
        shutil.copyfile(local, destination)
    else:
        url = selected_store.presign_get(
            ref.key,
            expires_seconds=settings.artifact_presigned_ttl_seconds,
        )
        with requests.get(url, stream=True, timeout=(20, 900)) as response:
            response.raise_for_status()
            with destination.open("wb") as output:
                for block in response.iter_content(chunk_size=_ONE_MIB):
                    if block:
                        output.write(block)
                output.flush()
                os.fsync(output.fileno())
    if sha256_file(destination) != ref.sha256:
        destination.unlink(missing_ok=True)
        raise ValueError(f"Artifact {ref.key} failed its SHA-256 check.")
    return destination


def cached_json_pair(
    *,
    output_key: str,
    summary_key: str,
    expected_model_signature: str,
    expected_source_sha256: str | None = None,
) -> dict[str, Any] | None:
    store = get_artifact_store()
    output_ref = store.head(output_key)
    summary_ref = store.head(summary_key)
    if output_ref is None or summary_ref is None:
        return None
    try:
        summary = store.read_json(summary_key)
    except Exception:
        return None
    if str(summary.get("model_signature") or "") != expected_model_signature:
        return None
    if expected_source_sha256:
        source = summary.get("source")
        source = source if isinstance(source, Mapping) else {}
        actual = str(source.get("sha256") or "")
        if actual and actual != expected_source_sha256:
            return None
    return {
        "artifact": output_ref.to_dict(),
        "summary_artifact": summary_ref.to_dict(),
        "summary": summary,
        "stats": dict(summary.get("stats") or {}),
        "reused": True,
    }


__all__ = [
    "build_custom_stage1",
    "cached_json_pair",
    "default_stage1_signature",
    "get_or_build_default_stage1",
    "materialize_artifact",
    "run_scoped_signature",
]
