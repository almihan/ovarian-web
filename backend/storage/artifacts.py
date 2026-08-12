"""Small S3-compatible artifact layer.

Railway Buckets expose an S3-compatible endpoint.  The web process stores only
one canonical compressed artifact for each completed stage and passes short-
lived presigned URLs to the Modal worker.  A local implementation is retained
for development and tests.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping

from backend.config import settings

_ONE_MIB = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_ONE_MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_key(value: str) -> str:
    candidate = str(PurePosixPath(str(value).strip().lstrip("/")))
    if not candidate or candidate in {".", ".."}:
        raise ValueError("Artifact key is empty.")
    parts = PurePosixPath(candidate).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Artifact key contains an unsafe path component.")
    return "/".join(parts)


def prefixed_key(relative_key: str) -> str:
    relative = _clean_key(relative_key)
    prefix = settings.artifact_prefix.strip("/")
    return f"{prefix}/{relative}" if prefix else relative


@dataclass(slots=True, frozen=True)
class ArtifactRef:
    key: str
    size_bytes: int
    sha256: str
    content_type: str = "application/octet-stream"
    content_encoding: str | None = None
    backend: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            key=_clean_key(str(payload["key"])),
            size_bytes=max(0, int(payload.get("size_bytes") or 0)),
            sha256=str(payload.get("sha256") or ""),
            content_type=str(
                payload.get("content_type") or "application/octet-stream"
            ),
            content_encoding=(
                str(payload["content_encoding"])
                if payload.get("content_encoding")
                else None
            ),
            backend=str(payload.get("backend") or settings.artifact_backend),
        )


class ArtifactStore(ABC):
    """Interface shared by the local and Railway Bucket implementations."""

    backend: str

    @abstractmethod
    def head(self, key: str) -> ArtifactRef | None:
        raise NotImplementedError

    @abstractmethod
    def put_file(
        self,
        path: Path,
        *,
        key: str,
        content_type: str | None = None,
        content_encoding: str | None = None,
        sha256: str | None = None,
    ) -> tuple[ArtifactRef, bool]:
        """Upload a file and return ``(reference, reused_existing)``."""

    @abstractmethod
    def presign_get(
        self,
        key: str,
        *,
        download_name: str | None = None,
        expires_seconds: int | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def presign_put(
        self,
        key: str,
        *,
        content_type: str,
        content_encoding: str | None = None,
        expires_seconds: int | None = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def local_path(self, key: str) -> Path | None:
        raise NotImplementedError

    @abstractmethod
    def read_json(self, key: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    backend = "local"

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or settings.artifact_local_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        resolved = (self.root / _clean_key(key)).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Artifact path escapes the local artifact root.")
        return resolved

    @staticmethod
    def _metadata_path(path: Path) -> Path:
        return path.with_name(path.name + ".metadata.json")

    def head(self, key: str) -> ArtifactRef | None:
        path = self._path(key)
        if not path.is_file():
            return None
        metadata_path = self._metadata_path(path)
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    metadata = loaded
            except (OSError, json.JSONDecodeError):
                metadata = {}
        digest = str(metadata.get("sha256") or sha256_file(path))
        content_type = str(
            metadata.get("content_type")
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )
        return ArtifactRef(
            key=_clean_key(key),
            size_bytes=path.stat().st_size,
            sha256=digest,
            content_type=content_type,
            content_encoding=(
                str(metadata["content_encoding"])
                if metadata.get("content_encoding")
                else None
            ),
            backend=self.backend,
        )

    def put_file(
        self,
        path: Path,
        *,
        key: str,
        content_type: str | None = None,
        content_encoding: str | None = None,
        sha256: str | None = None,
    ) -> tuple[ArtifactRef, bool]:
        source = path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        clean_key = _clean_key(key)
        digest = sha256 or sha256_file(source)
        existing = self.head(clean_key)
        if existing is not None and existing.sha256 == digest:
            return existing, True

        destination = self._path(clean_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle, length=_ONE_MIB)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)

        resolved_type = content_type or mimetypes.guess_type(destination.name)[0]
        resolved_type = resolved_type or "application/octet-stream"
        metadata = {
            "sha256": digest,
            "content_type": resolved_type,
            "content_encoding": content_encoding,
        }
        metadata_path = self._metadata_path(destination)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return (
            ArtifactRef(
                key=clean_key,
                size_bytes=destination.stat().st_size,
                sha256=digest,
                content_type=resolved_type,
                content_encoding=content_encoding,
                backend=self.backend,
            ),
            False,
        )

    def presign_get(
        self,
        key: str,
        *,
        download_name: str | None = None,
        expires_seconds: int | None = None,
    ) -> str:
        del download_name, expires_seconds
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.as_uri()

    def presign_put(
        self,
        key: str,
        *,
        content_type: str,
        content_encoding: str | None = None,
        expires_seconds: int | None = None,
    ) -> str:
        del content_type, content_encoding, expires_seconds
        return self._path(key).as_uri()

    def local_path(self, key: str) -> Path | None:
        path = self._path(key)
        return path if path.is_file() else None

    def read_json(self, key: str) -> dict[str, Any]:
        path = self._path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read JSON artifact {key}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"JSON artifact {key} is not an object.")
        return payload

    def delete(self, key: str) -> None:
        path = self._path(key)
        path.unlink(missing_ok=True)
        self._metadata_path(path).unlink(missing_ok=True)


class S3ArtifactStore(ArtifactStore):
    backend = "s3"

    def __init__(self) -> None:
        if not settings.object_store_configured:
            raise RuntimeError(
                "The S3 artifact backend is selected but Railway Bucket credentials "
                "are incomplete."
            )
        import boto3
        from botocore.config import Config

        self.bucket = settings.artifact_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.artifact_endpoint,
            aws_access_key_id=settings.artifact_access_key_id,
            aws_secret_access_key=settings.artifact_secret_access_key,
            region_name=settings.artifact_region,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": settings.artifact_addressing_style},
                retries={"max_attempts": 4, "mode": "standard"},
            ),
        )

    def head(self, key: str) -> ArtifactRef | None:
        from botocore.exceptions import ClientError

        clean_key = _clean_key(key)
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=clean_key)
        except ClientError as exc:
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if status == 404 or code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        metadata = response.get("Metadata") or {}
        return ArtifactRef(
            key=clean_key,
            size_bytes=int(response.get("ContentLength") or 0),
            sha256=str(metadata.get("sha256") or ""),
            content_type=str(response.get("ContentType") or "application/octet-stream"),
            content_encoding=(
                str(response["ContentEncoding"])
                if response.get("ContentEncoding")
                else None
            ),
            backend=self.backend,
        )

    def put_file(
        self,
        path: Path,
        *,
        key: str,
        content_type: str | None = None,
        content_encoding: str | None = None,
        sha256: str | None = None,
    ) -> tuple[ArtifactRef, bool]:
        source = path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        clean_key = _clean_key(key)
        digest = sha256 or sha256_file(source)
        existing = self.head(clean_key)
        if existing is not None and existing.sha256 == digest:
            return existing, True

        resolved_type = content_type or mimetypes.guess_type(source.name)[0]
        resolved_type = resolved_type or "application/octet-stream"
        extra: dict[str, Any] = {
            "ContentType": resolved_type,
            "Metadata": {"sha256": digest},
        }
        if content_encoding:
            extra["ContentEncoding"] = content_encoding
        self.client.upload_file(
            str(source),
            self.bucket,
            clean_key,
            ExtraArgs=extra,
        )
        return (
            ArtifactRef(
                key=clean_key,
                size_bytes=source.stat().st_size,
                sha256=digest,
                content_type=resolved_type,
                content_encoding=content_encoding,
                backend=self.backend,
            ),
            False,
        )

    def presign_get(
        self,
        key: str,
        *,
        download_name: str | None = None,
        expires_seconds: int | None = None,
    ) -> str:
        params: dict[str, Any] = {"Bucket": self.bucket, "Key": _clean_key(key)}
        if download_name:
            safe_name = Path(download_name).name.replace('"', "")
            params["ResponseContentDisposition"] = (
                f'attachment; filename="{safe_name}"'
            )
        return str(
            self.client.generate_presigned_url(
                "get_object",
                Params=params,
                ExpiresIn=expires_seconds
                or settings.artifact_presigned_ttl_seconds,
            )
        )

    def presign_put(
        self,
        key: str,
        *,
        content_type: str,
        content_encoding: str | None = None,
        expires_seconds: int | None = None,
    ) -> str:
        params: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": _clean_key(key),
            "ContentType": content_type,
        }
        if content_encoding:
            params["ContentEncoding"] = content_encoding
        return str(
            self.client.generate_presigned_url(
                "put_object",
                Params=params,
                ExpiresIn=expires_seconds
                or settings.artifact_presigned_ttl_seconds,
            )
        )

    def local_path(self, key: str) -> Path | None:
        del key
        return None

    def read_json(self, key: str) -> dict[str, Any]:
        response = self.client.get_object(Bucket=self.bucket, Key=_clean_key(key))
        body: BinaryIO = response["Body"]
        payload = json.loads(body.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"JSON artifact {key} is not an object.")
        return payload

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=_clean_key(key))


def get_artifact_store() -> ArtifactStore:
    if settings.artifact_backend == "s3":
        return S3ArtifactStore()
    return LocalArtifactStore()


__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "LocalArtifactStore",
    "S3ArtifactStore",
    "get_artifact_store",
    "prefixed_key",
    "sha256_file",
]
