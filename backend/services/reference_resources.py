"""Seed the writable reference cache on first Railway volume use."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from backend.config import PROJECT_ROOT, settings


def seed_reference_resources(source_dir: Path | None = None, destination_dir: Path | None = None) -> None:
    source = Path(source_dir or PROJECT_ROOT / "data" / "reference_data").resolve()
    destination = Path(destination_dir or settings.reference_data_cache_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if source == destination or not source.is_dir():
        return
    for original in source.iterdir():
        if not original.is_file() or original.name.startswith("."):
            continue
        target = destination / original.name
        if target.exists():
            continue
        fd, name = tempfile.mkstemp(prefix=f".{original.name}.", suffix=".tmp", dir=destination)
        os.close(fd)
        temporary = Path(name)
        try:
            shutil.copy2(original, temporary)
            with temporary.open("rb") as copied:
                os.fsync(copied.fileno())
            try:
                # Publish atomically without replacing an existing live cache.
                os.link(temporary, target)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
