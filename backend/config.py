"""Application configuration and persistent storage paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    data_dir: Path
    database_path: Path
    papers_dir: Path
    results_dir: Path
    model_cache_dir: Path
    cell_model_id: str
    load_cell_model: bool
    openai_api_key: str | None

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.papers_dir,
            self.results_dir,
            self.model_cache_dir,
            self.database_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    # Railway defines RAILWAY_VOLUME_MOUNT_PATH automatically when a volume is
    # attached. APP_DATA_DIR allows the same layout to be used locally.
    data_dir_raw = (
        os.getenv("APP_DATA_DIR")
        or os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
        or str(PROJECT_ROOT / "data")
    )
    data_dir = Path(data_dir_raw).expanduser().resolve()

    return Settings(
        app_name=os.getenv("APP_NAME", "Ovarian Network"),
        environment=os.getenv("APP_ENV", "development"),
        data_dir=data_dir,
        database_path=Path(
            os.getenv("DATABASE_PATH", str(data_dir / "database.sqlite"))
        ).expanduser().resolve(),
        papers_dir=Path(
            os.getenv("PAPERS_DIR", str(data_dir / "papers"))
        ).expanduser().resolve(),
        results_dir=Path(
            os.getenv("RESULTS_DIR", str(data_dir / "results"))
        ).expanduser().resolve(),
        model_cache_dir=Path(
            os.getenv("MODEL_CACHE_DIR", str(data_dir / "model_cache"))
        ).expanduser().resolve(),
        cell_model_id=os.getenv(
            "CELL_MODEL_ID", "almire/CellExLink-bioformer16L"
        ),
        load_cell_model=_as_bool(os.getenv("LOAD_CELL_MODEL"), default=False),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
    )


settings = get_settings()
