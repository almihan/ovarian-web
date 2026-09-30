"""Application configuration for the Railway controller and Modal GPU worker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from backend import deployment_flags

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _as_int(
    value: str | None,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def _as_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _as_choice(
    value: str | None,
    *,
    default: str,
    choices: set[str],
) -> str:
    if value is None or not value.strip():
        return default
    normalized = value.strip().casefold()
    return normalized if normalized in choices else default


def _public_base_url() -> str:
    explicit = (os.getenv("PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if explicit:
        return explicit
    railway_domain = (os.getenv("RAILWAY_PUBLIC_DOMAIN") or "").strip().strip("/")
    return f"https://{railway_domain}" if railway_domain else ""


@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    public_base_url: str
    public_precomputed_only: bool
    precomputed_corpora_dir: Path
    monthly_updates_enabled: bool
    monthly_update_max_new_papers: int
    monthly_update_poll_seconds: int
    monthly_update_timeout_seconds: int
    monthly_update_token: str
    monthly_update_text_mode: str

    data_dir: Path
    artifact_local_dir: Path
    cell_model_cache_dir: Path
    reference_data_cache_dir: Path
    local_annotation_jobs_dir: Path
    relation_jobs_dir: Path
    run_retention_seconds: int

    artifact_backend: str
    artifact_bucket: str
    artifact_endpoint: str
    artifact_access_key_id: str
    artifact_secret_access_key: str
    artifact_region: str
    artifact_addressing_style: str
    artifact_prefix: str
    artifact_presigned_ttl_seconds: int

    ncbi_email: str
    ncbi_tool: str
    ncbi_api_key: str | None
    retrieval_keyword_limit: int
    retrieval_batch_size: int
    retrieval_request_timeout: int

    pubtator3_batch_size: int
    pubtator3_request_timeout: int
    pubtator3_required: bool
    pubtator3_resolve_preferred_labels: bool

    cell_annotation_backend: str
    cell_local_device: str
    modal_app_name: str
    modal_function_name: str
    modal_environment: str | None
    cell_ner_model: str
    cell_ner_revision: str | None
    cell_nen_model: str
    cell_nen_revision: str | None
    cell_disable_abbreviations: bool
    cell_normalization_method_log: bool
    cell_cpu_threads: int
    cell_ner_text_batch_size: int
    cell_ner_window_batch_size: int
    cell_nen_batch_size: int
    cell_nen_request_batch_size: int
    cell_job_timeout_seconds: int

    openai_api_key: str | None
    openai_organization: str | None
    openai_project: str | None
    relation_model: str
    relation_window_size: int
    relation_concurrency: int
    relation_request_timeout_seconds: int
    relation_max_request_retries: int
    relation_retry_base_seconds: int
    relation_progress_update_every: int
    relation_max_output_tokens: int
    relation_max_output_tokens_ceiling: int
    relation_reasoning_effort: str
    relation_prompt_cache_key: str
    relation_prompt_cache_shards: int

    network_initial_nodes: int
    network_max_initial_nodes: int
    network_expansion_limit: int
    network_search_limit: int
    network_evidence_limit: int
    network_hierarchy_max_paths: int

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.artifact_local_dir,
            self.cell_model_cache_dir,
            self.reference_data_cache_dir,
            self.local_annotation_jobs_dir,
            self.relation_jobs_dir,
            self.precomputed_corpora_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def modal_configured(self) -> bool:
        return bool(
            self.cell_annotation_backend == "modal"
            and self.modal_app_name
            and self.modal_function_name
            and os.getenv("MODAL_TOKEN_ID")
            and os.getenv("MODAL_TOKEN_SECRET")
        )

    @property
    def relation_configured(self) -> bool:
        return bool(self.openai_api_key and self.relation_model)

    @property
    def object_store_configured(self) -> bool:
        if self.artifact_backend == "local":
            return True
        return bool(
            self.artifact_bucket
            and self.artifact_endpoint
            and self.artifact_access_key_id
            and self.artifact_secret_access_key
        )


def get_settings() -> Settings:
    on_railway = any(
        os.getenv(name)
        for name in ("RAILWAY_ENVIRONMENT_ID", "RAILWAY_SERVICE_ID", "RAILWAY_PUBLIC_DOMAIN")
    )
    environment = (os.getenv("APP_ENV") or ("production" if on_railway else "development")).strip()
    automatic_public_mode = on_railway or environment.casefold() in {"production", "staging"}
    public_default = (
        automatic_public_mode
        if deployment_flags.PUBLIC_PRECOMPUTED_ONLY is None
        else deployment_flags.PUBLIC_PRECOMPUTED_ONLY
    )
    data_dir_raw = (
        os.getenv("APP_DATA_DIR")
        or os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
        or str(PROJECT_ROOT / "data")
    )
    data_dir = Path(data_dir_raw).expanduser().resolve()

    bucket = (os.getenv("ARTIFACT_BUCKET") or os.getenv("BUCKET") or "").strip()
    endpoint = (os.getenv("ARTIFACT_ENDPOINT") or os.getenv("ENDPOINT") or "").strip()
    access_key = (
        os.getenv("ARTIFACT_ACCESS_KEY_ID") or os.getenv("ACCESS_KEY_ID") or ""
    ).strip()
    secret_key = (
        os.getenv("ARTIFACT_SECRET_ACCESS_KEY")
        or os.getenv("SECRET_ACCESS_KEY")
        or ""
    ).strip()
    detected_s3 = bool(bucket and endpoint and access_key and secret_key)
    artifact_backend = _as_choice(
        os.getenv("ARTIFACT_BACKEND"),
        default="s3" if detected_s3 else "local",
        choices={"local", "s3"},
    )

    artifact_prefix = (
        os.getenv("ARTIFACT_PREFIX") or "ovarian-network-hgnc-mesh"
    ).strip("/")

    # Structured relation output can exceed a small cap because GPT-5 output
    # accounting includes reasoning and formatting tokens. Keep a practical
    # first-attempt floor and allow bounded expansion on retries.
    relation_max_output_tokens = _as_int(
        os.getenv("RELATION_MAX_OUTPUT_TOKENS"),
        default=4_096,
        minimum=4_096,
        maximum=32_768,
    )
    relation_max_output_tokens_ceiling = max(
        relation_max_output_tokens,
        _as_int(
            os.getenv("RELATION_MAX_OUTPUT_TOKENS_CEILING"),
            default=32_768,
            minimum=4_096,
            maximum=32_768,
        ),
    )

    cell_annotation_backend = _as_choice(
        os.getenv("CELL_ANNOTATION_BACKEND"),
        default="modal",
        choices={"modal", "local", "disabled"},
    )
    explicit_cell_cache = (os.getenv("CELL_MODEL_CACHE_DIR") or "").strip()
    if explicit_cell_cache:
        cell_model_cache_dir = Path(explicit_cell_cache).expanduser().resolve()
    elif cell_annotation_backend == "local":
        # Keep large Hugging Face checkpoints and the generated Cell Ontology
        # embedding matrix outside the extracted project. Replacing the ZIP or
        # moving the source directory must not force an expensive rebuild.
        cache_home = Path(
            os.getenv("XDG_CACHE_HOME") or (Path.home() / ".cache")
        ).expanduser()
        cell_model_cache_dir = (
            cache_home / "ovarian-network" / "cellexlink"
        ).resolve()
    else:
        cell_model_cache_dir = (data_dir / "model_cache").resolve()

    return Settings(
        app_name=os.getenv("APP_NAME", "Ovarian Network — HGNC/MeSH"),
        environment=environment,
        public_base_url=_public_base_url(),
        public_precomputed_only=_as_bool(os.getenv("PUBLIC_PRECOMPUTED_ONLY"), default=public_default),
        precomputed_corpora_dir=Path(
            os.getenv("PRECOMPUTED_CORPORA_DIR") or str(data_dir / "precomputed_corpora")
        ).expanduser().resolve(),
        monthly_updates_enabled=_as_bool(
            os.getenv("MONTHLY_UPDATES_ENABLED"), default=deployment_flags.MONTHLY_UPDATES_ENABLED
        ),
        monthly_update_max_new_papers=_as_int(
            os.getenv("MONTHLY_UPDATE_MAX_NEW_PAPERS"),
            default=deployment_flags.MONTHLY_UPDATE_MAX_NEW_PAPERS,
            minimum=0, maximum=100_000,
        ),
        monthly_update_poll_seconds=_as_int(
            os.getenv("MONTHLY_UPDATE_POLL_SECONDS"), default=2, minimum=1, maximum=60,
        ),
        monthly_update_timeout_seconds=_as_int(
            os.getenv("MONTHLY_UPDATE_TIMEOUT_SECONDS"), default=86_400,
            minimum=60, maximum=604_800,
        ),
        monthly_update_token=(os.getenv("MONTHLY_UPDATE_TOKEN") or "").strip(),
        monthly_update_text_mode=_as_choice(
            os.getenv("MONTHLY_UPDATE_TEXT_MODE"), default="fulltext", choices={"abstract", "fulltext"},
        ),
        data_dir=data_dir,
        artifact_local_dir=Path(
            os.getenv("ARTIFACT_LOCAL_DIR", str(data_dir / "artifacts"))
        ).expanduser().resolve(),
        cell_model_cache_dir=cell_model_cache_dir,
        reference_data_cache_dir=Path(
            os.getenv(
                "REFERENCE_DATA_CACHE_DIR",
                str(data_dir / "reference_data"),
            )
        ).expanduser().resolve(),
        local_annotation_jobs_dir=Path(
            os.getenv(
                "LOCAL_ANNOTATION_JOBS_DIR",
                str(data_dir / "local_annotation_jobs"),
            )
        ).expanduser().resolve(),
        relation_jobs_dir=Path(
            os.getenv(
                "RELATION_JOBS_DIR",
                str(data_dir / "relation_jobs"),
            )
        ).expanduser().resolve(),
        run_retention_seconds=_as_int(
            os.getenv("RUN_RETENTION_SECONDS"),
            default=21_600,
            minimum=3_600,
            maximum=86_400,
        ),
        artifact_backend=artifact_backend,
        artifact_bucket=bucket,
        artifact_endpoint=endpoint,
        artifact_access_key_id=access_key,
        artifact_secret_access_key=secret_key,
        artifact_region=(
            os.getenv("ARTIFACT_REGION") or os.getenv("REGION") or "auto"
        ).strip(),
        artifact_addressing_style=_as_choice(
            os.getenv("ARTIFACT_ADDRESSING_STYLE"),
            default="virtual",
            choices={"virtual", "path", "auto"},
        ),
        artifact_prefix=artifact_prefix,
        artifact_presigned_ttl_seconds=_as_int(
            os.getenv("ARTIFACT_PRESIGNED_TTL_SECONDS"),
            default=86_400,
            minimum=900,
            maximum=604_800,
        ),
        ncbi_email=os.getenv("NCBI_EMAIL", "").strip(),
        ncbi_tool=os.getenv("NCBI_TOOL", "ovarian_network_hgnc_mesh").strip(),
        ncbi_api_key=os.getenv("NCBI_API_KEY") or None,
        retrieval_keyword_limit=_as_int(
            os.getenv("RETRIEVAL_KEYWORD_LIMIT"),
            default=2000,
            minimum=1,
            maximum=10_000,
        ),
        retrieval_batch_size=_as_int(
            os.getenv("RETRIEVAL_BATCH_SIZE"),
            default=100,
            minimum=1,
            maximum=200,
        ),
        retrieval_request_timeout=_as_int(
            os.getenv("RETRIEVAL_REQUEST_TIMEOUT"),
            default=120,
            minimum=20,
            maximum=300,
        ),
        pubtator3_batch_size=_as_int(
            os.getenv("PUBTATOR3_BATCH_SIZE"),
            default=100,
            minimum=1,
            maximum=100,
        ),
        pubtator3_request_timeout=_as_int(
            os.getenv("PUBTATOR3_REQUEST_TIMEOUT"),
            default=120,
            minimum=20,
            maximum=300,
        ),
        pubtator3_required=_as_bool(
            os.getenv("PUBTATOR3_REQUIRED"),
            default=True,
        ),
        pubtator3_resolve_preferred_labels=_as_bool(
            os.getenv("PUBTATOR3_RESOLVE_PREFERRED_LABELS"),
            default=True,
        ),
        cell_annotation_backend=cell_annotation_backend,
        cell_local_device=_as_choice(
            os.getenv("CELL_LOCAL_DEVICE"),
            default="cpu",
            choices={"cpu", "auto"},
        ),
        modal_app_name=(
            os.getenv("MODAL_APP_NAME", "ovarian-cellexlink-hgnc-mesh").strip()
            or "ovarian-cellexlink-hgnc-mesh"
        ),
        modal_function_name=(
            os.getenv("MODAL_FUNCTION_NAME", "annotate_bundle").strip()
            or "annotate_bundle"
        ),
        modal_environment=(os.getenv("MODAL_ENVIRONMENT") or "").strip() or None,
        cell_ner_model=(
            os.getenv("CELL_NER_MODEL", "almire/CellExLink-bioformer16L").strip()
            or "almire/CellExLink-bioformer16L"
        ),
        cell_ner_revision=(os.getenv("CELL_NER_REVISION") or "").strip() or None,
        cell_nen_model=(
            os.getenv("CELL_NEN_MODEL", "almire/CellExLink-Sapbert").strip()
            or "almire/CellExLink-Sapbert"
        ),
        cell_nen_revision=(os.getenv("CELL_NEN_REVISION") or "").strip() or None,
        cell_disable_abbreviations=_as_bool(
            os.getenv("CELL_DISABLE_ABBREVIATIONS"),
            default=False,
        ),
        cell_normalization_method_log=_as_bool(
            os.getenv("CELL_NORMALIZATION_METHOD_LOG"),
            default=False,
        ),
        cell_cpu_threads=_as_int(
            os.getenv("CELL_CPU_THREADS"),
            default=4,
            minimum=1,
            maximum=16,
        ),
        cell_ner_text_batch_size=_as_int(
            os.getenv("CELL_NER_TEXT_BATCH_SIZE"),
            default=8,
            minimum=1,
            maximum=64,
        ),
        cell_ner_window_batch_size=_as_int(
            os.getenv("CELL_NER_WINDOW_BATCH_SIZE"),
            default=4,
            minimum=1,
            maximum=32,
        ),
        cell_nen_batch_size=_as_int(
            os.getenv("CELL_NEN_BATCH_SIZE"),
            default=128,
            minimum=1,
            maximum=512,
        ),
        cell_nen_request_batch_size=_as_int(
            os.getenv("CELL_NEN_REQUEST_BATCH_SIZE"),
            default=128,
            minimum=1,
            maximum=1024,
        ),
        cell_job_timeout_seconds=_as_int(
            os.getenv("CELL_JOB_TIMEOUT_SECONDS"),
            default=21_600,
            minimum=600,
            maximum=86_400,
        ),
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
        openai_organization=(os.getenv("OPENAI_ORGANIZATION") or "").strip() or None,
        openai_project=(os.getenv("OPENAI_PROJECT") or "").strip() or None,
        relation_model=(
            os.getenv("OPENAI_RELATION_MODEL", "gpt-5.4-nano").strip()
            or "gpt-5.4-nano"
        ),
        relation_window_size=_as_int(
            os.getenv("RELATION_WINDOW_SIZE"),
            default=500,
            minimum=1,
            maximum=500,
        ),
        relation_concurrency=_as_int(
            os.getenv("RELATION_CONCURRENCY"),
            default=8,
            minimum=1,
            maximum=32,
        ),
        relation_request_timeout_seconds=_as_int(
            os.getenv("RELATION_REQUEST_TIMEOUT_SECONDS"),
            default=180,
            minimum=30,
            maximum=900,
        ),
        relation_max_request_retries=_as_int(
            os.getenv("RELATION_MAX_REQUEST_RETRIES"),
            default=5,
            minimum=0,
            maximum=10,
        ),
        relation_retry_base_seconds=_as_int(
            os.getenv("RELATION_RETRY_BASE_SECONDS"),
            default=1,
            minimum=1,
            maximum=30,
        ),
        relation_progress_update_every=_as_int(
            os.getenv("RELATION_PROGRESS_UPDATE_EVERY"),
            default=5,
            minimum=1,
            maximum=100,
        ),
        relation_max_output_tokens=relation_max_output_tokens,
        relation_max_output_tokens_ceiling=relation_max_output_tokens_ceiling,
        relation_reasoning_effort=_as_choice(
            os.getenv("RELATION_REASONING_EFFORT"),
            default="low",
            choices={"none", "low", "medium", "high", "xhigh"},
        ),
        relation_prompt_cache_key=(
            os.getenv(
                "RELATION_PROMPT_CACHE_KEY",
                "ovarian-relations-four-predicates",
            ).strip()
            or "ovarian-relations-four-predicates"
        )[:64],
        relation_prompt_cache_shards=_as_int(
            os.getenv("RELATION_PROMPT_CACHE_SHARDS"),
            default=32,
            minimum=1,
            maximum=128,
        ),
        network_initial_nodes=_as_int(
            os.getenv("NETWORK_INITIAL_NODES"),
            default=120,
            minimum=10,
            maximum=1000,
        ),
        network_max_initial_nodes=_as_int(
            os.getenv("NETWORK_MAX_INITIAL_NODES"),
            default=1000,
            minimum=50,
            maximum=5000,
        ),
        network_expansion_limit=_as_int(
            os.getenv("NETWORK_EXPANSION_LIMIT"),
            default=150,
            minimum=10,
            maximum=1000,
        ),
        network_search_limit=_as_int(
            os.getenv("NETWORK_SEARCH_LIMIT"),
            default=30,
            minimum=5,
            maximum=100,
        ),
        network_evidence_limit=_as_int(
            os.getenv("NETWORK_EVIDENCE_LIMIT"),
            default=100,
            minimum=10,
            maximum=500,
        ),
        network_hierarchy_max_paths=_as_int(
            os.getenv("NETWORK_HIERARCHY_MAX_PATHS"),
            default=3,
            minimum=1,
            maximum=10,
        ),
    )


settings = get_settings()
