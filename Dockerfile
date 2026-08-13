FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_VERBOSITY=error \
    HF_HUB_VERBOSITY=error \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TQDM_DISABLE=1 \
    APP_DATA_DIR=/data

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Railway runs the UI/controller, retrieval, PubTator3 HTTP branch, and final
# streaming merge. PyTorch, Transformers, CUDA libraries, and CellExLink model
# checkpoints remain in Modal.
COPY backend ./backend

RUN mkdir -p /data/artifacts /data/model_cache /data/runs /data/work \
    /data/local_annotation_jobs /data/relation_jobs

EXPOSE 8000

# One replica is intentional because browser-run state is process-local.
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
