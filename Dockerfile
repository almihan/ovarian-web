FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Railway runs the UI/controller, retrieval, PubTator3 HTTP branch, and final
# streaming merge. PyTorch, Transformers, CUDA libraries, and CellExLink model
# checkpoints remain in Modal.
COPY backend ./backend

RUN mkdir -p /data/papers /data/results /data/artifacts /data/model_cache \
    /data/local_annotation_jobs /data/relation_jobs

EXPOSE 8000

# One replica is intentional while job state is stored in SQLite.
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
