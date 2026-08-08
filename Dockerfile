FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./

# Railway provides CPU compute for this service. Installing the CPU-only wheel
# avoids pulling several gigabytes of CUDA libraries into the image.
ARG PYTORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir torch==2.7.0 --index-url ${PYTORCH_CPU_INDEX} \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data/papers /data/results /data/model_cache

EXPOSE 8000

# Keep one worker: one process means one in-memory copy of each loaded model.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
