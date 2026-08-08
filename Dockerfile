FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./

# Railway uses CPU compute for this service. Install the CPU-only PyTorch wheel
# so the image does not include unnecessary CUDA libraries.
ARG PYTORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir torch==2.7.0 --index-url "${PYTORCH_CPU_INDEX}" \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data/papers /data/results /data/model_cache

EXPOSE 8000

# Railway supplies PORT at runtime. Proxy-header support lets FastAPI correctly
# recognize the original HTTPS request made through Railway's reverse proxy.
# One worker means only one in-memory copy of each model will be loaded later.
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
