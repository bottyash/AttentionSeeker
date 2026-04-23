# ── Stage 1: builder (installs heavy deps separately for layer caching) ──────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools needed by some deps (e.g. umap-learn / numba)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy app source
COPY app/ ./app/
COPY .env .env

# Pre-download the model so first request is instant
# (cache is baked into the image, no model download at runtime)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
