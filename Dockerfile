# Hugging Face Spaces — Docker deployment
# Builds frontend static files, then serves FastAPI + static via uvicorn/nginx pattern.
# HF Spaces exposes port 7860 by default.

# ── Stage 1: build React frontend ─────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci --legacy-peer-deps

COPY frontend/ .
# Point at same origin (HF serves everything on one port via proxy)
ENV VITE_API_BASE_URL=""
RUN npm run build

# ── Stage 2: Python backend + static files ────────────────────────────────────
FROM python:3.12-slim

# Install nginx to serve frontend, supervisor to manage both processes
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/app/ ./app/

# Pre-download model weights into image cache
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

# Copy built frontend into nginx directory
COPY --from=frontend-builder /frontend/dist /usr/share/nginx/html

# Nginx: serve frontend on 7860, proxy /health|/encode|/similarity to uvicorn:8000
RUN printf 'server {\n\
    listen 7860;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
\n\
    location ~ ^/(health|encode|similarity) {\n\
        proxy_pass http://127.0.0.1:8000;\n\
        proxy_set_header Host $host;\n\
        proxy_read_timeout 120s;\n\
    }\n\
}\n' > /etc/nginx/sites-available/default

# Supervisor: manage nginx + uvicorn together
RUN printf '[supervisord]\nnodaemon=true\n\n\
[program:backend]\ncommand=python -m uvicorn app.main:app --host 127.0.0.1 --port 8000\n\
directory=/app\nautostart=true\nautorestart=true\n\n\
[program:nginx]\ncommand=nginx -g "daemon off;"\nautostart=true\nautorestart=true\n' \
    > /etc/supervisor/conf.d/attentionseeker.conf

EXPOSE 7860

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/attentionseeker.conf"]
