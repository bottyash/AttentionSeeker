# ── Stage 1: Build React ──────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci --legacy-peer-deps

COPY frontend/ .
# Empty string = relative URLs, so axios hits same origin
ENV VITE_API_BASE_URL=""
RUN npm run build

# ── Stage 2: Python + nginx + supervisord ─────────────────────────────────────
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx supervisor && rm -rf /var/lib/apt/lists/*

# Create non-root user required by HF Spaces
RUN useradd -m -u 1000 user

# ── Python deps ───────────────────────────────────────────────────────────────
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download model into image (eliminates cold-start download)
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Backend source ────────────────────────────────────────────────────────────
COPY backend/ /app/backend/

# ── React build ───────────────────────────────────────────────────────────────
COPY --from=builder /frontend/dist /usr/share/nginx/html

# ── Nginx config ──────────────────────────────────────────────────────────────
RUN printf 'server {\n\
    listen 7860;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    \n\
    # Proxy API routes to uvicorn\n\
    location ~ ^/(health|encode|similarity) {\n\
        proxy_pass http://127.0.0.1:8000;\n\
        proxy_set_header Host $host;\n\
    }\n\
    \n\
    # Serve React SPA\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

# Remove default nginx config
RUN rm -f /etc/nginx/sites-enabled/default

# ── Supervisord config ────────────────────────────────────────────────────────
RUN printf '[supervisord]\n\
nodaemon=true\n\
user=root\n\
\n\
[program:backend]\n\
command=uvicorn backend.app.main:app --host 0.0.0.0 --port 8000\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/dev/stdout\n\
stdout_logfile_maxbytes=0\n\
stderr_logfile=/dev/stderr\n\
stderr_logfile_maxbytes=0\n\
\n\
[program:nginx]\n\
command=nginx -g "daemon off;"\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/dev/stdout\n\
stdout_logfile_maxbytes=0\n\
stderr_logfile=/dev/stderr\n\
stderr_logfile_maxbytes=0\n' > /etc/supervisor/conf.d/supervisord.conf

EXPOSE 7860
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]