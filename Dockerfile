# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Install runtime dependencies cleanly without caching wheels in image layers.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Create a dedicated non-root user and group.
RUN groupadd --system --gid 10001 appuser \
    && useradd --system --uid 10001 --gid appuser --create-home --home-dir /home/appuser --shell /usr/sbin/nologin appuser

# Copy only runtime application files (never .env, .git, .venv, or tests).
COPY --chown=appuser:appuser main.py models.lock.json ./
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser static ./static
COPY --chown=appuser:appuser data/raw ./data/raw
RUN mkdir -p /app/data/index && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, sys, urllib.request; url = 'http://127.0.0.1:${PORT:-8000}/health'; hosts = [h.strip() for h in os.getenv('EVIDENCERAG_PUBLIC_HOST', '').split(',') if h.strip()]; req = urllib.request.Request(url, headers={'Host': hosts[0]} if hosts else {}); resp = urllib.request.urlopen(req, timeout=4); sys.exit(0 if resp.status == 200 else 1)"

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-proxy-headers"]
