# ============================================================
# TTS Backend — Docker Build
# ============================================================
FROM python:3.11-slim AS deps

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Production runner ─────────────────────────────────────────
FROM python:3.11-slim AS runner
WORKDIR /app

# Install wget for healthcheck and espeak-ng for piper phonemes
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 1001 appgrp \
    && useradd --system --uid 1001 --gid appgrp -m -d /home/appusr appusr

# ── Copy Python venv ──────────────────────────────────────────
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Copy backend source ──────────────────────────────────────
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY main.py ./main.py

# Create the models directory (will be overridden by volume mount at runtime)
RUN mkdir -p /app/data/piper_models \
    && chmod +x ./scripts/entrypoint.sh ./scripts/download_piper_models.py

RUN chown -R appusr:appgrp /app

ENV PYTHONPATH="/app"
ENV PIPER_MODELS_DIR="/app/data/piper_models"

USER appusr

HEALTHCHECK --interval=60s --timeout=5s --start-period=120s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://127.0.0.1:8080/health || exit 1

EXPOSE 8080

# Entrypoint downloads missing models on first start, then launches server
ENTRYPOINT ["bash", "/app/scripts/entrypoint.sh"]
