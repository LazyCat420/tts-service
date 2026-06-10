#!/bin/bash
# ============================================================
# TTS Service — Entrypoint
# Downloads missing Piper voice models on first start, then
# launches the FastAPI server.
# ============================================================

MODELS_DIR="${PIPER_MODELS_DIR:-/app/data/piper_models}"

# Count existing .onnx model files
MODEL_COUNT=$(find "$MODELS_DIR" -name "*.onnx" -not -name "*.onnx.json" 2>/dev/null | wc -l)

if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "[Entrypoint] No voice models found in ${MODELS_DIR} — downloading..."
    python /app/scripts/download_piper_models.py --output-dir "$MODELS_DIR"
    echo "[Entrypoint] Download complete."
else
    echo "[Entrypoint] Found ${MODEL_COUNT} voice models in ${MODELS_DIR} — skipping download."
fi

exec python main.py
