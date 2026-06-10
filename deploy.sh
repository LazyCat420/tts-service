#!/bin/bash
# ============================================================
# TTS Service — Build & Deploy to Synology NAS
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="tts-service"
DISPLAY_NAME="🗣️ TTS Service"
SKIP_ENV_DEPLOY=true

PRE_BUILD() {
  local CENTRAL_ENV="${DEPLOY_KIT_DIR}/.env.deploy"
  if [ -f "$CENTRAL_ENV" ]; then
    set -a; source "$CENTRAL_ENV"; set +a
    info "Loaded deploy-kit/.env.deploy"
  fi
}

EXTRA_SSH_SYNC() {
  # No .env needed for TTS
  true
}

source "${SCRIPT_DIR}/../deploy-kit/lib.sh"
