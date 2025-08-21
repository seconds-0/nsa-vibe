#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/install_fa2.sh [max_jobs]
# Defaults to MAX_JOBS=2 when not provided.

MAX_JOBS=${1:-2}

echo "[fa2] Ensuring ninja is installed..." >&2
pip uninstall -y ninja >/dev/null 2>&1 || true
pip install -q ninja
ninja --version

echo "[fa2] Installing flash-attn with MAX_JOBS=${MAX_JOBS} (no build isolation)" >&2
MAX_JOBS=${MAX_JOBS} pip install flash-attn --no-build-isolation
echo "[fa2] flash-attn install complete" >&2

