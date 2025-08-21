#!/usr/bin/env bash
set -euo pipefail

# Launch TensorBoard for live monitoring with SSH-friendly defaults

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

LOGDIR=${1:-artifacts/m7c_125m/tb}
PORT=${PORT:-6006}
HOST=${HOST:-0.0.0.0}

if [ ! -d .venv ]; then
  echo "[err] .venv not found. Run bootstrap first." >&2
  exit 2
fi
. ./.venv/bin/activate

echo "[tb] logdir=$LOGDIR port=$PORT host=$HOST"
echo "[tb] Checking for existing event files..."
ls -la "$LOGDIR" 2>/dev/null || echo "[tb] No event files yet - will appear when training starts"
echo ""
echo "=============================================="
echo "SSH port forward from your laptop:"
echo "  ssh -L ${PORT}:localhost:${PORT} $(whoami)@$(hostname)"
echo ""
echo "Then open: http://localhost:${PORT}"
echo "=============================================="

exec tensorboard --logdir "$LOGDIR" --port "$PORT" --host "$HOST"

