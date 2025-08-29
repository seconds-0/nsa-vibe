#!/usr/bin/env bash
# NSA-Vibe Mentat setup script
# Purpose: prepare a fast, reproducible local environment for agents and CI-like CPU runs.
# - Installs uv if missing
# - Creates a Python 3.11 venv
# - Syncs dependencies (CPU baseline by default)
# - Prints common commands

set -euo pipefail

echo "==> NSA-Vibe .mentat/setup.sh starting"

# Detect python
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: python3 not found. Please install Python 3.11+." >&2
  exit 1
fi

# Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv (pipx fallback if available)"
  if command -v pipx >/dev/null 2>&1; then
    pipx install uv || true
  fi
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv via pip"
  $PYTHON_BIN -m pip install -U pip
  $PYTHON_BIN -m pip install -U uv
fi

# Create venv and sync deps (CPU baseline by default)
VENV_DIR="${VENV_DIR:-.venv}"
PY_VER="${PY_VER:-3.11}"
echo "==> Creating venv at ${VENV_DIR} (python ${PY_VER})"
uv venv -p "${PY_VER}" "${VENV_DIR}"

# Choose requirements file
REQ_FILE="${REQ_FILE:-requirements-cpu.txt}"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "Warning: ${REQ_FILE} not found, falling back to requirements.txt"
  REQ_FILE="requirements.txt"
fi

echo "==> Syncing deps from ${REQ_FILE}"
uv pip sync "${REQ_FILE}"

cat <<'EOT'

Setup complete.

Common commands:
- Activate venv:    source .venv/bin/activate
- Lint (ruff):      uv run ruff check .
- Type check:       uv run mypy -p nsa
- Tests (fast):     PYTHONPATH=. uv run -q pytest

GPU hosts (optional manual step):
- For CUDA 12.1 + Torch 2.4 wheels:
    uv run pip install --index-url https://download.pytorch.org/whl/cu121 -r requirements-gpu-cu121-torch24.txt

Notes:
- Pre-commit hooks: run 'uv run pre-commit install' after activation to enable git hooks.
- Mentat will run .mentat/format.sh automatically before committing to apply autofixes.

EOT

echo "==> NSA-Vibe .mentat/setup.sh done"
