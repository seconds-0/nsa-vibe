#!/usr/bin/env bash
set -euo pipefail

echo "=== NSA GPU FA-2 Smoke ==="
date

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; ensure GPU drivers are available" >&2
fi

echo "GPU info:" || true
nvidia-smi || true

# Environment
export PYTHONUNBUFFERED=1
export PYTHONPATH=.
export NSA_USE_FA2=${NSA_USE_FA2:-1}
export NSA_USE_FA2_WIN=${NSA_USE_FA2_WIN:-1}
export NSA_USE_FA2_CMP=${NSA_USE_FA2_CMP:-1}
export NSA_SDPA_AUDIT=${NSA_SDPA_AUDIT:-1}
export NSA_DEBUG_TIMING=${NSA_DEBUG_TIMING:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}

echo "Env:" \
 && echo "  NSA_USE_FA2=$NSA_USE_FA2 NSA_USE_FA2_WIN=$NSA_USE_FA2_WIN NSA_USE_FA2_CMP=$NSA_USE_FA2_CMP" \
 && echo "  NSA_SDPA_AUDIT=$NSA_SDPA_AUDIT NSA_DEBUG_TIMING=$NSA_DEBUG_TIMING" \
 && echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# Optional: use uv if present; otherwise rely on an existing env
if command -v uv >/dev/null 2>&1; then
  echo "Using uv to ensure deps (GPU)";
  uv venv -p 3.11 .venv || true
  # shellcheck disable=SC1091
  source .venv/bin/activate
  if [[ -f requirements-gpu-cu121-torch24.txt ]]; then
    uv pip sync requirements-gpu-cu121-torch24.txt
  else
    uv pip sync requirements.txt
  fi
fi

echo "Running FA-2 smoke tests..."

# Parity tests (skip harmlessly if markers skip)
PYTHONPATH=. pytest -q nsa/tests/test_fa2_parity.py || true
PYTHONPATH=. pytest -q nsa/tests/test_fa2_parity_improved.py || true

# Varlen GPU test (guarded by env in tests)
NSA_TEST_FA2=1 PYTHONPATH=. pytest -q nsa/tests/test_fa2_gpu_varlen.py || true

# Guard behavior (SM89 skip path etc.)
PYTHONPATH=. pytest -q nsa/tests/test_hw_guard_sm89.py || true

echo "Done. Check logs above for fa2.gate_skip/fa2.* timing entries." 

