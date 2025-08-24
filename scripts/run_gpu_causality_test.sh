#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_gpu_causality_test.sh [--install] [--full]
#
# What this does:
# - Ensures branch/commit context for the M8 batched prefill assert fix
# - Optionally installs GPU deps (Torch cu121 + flash-attn) when --install is passed
# - Exports recommended env flags
# - Runs the GPU causality asserts test and a quick M0 subset
#
# Branch + commit context:
#   Base branch: feat/m7c-perf-stability
#   You will end up on: fix/m8-batched-prefill-assert (created if absent)
#   The script commits only nsa/core/nsa_attention.py if it has uncommitted changes

BASE_BRANCH="feat/m7c-perf-stability"
FIX_BRANCH="fix/m8-batched-prefill-assert"

INSTALL=0
FULL=0
for arg in "$@"; do
  case "$arg" in
    --install) INSTALL=1 ;;
    --full) FULL=1 ;;
    *) echo "[WARN] Unknown arg: $arg" ;;
  esac
done

echo "[INFO] Verifying git repo state..."
cur_branch=$(git rev-parse --abbrev-ref HEAD)
cur_head=$(git rev-parse --short HEAD)
echo "[INFO] Current: branch=${cur_branch} head=${cur_head}"

if [[ "$cur_branch" != "$BASE_BRANCH" ]]; then
  echo "[INFO] Checking out base branch ${BASE_BRANCH}..."
  git switch "$BASE_BRANCH"
fi

if git rev-parse --verify "$FIX_BRANCH" >/dev/null 2>&1; then
  echo "[INFO] Switching to existing branch ${FIX_BRANCH}..."
  git switch "$FIX_BRANCH"
else
  echo "[INFO] Creating branch ${FIX_BRANCH} from $(git rev-parse --short HEAD)..."
  git switch -c "$FIX_BRANCH"
fi

if ! git diff --quiet -- nsa/core/nsa_attention.py; then
  echo "[INFO] Committing minimal fix in nsa/core/nsa_attention.py..."
  git add nsa/core/nsa_attention.py
  git commit -m "M8: fix NameError in batched prefill strict asserts; preserve causality checks"
else
  echo "[INFO] No pending changes in nsa/core/nsa_attention.py to commit."
fi

head_short=$(git rev-parse --short HEAD)
echo "[INFO] Using commit ${head_short} on branch ${FIX_BRANCH}"

if (( INSTALL == 1 )); then
  echo "[INFO] Installing GPU dependencies (CUDA 12.1 wheels)..."
  python -m pip install -U pip wheel setuptools
  pip install --index-url https://download.pytorch.org/whl/cu121 -r requirements-gpu-cu121-torch24.txt
fi

echo "[INFO] Verifying GPU availability..."
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA not available; please ensure GPU runtime and correct Torch wheels.')
print('device_count', torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print('device0', torch.cuda.get_device_name(0))
PY

echo "[INFO] Exporting recommended env flags..."
# Default: keep Triton selection off; SDPA packed/gather is stable across GPUs
export NSA_USE_TRITON_SEL=${NSA_USE_TRITON_SEL:-0}

# Detect GPU class to choose FA-2 default
GPU_NAME=$(python - <<'PY'
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
PY
)
if [[ "$GPU_NAME" == *"A100"* || "$GPU_NAME" == *"H100"* ]]; then
  export NSA_USE_FA2=${NSA_USE_FA2:-1}
  echo "[INFO] Detected $GPU_NAME → enabling FA-2 (NSA_USE_FA2=1)."
else
  export NSA_USE_FA2=${NSA_USE_FA2:-0}
  echo "[INFO] Detected $GPU_NAME → leaving FA-2 default OFF (NSA_USE_FA2=0)."
fi

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}

# Short SDPA routing logs for first run only to avoid noise
ORIG_TORCH_LOGS=${TORCH_LOGS:-}
export TORCH_LOGS="+sdp"

echo "[RUN] GPU causality asserts test (should be 3 passed)"
PYTHONPATH=. pytest -q nsa/tests/test_causality_asserts.py

# Restore user-provided TORCH_LOGS or clear
if [[ -n "$ORIG_TORCH_LOGS" ]]; then export TORCH_LOGS="$ORIG_TORCH_LOGS"; else unset TORCH_LOGS; fi

echo "[RUN] Quick M0 subset (masks, block math, group consistency, small equiv, counters)"
PYTHONPATH=. pytest -q nsa/tests/test_masks.py nsa/tests/test_block_math.py nsa/tests/test_group_consistency.py nsa/tests/test_equiv_small.py nsa/tests/test_decode_counters.py

if (( FULL == 1 )); then
  echo "[RUN] FULL MODE: Enabling stricter CUDA error surfacing (CUDA_LAUNCH_BLOCKING=1)"
  export CUDA_LAUNCH_BLOCKING=1

  echo "[RUN] Selection packed parity"
  PYTHONPATH=. pytest -q nsa/tests/test_selection_packed.py

  if [[ "$NSA_USE_FA2" == "1" ]]; then
    echo "[RUN] FA-2 GPU varlen parity"
    NSA_TEST_FA2=1 PYTHONPATH=. pytest -q -k fa2_gpu_varlen
  else
    echo "[SKIP] FA-2 GPU varlen parity (NSA_USE_FA2=0)"
  fi

  echo "[RUN] Decode step smoke and masked tiny"
  PYTHONPATH=. pytest -q nsa/tests/test_decode_step.py nsa/tests/test_masked_tiny.py

  echo "[RUN] Rope dtype sanity"
  PYTHONPATH=. pytest -q nsa/tests/test_rope_dtype.py

  echo "[RUN] Long context smoke"
  PYTHONPATH=. pytest -q nsa/tests/test_long_context_smoke.py
fi

echo "[OK] All requested tests completed on ${FIX_BRANCH}@${head_short}"
