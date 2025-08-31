#!/usr/bin/env bash
set -euo pipefail

echo "=== NSA M7C Production — Single A100 (80GB) ==="
date
echo

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[FATAL] nvidia-smi not found; GPU environment not ready" >&2
  exit 2
fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo

# Core env (SDPA-first; FA-2 & selection-varlen disabled)
export PYTHONPATH=${PYTHONPATH:-.}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_CUDNN_ALLOW_TF32=${TORCH_CUDNN_ALLOW_TF32:-1}
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}

export NSA_USE_FA2=${NSA_USE_FA2:-0}
export NSA_FA2_MIN_LEN_WIN=${NSA_FA2_MIN_LEN_WIN:--1}
export NSA_FA2_MIN_LEN_CMP=${NSA_FA2_MIN_LEN_CMP:--1}
export NSA_USE_SEL_VARLEN=${NSA_USE_SEL_VARLEN:-0}
export NSA_USE_TRITON_SEL=${NSA_USE_TRITON_SEL:-0}
export NSA_USE_SEL_PACK=${NSA_USE_SEL_PACK:-0}
export NSA_USE_SEL_MASK=${NSA_USE_SEL_MASK:-1}
export NSA_FORCE_PARITY=${NSA_FORCE_PARITY:-0}
export NSA_FORCE_SEL_MASK=${NSA_FORCE_SEL_MASK:-1}
export NSA_STRICT_ASSERTS=${NSA_STRICT_ASSERTS:-0}

export CONFIG=${CONFIG:-configs/m7c_125m_1xa100_prod_v1.yaml}
export PYTHONUNBUFFERED=1

echo "[ENV] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "[ENV] NSA_USE_FA2=$NSA_USE_FA2 NSA_FA2_MIN_LEN_WIN=$NSA_FA2_MIN_LEN_WIN NSA_FA2_MIN_LEN_CMP=$NSA_FA2_MIN_LEN_CMP"
echo "[ENV] NSA_USE_SEL_VARLEN=$NSA_USE_SEL_VARLEN NSA_USE_TRITON_SEL=$NSA_USE_TRITON_SEL NSA_USE_SEL_PACK=$NSA_USE_SEL_PACK NSA_USE_SEL_MASK=$NSA_USE_SEL_MASK NSA_FORCE_SEL_MASK=$NSA_FORCE_SEL_MASK NSA_FORCE_PARITY=$NSA_FORCE_PARITY"
echo "[ENV] CONFIG=$CONFIG"
echo

echo "[Step] Validate environment"
python scripts/validate_run_env.py --strict || true

if [[ "${RUN_WATCHDOG:-1}" == "1" ]]; then
  echo "[Step] Start watchdog (background)"
  python scripts/_watchdog.py --dir artifacts/m7c_125m_1xa100_prod --halt 1 --interval 30 &
  WATCHDOG_PID=$!
  echo "[watchdog] pid=$WATCHDOG_PID"
fi

echo "[Step] Loader smoke (FineWeb‑Edu, 1×1024)"
python scripts/automation/fwe_smoke.py --seq-len 1024 --batch 1 --timeout 60 --tokenizer byte

echo
echo "[Step] Launch 50k training (single GPU)"
python -u scripts/train_showcase.py \
  --dataset fineweb_edu \
  --ddp 0 \
  --fwe-report-docs 1000 \
  --loader-timeout 120 \
  --synthetic-on-fail \
  2>&1 | tee training.log

ART=artifacts/m7c_125m_1xa100_prod
echo
echo "[Step] Smoke validation on artifacts"
python scripts/run_smoke_tests.py \
  --csv "$ART/training.csv" \
  --heartbeat "$ART/heartbeat_rank0.jsonl" \
  --min-steps 200 \
  --min-tps 10 || true

if [[ -n "${WATCHDOG_PID:-}" ]]; then
  echo "[watchdog] stopping pid=$WATCHDOG_PID"
  kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
fi

echo
echo "[DONE] Check artifacts under: $ART"
