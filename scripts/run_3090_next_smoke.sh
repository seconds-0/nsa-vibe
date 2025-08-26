#!/usr/bin/env bash
set -euo pipefail

# 3090 next-step smoke runner: 1-step trace + 200-step stability
# Usage: bash scripts/run_3090_next_smoke.sh [synthetic|fineweb_edu]

MODE=${1:-synthetic}

echo "=== 3090 Smoke: Preflight ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[fatal] nvidia-smi not found; GPU environment missing" >&2
  exit 2
fi
nvidia-smi -L || true

echo "[preflight] Python/Torch/CUDA:" 
python - <<'PY'
import torch
print({
  'torch': torch.__version__,
  'cuda_available': torch.cuda.is_available(),
  'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  'capability': torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
})
PY

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export NSA_TB_DISABLE=1
export NSA_DISABLE_CSV_LOGS=1
export NSA_HEARTBEAT_EVERY=1
export NSA_DISABLE_AUX_STATS=1
export NSA_PREFILL_BATCHED=1
export CONFIG=configs/m7c_3090_smoke.yaml

ART_DIR=artifacts/3090_smoke
mkdir -p "$ART_DIR"

echo "=== Phase 0: 1-step ${MODE} sanity (fp16, GC off) ==="
set +e
python -u scripts/train_showcase.py \
  --dataset "$MODE" \
  --steps 1 \
  --ddp 0 \
  2>&1 | tee "$ART_DIR/phase0_1step_${MODE}.log"
PH0=$?
set -e
if [ $PH0 -ne 0 ]; then
  echo "[fail] Phase 0 failed; inspect $ART_DIR/phase0_1step_${MODE}.log" >&2
  exit $PH0
fi

echo "=== Phase 1: 200-step ${MODE} stability (fp16, GC off) ==="
set +e
python -u scripts/train_showcase.py \
  --dataset "$MODE" \
  --ddp 0 \
  2>&1 | tee "$ART_DIR/phase1_200step_${MODE}.log"
PH1=$?
set -e

echo "=== Decision Points ==="
if [ $PH1 -eq 0 ]; then
  echo "✅ Stability: passed 200 steps without stalls (TB/CSV disabled)."
  echo "Next: optionally re-enable TensorBoard only (NSA_TB_DISABLE=0) and rerun Phase 1."
else
  echo "❌ Stability: non-zero exit ($PH1)."
  echo "If hang suspected: rerun with NSA_TRACE_GRADS=1 NSA_TRACE_MODULE_BWD=1 --steps 1."
fi

echo "Artifacts: $ART_DIR"
