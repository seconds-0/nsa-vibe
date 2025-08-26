#!/usr/bin/env bash
set -euo pipefail

# Small-GPU smoke runner: validates training loop on 8–24GB GPUs
# Usage: bash scripts/small_gpu_smoke.sh [synthetic|fineweb]

DATASET=${1:-synthetic}
export CONFIG=configs/m7c_small_smoke.yaml
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export NSA_TB_DISABLE=1
export NSA_DISABLE_CSV_LOGS=1
export NSA_HEARTBEAT_EVERY=1
export NSA_DISABLE_AUX_STATS=1
export NSA_PREFILL_BATCHED=1

echo "[small-smoke] dataset=${DATASET} config=${CONFIG}"
mkdir -p artifacts/small_smoke
python -u scripts/train_showcase.py --dataset "${DATASET}" --ddp 0 2>&1 | tee artifacts/small_smoke/small_smoke.log

echo "[small-smoke] done; artifacts in artifacts/small_smoke/"
