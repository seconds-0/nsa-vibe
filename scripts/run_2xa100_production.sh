#!/usr/bin/env bash

# NSA Production Launcher — 2×A100 80GB (PCIe)
# Uses validated env from Validation v4 (batched prefill, selection v2, BF16 compression, PCIe-tuned NCCL)

set -euo pipefail

echo "=== NSA Production: 2×A100 Launch ==="
echo "Timestamp: $(date)"

CONFIG_DEFAULT="configs/m7c_125m_2xa100_production.yaml"
DATASET_DEFAULT="fineweb_edu"
NP_DEFAULT=2

# Allow quick overrides
CONFIG="${CONFIG:-$CONFIG_DEFAULT}"
DATASET="${DATASET:-$DATASET_DEFAULT}"
NP="${NP:-$NP_DEFAULT}"

echo "Config:  $CONFIG"
echo "Dataset: $DATASET"
echo "NP:      $NP"

# Core env (from Validation v4)
export NSA_PREFILL_BATCHED=1
export NSA_SEL_RANGES_V2=1
# Allow override of gradient compression (set to 'off' to disable). Default: bf16
export NSA_DDP_COMPRESS=${NSA_DDP_COMPRESS:-bf16}
export NSA_DDP_BUCKET_MB=${NSA_DDP_BUCKET_MB:-25}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export NCCL_PROTO=${NCCL_PROTO:-Simple}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
# Fail-fast NCCL & verbose distributed debug to surface deadlocks quickly
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export TORCH_BACKEND=${TORCH_BACKEND:-nccl}

# Stability: disable gradient checkpointing under DDP for initial production
export NSA_DDP_DISABLE_GC=${NSA_DDP_DISABLE_GC:-1}
# Harden DDP graph semantics to avoid per-step dynamic graph mismatches
export NSA_DDP_FIND_UNUSED=${NSA_DDP_FIND_UNUSED:-0}
export NSA_DDP_STATIC_GRAPH=${NSA_DDP_STATIC_GRAPH:-1}

# Memory/allocator tuning
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}

# Optional: enable FA-2 paths on A100 (safe fallbacks in code). Leave off to keep parity reference.
# export NSA_USE_FA2=1
# export NSA_USE_FA2_WIN=1
# export NSA_USE_FA2_CMP=1

# Optional: short SDPA routing probe for first ~100 steps only; comment out for quiet logs
# export TORCH_LOGS="+sdp"

export PYTHONPATH=.
export CONFIG="$CONFIG"

OUT_DIR=$(python - <<'PY'
from omegaconf import OmegaConf
import os
cfg = OmegaConf.load(os.environ.get('CONFIG','configs/m7c_125m_2xa100_production.yaml'))
print(cfg.train.out_dir)
PY
)
mkdir -p "$OUT_DIR"

echo "\n🚀 Launching training...\n"
set -x
torchrun --nproc_per_node="$NP" scripts/train_showcase.py \
  --dataset "$DATASET" \
  --ddp 1 2>&1 | tee "$OUT_DIR/production.log"
set +x

echo "\n✅ Done. Artifacts: $OUT_DIR"
