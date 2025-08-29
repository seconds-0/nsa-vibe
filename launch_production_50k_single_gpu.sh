#!/usr/bin/env bash
set -euo pipefail

echo "=== NSA 50K Production Run - Single GPU ==="
echo "Timestamp: $(date)"
echo "Following runbook Single-GPU Fallback (Stability First)"

# Configuration from runbook
CONFIG=configs/m7c_125m_2xa100_production.yaml

# Create output directory
mkdir -p artifacts/m7c_125m_2xa100_prod

echo ""
echo "Config: $CONFIG"
echo "Dataset: fineweb_edu"
echo "Mode: Single GPU"
echo ""

# Single-GPU launch per runbook
export CONFIG
# Training profile (single A100, reliable)
export NSA_BATCH_SIZE=1
export NSA_ACCUM=4

# NSA fast paths
export NSA_PREFILL_BATCHED=1
export NSA_USE_SEL_PACK=1
export NSA_FORCE_PARITY=0
export NSA_SEL_RANGES_V2_MIN_S=1024
export NSA_SEL_RANGES_V2=1
export NSA_USE_FA2=1
export NSA_USE_FA2_WIN=0  # Sliding FA-2 disabled: API causal semantics mismatch with pre-extracted windows
export NSA_USE_FA2_CMP=1

# Data loader perf
export NSA_FWE_DOC_BATCH=64
export NSA_FWE_PREFETCH=1
export NSA_FWE_Q=4

# Debug/alloc tuning
export NSA_SDPA_AUDIT=1
export NSA_DEBUG_TIMING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=.

echo "Environment variables set:"
echo "  NSA_BATCH_SIZE=$NSA_BATCH_SIZE"
echo "  NSA_ACCUM=$NSA_ACCUM"
echo "  NSA_PREFILL_BATCHED=$NSA_PREFILL_BATCHED"
echo "  NSA_USE_SEL_PACK=$NSA_USE_SEL_PACK"
echo "  NSA_SEL_RANGES_V2_MIN_S=$NSA_SEL_RANGES_V2_MIN_S"
echo "  NSA_SEL_RANGES_V2=$NSA_SEL_RANGES_V2"
echo "  NSA_USE_FA2=$NSA_USE_FA2 NSA_USE_FA2_WIN=$NSA_USE_FA2_WIN NSA_USE_FA2_CMP=$NSA_USE_FA2_CMP"
echo "  NSA_FWE_DOC_BATCH=$NSA_FWE_DOC_BATCH NSA_FWE_PREFETCH=$NSA_FWE_PREFETCH NSA_FWE_Q=$NSA_FWE_Q"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
echo ""

echo "🚀 Launching single-GPU training..."
python -u scripts/train_showcase.py \
  --dataset fineweb_edu \
  --ddp 0 2>&1 | tee artifacts/m7c_125m_2xa100_prod/production_single_gpu.log

echo ""
echo "✅ Done. Artifacts: artifacts/m7c_125m_2xa100_prod/"
