#!/usr/bin/env bash
set -euo pipefail

echo "=== NSA 50K Production Run Launch (Triage Mode) ==="
echo "Timestamp: $(date)"
echo "Following runbook section 6 - DDP triage with compression off"

# Configuration from runbook
CONFIG=configs/m7c_125m_2xa100_production.yaml

# Create output directory
mkdir -p artifacts/m7c_125m_2xa100_prod

echo ""
echo "Config: $CONFIG"
echo "Dataset: fineweb_edu"
echo "Nodes: 2×A100 DDP"
echo "TRIAGE: Compression OFF, Bucket 100MB"
echo ""

# Launch with triage settings per runbook section 6
export CONFIG
export NSA_PREFILL_BATCHED=1
export NSA_SEL_RANGES_V2=1
export NSA_DDP_COMPRESS=off  # TRIAGE: Disabled compression
export NSA_DDP_BUCKET_MB=100  # TRIAGE: Increased bucket size
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NSA_PG_TIMEOUT_MIN=5  # TRIAGE: Reduced timeout for faster failure
export NSA_DDP_DISABLE_GC=1
export NSA_DDP_FIND_UNUSED=0
export NSA_DDP_STATIC_GRAPH=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export PYTHONPATH=.

echo "Environment variables set (TRIAGE MODE):"
echo "  NSA_PREFILL_BATCHED=$NSA_PREFILL_BATCHED"
echo "  NSA_SEL_RANGES_V2=$NSA_SEL_RANGES_V2"
echo "  NSA_DDP_COMPRESS=$NSA_DDP_COMPRESS (DISABLED)"
echo "  NSA_DDP_BUCKET_MB=$NSA_DDP_BUCKET_MB (INCREASED)"
echo "  NSA_PG_TIMEOUT_MIN=$NSA_PG_TIMEOUT_MIN (REDUCED)"
echo "  NSA_DDP_DISABLE_GC=$NSA_DDP_DISABLE_GC"
echo "  NSA_DDP_FIND_UNUSED=$NSA_DDP_FIND_UNUSED"
echo "  NSA_DDP_STATIC_GRAPH=$NSA_DDP_STATIC_GRAPH"
echo ""

echo "🚀 Launching training (triage mode)..."
torchrun --nproc_per_node=2 scripts/train_showcase.py \
  --dataset fineweb_edu \
  --ddp 1 2>&1 | tee artifacts/m7c_125m_2xa100_prod/production_triage.log

echo ""
echo "✅ Done. Artifacts: artifacts/m7c_125m_2xa100_prod/"