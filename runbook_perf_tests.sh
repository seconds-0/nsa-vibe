#!/bin/bash
# Runbook performance tests - executed EXACTLY as specified

cd /home/ubuntu/nsa-vibe
source .venv_runbook/bin/activate

# Phase 4: Set environment variables
export NSA_PREFILL_BATCHED=1
export NSA_USE_SEL_PACK=1
export NSA_FORCE_PARITY=0
export NSA_SEL_RANGES_V2_MIN_S=1024
export NSA_USE_FA2=1
export NSA_USE_FA2_WIN=1
export NSA_USE_FA2_CMP=1
export TORCH_LOGS=+sdp
export PYTHONPATH=.

echo "=== Starting performance tests at $(date) ==="

# Phase 5: Single-GPU performance runs

# S=512, adaptive v2 (defaults to v1 at this S due to threshold)
echo "Running S=512 adaptive..."
unset NSA_SEL_RANGES_V2
CONFIG=configs/runbook_s512.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s512_adaptive.log

# S=512, force v1
echo "Running S=512 force v1..."
export NSA_SEL_RANGES_V2=0
CONFIG=configs/runbook_s512.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s512_v1.log

# S=512, force v2
echo "Running S=512 force v2..."
export NSA_SEL_RANGES_V2=1
CONFIG=configs/runbook_s512.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s512_v2.log

# S=1024, adaptive (will pick v2)
echo "Running S=1024 adaptive..."
unset NSA_SEL_RANGES_V2
CONFIG=configs/runbook_s1024.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s1024_adaptive.log

# S=1024, force v1
echo "Running S=1024 force v1..."
export NSA_SEL_RANGES_V2=0
CONFIG=configs/runbook_s1024.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s1024_v1.log

# S=1024, force v2
echo "Running S=1024 force v2..."
export NSA_SEL_RANGES_V2=1
CONFIG=configs/runbook_s1024.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s1024_v2.log

# S=2048, adaptive (v2)
echo "Running S=2048 adaptive..."
unset NSA_SEL_RANGES_V2
CONFIG=configs/runbook_s2048.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s2048_adaptive.log

# S=2048, force v1
echo "Running S=2048 force v1..."
export NSA_SEL_RANGES_V2=0
CONFIG=configs/runbook_s2048.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s2048_v1.log

# S=2048, force v2
echo "Running S=2048 force v2..."
export NSA_SEL_RANGES_V2=1
CONFIG=configs/runbook_s2048.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s2048_v2.log

echo "=== Single-GPU tests complete at $(date) ==="