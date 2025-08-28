#!/bin/bash
# Execute ALL runbook tests EXACTLY as specified

cd /home/ubuntu/nsa-vibe
source .venv_runbook/bin/activate

# Set environment variables EXACTLY as in runbook Phase 4
export NSA_PREFILL_BATCHED=1
export NSA_USE_SEL_PACK=1
export NSA_FORCE_PARITY=0
export NSA_SEL_RANGES_V2_MIN_S=1024
export NSA_USE_FA2=1
export NSA_USE_FA2_WIN=1
export NSA_USE_FA2_CMP=1
export PYTHONPATH=.

echo "=== Starting COMPLETE runbook performance tests at $(date) ==="
echo "Branch: feat/nsa-selection-varlen-packing"
echo "Commit: bb1c3bbe95a7"

# Phase 5: Execute 9 single-GPU performance runs

# Test 1: S=512, adaptive
echo "=== Test 1/9: S=512, adaptive ==="
unset NSA_SEL_RANGES_V2
CONFIG=configs/runbook_s512.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s512_adaptive.log

# Test 2: S=512, force v1
echo "=== Test 2/9: S=512, force v1 ==="
export NSA_SEL_RANGES_V2=0
CONFIG=configs/runbook_s512.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s512_v1.log

# Test 3: S=512, force v2
echo "=== Test 3/9: S=512, force v2 ==="
export NSA_SEL_RANGES_V2=1
CONFIG=configs/runbook_s512.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s512_v2.log

# Test 4: S=1024, adaptive
echo "=== Test 4/9: S=1024, adaptive ==="
unset NSA_SEL_RANGES_V2
CONFIG=configs/runbook_s1024.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s1024_adaptive.log

# Test 5: S=1024, force v1
echo "=== Test 5/9: S=1024, force v1 ==="
export NSA_SEL_RANGES_V2=0
CONFIG=configs/runbook_s1024.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s1024_v1.log

# Test 6: S=1024, force v2
echo "=== Test 6/9: S=1024, force v2 ==="
export NSA_SEL_RANGES_V2=1
CONFIG=configs/runbook_s1024.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s1024_v2.log

# Test 7: S=2048, adaptive
echo "=== Test 7/9: S=2048, adaptive ==="
unset NSA_SEL_RANGES_V2
CONFIG=configs/runbook_s2048.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s2048_adaptive.log

# Test 8: S=2048, force v1
echo "=== Test 8/9: S=2048, force v1 ==="
export NSA_SEL_RANGES_V2=0
CONFIG=configs/runbook_s2048.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s2048_v1.log

# Test 9: S=2048, force v2
echo "=== Test 9/9: S=2048, force v2 ==="
export NSA_SEL_RANGES_V2=1
CONFIG=configs/runbook_s2048.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s2048_v2.log

echo "=== Phase 5 complete at $(date) ==="

# Phase 6: 2×A100 DDP test
echo "=== Phase 6: DDP test S=2048 adaptive ==="
unset NSA_SEL_RANGES_V2
CONFIG=configs/runbook_s2048.yaml \
  torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu --ddp 1 --steps 200 | tee run_ddp2_s2048_adaptive.log

echo "=== ALL TESTS COMPLETE at $(date) ==="