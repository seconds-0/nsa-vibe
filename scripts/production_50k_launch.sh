#!/usr/bin/env bash
# Production 50K Training Launch Script
# Based on TRAINING_RUNBOOK_50K.md requirements
# Target: 2×A100 80GB PCIe (Prime Intellect)

set -euo pipefail

echo "==================================="
echo "NSA Production 50K Training Launch"
echo "==================================="
echo "Timestamp: $(date)"
echo ""

# Phase 1: Clean up and verify environment
echo "Phase 1: Cleaning up GPU resources..."
pkill -f train_showcase.py || true
sleep 2

echo "Checking GPU status..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

# Verify branch
echo ""
echo "Phase 2: Verifying branch and environment..."
EXPECTED_BRANCH="feat/nsa-training-breakthrough-stable-a100"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]; then
    echo "ERROR: Not on correct branch!"
    echo "Expected: $EXPECTED_BRANCH"
    echo "Current: $CURRENT_BRANCH"
    exit 1
fi

# Record git SHA
GIT_SHA=$(git rev-parse HEAD)
echo "Branch: $CURRENT_BRANCH"
echo "Git SHA: $GIT_SHA"
echo "$GIT_SHA" > artifacts/production_50k_git_sha.txt

# Activate environment
echo "Activating Python environment..."
source .venv/bin/activate

# Verify PyTorch and CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
assert torch.cuda.is_available(), 'CUDA not available!'
assert torch.cuda.device_count() >= 2, 'Need at least 2 GPUs!'
"

# Phase 3: Run one-time GPU validation
echo ""
echo "Phase 3: Running GPU validation tests..."
echo "This validates the M8 batched prefill causality fix"
bash scripts/run_gpu_causality_test.sh --full

echo ""
echo "GPU validation completed successfully!"

# Phase 4: Set up environment for production
echo ""
echo "Phase 4: Setting up production environment variables..."

# Critical environment variables from Validation v4 success
export NSA_PREFILL_BATCHED=1        # CRITICAL for seq_len ≥ 1024
export NSA_SEL_RANGES_V2=1          # V2 selection enabled
export NSA_DDP_COMPRESS=bf16        # BF16 compression
export NSA_DDP_BUCKET_MB=25         # Optimal from v4 testing (25 MB)
export NCCL_ALGO=Ring               # PCIe optimization
export NCCL_PROTO=Simple            # PCIe optimization
export NCCL_IB_DISABLE=1            # No InfiniBand on PCIe
export NSA_DDP_DISABLE_GC=1         # Disable grad checkpointing under DDP initially
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export PYTHONPATH=.
export CONFIG=configs/m7c_125m_2xa100_production.yaml

# Print environment
echo "Environment configured:"
echo "  NSA_PREFILL_BATCHED=$NSA_PREFILL_BATCHED (vectorized prefill)"
echo "  NSA_SEL_RANGES_V2=$NSA_SEL_RANGES_V2"
echo "  NSA_DDP_COMPRESS=$NSA_DDP_COMPRESS"
echo "  NSA_DDP_BUCKET_MB=$NSA_DDP_BUCKET_MB"
echo "  NCCL_ALGO=$NCCL_ALGO"
echo "  NCCL_PROTO=$NCCL_PROTO"
echo "  CONFIG=$CONFIG"

# Verify config file
echo ""
echo "Verifying production config..."
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('$CONFIG')
print(f'Config: {cfg.model.dim}D, {cfg.model.n_layers}L, {cfg.model.n_heads}H')
print(f'NSA: l={cfg.nsa.l}, d={cfg.nsa.d}, l_sel={cfg.nsa.l_sel}, n_sel={cfg.nsa.n_sel}, w={cfg.nsa.w}')
print(f'Training: {cfg.train.steps} steps, seq_len={cfg.train.seq_len}, batch_size={cfg.train.batch_size}')
print(f'Save every: {cfg.train.save_every} steps')
print(f'Output dir: {cfg.train.out_dir}')
assert cfg.runtime.precision == 'bf16', 'Must use BF16!'
assert cfg.train.save_every > 0, 'Must have checkpointing enabled!'
assert cfg.train.steps == 50000, 'Must be 50k steps!'
"

# Create output directory
OUT_DIR=$(python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('$CONFIG'); print(cfg.train.out_dir)")
mkdir -p "$OUT_DIR"

# Phase 5: Launch production training
echo ""
echo "========================================="
echo "🚀 LAUNCHING PRODUCTION 50K TRAINING RUN"
echo "========================================="
echo "Config: $CONFIG"
echo "Output: $OUT_DIR"
echo "Dataset: fineweb_edu"
echo "Nodes: 2×A100 80GB PCIe"
echo ""

# Save launch info
cat > "$OUT_DIR/launch_info.txt" <<EOF
Launch timestamp: $(date)
Git SHA: $GIT_SHA
Branch: $CURRENT_BRANCH
Config: $CONFIG
Environment:
  NSA_PREFILL_BATCHED=$NSA_PREFILL_BATCHED
  NSA_SEL_RANGES_V2=$NSA_SEL_RANGES_V2
  NSA_DDP_COMPRESS=$NSA_DDP_COMPRESS
  NSA_DDP_BUCKET_MB=$NSA_DDP_BUCKET_MB
  NCCL_ALGO=$NCCL_ALGO
  NCCL_PROTO=$NCCL_PROTO
EOF

# Launch with production script (includes all validated settings)
echo "Starting training with run_2xa100_production.sh..."
bash scripts/run_2xa100_production.sh

echo ""
echo "✅ Production training launched successfully!"
echo "📊 Monitor with:"
echo "  tail -f $OUT_DIR/production.log"
echo "  tail -f $OUT_DIR/training.csv"
echo "  tail -f $OUT_DIR/heartbeat_rank0.jsonl"