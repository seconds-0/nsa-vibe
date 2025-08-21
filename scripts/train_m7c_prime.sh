#!/usr/bin/env bash
set -euo pipefail

# One-command M7C training runner for Prime Intellect (single box, 1–2 GPUs)
# - Detects GPU VRAM and selects an appropriate config
# - Sets NCCL env for reliability over SSH
# - Creates a timestamped run dir and logs training output

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if [ ! -d .venv ]; then
  echo "[err] .venv not found. Run: bash scripts/prime_bootstrap.sh" >&2
  exit 2
fi
. ./.venv/bin/activate

# Reliable NCCL defaults for single-node SSH sessions
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# Pick config based on largest GPU VRAM
GPU_GB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -nr | head -n1)
fi

CFG="configs/m7c_125m.yaml"
if [ "$GPU_GB" -ge 75000 ]; then
  CFG="configs/m7c_125m_80g.yaml"
elif [ "$GPU_GB" -ge 35000 ]; then
  CFG="configs/m7c_125m_40g.yaml"
else
  CFG="configs/m7c_125m_24g.yaml"
fi

# Determine GPU count
NGPU=1
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
  [ "$NGPU" -lt 1 ] && NGPU=1
fi

# Timestamped run dir
STAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="artifacts/train_runs/m7c_${STAMP}"
mkdir -p "$RUN_DIR"

echo "run_dir: $RUN_DIR" | tee "$RUN_DIR/run.info"
echo "config: $CFG" | tee -a "$RUN_DIR/run.info"
echo "gpus: $NGPU (max_mem=${GPU_GB}MB)" | tee -a "$RUN_DIR/run.info"

# Optional quick smoke (synthetic, 50 steps) to validate env before streaming
echo "[info] quick synthetic smoke (50 steps)" | tee -a "$RUN_DIR/run.info"
CONFIG=configs/train_showcase.yaml PYTHONPATH=. \
  python scripts/train_showcase.py --ddp 0 --save "$RUN_DIR/smoke.pt" 2>&1 | tee "$RUN_DIR/smoke.log" || true

# Real training — prefer 2 GPUs; fall back to 1
CMD=(python scripts/train_showcase.py)
LAUNCH="python"
if [ "$NGPU" -ge 2 ]; then
  LAUNCH="torchrun --nproc_per_node=2"
fi

# Auto-resume: pick latest checkpoint in out_dir, if any
RESUME_FLAG=()
LATEST=$(ls -1t artifacts/m7c_125m/checkpoint_step*.pt 2>/dev/null | head -n1 || true)
if [ -n "$LATEST" ]; then
  echo "[info] found checkpoint: $LATEST" | tee -a "$RUN_DIR/run.info"
  RESUME_FLAG=(--resume "$LATEST")
fi

set -x
CONFIG="$CFG" PYTHONPATH=. $LAUNCH \
  scripts/train_showcase.py --dataset fineweb_edu "${RESUME_FLAG[@]}" 2>&1 | tee "$RUN_DIR/train.log"
set +x

echo "[done] logs at $RUN_DIR; training.csv under artifacts/m7c_125m/" | tee -a "$RUN_DIR/run.info"
