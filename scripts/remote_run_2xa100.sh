#!/usr/bin/env bash

# Remote 2×A100 production launcher via SSH.
# - Reads connection/env from an optional env file (e.g., .env.a100)
# - Kills stale jobs, validates GPUs, sets up venv, runs preflight, then launches
# - Mirrors scripts/run_2xa100_production.sh behavior on the remote

set -euo pipefail

ENV_FILE=""
DRY_RUN=0
SYNC_LOCAL=${SYNC_LOCAL:-0}        # If 1, rsync the local repo to remote
REMOTE_DIR_DEFAULT="~/nsa-vibe"

usage() {
  cat <<USAGE
Usage: $0 [--env-file PATH] [--dry-run]

Required (can be provided via env or env-file):
  REMOTE_HOST          e.g., user@a100-host
  SSH_KEY_PATH         e.g., ~/.ssh/id_ed25519 (optional if agent loaded)

Optional env (with defaults):
  SSH_OPTS             "-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
  REMOTE_DIR           "${REMOTE_DIR_DEFAULT}"
  SYNC_LOCAL           0 (if 1, rsync the current repo to REMOTE_DIR)

Examples:
  $0 --env-file .env.a100
  SYNC_LOCAL=1 $0 --env-file .env.a100
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

if [[ -n "$ENV_FILE" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC2046
    export $(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE" | cut -d= -f1) >/dev/null 2>&1 || true
    # shellcheck disable=SC1090
    . "$ENV_FILE"
  else
    echo "Env file not found: $ENV_FILE" >&2; exit 1
  fi
fi

REMOTE_HOST=${REMOTE_HOST:-}
SSH_KEY_PATH=${SSH_KEY_PATH:-}
SSH_OPTS=${SSH_OPTS:-"-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"}
REMOTE_DIR=${REMOTE_DIR:-$REMOTE_DIR_DEFAULT}

if [[ -z "$REMOTE_HOST" ]]; then
  echo "REMOTE_HOST is required (e.g., user@a100-host). Use --env-file or export it." >&2
  exit 1
fi

SSH_BASE=(ssh $SSH_OPTS)
if [[ -n "$SSH_KEY_PATH" ]]; then
  SSH_BASE+=( -i "$SSH_KEY_PATH" )
fi

echo "=== Remote production launch on $REMOTE_HOST ==="
echo "REMOTE_DIR=$REMOTE_DIR SYNC_LOCAL=$SYNC_LOCAL"

# Bootstrap on remote
BOOT_CMD='set -euo pipefail; \
echo "[remote] Hostname: '"'"'"'$(hostname)'"'"'"'"; \
echo "[remote] GPU snapshot:"; nvidia-smi || true; \
REMOTE_DIR='"$REMOTE_DIR"'; mkdir -p "$REMOTE_DIR"; cd "$REMOTE_DIR"; \
if [ ! -d .git ]; then echo "[remote] Git repo not found at $REMOTE_DIR (rsync or git clone before running)"; fi'

if [[ $DRY_RUN -eq 1 ]]; then
  echo "--- DRY RUN: would run SSH bootstrap ---"
  printf '%s\n' "$BOOT_CMD"
  exit 0
fi

"${SSH_BASE[@]}" "$REMOTE_HOST" "bash -lc '$BOOT_CMD'"

# Optional rsync of current repo (ensures exact local state on remote)
if [[ $SYNC_LOCAL -eq 1 ]]; then
  echo "[local] rsyncing repo to $REMOTE_HOST:$REMOTE_DIR"
  rsync -az --delete --exclude '.venv' --exclude 'artifacts' --exclude '.git' ./ "$REMOTE_HOST:$REMOTE_DIR/"
fi

# Run setup + training on remote
RUN_CMD='set -euo pipefail; \
REMOTE_DIR='"$REMOTE_DIR"'; cd "$REMOTE_DIR"; \
echo "[remote] Python/venv setup"; \
python3 -V || true; \
if ! command -v uv >/dev/null 2>&1; then pipx install uv >/dev/null 2>&1 || python3 -m pip install -U uv || true; fi; \
uv venv -p 3.11 .venv; \
. .venv/bin/activate; \
uv pip sync -r requirements-gpu-cu121-torch24.txt; \
echo "[remote] Clean stale jobs and artifacts"; \
pkill -f '"'"scripts/train_showcase.py"'"' || true; pkill -f '"'"torchrun"'"' || true; \
mkdir -p artifacts/m7c_125m_2xa100_prod; rm -f artifacts/m7c_125m_2xa100_prod/training.csv || true; \
echo "[remote] Preflight quick check (CUDA)"; \
python - <<"PY"\nimport torch\nprint('cuda_available=', torch.cuda.is_available(), 'device_count=', torch.cuda.device_count())\nPY\n\
echo "[remote] Launch production (2×A100)"; \
export PYTHONPATH=. CONFIG=configs/m7c_125m_2xa100_production.yaml; \
export NSA_PREFILL_BATCHED=1 NSA_SEL_RANGES_V2=1; \
export NSA_DDP_COMPRESS=${NSA_DDP_COMPRESS:-bf16} NSA_DDP_BUCKET_MB=${NSA_DDP_BUCKET_MB:-25}; \
export NCCL_ALGO=${NCCL_ALGO:-Ring} NCCL_PROTO=${NCCL_PROTO:-Simple} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}; \
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1} NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}; \
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL} NSA_PG_TIMEOUT_MIN=${NSA_PG_TIMEOUT_MIN:-15}; \
export NSA_DDP_DISABLE_GC=${NSA_DDP_DISABLE_GC:-1} NSA_DDP_FIND_UNUSED=${NSA_DDP_FIND_UNUSED:-0} NSA_DDP_STATIC_GRAPH=${NSA_DDP_STATIC_GRAPH:-1}; \
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}; \
set -x; \
torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu --ddp 1 2>&1 | tee artifacts/m7c_125m_2xa100_prod/production_remote.log; \
set +x; \
echo "[remote] Done. See artifacts/m7c_125m_2xa100_prod/production_remote.log"'

"${SSH_BASE[@]}" "$REMOTE_HOST" "bash -lc '$RUN_CMD'"

echo "=== Remote production completed (check remote artifacts/logs) ==="

