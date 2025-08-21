#!/usr/bin/env bash
set -euo pipefail

# Prime Intellect bootstrap: prepare venv, install deps, probe GPUs

log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

log "Creating venv and installing GPU deps..."
python -m venv .venv
. ./.venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements-gpu-cu121-torch24.txt
pip install transformers datasets

log "Probing GPU availability..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  log "nvidia-smi not found — ensure you are on a GPU host"
fi

python - << 'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"gpu[{i}]", p.name, f"{p.total_memory/1024**3:.1f}GB")
PY

log "Bootstrap complete. Activate with: . ./.venv/bin/activate"
