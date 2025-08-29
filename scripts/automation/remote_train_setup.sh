#!/bin/bash
set -euo pipefail
echo "🔧 Setting up NSA training environment..."
cd ~
sudo apt-get update -qq && sudo apt-get install -y -qq git python3-pip python3-venv ninja-build tmux
if [ ! -d nsa-vibe ]; then
  git clone https://github.com/seconds-0/nsa-vibe.git
fi
cd nsa-vibe
git fetch origin && git checkout main && git pull origin main
### M8: Prefer uv for reproducible setup (fallback to pip/venv if uv unavailable)
if ! command -v uv >/dev/null 2>&1; then
  echo "➡️  Installing uv (user)"
  python3 -m pip install -U uv || true
fi

if command -v uv >/dev/null 2>&1; then
  if [ ! -d .venv ]; then
    echo "🔧 Creating Python environment with uv..."
    uv venv -p 3.11 .venv
    echo "🐍 Environment created successfully"
  fi
  # Install GPU stack via constraints file when present
  if [ -f requirements-gpu-cu121-torch24.txt ]; then
    uv pip sync -r requirements-gpu-cu121-torch24.txt
  fi
  # Basic environment validation
  echo "🔍 Validating environment..."
  PYTHONPATH=. uv run python scripts/_env_guard.py || echo "⚠️  Environment validation failed"
  echo "🧪 Testing data loader..."
  PYTHONPATH=. uv run python scripts/datasets/check_fwe_stream.py
else
  # Fallback to plain venv/pip
  if [ ! -d .venv ]; then
    echo "🔧 Creating Python environment (venv/pip)..."
    python3 -m venv .venv
    . .venv/bin/activate
    pip install -U pip wheel setuptools
    if [ -f requirements-gpu-cu121-torch24.txt ]; then
      pip install -r requirements-gpu-cu121-torch24.txt
    fi
  fi
  echo "🔍 Validating environment..."
  PYTHONPATH=. .venv/bin/python scripts/_env_guard.py || echo "⚠️  Environment validation failed"
  echo "🧪 Testing data loader..."
  PYTHONPATH=. .venv/bin/python scripts/datasets/check_fwe_stream.py
fi
echo "🎯 Starting training in tmux session..."
tmux kill-session -t nsa-training 2>/dev/null || true
tmux new-session -d -s nsa-training
tmux send-keys -t nsa-training "cd ~/nsa-vibe" Enter
tmux send-keys -t nsa-training "bash scripts/train_m7c_prime.sh" Enter
echo "✅ Training started in tmux session 'nsa-training'"
echo "📊 To start TensorBoard, run on your laptop:"
echo "   REMOTE_HOST=$YOUR_HOST make monitor"
