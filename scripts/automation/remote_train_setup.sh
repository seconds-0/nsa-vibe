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
# M8: Use constraints file and env guard
if [ ! -d .venv ]; then
  echo "🔧 Creating Python environment..."
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -U pip wheel setuptools
  # Use constraints file for reproducible builds
  if [ -f requirements-gpu-cu121-torch24.txt ]; then
    pip install -r requirements-gpu-cu121-torch24.txt
  else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install triton packaging ninja
    pip install transformers datasets tensorboard
    pip install numpy hydra-core omegaconf pydantic pytest hypothesis ruff mypy
  fi
  echo "🐍 Environment created successfully"
fi
# M8: Run environment guard
echo "🔍 Validating environment..."
PYTHONPATH=. .venv/bin/python scripts/_env_guard.py || echo "⚠️  Environment validation failed"
echo "🧪 Testing data loader..."
PYTHONPATH=. .venv/bin/python scripts/datasets/check_fwe_stream.py
echo "🎯 Starting training in tmux session..."
tmux kill-session -t nsa-training 2>/dev/null || true
tmux new-session -d -s nsa-training
tmux send-keys -t nsa-training "cd ~/nsa-vibe" Enter
tmux send-keys -t nsa-training "bash scripts/train_m7c_prime.sh" Enter
echo "✅ Training started in tmux session 'nsa-training'"
echo "📊 To start TensorBoard, run on your laptop:"
echo "   REMOTE_HOST=$YOUR_HOST make monitor"
