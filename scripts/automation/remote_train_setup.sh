#!/bin/bash
set -euo pipefail
echo "🔧 Setting up M7C training environment..."
cd ~
sudo apt-get update -qq && sudo apt-get install -y -qq git python3-pip python3-venv ninja-build tmux
if [ ! -d nsa-vibe ]; then
  git clone https://github.com/seconds-0/nsa-vibe.git
fi
cd nsa-vibe
git fetch origin && git checkout test-plan/m7-training-readiness && git pull origin test-plan/m7-training-readiness
if [ ! -d .venv ]; then
  bash scripts/prime_bootstrap.sh
fi
echo "🧪 Testing data loader..."
PYTHONPATH=. .venv/bin/python scripts/datasets/check_fwe_stream.py
echo "🎯 Starting training in tmux session..."
tmux kill-session -t m7c 2>/dev/null || true
tmux new-session -d -s m7c
tmux send-keys -t m7c "cd ~/nsa-vibe" Enter
tmux send-keys -t m7c "bash scripts/train_m7c_prime.sh" Enter
echo "✅ Training started in tmux session 'm7c'"
echo "📊 To start TensorBoard, run on your laptop:"
echo "   make monitor-prime"
