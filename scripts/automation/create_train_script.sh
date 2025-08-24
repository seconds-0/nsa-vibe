#!/bin/bash
# Generate the remote training setup script

# M8: Use environment variables for parameterization
REPO_URL="${REPO_URL:-https://github.com/seconds-0/nsa-vibe.git}"
BRANCH_NAME="${BRANCH_NAME:-main}"
TMUX_SESSION="${TMUX_SESSION:-nsa-training}"

mkdir -p scripts/automation

cat > scripts/automation/remote_train_setup.sh << EOF
#!/bin/bash
set -euo pipefail
echo "🔧 Setting up NSA training environment..."
cd ~
sudo apt-get update -qq && sudo apt-get install -y -qq git python3-pip python3-venv ninja-build tmux
if [ ! -d nsa-vibe ]; then
  git clone ${REPO_URL}
fi
cd nsa-vibe
git fetch origin && git checkout ${BRANCH_NAME} && git pull origin ${BRANCH_NAME}
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
tmux kill-session -t ${TMUX_SESSION} 2>/dev/null || true
tmux new-session -d -s ${TMUX_SESSION}
tmux send-keys -t ${TMUX_SESSION} "cd ~/nsa-vibe" Enter
tmux send-keys -t ${TMUX_SESSION} "bash scripts/train_m7c_prime.sh" Enter
echo "✅ Training started in tmux session '${TMUX_SESSION}'"
echo "📊 To start TensorBoard, run on your laptop:"
echo "   REMOTE_HOST=\$YOUR_HOST make monitor"
EOF

chmod +x scripts/automation/remote_train_setup.sh