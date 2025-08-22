#!/bin/bash
# Generate the remote setup-only script

mkdir -p scripts/automation

cat > scripts/automation/remote_setup_only.sh << 'EOF'
#!/bin/bash
set -euo pipefail
echo "🔧 Setting up environment only..."
cd ~
sudo apt-get update -qq && sudo apt-get install -y -qq git python3-pip python3-venv ninja-build tmux
if [ ! -d nsa-vibe ]; then
  git clone https://github.com/seconds-0/nsa-vibe.git
fi
cd nsa-vibe
git fetch origin && git checkout test-plan/m7-training-readiness && git pull origin test-plan/m7-training-readiness
bash scripts/prime_bootstrap.sh
echo "✅ Environment ready. To start training: make train-prime"
EOF

chmod +x scripts/automation/remote_setup_only.sh