#!/bin/bash
set -e

echo "🔗 Starting TensorBoard connection to Prime Intellect..."
echo "📡 Connecting to ubuntu@216.81.248.82..."

# Test SSH connection first
if ssh -i ~/.ssh/primeintellect_ed25519 ubuntu@216.81.248.82 "echo 'SSH connection successful'"; then
    echo "✅ SSH connection working"
else
    echo "❌ SSH connection failed"
    exit 1
fi

# Check if TensorBoard directory exists and has data
echo "📊 Checking TensorBoard data..."
ssh -i ~/.ssh/primeintellect_ed25519 ubuntu@216.81.248.82 "cd nsa-vibe && ls -la artifacts/m7c_125m/tb/ 2>/dev/null || echo 'No m7c_125m/tb directory yet, checking train_showcase...'; ls -la artifacts/train_showcase/tb/ 2>/dev/null || echo 'No data directories found'"

echo "🚀 Starting TensorBoard tunnel on port 6006..."
echo "📈 After this starts, open: http://localhost:6006"
echo "🔄 Keep this terminal open - TensorBoard will run here"
echo ""

# Start the tunnel with TensorBoard
ssh -i ~/.ssh/primeintellect_ed25519 -L 6006:localhost:6006 ubuntu@216.81.248.82 'cd nsa-vibe && . .venv/bin/activate && echo "🎯 TensorBoard starting on remote..." && tensorboard --logdir artifacts/m7c_125m/tb --port 6006 --host 0.0.0.0 --reload_interval 5'