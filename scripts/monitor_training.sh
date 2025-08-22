#!/bin/bash
# Training monitoring script with alerts

REMOTE_HOST="ubuntu@216.81.248.82"
SSH_KEY="~/.ssh/primeintellect_ed25519"

echo "🔍 M7C Training Monitor - $(date)"
echo "================================="

# Check if training process is running
TRAINING_PID=$(ssh -i $SSH_KEY $REMOTE_HOST "ps aux | grep 'python scripts/train_showcase.py.*fineweb_edu' | grep -v grep | awk '{print \$2}'")

if [ -z "$TRAINING_PID" ]; then
    echo "🚨 ALERT: No training process found!"
    echo "❌ Training may have crashed or stopped"
    exit 1
else
    echo "✅ Training process running (PID: $TRAINING_PID)"
fi

# Check GPU memory usage
echo "📊 GPU Status:"
ssh -i $SSH_KEY $REMOTE_HOST "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits" | while IFS=, read gpu name mem_used mem_total util temp; do
    echo "  GPU $gpu: ${util}% util, ${mem_used}/${mem_total}MB, ${temp}°C"
    if [ "$util" -lt 50 ]; then
        echo "  ⚠️  WARNING: Low GPU utilization on GPU $gpu"
    fi
    if [ "$temp" -gt 80 ]; then
        echo "  🔥 WARNING: High temperature on GPU $gpu"
    fi
done

# Check training log for recent activity
echo ""
echo "📝 Recent Training Output:"
ssh -i $SSH_KEY $REMOTE_HOST "cd nsa-vibe && tail -5 live_training.log 2>/dev/null || echo 'No recent logs'"

# Check TensorBoard data
echo ""
echo "📊 TensorBoard Status:"
TB_FILES=$(ssh -i $SSH_KEY $REMOTE_HOST "ls ~/nsa-vibe/artifacts/m7c_125m/tb/*.tfevents.* 2>/dev/null | wc -l")
if [ "$TB_FILES" -gt 0 ]; then
    echo "  ✅ $TB_FILES TensorBoard event files found"
    LATEST_SIZE=$(ssh -i $SSH_KEY $REMOTE_HOST "ls -la ~/nsa-vibe/artifacts/m7c_125m/tb/*.tfevents.* | tail -1 | awk '{print \$5}'")
    echo "  📈 Latest file size: ${LATEST_SIZE} bytes"
    if [ "$LATEST_SIZE" -lt 100 ]; then
        echo "  ⚠️  WARNING: TensorBoard file very small - training may not be logging yet"
    fi
else
    echo "  ⚠️  WARNING: No TensorBoard files found"
fi

echo ""
echo "🔄 Monitor complete. Run again to check status."