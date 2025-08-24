#!/usr/bin/env bash
set -euo pipefail

# Parameterized training monitor for remote host. See .env.example for vars.
REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST, e.g. user@host}"
SSH_KEY_PATH="${SSH_KEY_PATH:-}"
SSH_OPTS="${SSH_OPTS:- -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4}"
REMOTE_CMD_PREFIX="${REMOTE_CMD_PREFIX:-cd nsa-vibe &&}"

if ! command -v ssh >/dev/null; then echo "ssh not found" >&2; exit 1; fi

echo "🔍 Training Monitor - $(date)"
echo "=============================="

SSH=(ssh $SSH_OPTS)
if [[ -n "${SSH_KEY_PATH}" ]]; then SSH+=(-i "${SSH_KEY_PATH}"); fi
SSH+=("${REMOTE_HOST}")

# Check if training process is running
TRAINING_PID=$("${SSH[@]}" "ps aux | grep -E 'python .*train_showcase\.py.*(fineweb|fineweb_edu)' | grep -v grep | awk '{print \$2}'" || true)
if [[ -z "${TRAINING_PID}" ]]; then
  echo "🚨 ALERT: No training process found!"; echo "❌ Training may have crashed or stopped"; exit 1
else
  echo "✅ Training process running (PID: $TRAINING_PID)"
fi

echo "📊 GPU Status:"
"${SSH[@]}" "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits" \
| while IFS=, read -r gpu name mem_used mem_total util temp; do
  echo "  GPU $gpu: ${util}% util, ${mem_used}/${mem_total}MB, ${temp}°C"
  [[ "${util}" -lt 50 ]] && echo "  ⚠️  WARNING: Low GPU utilization on GPU $gpu"
  [[ "${temp}" -gt 80 ]] && echo "  🔥 WARNING: High temperature on GPU $gpu"
done

echo ""; echo "📝 Recent Training Output:"
"${SSH[@]}" "${REMOTE_CMD_PREFIX} tail -5 live_training.log 2>/dev/null || echo 'No recent logs'"

echo ""; echo "📊 TensorBoard Status:"
TB_LOGDIR_REMOTE="${TB_LOGDIR:-artifacts/m7c_125m/tb}"
TB_GLOB_REMOTE="${TB_LOGDIR_REMOTE%/}/*.tfevents.*"
TB_FILES=$("${SSH[@]}" "${REMOTE_CMD_PREFIX} ls ${TB_GLOB_REMOTE} 2>/dev/null | wc -l" || echo 0)
if [[ "${TB_FILES}" -gt 0 ]]; then
  echo "  ✅ $TB_FILES TensorBoard event files found in ${TB_LOGDIR_REMOTE}"
  LATEST_SIZE=$("${SSH[@]}" "${REMOTE_CMD_PREFIX} ls -la ${TB_GLOB_REMOTE} | tail -1 | awk '{print \$5}'" || echo 0)
  echo "  📈 Latest file size: ${LATEST_SIZE} bytes"
  if [[ "${LATEST_SIZE}" -lt 100 ]]; then echo "  ⚠️  WARNING: TB file very small - training may not be logging yet"; fi
else
  echo "  ⚠️  WARNING: No TensorBoard files found in ${TB_LOGDIR_REMOTE}"
fi

echo ""; echo "🔄 Monitor complete. Run again to check status."
