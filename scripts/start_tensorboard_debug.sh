#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST, e.g. user@host}"
SSH_KEY_PATH="${SSH_KEY_PATH:-}"
TB_PORT="${TB_PORT:-6006}"
SSH_OPTS="${SSH_OPTS:- -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4}"
REMOTE_CMD_PREFIX="${REMOTE_CMD_PREFIX:-cd nsa-vibe &&}"
LOGDIR="${TB_LOGDIR:-artifacts/m7c_125m/tb}"

[[ ${TB_PORT} -ge 1024 && ${TB_PORT} -le 65535 ]] || { echo "Invalid TB_PORT" >&2; exit 2; }
command -v ssh >/dev/null || { echo "ssh not found" >&2; exit 1; }

echo "🔗 Starting TensorBoard connection..."
echo "📡 Connecting to ${REMOTE_HOST}..."

SSH=(ssh $SSH_OPTS)
if [[ -n "${SSH_KEY_PATH}" ]]; then SSH+=(-i "${SSH_KEY_PATH}"); fi
SSH+=("${REMOTE_HOST}")

if "${SSH[@]}" "echo 'SSH connection successful'" >/dev/null 2>&1; then
  echo "✅ SSH connection working"
else
  echo "❌ SSH connection failed"; exit 1
fi

echo "📊 Checking TensorBoard data..."
"${SSH[@]}" "${REMOTE_CMD_PREFIX} ls -la ${LOGDIR} 2>/dev/null || echo 'No ${LOGDIR} directory yet'"

echo "🚀 Starting TensorBoard tunnel on port ${TB_PORT}..."
echo "📈 After this starts, open: http://localhost:${TB_PORT}"
echo "🔄 Keep this terminal open - TensorBoard will run here"
echo ""

"${SSH[@]}" -L "${TB_PORT}:localhost:${TB_PORT}" "${REMOTE_HOST}" \
  "${REMOTE_CMD_PREFIX} . .venv/bin/activate && echo '🎯 TensorBoard starting on remote...' && tensorboard --logdir ${LOGDIR} --port ${TB_PORT} --host 0.0.0.0 --reload_interval 5"
