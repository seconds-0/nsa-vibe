#!/usr/bin/env bash
set -euo pipefail

# Auto-tune FA-2 thresholds by running bench_fa2 and applying results to configs/base.yaml.
#
# Usage:
#   bash scripts/automation/fa2_autotune.sh [--config configs/base.yaml] [--out artifacts/fa2_autotune]
#
# Requirements: CUDA GPU with flash-attn installed and Torch GPU build.

CONFIG=${1:-configs/base.yaml}
OUT_DIR=${2:-artifacts/fa2_autotune}

mkdir -p "${OUT_DIR}"

echo "[fa2-autotune] Running FA-2 benchmarks..." >&2
PYTHONPATH=. python bench/bench_fa2.py --mode win --heads 8 --dk 64 --dv 64 --S_list 256,512,1024 --w 256 --iters 50 --device auto | tee "${OUT_DIR}/fa2_bench_win.txt" || true
PYTHONPATH=. python bench/bench_fa2.py --mode cmp --heads 8 --dk 64 --dv 64 --S_list 256,512,1024 --l 32 --d 16 --iters 50 --device auto | tee "${OUT_DIR}/fa2_bench_cmp.txt" || true

cat "${OUT_DIR}/fa2_bench_win.txt" "${OUT_DIR}/fa2_bench_cmp.txt" > "${OUT_DIR}/fa2_bench.txt"

echo "[fa2-autotune] Applying thresholds to ${CONFIG}..." >&2
PYTHONPATH=. python bench/threshold_optimizer.py "${OUT_DIR}/fa2_bench.txt" --config "${CONFIG}" --report "${OUT_DIR}/fa2_thresholds.md" || true

echo "[fa2-autotune] Done. Outputs:"
echo "  - ${OUT_DIR}/fa2_bench.txt"
echo "  - ${OUT_DIR}/fa2_thresholds.md"
echo "  - Updated: ${CONFIG}"

