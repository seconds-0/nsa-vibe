#!/usr/bin/env bash
set -euo pipefail

# Runner one-shot script: captures a full validation sweep and gathers artifacts
# Usage: bash scripts/runner_oneshot.sh [out_dir]

# Sanitize output directory argument
RAW_OUT=${1:-artifacts/runner}
SAFE_RE='^[A-Za-z0-9_./-]+$'
if [[ "$RAW_OUT" =~ $SAFE_RE ]]; then
  OUT="$RAW_OUT"
else
  echo "[oneshot] invalid output dir '$RAW_OUT', defaulting to artifacts/runner" >&2
  OUT="artifacts/runner"
fi
mkdir -p -- "$OUT"

echo "[oneshot] recording git commit" >&2
COMMIT=$(git rev-parse --short HEAD || echo unknown)
RUN_DIR="$OUT/$COMMIT"
mkdir -p -- "$RUN_DIR"

echo "[oneshot] versions and routing" >&2
(
  echo "== nvidia-smi =="; nvidia-smi || true;
  echo;
  echo "== versions ==";
  python - << 'PY'
import sys, torch
print('python', sys.version)
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
try:
  import triton
  print('triton', triton.__version__)
except Exception:
  print('triton', '<none>')
PY
) | tee "$RUN_DIR/env.txt"

PYTHONPATH=. python scripts/print_routing.py > "$RUN_DIR/routing.json" || true

echo "[oneshot] gpu sanity" >&2
PYTHONPATH=. python scripts/gpu_sanity.py | tee "$RUN_DIR/sanity.out" || true

echo "[oneshot] triton forward parity" >&2
NSA_USE_TRITON_SEL=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. \
  pytest -q nsa/tests/test_triton_sel_parity_gpu.py | tee "$RUN_DIR/triton_fwd.txt" || true

echo "[oneshot] triton backward parity" >&2
NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_ALLOW_GRAD=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. \
  pytest -q nsa/tests/test_triton_sel_backward_gpu.py | tee "$RUN_DIR/triton_bwd.txt" || true

echo "[oneshot] decode bench + summary" >&2
PYTHONPATH=. python bench/bench_decode.py --B 1 --dim 256 --heads 8 --groups 2 --dk 32 --dv 32 \
  --l 32 --d 16 --l_sel 64 --n_sel 16 --w 512 --S_list 512,1024,2048,4096 --iters 64 --warmup 8 \
  --csv "$RUN_DIR/decode_gpu_final.csv" --branch_force_mode env | tee "$RUN_DIR/decode_bench.txt" || true
python bench/summarize_decode_csv.py "$RUN_DIR/decode_gpu_final.csv" | tee "$RUN_DIR/decode_summary.txt" || true

echo "[oneshot] FA-2 availability + optional parity" >&2
python - << 'PY' | tee "$RUN_DIR/fa2_probe.txt" || true
from nsa.kernels.flash_wrappers import is_flash_varlen_available
import torch
print('fa2_varlen_available', is_flash_varlen_available())
print('cuda_is_available', torch.cuda.is_available())
PY

if grep -q "fa2_varlen_available True" "$RUN_DIR/fa2_probe.txt"; then
  NSA_TEST_FA2=1 NSA_USE_FA2=1 PYTHONPATH=. \
    pytest -q -k fa2_gpu_varlen | tee "$RUN_DIR/fa2_varlen.txt" || true
  # Optional FA-2 benches (if present) for threshold suggestions
  if [ -f bench/bench_fa2.py ]; then
    PYTHONPATH=. python bench/bench_fa2.py | tee "$RUN_DIR/fa2_bench.txt" || true
  fi
fi

echo "[oneshot] selection ranges sample" >&2
python scripts/print_selection_ranges.py --S 32 --heads 4 --groups 2 --dk 16 --dv 16 \
  --l 8 --d 4 --l_sel 16 --n_sel 4 --w 16 --json | head -n 20 > "$RUN_DIR/sel_ranges.jsonl" || true

echo "[oneshot] training showcase (short)" >&2
CONFIG=configs/train_showcase.yaml python scripts/train_showcase.py | tee "$RUN_DIR/train_showcase.txt" || true

echo "[oneshot] done -> $RUN_DIR" >&2
