#!/usr/bin/env bash
set -euo pipefail

echo "================ NSA Brag Report ================"
echo "Repo: $(basename "$(pwd)")"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse --short HEAD) — $(git show -s --format=%s)"
echo "Python: $(python -V 2>&1)"
python - <<'PY'
import torch
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

echo
echo "-- Core correctness (CPU) --"
PYTHONPATH=. pytest -q \
  nsa/tests/test_equiv_small.py \
  nsa/tests/test_group_consistency.py \
  nsa/tests/test_decode_counters.py \
  nsa/tests/test_masks.py \
  || true

echo
echo "-- Long context selection (needle) --"
PYTHONPATH=. pytest -q nsa/tests/test_long_context_needle.py::test_selection_mapping_includes_needle_cpu_small || true
if python - <<'PY'
import torch; print('cuda' if torch.cuda.is_available() else 'cpu')
PY
then
  echo "(CUDA) Running 64k needle test"
  PYTHONPATH=. pytest -q nsa/tests/test_long_context_needle.py::test_selection_mapping_includes_needle_64k_cuda || true
fi

echo
echo "-- Decode bench (reads accuracy) --"
TMPFILE=$(mktemp /tmp/nsa_decode.XXXXXX)
OUT_CSV="${TMPFILE}.csv"
PYTHONPATH=. python bench/bench_decode.py \
  --B 1 --dim 32 --heads 2 --groups 1 --dk 16 --dv 16 \
  --l 8 --d 4 --l_sel 8 --n_sel 4 --w 16 \
  --S_list 8,16 --iters 4 --warmup 1 \
  --csv "$OUT_CSV"
python scripts/summarize_bench.py "$OUT_CSV"
echo "CSV head:"; head -n 3 "$OUT_CSV"

if [ -d artifacts-accuracy-224091b ]; then
  echo
  echo "-- Accuracy artifacts (224091b) --"
  sed -n '1,5p' artifacts-accuracy-224091b/decode_summary.txt || true
  head -n 5 artifacts-accuracy-224091b/decode_gpu_final.csv || true
fi

echo
echo "-- Execution policy (ADR) --"
echo "Triton selection disabled by default on RTX 4090; clean SDPA fallback."
PYTHONPATH=. pytest -q nsa/tests/test_triton_fallback_dtype.py || true

echo "==================== End Report ===================="
