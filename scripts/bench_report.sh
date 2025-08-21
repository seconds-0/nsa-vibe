#!/usr/bin/env bash
set -euo pipefail

# Minimal, standard-looking benchmark summary

echo "NSA Bench Summary"

repo=$(basename "$(pwd)")
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '-')
commit_short=$(git rev-parse --short HEAD 2>/dev/null || echo '-')
commit_msg=$(git show -s --format=%s 2>/dev/null || echo '-')

echo "repo:    ${repo}"
echo "branch:  ${branch}"
echo "commit:  ${commit_short} — ${commit_msg}"

python - <<'PY'
import torch, sys
print(f"python:  {sys.version.split()[0]}")
print(f"torch:   {torch.__version__}")
print(f"cuda:    {bool(torch.cuda.is_available())}")
if torch.cuda.is_available():
    print(f"device:  {torch.cuda.get_device_name(0)}")
PY

echo
echo "tests:"
set +e
PYTHONPATH=. pytest -q nsa/tests/test_equiv_small.py >/dev/null 2>&1
echo "  small_s_equiv:        $([[ $? -eq 0 ]] && echo PASS || echo FAIL)"
PYTHONPATH=. pytest -q nsa/tests/test_group_consistency.py >/dev/null 2>&1
echo "  group_consistency:    $([[ $? -eq 0 ]] && echo PASS || echo FAIL)"
PYTHONPATH=. pytest -q nsa/tests/test_decode_counters.py >/dev/null 2>&1
echo "  decode_counters:      $([[ $? -eq 0 ]] && echo PASS || echo FAIL)"
PYTHONPATH=. pytest -q nsa/tests/test_masks.py >/dev/null 2>&1
echo "  causal_masks:         $([[ $? -eq 0 ]] && echo PASS || echo FAIL)"

# Needle tests with simple timing
function run_timed() {
  local label=$1; shift
  local start end dur rc
  start=$(python - <<'PY'
import time; print(time.time())
PY
)
  "$@" >/dev/null 2>&1
  rc=$?
  end=$(python - <<'PY'
import time; print(time.time())
PY
)
  # Use safe calculation with argument passing
  dur=$(python3 -c "
import sys
start_time = float('$start')
end_time = float('$end')
print(f'{end_time - start_time:.2f}s')
")
  if [[ $rc -eq 0 ]]; then echo "  ${label}: PASS (${dur})"; else echo "  ${label}: FAIL"; fi
}

run_timed "needle_cpu_4k" bash -lc "PYTHONPATH=. pytest -q nsa/tests/test_long_context_needle.py::test_selection_mapping_includes_needle_cpu_small"

CUDA_AVAIL=$(python - <<'PY'
import torch; print(int(torch.cuda.is_available()))
PY
)
if echo "$CUDA_AVAIL" | grep -q 1; then
  run_timed "needle_cuda_64k" bash -lc "PYTHONPATH=. pytest -q nsa/tests/test_long_context_needle.py::test_selection_mapping_includes_needle_64k_cuda"
else
  echo "  needle_cuda_64k: SKIP (no CUDA)"
fi
set -e

echo
echo "decode_bench:"
TMP=$(mktemp /tmp/nsa_bench.XXXXXX)
CSV="${TMP}.csv"

# Set up cleanup trap
cleanup() {
    rm -f "$TMP" "$CSV"
}
trap cleanup EXIT

PYTHONPATH=. python bench/bench_decode.py \
  --B 1 --dim 32 --heads 2 --groups 1 --dk 16 --dv 16 \
  --l 8 --d 4 --l_sel 8 --n_sel 4 --w 16 \
  --S_list 8,16 --iters 4 --warmup 1 \
  --csv "$CSV"
python scripts/summarize_bench.py "$CSV"
echo "csv_head:"
head -n 2 "$CSV"

if [ -f artifacts/tracked/artifacts-accuracy-224091b/decode_summary.txt ]; then
  echo
  echo "artifacts:"
  sed -n '1,4p' artifacts/tracked/artifacts-accuracy-224091b/decode_summary.txt || true
elif [ -f artifacts-accuracy-224091b/decode_summary.txt ]; then
  echo
  echo "artifacts:"
  sed -n '1,4p' artifacts-accuracy-224091b/decode_summary.txt || true
fi
