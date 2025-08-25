#!/usr/bin/env bash
set -euo pipefail

# End-to-end backward scaling + backend sweep + summary

OUT_ROOT=${OUT_ROOT:-artifacts/nsa_suite}
TS=$(date +%Y%m%d_%H%M%S)

base() {
  python -u scripts/nsa_backward_repro.py --dim 768 --heads 12 --groups 12 \
    --d-k 64 --d-v 64 --l 16 --d 16 --l-sel 64 --n-sel 16 --w 512 \
    --layers "$1" --seq-len "$2" --out-dir "$3" --tag "$4"
}

echo "Running scaling set..."
base 5 128  "$OUT_ROOT" "S128_${TS}"
base 5 512  "$OUT_ROOT" "S512_${TS}"
base 5 1024 "$OUT_ROOT" "S1024_${TS}" || true
base 5 2048 "$OUT_ROOT" "S2048_${TS}" || true

echo "Selection backend sweep (S=2048)..."
NSA_FORCE_BRANCH=sel python -u scripts/nsa_backward_repro.py --branch sel --sel masked --layers 5 --seq-len 2048 --dim 768 --heads 12 --groups 12 --d-k 64 --d-v 64 --l 16 --d 16 --l-sel 64 --n-sel 16 --w 512 --out-dir "$OUT_ROOT" --tag sel_masked_${TS} --profile || true
NSA_FORCE_BRANCH=sel python -u scripts/nsa_backward_repro.py --branch sel --sel packed --layers 5 --seq-len 2048 --dim 768 --heads 12 --groups 12 --d-k 64 --d-v 64 --l 16 --d 16 --l-sel 64 --n-sel 16 --w 512 --out-dir "$OUT_ROOT" --tag sel_packed_${TS} --profile || true
NSA_FORCE_BRANCH=sel python -u scripts/nsa_backward_repro.py --branch sel --sel gather --layers 5 --seq-len 2048 --dim 768 --heads 12 --groups 12 --d-k 64 --d-v 64 --l 16 --d 16 --l-sel 64 --n-sel 16 --w 512 --out-dir "$OUT_ROOT" --tag sel_gather_${TS} --profile || true

echo "Allocator sensitivity..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 \
  NSA_FORCE_BRANCH=sel python -u scripts/nsa_backward_repro.py --branch sel --sel masked --layers 5 --seq-len 2048 --dim 768 --heads 12 --groups 12 --d-k 64 --d-v 64 --l 16 --d 16 --l-sel 64 --n-sel 16 --w 512 --out-dir "$OUT_ROOT" --tag alloc256_${TS} || true

echo "Summarizing..."
python scripts/summarize_backward_runs.py "$OUT_ROOT" --glob "**/run_*" --save-json "$OUT_ROOT/summary_${TS}.json"
echo "Done. See $OUT_ROOT and summary JSON."

