#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Routing snapshot" >&2
python scripts/print_routing.py || true

echo "[2/3] CPU test sanity (skips GPU)" >&2
PYTHONPATH=. pytest -q || true

echo "[3/3] Decode bench (tiny, CPU/GPU)" >&2
PYTHONPATH=. python bench/bench_decode.py --B 1 --dim 32 --heads 2 --groups 1 --dk 16 --dv 16 \
  --l 8 --d 4 --l_sel 8 --n_sel 4 --w 16 --S_list 8 --iters 3 --warmup 1 \
  --csv decode_test.csv --branch_force_mode env

echo "Summary:" >&2
python bench/summarize_decode_csv.py decode_test.csv || true

echo "Done." >&2

