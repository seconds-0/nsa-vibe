# Decode Benchmark Guide (RTX 4090 or similar)

This guide runs the end-to-end decode microbenchmark, captures per-branch timings (compressed/selection/window), and summarizes results to decide if we should invest in a custom selection kernel.

## Prerequisites
- Repo at `/root/nsa-vibe` and Python venv activated (`source .venv/bin/activate`).
- CUDA GPU (4090 or similar). CPU will run but won’t be meaningful for perf.
- Optional: `uv` installed (you can substitute with `python`).

Sanity check:
```bash
python - << 'PY'
import torch
print({
  'cuda': torch.cuda.is_available(),
  'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  'capability': torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
  'torch': torch.__version__,
  'cuda_version': torch.version.cuda,
})
PY
```
Expected: `cuda: True`, valid device + capability (4090 is (8, 9)).

## Default Production Guardrails
- SDPA is used for selection; FA‑2 is disabled on Ada (SM 8.9) by code guard.
- Triton selection is disabled on 4090 by wrapper guard + high `sel_triton_min_L`.
- You do not need to set any env vars for the baseline run.

## Run the Decode Bench (CSV output)
This runs prefill to build caches, then measures `iters` single‑token decode steps and reports ms/token.

- Columns: `S` (context), `ms_total`, `ms_cmp`, `ms_sel`, `ms_win`, `reads_actual`, `reads_expected`.
- The three branch columns are measured by forcing one‑hot gates (cmp‑only, sel‑only, win‑only) to estimate relative cost.

Command (recommended):
```bash
PYTHONPATH=. uv run -q python bench/bench_decode.py \
  --S_list 128,256,512,1024 \
  --iters 32 --warmup 8 \
  --csv decode.csv
```
Console shows a quick table and writes `decode.csv`.

Example console:
```
Context    Total(ms)   cmp(ms)   sel(ms)   win(ms)   Reads
128        1.23        0.41      0.55      0.27      704/704
256        2.86        0.92      1.38      0.56      960/960
...
```

## Summarize Results
Print per‑context breakdown with percentages for each branch:
```bash
PYTHONPATH=. uv run -q python bench/summarize_decode_csv.py decode.csv
```
Sample output:
```
     S     total    cmp%    sel%    win%  reads
   128     1.23    33.1    44.7    22.2  704/704
   256     2.86    32.2    48.3    19.5  960/960
```

## How to Interpret
- Focus on `sel%` (selection share). If selection ≥ 25–30% of total at target contexts, a custom CUDA selection kernel can deliver meaningful end‑to‑end gains.
- If `sel%` is low (e.g., < 15–20%), the system is dominated by other branches; prioritize other optimizations.
- `reads_actual/reads_expected` should match exactly; if they diverge, report it (causality or schedule issue).

## Optional: Experiment Flags (use with care)
- Force CUDA selection route (still behind safe fallback):
  - Build and route: `NSA_SEL_CUDA_BUILD=1 NSA_SEL_CUDA=1` (wrapper will fall back safely on error)
- Force FA‑2 (not recommended on 4090):
  - `NSA_FA2_FORCE=1` (code guard is bypassed; for experiments only)
- Force Triton (not recommended on 4090):
  - `NSA_TRITON_SEL_FORCE=1 NSA_USE_TRITON_SEL=1` (wrapper still may fall back)

Example (CUDA selection experiment):
```bash
NSA_SEL_CUDA_BUILD=1 NSA_SEL_CUDA=1 \
PYTHONPATH=. uv run -q python bench/bench_decode.py --S_list 128,256,512,1024 --iters 32 --warmup 8 --csv decode_cuda.csv
```

## Deliverables to Share
- `decode.csv` (raw timings)
- One‑line summary for each S: `total ms`, and branch percentages
- Any anomalies (mismatched reads, large variance across runs)

## Acceptance Gate for M4 Kernel Work
- Proceed with custom CUDA selection kernel (decode S=1) only if `sel%` ≥ 25–30% at target contexts.
- Target for kernel acceptance: MAE ≤ 1e‑3 vs packed SDPA; ≥ 1.2× speedup on 4090 decode shapes (l′=64, n∈{8,16}, D/Dv∈{64,128}, H∈{4,8}, N∈{256,1024}).

*** End Guide ***
