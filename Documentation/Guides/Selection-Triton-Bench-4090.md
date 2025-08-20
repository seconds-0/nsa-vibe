# Selection (Triton) Benchmark Instructions — RTX 4090

This guide runs Triton selection parity smokes and microbenchmarks on the Prime Intellect RTX 4090 to determine a safe `sel_triton_min_L` threshold and validate stability.

## Goals
- Verify Triton selection compiles and runs (dense single‑span and varlen multi‑span).
- Validate numerical parity vs packed SDPA (MAE ≤ 1e‑3 in fp16/bf16).
- Benchmark speed vs packed SDPA across realistic NSA shapes and span distributions.
- Propose a data‑driven `runtime.sel_triton_min_L` for the target GPU.

## What’s in this repo (relevant)
- Group‑centric Triton kernels (opt‑in): `nsa/kernels/triton_sel_kernel/sel_fwd.py`
  - Dense: `_sel_attn_fwd_dense_group_kernel` via `sel_attn_fwd_dense_group`
  - Varlen: `_sel_attn_fwd_varlen_group_kernel` via `sel_attn_fwd_varlen_group`
  - Probability reuse across Dv tiles; expanded autotune configs.
- Safe defaults and fallbacks
  - Triton off by default (`sel_triton_min_L` high); try/except fallbacks to packed SDPA on any error.
  - Varlen MLIR mask fix (no pointer‑shape broadcasting).
- Benchmark script: `bench/bench_sel_triton.py` (CSV output, concurrency via `--streams`).
- Tests: `nsa/tests/test_triton_sel_parity*.py`, `nsa/tests/test_triton_sel_edge_cases.py`.

## Branches and Flags
- Branch policy
  - main: current changes live here, behind flags and with safe fallbacks.
  - feat/m4-group-kernel: reserved for riskier perf refactors (double‑buffering, backward); not required to run this bench.
- Runtime/env flags
  - `NSA_USE_TRITON_SEL=1` — enable Triton selection path.
  - `NSA_SEL_TRITON_GROUP=1` — use group‑centric kernels (recommended).
  - `NSA_SEL_TRITON_MIN_L=64` — low threshold for bench only (default is high for safety).
  - Debug (optional): `NSA_DEBUG_TIMING=1`, `NSA_DEBUG_COMPARE=1`, `NSA_DEBUG_SHAPES=1`.

## Prerequisites
- SSH to Prime Intellect 4090
  - Quick: `ssh root@47.47.180.127 -p 12181`
  - With key: `ssh -i ~/.ssh/primeintellect_ed25519 root@47.47.180.127 -p 12181`
- Code sync options
  - Option A (bench EXACT local code): `scp -P 12181 -r . root@47.47.180.127:/root/nsa-vibe`
  - Option B (clone main branch on host): `git clone https://github.com/seconds-0/nsa-vibe.git /root/nsa-vibe`

## One‑Time Environment Setup (on 4090)
Run inside the 4090 shell:

```
bash -lc '
set -euxo pipefail
cd /root/nsa-vibe
apt-get update
apt-get install -y python3-venv python3-pip ninja-build git
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
# CUDA PyTorch 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.3.*" torchvision torchaudio
# Repo deps (triton pinned on Linux in requirements)
pip install -r requirements.txt || true
# Optional: FlashAttention-2 (not required for selection)
pip install "flash-attn==2.*" --no-build-isolation || true
# Device sanity
python - << PY
import torch
print("CUDA:", torch.cuda.is_available(), "Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
'
```

## Parity Smoke Tests (GPU)
Dense single‑span and varlen multi‑span, opt‑in:

```
# Dense
NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_GROUP=1 NSA_SEL_TRITON_MIN_L=64 \
PYTHONPATH=. .venv/bin/python -m pytest -q -k triton_sel_parity || true

# Varlen
NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_GROUP=1 NSA_SEL_TRITON_MIN_L=64 NSA_TEST_TRITON_SEL=1 \
PYTHONPATH=. .venv/bin/python -m pytest -q -k triton_sel_parity_gpu || true
```

Pass criteria: compiles without MLIR errors; MAE ≤ 1e‑3.

## Microbenchmarks (Triton vs Packed SDPA)
Enable group kernels and lower Triton min‑L for bench runs only.

- Dense (few spans; single contiguous range):
```
NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_GROUP=1 NSA_SEL_TRITON_MIN_L=64 \
PYTHONPATH=. .venv/bin/python bench/bench_sel_triton.py \
  --N 1024 --H 8 --D 128 --Dv 128 \
  --L_list 64,128,256,512,1024 \
  --dist few --iters 50 --warmup 5 --streams 1 --csv sel_dense.csv
```

- Varlen (many spans; n=8) with concurrency stress:
```
NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_GROUP=1 NSA_SEL_TRITON_MIN_L=64 \
PYTHONPATH=. .venv/bin/python bench/bench_sel_triton.py \
  --N 1024 --H 8 --D 128 --Dv 128 \
  --L_list 128,256,512,1024 \
  --dist many --iters 50 --warmup 5 --streams 2 --csv sel_varlen.csv
```

- Print results:
```
 tail -n +1 sel_dense.csv || true
 tail -n +1 sel_varlen.csv || true
```

Each CSV row includes: mode, N, H, D, Dv, L, nspans (varlen), streams, tri_ms, ref_ms, speedup, mae.

## How To Interpret Results
- Parity: ensure MAE ≤ 1e‑3 (fp16/bf16).
- Speedup: compute SDPA_time / Triton_time. Use a safety margin ≥ 1.2×.
- Threshold selection (4090): choose minimal L where speedup ≥ 1.2× across tested heads/dims and span distributions. If none meet the margin, keep Triton off by setting a high `sel_triton_min_L`.
- Update (manual): set `runtime.sel_triton_min_L` in `configs/base.yaml` to the chosen L.

## Optional: Automate via Modal (CI)
This repo includes a GitHub workflow and Modal job that benchmarks FA‑2 and selection and proposes thresholds via PR.
- Trigger: GitHub Actions → GPU Benchmark (workflow_dispatch).
- It runs selection benches, parses output, and proposes `sel_triton_min_L` in the PR body.

## Troubleshooting
- MLIR crashes: ensure `NSA_SEL_TRITON_GROUP=1`; our kernels avoid pointer‑shape broadcast masks.
- Fallbacks: any exception falls back to packed SDPA; enable `NSA_DEBUG_TIMING=1` to log path decisions.
- Contiguity: inputs are forced contiguous in wrappers; avoid non‑contiguous tensors at call sites.
- First‑run JIT: expect initial compile latency; bench script includes warmups.

## Deliverables
- Paste the tail of `sel_dense.csv` and `sel_varlen.csv` in your report.
- State the chosen `sel_triton_min_L` for 4090 (or “keep disabled” if no ≥1.2× region).
- Note any errors or anomalies (compile, runtime, large MAE, instability).

## Appendix: Test Matrix (suggested)
- H ∈ {4, 8}; D, Dv ∈ {64, 128}; N ∈ {256, 1024}; L ∈ {64,128,256,512,1024}.
- Dist ∈ {few (dense), many (varlen n=8)}; streams ∈ {1 (baseline), 2 (concurrency)}.
