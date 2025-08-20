# M4-02 — Triton Selection Revival: Execution Playbook (RTX 4090)

Status: Paused per ADR-2025-08-M4-02 (non-viable on RTX 4090). Keep for archival/reference only.

A concise, step-by-step plan for the agent executing M4-02 on the Prime Intellect RTX 4090 pod. Keep production safe via fallbacks and thresholds while iterating.

## Objective
- Validate Triton selection kernels (dense + varlen) for stability and parity.
- Benchmark against packed SDPA and compute a safe `sel_triton_min_L` for 4090.
- If failures/underperformance occur, apply targeted fixes per M4-02 and re‑bench.

## Guardrails
- Default path is packed SDPA; Triton is opt‑in via `NSA_USE_TRITON_SEL=1` and threshold gate.
- Always keep try/except fallbacks intact. Do not weaken causality or GQA invariants.
- Numerics target (fp16/bf16): MAE ≤ 1e-3 vs packed SDPA on test shapes.
- Only change code under `nsa/kernels/triton_sel_kernel/*` and config thresholds unless explicitly asked.

## Prerequisites
- SSH reachable: `ssh root@47.47.180.127 -p 12181` (see `CLAUDE.md` for key config).
- Repo synced on the pod at `/root/nsa-vibe`.
- Python venv with CUDA PyTorch 12.1 and Triton per the setup below.

## Phase 0 — Environment Setup (One-Time)
```bash
cd /root
apt-get update && apt-get install -y python3-venv python3-pip ninja-build git
git clone https://github.com/seconds-0/nsa-vibe.git || true
cd /root/nsa-vibe
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.3.*" torchvision torchaudio
pip install -r requirements.txt || true
pip install "flash-attn==2.*" --no-build-isolation || true
python - << 'PY'
import torch
print('CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## Phase 1 — Parity Smoke Tests (GPU)
Run with group kernels and low threshold to exercise Triton paths.
```bash
# Dense single-span parity
NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_GROUP=1 NSA_SEL_TRITON_MIN_L=64 \
PYTHONPATH=. .venv/bin/python -m pytest -q -k triton_sel_parity

# Varlen multi-span parity
NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_GROUP=1 NSA_SEL_TRITON_MIN_L=64 NSA_TEST_TRITON_SEL=1 \
PYTHONPATH=. .venv/bin/python -m pytest -q -k triton_sel_parity_gpu
```
Pass if: compiles without MLIR errors and MAE ≤ 1e‑3.

## Phase 2 — Microbenchmarks + Threshold
Run both distributions and compute threshold.
```bash
# Dense (few spans)
NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_GROUP=1 NSA_SEL_TRITON_MIN_L=64 \
PYTHONPATH=. .venv/bin/python bench/bench_sel_triton.py \
  --N 1024 --H 8 --D 128 --Dv 128 \
  --L_list 64,128,256,512,1024 \
  --dist few --iters 50 --warmup 5 --streams 1 --csv sel_dense.csv

# Varlen (many spans, n=8)
NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_GROUP=1 NSA_SEL_TRITON_MIN_L=64 \
PYTHONPATH=. .venv/bin/python bench/bench_sel_triton.py \
  --N 1024 --H 8 --D 128 --Dv 128 \
  --L_list 128,256,512,1024 \
  --dist many --iters 50 --warmup 5 --streams 2 --csv sel_varlen.csv

# Compute recommended threshold at 1.2x safety margin
PYTHONPATH=. .venv/bin/python bench/sel_threshold_from_csv.py \
  --dense sel_dense.csv --varlen sel_varlen.csv --margin 1.2 --out selection_report.md
```
Decision:
- If speedup ≥ 1.2x exists, update `configs/base.yaml: runtime.sel_triton_min_L` to the minimal L meeting it.
- Otherwise leave high (e.g., 2048) to keep Triton effectively off on 4090.

## Phase 3 — Diagnostics (If Failures or Instability)
- Enable logs: `NSA_DEBUG_TIMING=1 NSA_DEBUG_COMPARE=1` to record path, bucket timings, and MAE.
- Common issues and remedies:
  - MLIR/broadcast shape issues in varlen: ensure boolean tile masks only; avoid pointer-shaped broadcast.
  - Large MAE at L ≥ 128: verify two-pass FP32 LSE and scaling; check `inv_sqrt_d` use; ensure no NaNs.
  - Underperformance: confirm group-centric path (`NSA_SEL_TRITON_GROUP=1`), review autotune ranges (`num_warps/stages`), and reduce read-modify-write on O.

## Phase 4 — Targeted Fixes (M4-02 Scope)
Primary files: `nsa/kernels/triton_sel_kernel/sel_fwd.py`, wrapper `__init__.py`.
- Numerics: tighten two-pass softmax; ensure accumulators are FP32; clamp lengths and indices defensively.
- Performance: eliminate read-modify-write on O by accumulating in registers per tile; add double-buffering for K/V tiles behind `NSA_SEL_TRITON_DB=1`.
- Autotune: expand configs for 4090 (e.g., `{warps: 4,8}, {stages: 2,3}`) keyed on D/Dv.
- Varlen: re-check cu_seqlens bounds and L estimation for tiling; keep masks as tile-shaped booleans.

Re-run Phase 1 and 2 after changes.

## Phase 5 — Update Thresholds and Land
- If results show stable ≥1.2x region, update `configs/base.yaml: runtime.sel_triton_min_L` and open a PR with CSVs and `selection_report.md` attached.
- If not, keep Triton off for 4090 and document the rationale in `Documentation/M4-RTX-4090-Triton-Selection-Benchmark-Report.md`.

## Deliverables
- `sel_dense.csv`, `sel_varlen.csv`, and `selection_report.md` (threshold decision).
- Short summary of parity (max MAE) and any MLIR/runtime issues encountered.
- If thresholds changed, the config diff for `configs/base.yaml`.

## References
- SSH/setup/bench guide: `Documentation/Guides/Selection-Triton-Bench-4090.md`
- Kernels and wrapper: `nsa/kernels/triton_sel_kernel/sel_fwd.py`, `nsa/kernels/triton_sel_kernel/__init__.py`
- Fallbacks/reference paths: `nsa/core/attention_kernels.py`
