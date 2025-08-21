# M4 — Selection Acceleration (Consolidated Plan)

Status: Paused on RTX 4090 per ADR‑2025‑08‑M4‑02; SDPA remains production. This consolidates M4 Selection Triton plans, execution playbooks, CUDA spike, and benchmark plan into a single reference aligned to PRD.md.

## Scope & Goals
- Implement a Triton forward kernel for selection with group‑centric schedule (Figure 3): load Q once per (B,G), reuse K/V across heads; inner loop over selected KV blocks.
- Numerics match SDPA gather (FP32 tol ≤ 1e‑4) and reduce HBM reads for large L.
- Robust fallbacks to packed SDPA; device/dtype/shape guards.

## Constraints & Policy (ADR)
- RTX 4090 (SM 8.9): Triton selection is disabled by default; wrapper enforces fallback unless `NSA_TRITON_SEL_FORCE=1`.
- Threshold gating: prefer Triton only when total selected length per row ≥ `NSA_SEL_TRITON_MIN_L` (default 4096).
- Dtypes/head‑dim: BF16/FP16 with FP32 accum only; align dims to multiples of 8/16; else fallback.

## Deliverables
- Wrapper: `nsa/kernels/triton_sel_kernel/__init__.py` with dense/varlen packing, bucketed execution, logging, and strict fallback ladder.
- Group‑centric variants (opt‑in) for better KV reuse; timing/reads logs (`sel.triton.*`).
- Benchmarks + parity tests for supported GPUs.

## Acceptance
- Parity: FP32 MAE ≤ 1e‑4 vs SDPA gather across dense/varlen, including empty/single/multi‑span and mixed‑L buckets.
- Reduced reads: for high‑L selections on supported GPUs, Triton logs `sel.triton.reads` with total selected tokens ≤ packed SDPA’s total (validated in benches/tests).
- Fallbacks: safe and explicit (dtype/alignment/threshold/ADR); CPU path uses SDPA; SM 8.9 (RTX 4090) disabled by ADR unless forced.

## Tests & Benches
- Parity: `nsa/tests/test_triton_sel_parity*.py`, `test_triton_sel_edge_cases.py`.
- CPU wrapper parity/backward fallbacks: `test_triton_sel_wrapper_cpu.py`, `test_triton_sel_autograd_cpu.py`.
- GPU runs on supported devices only.
- Benches: `bench/bench_sel_triton.py` and decode benches; config threshold tuning via `bench/threshold_optimizer.py`.

## Config & Flags
- Enable: `NSA_USE_TRITON_SEL=1`; force on SM 8.9: `NSA_TRITON_SEL_FORCE=1` (experiments only).
- Group kernels: `NSA_SEL_TRITON_GROUP=1`.
- Threshold: `NSA_SEL_TRITON_MIN_L` (plumb from `configs/base.yaml` runtime.sel_triton_min_L).
- Debug: `NSA_DEBUG_TRITON=1` or `NSA_DEBUG_LOG=1` to enable wrapper logs (`sel.triton.*`).

## Outcome
On RTX 4090, SDPA packed remains superior; Triton is disabled (ADR‑2025‑08‑M4‑02). On A100/H100, revisit with the same acceptance gates and thresholds.

## Usage Matrix
- RTX 4090 (SM 8.9): Triton disabled by default (ADR). Use SDPA packed. For experiments only, force via `NSA_TRITON_SEL_FORCE=1` and expect fallback.
- A100/H100: Optional. Enable with `NSA_USE_TRITON_SEL=1`; set `sel_triton_min_L` via benches; use `NSA_SEL_TRITON_GROUP=1` optionally.
- CPU: Always SDPA packed; Triton never imported.

## CI
- A100/H100 lane: weekly or on-demand smoke (parity + fallbacks). See `.github/workflows/triton-gpu-smoke.yml`.
- SM 8.9 lane: assert ADR fallback; no Triton execution.
