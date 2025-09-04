# 2025-09-04 Core Engineer Report - FA‑2 Stabilization and Triage Plan v1

## Executive Summary

Goal: Make FlashAttention‑2 (FA‑2) the default fast path for NSA compressed and sliding branches on supported GPUs (H100/A100), with strict guards and hard SDPA fallbacks. Eliminate kernel crashes (FPE) via dtype/shape/SM validation, contiguity, and varlen invariants; ship a reproducible triage harness and acceptance tests.

Current status: FA‑2 sometimes crashes on H100 at init in this environment. SDPA fallback works, but throughput is insufficient for production. We hardened wrappers (dtype cast, contiguity, cu_seqlens checks, SM/head_dim guards) and updated Architecture Overview to enable FA‑2 by default on supported GPUs.

## Objectives

- Reliability: Zero FA‑2 crashes in decode and batched paths for cmp/win.
- Parity: Numeric parity within tolerance vs SDPA references on small shapes.
- Performance: Achieve FA‑2 speedups on H100/A100; enforce min-length thresholds.
- Observability: Sufficient logs/artifacts to reproduce any regression in <10 minutes.

## Scope and Non‑Goals

- In scope: FA‑2 dense/varlen integration for compressed/sliding branches; selection remains SDPA (masked/packed). Harden guards, dtype/shape handling.
- Out of scope: Triton selection kernels (M4/M5), tokenizer/byte model changes, unrelated training scripts.

## Changes Implemented (this PR)

- Guardrails and safety checks in `nsa/kernels/flash_wrappers.py`:
  - Dtype enforcement: cast Q/K/V to fp16 or bf16 before FA‑2; `NSA_FA2_PREF_FP16=1` controls preference; `NSA_FA2_ALLOW_FP32=0` by default.
  - Contiguity: enforce `.contiguous()` on all FA‑2 inputs and KV‑packed buffer.
  - Varlen invariants: assert `cu_seqlens_*` int32, CUDA, non‑decreasing; validate sizes.
  - SM/head_dim guards: disallow unsupported combos (≤128 on SM8x, ≤256 on SM9x; head_dim%8==0).
  - Telemetry: log path, shapes, dtype when `NSA_DEBUG_TIMING=1`.
- Architecture doc updated (FA‑2 default on H100/A100; strict guards; SDPA fallbacks).

## Work Plan (Consultant‑Ready)

1) Triage & Instrumentation
- Reproduce on H100 with minimal harness; capture shapes/dtypes/SM:
  - `NSA_TEST_FA2=1 PYTHONPATH=. uv run -q pytest -k fa2_gpu_varlen`
  - `PYTHONPATH=. uv run python bench/bench_fa2.py`
- Enable logs: `NSA_DEBUG_TIMING=1 NSA_SDPA_AUDIT=1`.
- Collect: driver/runtime versions, FA‑2 version, head_dim, cu_seqlens stats.

2) Preconditions & Guards
- Enforce head_dim%8==0 and SM‑specific caps (≤128 on A100, ≤256 on H100).
- Force contiguity and dtype (fp16/bf16) at all FA‑2 callsites.
- Validate cu_seqlens monotonicity, correct totals; reject early with clear errors.

3) Parity & Safety Nets
- Keep hard SDPA fallback on any FA‑2 exception; log reason.
- Use conservative `NSA_FA2_MIN_LEN_{WIN,CMP}` thresholds (start at 16; tune via benches).
- Verify masked SDPA fallback parity on tiny shapes and empty‑row cases.

4) Acceptance Tests
- GPU parity (opt‑in): `-k fa2_gpu_varlen` and small decode cases for cmp/win.
- Batched parity: `test_batched_parity`, `test_masked_tiny` with FA‑2 enabled.
- Long context smoke (decode counters) with FA‑2 toggled; assert no NaNs/FPE.

5) Performance Validation
- Run `bench/bench_fa2.py` on H100/A100; record speedups and pick thresholds:
  - Produce CSVs and a short artifact report.
  - Update `NSA_FA2_MIN_LEN_{WIN,CMP}` based on knee points.

6) Rollout & Flags
- Default: `NSA_USE_FA2=1 NSA_USE_FA2_WIN=1 NSA_USE_FA2_CMP=1` on supported GPUs.
- CI/CPU: FA‑2 off by default; SDPA parity only.
- Debug: `NSA_FA2_PREF_FP16=1`, `NSA_FA2_ALLOW_FP32=0`, `NSA_FA2_FORCE` for forced tests.

## Consultant Engagement Brief

Deliverables (1–2 weeks):
- Diagnose and fix any remaining FA‑2 crash with reproducible MRE and patch.
- Solidify varlen packing semantics if needed; document constraints.
- Produce parity and bench reports; recommend production thresholds.

Required Access/Artifacts:
- H100 and A100 nodes; CUDA and driver versions; FA‑2 version pin.
- Artifacts path: `artifacts/2025-09-04/fa2_harden/` with logs, CSVs, env dumps.

Success Criteria:
- Zero FA‑2 crashes on targeted shapes/hardware; automatic fallback only on unsupported shapes.
- Parity tests green; performance >1.2× vs SDPA on thresholds.

## Risks & Mitigations

- Kernel API drift: pin FA‑2 version; use guarded imports; add smoke at startup.
- Numerical differences: default to fp16 preference with optional bf16; assert finite outputs; fallback on anomalies.
- Shape corner cases: maintain masked SDPA fallback; incremental thresholds.

## How to Run

```bash
export NSA_USE_FA2=1 NSA_USE_FA2_WIN=1 NSA_USE_FA2_CMP=1
export NSA_FA2_MIN_LEN_WIN=16 NSA_FA2_MIN_LEN_CMP=16
export NSA_FA2_PREF_FP16=1 NSA_DEBUG_TIMING=1 NSA_SDPA_AUDIT=1

# Parity tests (GPU)
NSA_TEST_FA2=1 PYTHONPATH=. uv run -q pytest -k fa2_gpu_varlen

# Benches (GPU)
PYTHONPATH=. uv run python bench/bench_fa2.py
```

## Change Log

- `nsa/kernels/flash_wrappers.py`: dtype/contiguity enforcement, varlen cu_seqlens checks, SM/head_dim guards, logging.
- `Documentation/Architecture/Overview.md`: FA‑2 default policy on supported GPUs; guards and fallbacks.

## Links

- Artifacts (to be populated): `artifacts/2025-09-04/fa2_harden/`
- Execution rules (M0/M1): `.cursor/rules/20-m0-execution.mdc`

