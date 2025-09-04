# 2025-09-04 Core Engineer Report - FA‑2 Integration Analysis and Consultant Request v1

## Executive Summary

We need FlashAttention‑2 (FA‑2) enabled by default for NSA’s compressed (cmp) and sliding (win) branches on supported GPUs to meet production throughput targets. The current crashes reported by Test Engineering are attributable to brittle preconditions (dtype, contiguity, cu_seqlens validity, head_dim vs SM constraints), not a fundamental architectural mismatch. I’ve added guardrails and safety checks to the FA‑2 wrappers and aligned our Architecture Overview to position FA‑2 as the default fast path (with strict guards and hard SDPA fallbacks). This document consolidates Test Engineering’s findings, my analysis, gaps, and a precise request list for a specialized FA‑2 consultant to drive stabilization and performance validation to closure.

## Reviewed Inputs (Test Engineer Reports)

- 2025-09-04 Test Engineer Report - NSA FlashAttention-2 Incompatibility Analysis v3.md
- 2025-09-04 Test Engineer Report - NSA H100 FlashAttention Investigation v2.md
- 2025-09-04 Test Engineer Report - NSA H100 Training Performance Analysis v1.md

Key claims and my assessment:
- “Architectural incompatibility” with FA‑2: Not supported by code or routing. NSA uses SDPA everywhere with FA‑2 opt‑in for cmp/win; selection remains SDPA. Crashes likely stem from input preconditions, not design.
- FA‑2 enabled causes immediate FPE on H100: Plausible. Repro likely tied to dtype (fp32), non‑contiguous views, invalid varlen cu_seqlens, or head_dim vs SM.
- “NSA untrainable without optimized kernels”: Exaggerated. SDPA routes are correct and trainable; however, production throughput requires FA‑2 to meet schedules and cost goals. So FA‑2 is practically required for performance, not for correctness.

## Current Code State (Core changes in this branch)

- Enforced dtype and contiguity before FA‑2 calls (dense and varlen); preference for fp16 (`NSA_FA2_PREF_FP16=1`), optional bf16.
- Validated varlen cu_seqlens (int32 on CUDA, monotonic non‑decreasing, size checks) to prevent kernel crashes.
- Added SM/head_dim guards (≤128 on SM8x, ≤256 on SM9x; head_dim%8==0) in `fa2_supported_verbose`.
- Logging improvements to surface shapes/dtypes/paths when `NSA_DEBUG_TIMING=1`.
- Architecture Overview updated: FA‑2 default ON for cmp/win on supported GPUs; strict guards; automatic SDPA fallback; selection remains SDPA.

## Gaps and Open Questions

- Causal semantics vs FA‑2 API: decode paths pass `causal=False` for per‑row batched calls (since we slice keys to the allowed range), but need to ensure no off‑by‑one when switching between dense/varlen and SDPA fallbacks. Verified in unit tests, but consultant should re‑check for corner cases.
- Varlen packing for selection (future): selection stays SDPA; we are not routing selection through FA‑2 varlen today by policy. Ensure that any future effort to use FA‑2 varlen for selection has a clear contract on contiguous packed layouts and cu_seqlens.
- Performance thresholds: Conservative min lengths set to 16; need empirical knee points per GPU/head_dim.
- Mixed precision policy: Prefer fp16 for FA‑2 kernels for throughput unless numerics demand bf16. Consultant input desired.

## Risks

- Kernel API drift across FA‑2 versions; behavior changes in varlen vs kvpacked APIs.
- Edge shapes (tiny windows/lengths) causing non‑finite outputs; we guard and fallback but want to minimize these paths.
- SM‑specific constraints (Ada vs Hopper) not fully mapped for every head_dim.

## Acceptance Criteria (Definition of Done)

- Stability: Zero FA‑2 crashes (FPE or illegal memory access) across our supported shapes on H100/A100.
- Parity: All FA‑2 parity tests green (tiny shapes, varlen decode, cmp/win decode) within numeric tolerances; no NaNs after FA‑2 calls.
- Performance: Demonstrated >1.2× speedup over SDPA on cmp/win at or above tuned min‑length thresholds; recommended thresholds captured in config.
- Observability: Logs include path/shape/dtype; artifacts recorded in repo‑local `artifacts/` with a short index report.

## Explicit Consultant Requests

Please deliver the following (with links/artifacts under `artifacts/2025-09-04/fa2_harden/`):

1) Crash Diagnostics and Minimal Repro
- Produce a minimal script (single file) reproducing any FA‑2 crash, with exact shapes, dtypes, strides, head_dim, and cu_seqlens. Include device model, driver, CUDA, PyTorch, and FA‑2 version.
- Identify the exact kernel path (dense vs varlen vs kvpacked), and whether the failure correlates with dtype, contiguity, or head_dim/SM limits.
- Recommend permanent guard conditions if applicable (e.g., disallow head_dim=192 on SM8x varlen).

2) Input Contracts (Authoritative)
- Document required input layout for FA‑2 dense and varlen calls we use:
  - Tensor dtypes (fp16/bf16 only?), alignment, stride/contiguity expectations.
  - cu_seqlens invariants and size relationships for varlen and kvpacked APIs.
  - Causal semantics expectations when Tq=1, Tk=L and we pass `causal=False` vs `True`.
- Propose code‑level assertions we should add to enforce the contract at runtime (cheap checks) and in tests.

3) Varlen Packing Review
- Review our varlen packing in `attention_kernels.py` and the SDPA masked fallback for selection. Validate that our cu_seqlens and packed buffers are compatible with FA‑2 if/when we route selection through varlen.
- Advise on safe next steps to enable selection varlen via FA‑2 (if recommended) or confirm staying SDPA for selection.

4) Performance Tuning and Thresholds
- Run `bench/bench_fa2.py` on H100 and A100 across representative head_dims (e.g., 64/96/128/192) and sequence/window lengths. Provide CSVs and a short analysis.
- Recommend `NSA_FA2_MIN_LEN_{WIN,CMP}` thresholds for each GPU/head_dim pairing.
- If relevant, suggest tile sizes or flags (e.g., kvpacked vs non‑packed) to maximize throughput.

5) Parity and QA Enhancements
- Augment our parity tests to catch common pitfalls (empty rows; extreme small L; large L near shared memory limits; bf16 vs fp16 differences).
- Provide numeric tolerance guidance for comparisons against SDPA.

6) Rollout Guidance
- Propose a guarded rollout plan (feature flags, telemetry to confirm path usage, fallback ratios we should watch).
- Identify any FA‑2 version pins and known good configs for H100/A100, including driver/CUDA combinations.

7) Documentation Deliverables
- An engineer‑readable “FA‑2 Integration Guide for NSA” summarizing contracts, guards, and runbook steps; lives in `Documentation/Architecture/`.
- A short “Troubleshooting FA‑2” doc with common symptoms and fixes.

## What We’ve Done Already (for your starting point)

- Hardened FA‑2 wrapper:
  - Dtype casting to fp16/bf16, contiguity enforcement.
  - cu_seqlens validation for varlen.
  - SM/head_dim guardrails in `fa2_supported_verbose`.
  - Detailed logging (shapes/dtypes/paths) when `NSA_DEBUG_TIMING=1`.
- Architecture doc aligned to enable FA‑2 by default (cmp/win) with strict fallbacks.
- Added a stabilization plan report and this consolidated request for engagement.

## How to Reproduce and Validate

Environment flags:
```bash
export NSA_USE_FA2=1 NSA_USE_FA2_WIN=1 NSA_USE_FA2_CMP=1
export NSA_FA2_MIN_LEN_WIN=16 NSA_FA2_MIN_LEN_CMP=16
export NSA_FA2_PREF_FP16=1 NSA_DEBUG_TIMING=1 NSA_SDPA_AUDIT=1
```

Commands:
- GPU parity (opt‑in): `NSA_TEST_FA2=1 PYTHONPATH=. uv run -q pytest -k fa2_gpu_varlen`
- Benches (GPU): `PYTHONPATH=. uv run python bench/bench_fa2.py`

Artifacts:
- Please place logs, CSVs, and environment summaries under `artifacts/2025-09-04/fa2_harden/` and link them from a brief README in that directory.

## Timeline Proposal

- Days 1–2: Repro and input contracts; crash fixes/guards proposed.
- Days 3–4: Parity suite enhancements; initial performance bench; threshold recommendations.
- Days 5–7: Rollout plan, docs, and final patch series; sign‑off on acceptance criteria.

---
Prepared by: Core Engineer (GPT‑5)
Date: 2025‑09‑04

