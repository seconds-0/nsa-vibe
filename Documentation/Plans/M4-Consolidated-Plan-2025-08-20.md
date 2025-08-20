# M4 — Consolidated Plan (Selection Acceleration)

Status: Paused on RTX 4090 (SM 8.9); Conditional for A100/H100.
Scope: Selection branch acceleration (decode-first), routing/guards, benchmarks, and decision gates.

## Context & Findings (4090)
- Decode benchmark shows selection share ≈ 1–3% on RTX 4090 across S=128–1024.
- FA‑2 is 5–100× slower than SDPA on 4090; SDPA dominates.
- Triton selection is non‑viable on 4090 (slower/unstable); disabled by ADR-2025-08-M4-02.
- Production defaults set SDPA everywhere on 4090; FA‑2/Triton guarded off unless forced.

## Goal
Accelerate selection only when it materially improves end‑to‑end decode (≥1.2× over SDPA on target shapes) and selection accounts for a meaningful share (≥25–30%).

## Decision Gates
1) Hotspot share: selection% ≥ 25–30% of decode time on target hardware/shapes.
2) Kernel ROI: custom kernel ≥ 1.2× vs packed SDPA on those shapes.
3) Stability: numerics within MAE ≤ 1e‑3 (forward), safe fallbacks on error.

If any gate fails, do not ship a custom kernel; stay on SDPA.

## Hardware Matrix (Initial)
- RTX 4090 (SM 8.9): selection share low; kernel ROI low → Stay on SDPA (default).
- A100/H100: Unknown → Run decode bench; if gates pass, proceed with CUDA selection kernel.

## Implementation Tracks

Track A — Measurement & Guardrails (all GPUs)
- Decode bench: use updated bench with correct reads; summarize selection% per S; seed runs.
- Config: keep FA‑2/Triton off on 4090; expose env overrides for experiments.
- Tests: CPU guard tests; CUDA loader fallback test; optional CLI smoke.

Track B — CUDA Selection (Conditional)
- Scope: Forward-only, decode S=1 initially. Keep behind `NSA_SEL_CUDA=1` & build flag.
- Kernel sketch:
  - Persistent blocks per (B×G); group-centric Q load once.
  - Two-pass FP32 LSE (m,l) with register accumulation; single writeback.
  - Iterate contiguous spans; coalesce loads; avoid repeated indexing.
  - Dtypes: FP16/BF16 compute with FP32 accum.
- Wrapper & routing:
  - Python wrapper with strict fallbacks to packed SDPA on errors/unsupported shapes.
  - Device guards: off by default on 4090; opt-in on other GPUs.
- Validation:
  - Parity tests vs packed SDPA (MAE ≤ 1e‑3) on realistic shapes.
  - Microbench harness: CUDA kernel vs SDPA on target S/G/h/Dk/Dv & span mixes.
- Acceptance:
  - ≥1.2× vs SDPA on target shapes and selection% ≥ 25–30%. Otherwise, leave off by default.

Track C — Triton (Reference Only)
- Keep Triton selection paused (ADR-2025-08-M4-02). Revisit only on newer Triton/CUDA.

## Phased Plan
1) Confirm hotspot (all target GPUs)
   - Run decode bench; compute selection% across S; record CSV & summary.
   - If selection% < 25–30% → stop; document and keep SDPA.

2) Prototype CUDA kernel (only if gate 1 passes)
   - Implement forward; add tests; microbench vs SDPA.
   - If speedup < 1.2× → stop; document and keep SDPA.

3) Integrate & Guard (if both gates pass)
   - Wire routing with flags; add CI smoke; docs and ADR update per GPU.

## Deliverables
- Bench CSVs + summary for each GPU.
- CUDA selection forward (optional) + tests + microbench.
- ADR update enabling/disabling per GPU.
- Docs: user guide for running decode bench and interpreting thresholds.

## References
- ADR-2025-08-M4-02 — Deprecate Triton Selection on RTX 4090.
- Guides/Decode-Benchmark-Guide.md — Benchmark steps and acceptance thresholds.
- Plans: M4-Selection-Triton-2025-08-19.md, M4-02-Triton-Selection-Execution-Playbook-4090.md, M4-02-Triton-Selection-Revival.md, M4-CUDA-Selection-Spike.md, M4-Triton-Selection-Benchmark-Plan.md.

