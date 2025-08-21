# M5 — Selection Backward & Autotune (Consolidated Plan)

Status: Implemented (functional, safe); performance Triton kernel reserved for follow-up.

## Scope & Goals
- Backward path for selection with FA‑style numerics (recomputed softmax using LSE) and sparse dK/dV writes limited to selected ranges.
- Varlen packing by identical L for efficiency; dense single-span path supported.
- Grad parity vs SDPA reference; gradcheck on tiny shapes; gating remains behind env.

## Deliverables
- Autograd wrapper: `_SelAttnTritonFn` now computes backward via vectorized PyTorch with varlen packing and sparse scatters.
- Safety: clear fallbacks to SDPA packed autograd on errors; CPU/GPU compatible.
- Tests: CUDA grad parity vs reference; selection gradcheck tiny case.

## Acceptance
- Gradcheck passes on tiny dims (double precision) for selection packed reference.
- CUDA backward parity vs reference within tight tolerances on small random shapes.
- Deterministic numerics on fixed seeds; device/dtype alignment guards remain in wrapper.

## Flags
- `NSA_SEL_TRITON_ALLOW_GRAD=1` to enable the autograd wrapper path.
- `NSA_TRITON_SEL_FORCE=1` to bypass SM 8.9 ADR in tests only.

## Outcome
- Backward enabled and correct when gated; still defaults to packed SDPA autograd in production.
- Triton-specific performance kernel (sel_bwd.triton) remains a placeholder; can be added without changing the public API.
