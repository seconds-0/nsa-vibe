# M5: Selection Backward + Autotune + Productionization

## Task ID
M5-Selection-Backward-Autotune

## Status
Not Started

## Problem Statement
M4 delivers a robust forward Triton selection kernel with integration, guards, and benchmarking. To make NSA training-ready and performant end-to-end, we need a numerically stable, efficient backward implementation for the selection branch, along with autotuning and productionization (observability, config defaults, and CI coverage). This milestone completes the training path and locks in performance defaults.

## Proposed Solution
- Implement Triton backward kernels (dense and varlen) for selection attention using the standard two-pass softmax derivatives. Provide a custom `autograd.Function` with fused forward/backward when possible; fall back to packed SDPA backward if Triton disabled or for unsupported shapes.
- Add `@triton.autotune` configs for forward and backward, keyed on `(D, Dv, avg_L, dtype)`. Persist best configs per device and expose env/config overrides. Provide a light-weight warmup autotune at startup or first call per device.
- Integrate AMP/bfloat16/fp16 safe paths with mixed-precision autocast. Validate numerics against packed SDPA in fp32 reference mode on small shapes.
- Extend observability: log per-bucket timings, tokens/bytes read, kernel config used, and parity MAE (opt-in). Add counters for backward pass FLOPs and memory bandwidth estimates.
- Update config defaults (`configs/base.yaml`) with benchmark-driven `sel_triton_min_L` (from M4 benches) and tuned block sizes. Keep environment variables as overrides.
- Expand tests: forward/backward parity, gradcheck on tiny shapes, GPU parity matrix (opt-in), and CI hooks to run GPU tests when available.
- Document constraints in PRD (supported dtypes, head_dim multiples, min L thresholds, fallback policy) and add a "Selection Triton Usage" section.

## Automated Test Plan
- Forward parity (GPU opt-in):
  - Matrix over `L ∈ {64, 128, 256, 512}`, `D ∈ {64, 128}`, `Dv ∈ {64, 128}`, `H ∈ {2, 8}`; multi-range patterns (few/many/mixed); compare Triton wrapper vs packed SDPA (MAE < 1e-3).
  - Edge cases: empty ranges, invalid ranges (clamped), tiny shapes, contiguous single-range dense fast-path.
- Backward parity:
  - autograd grad parity vs packed SDPA on tiny shapes; assert close for dQ, dK, dV.
  - `torch.autograd.gradcheck` on double-precision tiny configs when feasible.
  - AMP paths smoke tests (bf16/fp16 autocast enabled) verifying finite outputs and stable loss.
- Performance smoke:
  - Ensure `sel_triton_min_L` gate routes to Triton for large L; verify wrapper runs without fallback.
  - Measure per-bucket timing logs emitted (sanity only; not asserting thresholds in CI).

## Components Involved
- `nsa/kernels/triton_sel_kernel/sel_fwd.py` (extend with backward or add `sel_bwd.py`)
- `nsa/kernels/triton_sel_kernel/sel_bwd.triton` (implement)
- `nsa/kernels/triton_sel_kernel/__init__.py` (autograd integration, guards, buffers, observability)
- `nsa/core/attention_kernels.py` (reference packed SDPA paths)
- `configs/base.yaml` (thresholds and defaults)
- Tests under `nsa/tests/` (GPU opt-in parity, grad tests)
- `PRD.md` (constraints, toggles, thresholds)

## Dependencies
- Output from M4 benchmarks to determine `sel_triton_min_L` and good initial block sizes.
- CUDA/Triton environment for GPU tests and autotune.

## Implementation Checklist
- [ ] Backward math derivation and kernel API design (dense + varlen)
- [ ] Implement Triton backward kernels (`sel_bwd.triton`) with two-pass softmax derivatives
- [ ] Integrate custom `autograd.Function` with fused backward; fall back to packed SDPA when disabled/unsupported
- [ ] Add autotune configs for forward/backward; cache best per device; env/config overrides
- [ ] AMP integration tests (bf16/fp16 autocast) and fallback policy for unsupported dtypes
- [ ] Observability: per-bucket timings, kernel config id, tokens/bytes, parity MAE (opt-in)
- [ ] Update `configs/base.yaml` with `sel_triton_min_L` and block defaults from benches
- [ ] Update `PRD.md` with M5 constraints and usage notes
- [ ] Tests: forward/backward parity (CPU/GPU), gradcheck tiny, GPU parity matrix (opt-in)
- [ ] Optional: workspace buffer pooling for backward temporaries

## Verification Steps
- Run forward/backward parity tests; ensure MAE and grad diffs within tolerances.
- Run gradcheck on at least one tiny config per dtype.
- Run opt-in GPU parity matrix; spot-check logs show Triton path used and timing per bucket.
- Validate AMP paths numerically stable on small synthetic training step.
- Confirm `sel_triton_min_L` routes as expected via logs.

## Decision Authority
- Engineering choices (kernel internals, autotune parameters) at engineer discretion.
- Thresholds and config defaults decided by benchmark data; escalate if significant PRD changes are needed.

## Questions / Uncertainties
Blocking:
- None (dependent on M4 benchmark results to set defaults; work can proceed with placeholders).
Non-blocking:
- How aggressively to autotune at runtime vs static defaults; acceptable startup overhead.
- Gradcheck feasibility for larger shapes (likely limited to tiny configs).

## Acceptable Tradeoffs
- Use packed SDPA as backward fallback where Triton unsupported (temporary, acceptable if forward is Triton for large L).
- Limit autotune search space to a small curated set to keep overhead low.

## Notes
- Keep env toggles: `NSA_USE_TRITON_SEL`, `NSA_SEL_TRITON_MIN_L`, block size overrides, `NSA_SEL_TRITON_ALLOW_GRAD`, `NSA_DEBUG_COMPARE`.
- Maintain deterministic behavior for tests; avoid nondeterministic reductions.


