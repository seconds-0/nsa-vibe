# ADR: Deprecate Triton Selection on RTX 4090 (M4-02 Closure)

- Date: 2025-08-20
- Status: Accepted — Non‑viable on RTX 4090
- Owners: NSA Architect; TritonKernel Agent

## Context

The M4-02 milestone aimed to revive Triton-based selection attention kernels (dense and varlen) to achieve performance parity or wins versus the packed SDPA fallback on an RTX 4090 (Ada, SM 8.9). The codebase includes group-centric kernels, FP32 LSE, safe fallbacks, and a benchmark suite.

## Findings

Comprehensive benches and parity tests on RTX 4090 show:
- Performance: 4–25× slower than packed SDPA across realistic shapes; no ≥1.2× region.
- Numerics: up to ~150× worse MAE in adverse cases despite two-pass FP32 LSE.
- Stability: MLIR compilation crashes on varlen kernels due to core type system limitations (2D pointer arithmetic patterns), not easily resolved at the kernel level.

These gaps are too large for incremental optimizations (register accumulation, double buffering, autotune) to bridge.

## Decision

- Close M4-02 as “investigated and determined non‑viable” for RTX 4090.
- Keep Triton selection permanently disabled on 4090 (Ada, SM 8.9) with hard runtime fallback to packed SDPA.
- Maintain a high activation threshold globally (`sel_triton_min_L` high) and require an explicit override to run Triton on 4090 for experimentation.

## Consequences

- Production path remains packed SDPA for selection on consumer GPUs.
- Future optimization efforts should focus on:
  - Proven kernels like FlashAttention‑2 where applicable, or
  - Purpose‑built CUDA C++ kernels if sparse selection becomes a critical requirement.
- The Triton selection code remains in-tree for research behind explicit flags, with strong safeguards and documentation clarifying its non‑viability on 4090.

## Implementation

- Config: raise default `runtime.sel_triton_min_L` to keep Triton effectively off.
- Wrapper: detect SM 8.9 (RTX 4090) and force fallback unless `NSA_TRITON_SEL_FORCE=1`.
- Docs: update playbooks and guides to reflect the decision.

## References
- Guides: `Documentation/Guides/Selection-Triton-Bench-4090.md`
- Kernels: `nsa/kernels/triton_sel_kernel/sel_fwd.py`
- Wrapper: `nsa/kernels/triton_sel_kernel/__init__.py`
- Benchmarks: `bench/bench_sel_triton.py`, `bench/sel_threshold_from_csv.py`

