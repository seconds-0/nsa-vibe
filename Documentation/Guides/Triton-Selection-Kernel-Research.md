# Triton Selection Kernel — Fundamentals, Paper Implications, and Best Practices

This document consolidates practical guidance for writing a performant, numerically stable selection (sparse) attention kernel in Triton, grounded in the NSA paper’s requirements and general GPU/Triton best practices. It is intended to guide our rewrite and diagnosis after initial Triton attempts proved non‑viable on RTX 4090.

## 1) NSA Paper Implications for the Kernel
- Group‑centric execution (Figure 3):
  - Load all heads’ queries (within a GQA group) once; reuse K/V fetch across heads.
  - Iterate over selected KV ranges in contiguous tiles (chunk size divides selection block length).
- Two‑pass softmax numerics:
  - Pass 1: accumulate per‑head running max `m` and running sum of exp (LSE) across L tiles.
  - Pass 2: compute probabilities `p = exp((Q·K^T − m) / 1) / lse` and accumulate `p·V`.
  - FP32 accumulation; bf16/fp16 inputs acceptable with FP32 compute.
- Determinism & causality:
  - Respect causal clamps (`end ≤ t+1`); ranges are pre‑merged upstream.
  - Same ranges across heads in a group (GQA group consistency).
- Varlen packing:
  - Multi‑range rows should be packed host‑side into contiguous slices with `cu_seqlens`; kernel consumes packed buffers.

Takeaway: The kernel must amortize K/V global loads across heads and across Dv tiles, and implement a stable two‑pass softmax per head.

## 2) Triton Fundamentals (for performance and stability)
- Programs and tiling:
  - Each `triton.jit` program handles a logical tile (e.g., one row × BLOCK sizes). `tl.program_id(dim)` indexes a program.
  - Choose BLOCK sizes such that on‑chip SRAM (shared) holds working tiles, and memory accesses are coalesced.
- Memory access:
  - Favor contiguous, aligned loads/stores; arrange tensors so the fastest‑varying stride matches the loaded axis.
  - Use boolean masks matching the tile’s value shape, not pointer shape. Avoid broadcasting pointer expressions.
  - Vectorization/alignment: try to ensure `stride_* == 1` along the loaded dimension and use multiples of 8/16.
- Softmax numerics:
  - Two‑pass LSE (running max `m` and sum of exp) for stability; FP32 accumulations.
- Pipelining:
  - `num_stages>1` enables software pipelining; combine with explicit double‑buffering to overlap load/compute.
  - `num_warps` tunes parallelism within a program; typical attention tiles use 4–8 warps.
- Reductions:
  - Prefer block reductions (`tl.sum`, `tl.max`) over scalar loops; keep reduction axes small and contiguous for bandwidth.
- Control flow and masks:
  - Keep divergent branches minimal; apply masks to values instead of branching lanes.
  - Ensure mask shapes equal the tile load/store shapes.

## 3) Common Pitfalls (seen in our attempts)
- Pointer‑shape broadcasting in masks (MLIR errors):
  - Avoid `tl.broadcast_to(..., k_ptrs.shape)` when `k_ptrs` is a pointer expression; use `(Lmask[:, None] & Dmask[None, :])` sized to tile.
- Excessive global read‑modify‑write on O:
  - Repeatedly loading/storing `O` per L‑tile (RMW) causes bandwidth blow‑ups. Accumulate in registers/shared for current `dv0` and store once.
- Per‑head K/V reloads:
  - Launching per‑head kernels duplicates K/V global loads; group‑centric kernels must reuse K/V across heads.
- Tiny tiles and poor occupancy:
  - BLOCK sizes that are too small, or shared memory that is too large, can tank occupancy. Sweep BLOCK_L=128/256, BLOCK_D=64, BLOCK_DV=64 with 4–8 warps.

## 4) Best‑Practice Skeleton for Selection (Forward)
- Launch policy:
  - Grid over packed rows `N_rows` (each is one (b,t,g)). Compute all `H` heads per program (BLOCK_H ≤ H, often 8–16). Fallback to per‑head if H is larger.
- Pass 1 (LSE):
  - For `l0` in tiles of L: load `K_tile[L,BLOCK_D]` and `Q_tile[H,BLOCK_D]`, update `logits_tile[H,L] += sum(Q_tile*K_tile)`.
  - Reduce across D for dot; across L for max/sum. Track `m[H], lse[H]`.
- Pass 2 (p reuse):
  - For `l0` in tiles of L: recompute `logits_tile`, then `p[H,L] = exp(logits−m)/lse`.
  - For `dv0` in tiles of Dv: load `V_tile[L,BLOCK_DV]`, compute `acc[H,BLOCK_DV] += p @ V_tile`. Store `acc` once per `dv0`.
- Memory & pipelining:
  - Double‑buffer `K_tile`/`V_tile`: prefetch next tile while computing current.
  - Keep `Q_tile` in registers; ensure strides are contiguous along D.

## 5) Varlen Path Considerations
- Host packing:
  - Build packed `K/V` and `cu_seqlens` so each row is contiguous in memory. Avoid tiny slices that fragment memory.
- Kernel reuse:
  - Use the same group‑centric schedule treating `[row_start:row_end)` as L; identical tiling and numerics.

## 6) Diagnostics & Profiling
- Time components:
  - Host packing time vs kernel time (CUDA events); bucket timings per L_i and N to spot skew.
- Path logging:
  - Log path decisions (dense_group/varlen_group/per‑head/fallback) to ensure intended routes are used.
- Warmups:
  - Exclude first‑run JIT time from measurements (do 3–5 warmups).
- Sanity knobs:
  - Compare a small subset against packed SDPA (MAE) for each change.

## 7) Concrete Fixes for Our Kernel
- Eliminate O RMW in pass‑2:
  - Accumulate per‑`dv0` into a register/shared `acc[H,BLOCK_DV]` across all L tiles, then store once.
- Enable double‑buffering:
  - Alternate two `K_tile`/`V_tile` buffers across L tiles; set `num_stages` to 2–3.
- Expand autotune:
  - Try `num_warps ∈ {4,8}`, `num_stages ∈ {2,3}`, `BLOCK_L ∈ {128,256}`, `BLOCK_D=64`, `BLOCK_DV=64`.
- Enforce alignment/contiguity:
  - Ensure `stride_kd == 1`, `stride_vd == 1`, and inputs are `contiguous()` along feature dims.
- Guard path selection:
  - Keep `sel_triton_min_L` high by default; only enable kernel where speedups are consistently ≥ margin.

## 8) Alternatives When Triton Fails
- CUDA C++ (CUTLASS‑style):
  - Write a custom kernel with explicit shared memory control and vectorized loads; more work but stable and faster.
- Permanent SDPA fallback:
  - Keep packed SDPA as the production path; use Triton only for experimentation.

## 9) Action Plan
1. Implement double‑buffering and register accumulation (feature flag `NSA_SEL_TRITON_DB=1`).
2. Remove `O` RMW; store once per `dv0` after summing across L.
3. Expand autotune configurations; record win/loss per bucket.
4. Re‑bench on 4090; pick `sel_triton_min_L` only if ≥1.2× consistently.
5. If still non‑viable, prioritize a CUDA C++ kernel or keep SDPA permanently.

## 10) References (conceptual)
- Group‑centric kernel schedule (NSA Figure 3): share KV fetch across heads; iterate contiguous L tiles.
- Triton best practices: coalesced loads/stores, boolean masks with tile shapes, two‑pass softmax, pipelining with `num_stages`, autotuning `num_warps` and tile sizes.
