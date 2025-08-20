# M4 — CUDA Selection Spike (Forward Only)

Status: In Progress
Scope: Implement a forward-only CUDA selection kernel for decode (S=1) behind a flag, with strict fallbacks to packed SDPA. Target ≥1.2× speedup on RTX 4090 for realistic decode shapes.

## Goals
- Forward CUDA kernel for selection attention (decode S=1 initial), inputs `[B,1,G,h,Dk]`, KV `[B,G,S_kv,D*]`, ranges `[B,1,G,n,2]`.
- Numerics parity: MAE ≤ 1e-3 vs packed SDPA (fp16/bf16 inputs, FP32 accumulators).
- Performance: ≥1.2× vs packed SDPA for l′=64, n∈{8,16}, D/Dv∈{64,128}, H∈{4,8}, N∈{256,1024}` on RTX 4090.
- Safety: Zero output for empty ranges; clamp reads ≤ t; robust fallback on any error.

## Non-Goals
- Backward kernel; prefill batch (will follow after decode S=1 success).
- Removing SDPA fallback.

## Design (Decode S=1)
- Grid: one persistent block per (row) of B×G; each block processes all heads h.
- Two-pass FP32 LSE numerics over concatenated selected ranges (contiguous segments).
- Memory: load contiguous K/V tiles into shared memory; accumulate `p·V` in registers per Dv tile; single writeback per tile.
- Dtype: fp16/bf16 inputs, cast to fp32 for accum; cast output to input dtype of V.

## API & Flags
- Python wrapper: `nsa.kernels.cuda_sel_kernel.selection_attention_cuda(Q,K,V,ranges)` behind `NSA_SEL_CUDA=1`.
- Build on demand: `NSA_SEL_CUDA_BUILD=1` triggers extension compilation.
- Fallback: default to `grouped_selection_attention_packed` on any error or when flags not set.

## Benchmarks
- Script: `bench/bench_sel_cuda.py` (added), prints CSV-like lines with `{cuda_ms, ref_ms, speedup, mae}`.
- Matrix: N∈{256,1024}, H∈{4,8}, D/Dv∈{64,128}, L∈{128,256,512}, n ∈ {8,16} collapsed into a single contiguous range for first pass.

## Tests
- CPU parity smoke via wrapper fallback (added): `nsa/tests/test_sel_cuda_wrapper.py`.
- GPU parity (opt-in once kernel lands): compare vs packed SDPA on tiny grids; MAE ≤ 1e-3.

## Rollout
- Keep off by default; enable via `NSA_SEL_CUDA=1` for experiments.
- If acceptance met, consider gating on decode for 4090 with conservative min-L; retain packed SDPA fallback.

