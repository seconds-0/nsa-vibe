# M4-02 — Triton Selection Revival Plan (Paper-Aligned, 4090-Ready)

Status: Paused (ADR-2025-08-M4-02: Non-viable on RTX 4090). Keep as reference for non-Ada GPUs only; do not execute on SM 8.9.
Owner: Lead Engineer
Scope: Triton selection kernels (forward), routing, diagnostics, benches, and thresholds
Targets: RTX 4090 (primary), others later

## Why Triton (paper intent)
- NSA paper (Figure 3) assumes a custom, group-centric selection kernel that shares KV fetch across heads in a GQA group, iterates contiguous selected ranges, and uses two-pass LSE numerics. Generic SDPA cannot realize the same bandwidth economics or group reuse.
- Owning this kernel unlocks decode efficiency, sparse backward (later), and device-specific tuning.

## Current State (4090 findings)
- Dense (few spans): Triton 2–25x slower than direct SDPA; MAE only good at L=64, degrades to ~0.1–0.15 for L≥128.
- Varlen (many spans): MLIR compilation crashes across shapes.
- Group kernels: compilation timeouts for benches; per-head variants used.
- Production: Keep Triton disabled (high min-L); SDPA path is accurate and fast.

## Acceptance Criteria
- Correctness: MAE ≤ 1e-3 (fp16/bf16 inputs, FP32 accum) up to L=1024.
- Performance: Find min-L where Triton ≥ 1.2x SDPA on 4090 (dense and varlen).
- Stability: No MLIR crashes for tested matrices; deterministic routing; robust fallbacks remain.
- Alignment: Group-centric schedule, two-pass LSE, varlen packing with cu_seqlens.

## Technical Workstream

### A) Numerics & Correctness
- Enforce FP32 accumulators for logits/LSE and p·V; inputs fp16/bf16; cast outputs back.
- Two-pass LSE across ALL L tiles per head; reuse m,lse consistently in pass-2.
- Verify scaling (inv_sqrt_d) applied consistently.
- Add GPU parity tests sweeping L/H/D/Dv small grid (opt-in).

### B) Scheduling & Memory
- Group-centric execution: one program per (row), compute all heads (BLOCK_H≤H) to reuse K/V across heads. Same for varlen rows via cu_seqlens.
- Eliminate O read-modify-write: accumulate per Dv tile across L tiles in registers/shared; store once per dv0.
- Double-buffer K/V tiles across L tiles; tune num_stages∈{2,3}.
- Expand autotune: num_warps∈{4,8}, num_stages∈{2,3}, BLOCK_L∈{128,256}, BLOCK_D=64, BLOCK_DV=64. Remove conflicting explicit launch params.

### C) Varlen Robustness
- Strict tile-shaped boolean masks only; no pointer-shape broadcasting.
- Pack adjacent spans; minimize tiny fragments; time packing vs kernel.
- Single varlen kernel mirrors dense schedule over [row_start,row_end).

### D) Routing & Flags
- Force-path: `NSA_SEL_TRITON_FORCE_PATH={dense,varlen,auto}`.
- Device gate: keep `sel_triton_min_L` high on 4090 until ≥1.2x wins; lower per data.
- Maintain try/except fallbacks to packed SDPA.

### E) Diagnostics
- Log kernel path (dense_group/varlen_group/per_head), bucket timings (ms), pack time (ms).
- Kernel-only microbench (no packing) to measure pure kernel throughput and validate autotune.

## Milestones
- M4-02.A (1–2d): Fix numerics (LSE/FP32/scaling); pass MAE ≤1e-3 up to L=1024.
- M4-02.B (3–5d): Reg/shared accumulation, double-buffering, autotune expansion; bench vs SDPA.
- M4-02.C (2–3d): Varlen stability, packing optimization; bench.
- M4-02.D (1d): Threshold selection on 4090 (≥1.2x), PR updates configs; otherwise keep disabled.
- Contingency (1–2w): CUDA kernel spike (CUTLASS-style) or finalize SDPA fallback for 4090.

## Bench & Thresholds
- Dense few spans: L∈{64,128,256,512,1024}, N∈{256,1024}, H∈{4,8}, D/Dv∈{64,128}.
- Varlen many spans (n=8), streams∈{1,2}.
- Use `bench/bench_sel_triton.py` and `bench/sel_threshold_from_csv.py --margin 1.2`.

## Risks & Mitigations
- MLIR instability: simplify pointer math; strict boolean masks.
- Occupancy vs SRAM: tile size tuning; autotune; measure.
- Device variance: per-GPU thresholds; do not enable globally without wins.

## Deliverables
- Paper-aligned forward kernels (dense/varlen) with numerics and DB improvements.
- Bench report & CSVs; chosen `sel_triton_min_L` for 4090.
- Updated config, flags, and docs.
