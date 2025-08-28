NSA Performance Optimization Plan (M8)

Source: recovered from PR commit 4fc1e25e (2025-08-28 Performance Optimization Plan.md), moved out of Reports to comply with linting and scope. This document tracks performance phases beyond FlashAttention enablement.

# NSA Performance Optimization Plan

Date: 2025-08-28
Author: Test Engineer
Current Performance: 153 tok/s (S=128), ~66 tok/s (S=256), Hangs (S=2048)
Target Performance: 300-800 tok/s
GPU: NVIDIA A100 80GB PCIe

Executive Summary
- Primary bottleneck: FlashAttention-2 not actually being used; code fell back to SDPA and repeat_interleave.
- Fixing FA‑2 usage yields 10–20x speedup; then proceed with memory and vectorization phases.

Root Cause Analysis
- FlashAttention not active: capability detection, tensor layout, and import fallbacks caused SDPA path.
- Memory fragmentation: workspace growth and repeated allocations in hot path.
- Python loop overhead: O(B*S*G*n) loops in selection.

Optimization Plan
- Phase 1: Fix FA‑2 enablement (dense/varlen), layout via view/expand, force usage envs, install/verify.
- Phase 2: Memory—eliminate repeat_interleave, preallocate workspaces, allocator tuning.
- Phase 3: Vectorize selection—bucketed/varlen packing and batched gathers; reduce Python loops.
- Phase 4: Kernel fusion—gate+combine, optional RoPE+proj fusions.
- Phase 5: Algorithmic—adaptive selection, hierarchical selection, mixed precision for scoring.
- Phase 6: Configuration—env defaults and model runtime knobs for S≈2048.

Key Snippets
- Dense FA‑2 with view/expand (no materialization)
  - Use `Q.transpose(1,2).reshape(B,G*h,...)` and `K/V.unsqueeze(2).expand(...).reshape(...)`.
- Preallocation knobs
  - `NSA_PREALLOC_VARLEN_N`, `NSA_PREALLOC_VARLEN_K` for workspace caps.
- Env recommendations (A100)
  - `NSA_USE_FA2=1`, `NSA_FA2_MIN_LEN_WIN=1`, `NSA_FA2_MIN_LEN_CMP=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512`.

Notes
- Full original details were preserved in the PR history (commit 4fc1e25e). This document is a condensed plan aligned with current code paths and CI constraints. We will expand sections 3–6 as we land subsequent phases on a dedicated perf branch.

