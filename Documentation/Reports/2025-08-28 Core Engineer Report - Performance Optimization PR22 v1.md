# Core Engineer Report — Performance Optimization (PR22)

Date: 2025-08-28  
Role: Core Engineer (GPT‑5)  
Branch: `feat/pr-22-uv-polish`

## Scope
- Map items in Documentation/Plans/M8-Performance-Optimization-Plan-FULL.md to what exists on PR22.  
- Identify gaps and propose the fastest, lowest‑risk path to target performance.

## Executive Summary
- PR22 wires robust FA‑2 routing with safe fallbacks, batched selection scoring/mapping, and workspace caching.  
- Major remaining costs come from repeat_interleave SDPA fallbacks, partially vectorized selection attention, and lack of kernel fusions.  
- Priorities: make FA‑2 the default on A100/H100, remove repeat_interleave from SDPA fallbacks, finish vectorized selection attention, then pursue fusions/mixed precision.

## What’s Done on PR22 (relative to plan)
- FA‑2 integration and routing:
  - Sliding/compressed FA‑2 forward paths with varlen and dense fallbacks: `nsa/core/attention_kernels.py` (sliding_window_attention_fa2, compressed_attention_fa2) and `nsa/kernels/flash_wrappers.py` (attention_fa2_varlen, attention_fa2_dense_batch).  
  - Device/arch guard for Ada (SM 8.9) and env overrides (`NSA_FA2_FORCE`, `NSA_FA2_MIN_LEN_{WIN,CMP}`) with robust try/except fallbacks.  
  - Decode FA‑2 paths present for both sliding and compressed (`*_fa2_decode`).
- Selection scoring/mapping (vectorized core):
  - Batched p_cmp via einsum: `compute_pcmp_all` (no Python loops).  
  - CSR mapping batched path: `map_pcmp_to_pslc_batched` using COO scatter_add.  
  - Deterministic top‑k with explicit tie‑breakers and forced initial + local blocks: `select_topn_ranges` / `_batched`.
- Selection attention fast paths:
  - Packed SDPA with length buckets and reusable workspaces: `grouped_selection_attention_packed` + `_SEL_PACK_WS`.  
  - Masked SDPA alternatives for sliding/compressed selection: `sliding_window_attention_masked`, `batched_causal_attention_compressed_masked` (parity‐safe simplifications).  
  - Triton/CUDA selection wrappers with high‑threshold, hardened fallbacks (disabled by default on 4090 per ADR).
- Workspaces / memory hygiene:
  - Varlen pack workspaces `_VARLEN_WS` and selection pack workspaces `_SEL_PACK_WS` to reduce allocations.  
  - Env audit/observability and fallback counters across branches; gate stats and selection stats maintained in `NSAAttention`.
- Decode caches and counters:
  - Correct emission schedule for compressed branch; per‑step predicted/actual reads tracked; causality assertions guarded by `NSA_STRICT_ASSERTS`.
- Training scripts:
  - Selective gradient checkpointing patterns present in `scripts/train_showcase.py` and FSDP variant.

## What’s Not Done (or only partial)
- Repeat_interleave hot paths:
  - `attention_bgh` and `_sdpa_full` still expand K/V via `.repeat_interleave(...)` before SDPA; this is both slow and memory‑fragile on long sequences.  
  - Some tests/utility paths still use repeat_interleave for shaping.
- Selection attention vectorization:
  - `grouped_selection_attention_packed` reduces overhead but still builds per‑row index lists and scatters in Python.  
  - `grouped_selection_attention_masked` builds the allowed mask with nested Python loops over (B,S,G,n).  
  - No single‑kernel/selective gather implementation; no FA‑2 varlen reuse for selection ranges.
- Kernel fusion:
  - No fusion of Gate MLP + softmax + branch combine; no RoPE+proj fusion.
- Mixed precision for selection scoring:
  - `compute_pcmp_all` runs in the ambient dtype; no autocast targeting BF16/FP16 for logits while preserving final selection stability.
- Algorithmic knobs:
  - Adaptive `n_sel` by seq length not implemented; no hierarchical selection stage.
- FA‑2 readiness checks:
  - `is_flash_available()`/`is_flash_varlen_available()` are permissive stubs; real import/version probes exist only inside try/except at call sites.

## Gaps/Risks
- If FA‑2 is unavailable, large portions of prefill/decoder fall back to SDPA paths that still use repeat_interleave, risking fragmentation and OOM on long contexts.  
- Selection attention remains a mix of Python control flow plus tensor ops, limiting GPU utilization at S=2k–8k even when scoring/mapping are vectorized.  
- Without fusions, gate and branch combine incur extra kernel launches and memory traffic.

## Recommended Path Forward (priority ordered)
1) Make FA‑2 the default on A100/H100 for cmp/win
   - Harden capability checks to real import+version gating; keep SM 8.9 guard unless forced.  
   - Keep `NSA_FA2_MIN_LEN_{WIN,CMP}` thresholds; add debug timing logs behind `NSA_DEBUG_TIMING`.
2) Remove repeat_interleave from SDPA fallbacks
   - In `attention_bgh`/`_sdpa_full`, expand heads via view/unsqueeze/expand and reshape without materializing copies.  
   - This lowers memory traffic even when FA‑2 is disabled and improves CPU parity paths.
3) Finish vectorized selection attention
   - Build allowed masks or flattened indices from `ranges` without Python loops (pure tensor ops).  
   - Prefer a single SDPA per (B,G*h) with additive mask, or pack to FA‑2 varlen by converting ranges→indices and reuse `attention_fa2_varlen` machinery.
4) Consolidate and pre‑size workspaces for decode
   - Promote `_VARLEN_WS`/`_SEL_PACK_WS` sizing to cover worst‑case seen during a run; avoid per‑step growth checks.  
   - Add lightweight telemetry to confirm reuse and absence of reallocs.
5) Gate MLP + combine fusion (safe, incremental)
   - TorchScript/compile a fused path for `fc1 → silu → fc2 → softmax → weighted_sum`.  
   - Keep last‑layer zero‑init invariant; add numerics checks in tests.
6) Mixed precision for selection scoring
   - Under `autocast`, run p_cmp logits in BF16/FP16 and upcast only for tie‑broken top‑k; verify with existing Eq.9/Eq.10 tests.  
   - Expectable 1.2–1.5× for large S/G/H.
7) Optional: Adaptive n_sel for short sequences
   - Clamp to 8/12 when S≤512/1024; keep invariants and counters consistent.  
   - Guard via config/env; off by default for determinism.

## Quick Validation Plan
- Unit: run existing FA‑2 parity suites with `NSA_TEST_FA2=1` on A100/H100.  
- Microbench: `bench/bench_fa2.py`, `bench/bench_decode.py` at S={128,512,2048}.  
- Stability: enable `NSA_STRICT_ASSERTS=1` on small S to re‑check causality after refactors.  
- Memory: sample `gpu_mem_alloc/reserved` heartbeat and confirm no alloc growth across long decode.

## Expected Impact
- Steps (1)–(3) should close most of the gap to the 300–800 tok/s target on A100 by eliminating slow fallbacks and Python control flow in selection.  
- Steps (5)–(6) add incremental wins and reduce kernel launch overhead in steady state.

