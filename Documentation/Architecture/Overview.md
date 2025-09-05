# NSA Architecture Overview

Purpose
- Canonical overview of the implemented NSA (Native Sparse Attention) design, invariants, and defaults. Supersedes PRD.md for day‑to‑day correctness references; PRD.md remains historical.

## Critical Performance Requirements

**Minimum Viable Production Flags**:
```bash
export NSA_FORCE_SEL_MASK=1     # MANDATORY: Enables 9,200+ toks/s (vs 0)
export NSA_PREFILL_BATCHED=1    # 10-15% throughput improvement
# SDPA flash is the default fast path. FA‑2 is optional/guarded and OFF by default.
# Enable FA‑2 only when perf sweeps show wins; otherwise omit these flags.
# (If enabling experimentally)
# export NSA_USE_FA2=1
# export NSA_USE_FA2_WIN=1
# export NSA_USE_FA2_CMP=1
# export NSA_FA2_MIN_LEN_WIN=8192   # set high to avoid accidental use
# export NSA_FA2_MIN_LEN_CMP=8192
```

**Expected Performance Baselines** (A100 80GB, 117M model, S=2048):
- With gradient checkpointing: 9,200 toks/s at 3GB memory
- Without gradient checkpointing: 16,000 toks/s at 8GB memory  
- Batch size 16 (optimal): 23,100 toks/s at 55GB memory

Summary
- Three causal branches — Compressed (cmp), Selected (sel), and Sliding window (win) — combined by a learned gate (softmax, τ=1.0; last layer zero‑init). All branches are strictly causal and respect GQA group consistency.
- SDPA flash is the default fast path. FA‑2 is optional, guarded, and OFF by default. Selection remains SDPA (masked/packed). All FA‑2 calls have hard SDPA fallbacks.
- Observability is built‑in: per‑branch read counters, gate stats, and debug heatmaps when enabled.
- **Complexity Reduction**: O(S²) → O(n_sel·l_sel + w + compressed), achieving ~40x reduction at S=2048.

Branches
- **Compressed (cmp)**: Emits compressed K/V blocks after a warmup. Prefill can tile to bound memory. Decode updates follow the emission schedule (every d after initial l).
  - *Performance Impact*: Reduces KV cache by factor of d/l (typically 2x)
- **Selected (sel)**: Chooses non‑overlapping selection blocks and gathers K/V to run SDPA over ranges. Group‑consistent decisions shared across heads in a KV group.
  - *Performance Impact*: Critical path - must use `NSA_FORCE_SEL_MASK=1` for masked attention optimization
  - *Decode Efficiency*: GQA group consistency enables single selection computation per group
- **Sliding window (win)**: Causal window of recent tokens of width w with SDPA masking.
  - *Performance Impact*: Constant-time attention to recent context, independent of sequence length

Selection Semantics (key rules)
- Eq.9 mapping: Build a CSR mapping from compressed blocks to selection blocks with fractional‑overlap weights.
- Eq.10 group consistency: Reduce per‑head scores to per‑group scores and select identically across heads in the group.
- Deterministic top‑n: Break ties by lower index; always include forced block 0 and two local blocks; de‑duplicate; merge adjacent; clamp to ≤ current time t.

Gating
- Gate MLP outputs logits per branch; last linear layer is zero‑initialized; softmax temperature τ=1.0. Log means/std for collapse detection.
- Debug env: `NSA_FORCE_BRANCH={cmp|sel|win}` sets one‑hot gating for tests.
- *Performance Note*: Gate collapse (entropy < 0.5) indicates training issues - monitor via heartbeat logs

Caches and Masks
- Separate decode caches: `K_win,V_win` and `K_sel,V_sel`. Compressed path stores compressed K/V (and raw for prefill where needed).
- Strict causality masks for all branches; defensive clamping of any index/range to ≤ t.

Counters (decode memory/reads)
- num_cmp(S) = 0 if S < l else floor((S − l)/d) + 1
- reads(S) = num_cmp(S) + n·l′ + min(w, S)
- Expose per‑step values for observability and tests.

Defaults and Constraints
- Blocks: l=32, d=16, l′=64, n=16 (forced initial + 2 local included), w=512.
- GQA: G=4, H=64, d_k=192, d_v=128.
- Divisibility: enforce d | l and d | l′.

Execution Policy
- GPU (production): Use SDPA flash by default. FA‑2 is optional and disabled by default; enable only where device/head_dim/length sweeps demonstrate clear wins. Selection remains SDPA (masked/packed). On any FA‑2 guard failure, fallback to SDPA automatically.
- CPU/CI: SDPA only, deterministic path. FA‑2 disabled in CI.

Observability and Debug
- Gate stats (mean/std), per‑branch token reads, optional NVTX ranges for prefill stages, selection mapping tracing.
- Key env flags: see Execution Routing.

Where to Go Next
- **Performance optimization**: [Performance-Critical Flags](../Operations/Performance-Critical-Flags.md)
- **Troubleshooting**: [Troubleshooting Index](../Troubleshooting/Index.md)
- Execution routing and flags: Documentation/Architecture/execution-routing.md
- Selection details/spec: docs/NSA_CHUNKED_SELECTION_SPEC.md
- Test mapping: Documentation/Tests/Index.md
- Decisions/ADRs: Documentation/Decisions/INDEX.md
