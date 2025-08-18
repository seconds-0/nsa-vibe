# M0-S2: Performance Fix & Validation Plan (Amended)
**Date**: January 18, 2025  
**Status**: Planning  
**Driver**: Lead Engineer  
**Scope**: Fix critical per-token loop bottleneck and establish performance validation infrastructure

## Executive Summary

The NSA implementation has a critical performance bottleneck in the prefill forward path caused by a per-token loop, yielding effectively O(S²) work. This plan refactors the prefill path into batched computations while preserving the M0 rules (SDPA only; deterministic tests; fixed sequence length) and the PRD’s selection semantics. We also add robust validation (numerical equivalence vs sequential reference, scaling benchmarks) and a structural long-context test.

## Problem Analysis

### Current State
Location: `nsa/core/nsa_attention.py` (sequential prefill)

```
for t in range(S):  # Sequential processing
    p_grp = p_grp_all[:, t]
    sel_ranges = select_topn_ranges(p_grp, kv.meta, self.n_sel, t, True, 2)
    # ... per-branch attention per token ...
```

### Impact
- O(S) sequential iterations, many small GPU kernel launches
- Non-optimal memory access patterns, underutilized kernels
- Sequences ≥ 1k tokens are too slow vs expectations

### Root Cause
Selection ranges depend on the query position due to causality. This does not require sequential processing; we can compute all ranges and masks in parallel and use SDPA with explicit masks to respect causality.

### Performance Measurements (est.)
- S=512: ~500ms (target ~50–100ms)
- S=1024: ~2000ms (target ~100–200ms)

## Solution Architecture

### Key Insight
Vectorize prefill: compute selection scores and ranges for all positions at once, build boolean/additive masks for SDPA, and run each branch with batched calls.

### Design Principles
1. Separation of concerns: distinct prefill vs decode paths
2. Batched by default: no per-token loops in prefill fast path
3. Hardware alignment: reuse SDPA; prepare for FA‑2 later (not in tests)
4. Maintainability: explicit masks and assertions; deterministic semantics

### Mask Semantics and Causality (Critical)
- SDPA mask rules (PyTorch):
  - Boolean mask: True means “disallowed/masked”, False means “allowed”.
  - Additive mask: 0 for allowed, −inf for disallowed.
- Selection and sliding masks must therefore mark allowed positions as False and disallowed as True.
- Compressed-branch causality cannot naively use `is_causal=True` when `S_q != S_kv` or when allowed compressed tokens vary by t. Build a per-row mask for compressed K/V (or pack per-row segments) and pass it as `attn_mask` to SDPA.

## Detailed Implementation Plan

### Phase 1: Batched Selection Infrastructure

#### 1.1 Batched Range Selection Function
File: `nsa/core/selection_scorer.py` (new function)

```python
def select_topn_ranges_batched(
    p_grp_all: torch.Tensor,  # [B,S,G,S_sel]
    meta: BlockMeta,
    n_top: int,
    S: int,
    force_init: bool = True,
    force_local: int = 2,
) -> torch.Tensor:  # [B,S,G,n_ranges,2]
    """
    Compute top-n block indices for all positions in parallel, then convert
    to merged contiguous ranges [start,end) with clamping to ≤ t+1.
    Deterministic tie-break: lower index wins on equal score.
    Forced blocks (0 and last k) are added, then excluded from scored top‑k.
    """
    B, S_q, G, S_sel = p_grp_all.shape
    device = p_grp_all.device

    sel_starts = meta.sel_starts.to(device)            # [S_sel]
    sel_ends = sel_starts + meta.l_sel                 # [S_sel]
    t_positions = torch.arange(S, device=device).view(S, 1)

    # Valid if block ends at or before t+1 (causal)
    valid = sel_ends.view(1, -1) <= (t_positions + 1)  # [S, S_sel], True=valid

    # Build disallowed mask (True=mask out) for masked_fill
    disallowed = ~valid  # [S,S_sel]
    masked_scores = p_grp_all.masked_fill(
        disallowed.view(1, S, 1, S_sel), float("-inf")
    )

    # Forced indices (deduped)
    forced_list: list[torch.Tensor] = []
    if force_init:
        init_blocks = torch.zeros((B, S, G, 1), dtype=torch.long, device=device)
        forced_list.append(init_blocks)
    if force_local > 0:
        tpos = torch.arange(S, device=device)
        # last k blocks aligned to selection block size
        last_block = (tpos // meta.l_sel).clamp_min(0)
        locals = [ (last_block - k).clamp_min(0) for k in range(force_local) ]
        locals = [ idx.view(1, S, 1, 1).expand(B, S, G, 1) for idx in locals ]
        forced_list.extend(locals)
    forced = (
        torch.cat(forced_list, dim=-1).unique(sorted=True, dim=-1)
        if forced_list else torch.empty((B,S,G,0), dtype=torch.long, device=device)
    )

    # Exclude forced from scored candidates
    if forced.numel() > 0:
        forced_mask = torch.zeros_like(masked_scores, dtype=torch.bool)
        forced_mask.scatter_(-1, forced, True)
        masked_scores = masked_scores.masked_fill(forced_mask, float("-inf"))

    # Deterministic top-k with tie-break on lower index.
    # Two-pass: first get indices sorted by index asc (stable base), then argsort by score desc.
    k_rest = max(0, n_top - forced.shape[-1])
    if k_rest > 0:
        # Base order by index asc
        base_idx = torch.arange(S_sel, device=device).view(1,1,1,S_sel)
        base_idx = base_idx.expand(B,S,G,S_sel)

        # Sort scores descending, tie-break via index asc by constructing a composite key
        # Use a small epsilon to bias by index deterministically without changing ordering materially
        eps = torch.finfo(masked_scores.dtype).eps
        bias = (base_idx.to(masked_scores.dtype) * eps)
        composite = masked_scores + bias  # still -inf where disallowed/forced
        _, top_idx = torch.topk(composite, k=min(k_rest, S_sel), dim=-1, largest=True)

        selected_idx = top_idx
        if forced.shape[-1] > 0:
            selected_idx = torch.cat([forced, top_idx], dim=-1)
    else:
        selected_idx = forced[..., :n_top]

    # Sort indices asc, dedup
    selected_idx = torch.sort(selected_idx, dim=-1).values

    # Convert indices to merged ranges (exclusive end), clamped to ≤ t+1
    ranges = convert_indices_to_ranges_batched(selected_idx, meta, S)
    return ranges  # [B,S,G,n_ranges,2]
```

#### 1.2 Helper: Convert Indices to Ranges (Merged)
File: `nsa/core/selection_scorer.py` (helper)

```python
def convert_indices_to_ranges_batched(
    indices: torch.Tensor,  # [B,S,G,k]
    meta: BlockMeta,
    S: int,
) -> torch.Tensor:  # [B,S,G,n_max,2]
    """
    Convert block indices to contiguous [start,end) ranges per (B,S,G).
    Merge adjacent blocks; clamp end to ≤ t+1; deterministic ordering.
    Note: Vectorization is a later milestone; acceptable to loop here given small k.
    """
    B, S_q, G, k = indices.shape
    device = indices.device
    sel_starts = meta.sel_starts.to(device)

    ranges_list = []
    for b in range(B):
        for t in range(S_q):
            clamp_end = int(t) + 1
            for g in range(G):
                block_ids = indices[b, t, g].tolist()
                # Expand to [start,end) and merge
                spans = []
                for bid in block_ids:
                    start = int(sel_starts[bid].item())
                    end = start + int(meta.l_sel)
                    end = min(end, clamp_end)
                    if end <= start:
                        continue
                    if not spans or start > spans[-1][1]:
                        spans.append([start, end])
                    else:
                        spans[-1][1] = max(spans[-1][1], end)
                ranges_list.append(spans)

    max_ranges = max(len(r) for r in ranges_list) if ranges_list else 0
    out = torch.zeros((B, S_q, G, max_ranges, 2), dtype=torch.int32, device=device)
    idx = 0
    for b in range(B):
        for t in range(S_q):
            for g in range(G):
                spans = ranges_list[idx]
                for i, (s0, e0) in enumerate(spans):
                    out[b, t, g, i, 0] = s0
                    out[b, t, g, i, 1] = e0
                idx += 1
    return out
```

### Phase 2: Batched Attention Kernels (SDPA-only for M0)

File: `nsa/core/attention_kernels.py` (new)

```python
def batched_causal_attention_compressed(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K_cmp: torch.Tensor,  # [B,G,S_cmp,Dk]
    V_cmp: torch.Tensor,  # [B,G,S_cmp,Dv]
    l: int,
    d: int,
) -> torch.Tensor:  # [B,S,G,h,Dv]
    """
    Compressed branch with per-row causal mask. Cannot use is_causal=True because
    S_q != S_kv and allowed tokens vary by t.
    """
    B, S, G, h, Dk = Q.shape
    S_cmp = K_cmp.shape[2]
    device = Q.device

    # num_cmp(t) = 0 if t+1 < l else floor((t+1 - l) / d) + 1
    tpos = torch.arange(S, device=device)
    num_cmp = torch.where(tpos + 1 < l, 0, ((tpos + 1 - l) // d) + 1).clamp(max=S_cmp)
    # Build boolean mask [S, S_cmp]: True = disallowed
    col = torch.arange(S_cmp, device=device).view(1, S_cmp)
    disallowed = col >= num_cmp.view(S, 1)

    # Flatten to SDPA shapes and apply mask
    Qf = Q.reshape(B*G, S, h, Dk).transpose(1, 2)         # [B*G,h,S,Dk]
    Kf = K_cmp.unsqueeze(2).expand(-1,-1,h,-1,-1).reshape(B*G, h, S_cmp, Dk)
    Vf = V_cmp.unsqueeze(2).expand(-1,-1,h,-1,-1).reshape(B*G, h, S_cmp, V_cmp.shape[-1])
    Mf = disallowed.unsqueeze(0).unsqueeze(0).expand(B*G, h, S, S_cmp)

    Of = torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=Mf)
    O = Of.transpose(1, 2).reshape(B, S, G, h, V_cmp.shape[-1])
    return O


def sliding_window_attention(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K: torch.Tensor,  # [B,G,S,Dk]
    V: torch.Tensor,  # [B,G,S,Dv]
    w: int,
) -> torch.Tensor:  # [B,S,G,h,Dv]
    B, S, G, h, Dk = Q.shape
    device = Q.device
    row = torch.arange(S, device=device).view(S, 1)
    col = torch.arange(S, device=device).view(1, S)
    allowed = (row >= col) & (row - col < w)
    disallowed = ~allowed  # True = mask out

    Qf = Q.reshape(B*G, S, h, Dk).transpose(1, 2)
    Kf = K.unsqueeze(2).expand(-1,-1,h,-1,-1).reshape(B*G, h, S, Dk)
    Vf = V.unsqueeze(2).expand(-1,-1,h,-1,-1).reshape(B*G, h, S, V.shape[-1])
    Mf = disallowed.unsqueeze(0).unsqueeze(0).expand(B*G, h, S, S)

    Of = torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=Mf)
    O = Of.transpose(1, 2).reshape(B, S, G, h, V.shape[-1])
    return O


def grouped_selection_attention(
    Q: torch.Tensor,      # [B,S,G,h,Dk]
    K: torch.Tensor,      # [B,G,S_kv,Dk]
    V: torch.Tensor,      # [B,G,S_kv,Dv]
    ranges: torch.Tensor, # [B,S,G,n,2]
) -> torch.Tensor:       # [B,S,G,h,Dv]
    B, S, G, h, Dk = Q.shape
    S_kv = K.shape[2]
    device = Q.device

    # Build boolean mask [B,S,G,S_kv], False=allowed, True=disallowed
    mask = torch.ones((B, S, G, S_kv), dtype=torch.bool, device=device)
    n = ranges.shape[3]
    for b in range(B):
        for t in range(S):
            for g in range(G):
                for i in range(n):
                    s0 = int(ranges[b,t,g,i,0].item())
                    e0 = int(ranges[b,t,g,i,1].item())
                    if e0 > s0:
                        mask[b, t, g, s0:e0] = False

    Qf = Q.reshape(B, S, G*h, Dk).transpose(1, 2)             # [B,G*h,S,Dk]
    Kf = K.unsqueeze(2).expand(-1,-1,h,-1,-1).reshape(B, G*h, S_kv, Dk)
    Vf = V.unsqueeze(2).expand(-1,-1,h,-1,-1).reshape(B, G*h, S_kv, V.shape[-1])
    Mf = mask.reshape(B, S, G*h, S_kv).transpose(1, 2)         # [B,G*h,S,S_kv]

    Of = torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=Mf)
    O = Of.transpose(1, 2).reshape(B, S, G, h, V.shape[-1])
    return O
```

Notes:
- Heads are expanded only at the last moment to reduce memory pressure.
- FA‑2 is intentionally not used in tests for M0. We will add a flag for benchmarks in later milestones.

### Phase 3: Refactor NSAAttention Forward Pass

File: `nsa/core/nsa_attention.py`

```python
def forward(self, x: torch.Tensor, kv: NSA_KV, *, prefill: bool):
    if prefill:
        return self._forward_prefill_batched(x, kv)
    else:
        return self._forward_decode_step(x, kv)

def _forward_prefill_batched(self, x: torch.Tensor, kv: NSA_KV):
    """Vectorized prefill: projections, caches, batched scores, batched ranges, batched SDPA per branch, gating, output."""
    # 1) Projections with RoPE(Q), RoPE(K before ϕ), cache updates (sel/win/cmp)
    # 2) p_cmp_all (einsum), p_slc_all (CSR map, batched), p_grp_all (sum)
    # 3) select_topn_ranges_batched → [B,S,G,n,2]
    # 4) O_cmp via batched_causal_attention_compressed (per-row mask)
    # 5) O_sel via grouped_selection_attention (block masks)
    # 6) O_win via sliding_window_attention (window mask)
    # 7) Gate (softmax(g/τ)) and combine, output projection
    # Add asserts for shapes and determinism where relevant
```

Decode path remains sequential by design (one token), retaining existing correctness and counters.

### Phase 4: Benchmark Suite

Files: `bench/bench_prefill.py`, `bench/bench_decode.py`

Adjustments:
- Print device and dtype; support CPU fallback
- Synchronize only when CUDA available
- Analyze scaling ratios (2× S → ~2× time)

### Phase 5: Long-Context Structural Validation (Needle)

File: `nsa/tests/test_needle.py`

Replace probabilistic similarity test with structural selection validation:
- Force selection branch (e.g., `NSA_FORCE_BRANCH=sel` or gate override), or directly analyze selection masks/ranges.
- Construct synthetic tensors so a known “needle” block attains the highest selection score (e.g., set K/Q such that p_cmp peaks at the needle’s block; or inject a synthetic `p_grp_all`).
- Assert that `select_topn_ranges_batched` includes the needle block in the ranges for downstream positions and that the attention mass over the selection mask concentrates on that block (> given threshold).
- Parameterize over sequence lengths (up to 8k) and depths.

### Phase 6: Validation & Integration Testing

- Batched vs sequential equivalence: keep the previous sequential prefill method (private) to compare outputs; assert MAE < 1e‑5.
- Determinism fixture: add `conftest.py` setting seeds and `torch.backends.cudnn.benchmark = False`.
- M0 rule checks: SDPA-only in tests; fixed sequence length; no Triton/FA‑2 imports active in tests.

## Success Metrics

### Critical (Must Pass)
- [ ] All existing tests pass (M0 rules)
- [ ] Per-token loop eliminated from prefill
- [ ] Prefill shows ~linear scaling (2× S → ~2× time)
- [ ] Decode token reads match formula exactly

### Important (Should Pass)
- [ ] Structural needle test >90% success at 8k
- [ ] Performance within ~3× of dense SDPA baseline at equal dims
- [ ] No memory leaks or OOM; mask builds bounded by `n_sel` and reasonable S

### Nice to Have
- [ ] FA‑2 integration stubs behind flags (benchmarks only)
- [ ] 64k context support via packed selection (future milestone)
- [ ] Triton block-sparse prototype (future milestone)

## Risk Analysis & Mitigation

### Risk 1: Mask Semantics Errors
Probability: Medium · Impact: High  
Mitigation: Unit tests for mask truth tables; shape asserts; equivalence vs sequential.

### Risk 2: Deterministic Top‑k Tie‑Break
Probability: Medium · Impact: Medium  
Mitigation: Two-pass deterministic selection; tests with equal-score cases.

### Risk 3: Memory Pressure from Masks
Probability: Medium · Impact: Medium  
Mitigation: Build masks per (B,G) slice; avoid head expansion until needed; keep `n_sel` modest; plan packing for later milestones.

### Risk 4: Performance Regression
Probability: Low · Impact: High  
Mitigation: Keep sequential fallback for A/B; benchmark at each step; CI sanity checks.

## Execution Checklist (Status-Driven)

- [ ] Implement `select_topn_ranges_batched` with deterministic tie‑break and dedup
- [ ] Implement `convert_indices_to_ranges_batched` with merge + clamp
- [ ] Add `attention_kernels.py` with corrected SDPA mask semantics for all branches
- [ ] Refactor `_forward_prefill_batched` and preserve sequential reference
- [ ] Add determinism fixture (`conftest.py`)
- [ ] Add batched vs sequential equivalence test
- [ ] Add scaling benchmark scripts (CPU/GPU aware)
- [ ] Add structural needle test (forced selection; synthetic tensors)
- [ ] CI: uv + ruff + mypy + pytest on CPU

## Testing Protocol

```bash
# Baseline
uv run -q pytest -xvs --tb=short > baseline_tests.txt

# After Phase 1 (selection)
uv run -q pytest -xvs -k "selection or equiv"

# After Phase 3 (forward refactor)
uv run -q pytest -xvs

# Benchmarks
uv run python bench/bench_prefill.py
uv run python bench/bench_decode.py

# Needle (long)
uv run -q pytest -xvs -k test_needle -m long

# Final validation
uv run -q pytest --tb=short --cov=nsa
```

## Acceptance Criteria

M0‑S2 is complete when:
1. ✅ Prefill loop removed; batched path in place
2. ✅ All original tests pass under M0 constraints; new equivalence tests pass
3. ✅ Prefill shows ~linear scaling
4. ✅ Decode counters equal formula; branch breakdowns consistent
5. ✅ Structural needle test passes at 8k

Ready for M1 when:
1. ✅ M0‑S2 complete
2. ✅ Performance documented and baselined
3. ✅ FA‑2 stubs in place and gated (not used in tests)
4. ✅ Architecture supports block-sparse/packed kernels later


