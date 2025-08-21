# M0 — Steel Thread (Consolidated Plan)

Status: Completed (per repo); this consolidates M0-Steel-Thread, M0-S2, M0-S3, and the M0–M3 Validation Test Plan in alignment with PRD.md.

## Scope & Goals
- SDPA-only implementation of three branches (compressed, selection, sliding) with learned gates.
- Fixed-length batches; strict causality; GQA group consistency (Eq.10).
- CSR mapping Eq.9 (cmp→sel fractional-overlap). Deterministic top‑n with forced block 0 + 2 locals; de‑dup and merge adjacent; clamp ≤ t.
- CPU fallback paths for all branches.

## Key Deliverables
- NSAAttention M0 forward paths (prefill + 1-step decode).
- Block indexing + mapping (`BlockMeta`, CSR/COO) with divisibility guards (d|l, d|l′).
- RoPE applied to Q and branch K before ϕ; M0 ϕ = average pooling over (l, d).
- Counters: exact decode read counts per step (cmp/sel/win + total).
- Observability: logs for gates, counters, selection ranges.

## Acceptance (from PRD)
- Causality: no branch reads indices > t.
- Eq.9 mapping property tests; Eq.10 group consistency tests.
- Small‑S equivalence (configure w ≥ S and n·l′ ≥ S): MAE < 1e‑5 FP32.
- Decode counter formula exact: num_cmp(S) = 0 if S<l else floor((S−l)/d)+1; reads(S) = num_cmp(S) + n·l′ + min(w,S).

## Execution Notes (M0‑S2/S3 amendments)
- Batched prefill: vectorize p_cmp→p_slc→group reduce and ranges for all t; remove per‑token loops in prefill.
- Masking semantics: boolean/additive masks used where `is_causal=True` is insufficient (cmp, varlen cases).
- Selection packing: bucket rows by gathered length L and run SDPA per bucket; parity with gather path.

## Tests (consolidated)
- Unit/property: `test_masks.py`, `test_block_math.py`, `test_group_consistency.py`.
- Equivalence: `test_equiv_small.py`.
- Counters: `test_decode_counters.py`.
- Selection packing parity: `test_selection_packed.py` (M0 fast path, optional).

## Flags & Defaults
- `NSA_FORCE_PARITY=1` enforces reference paths (CI parity).
- Fast paths default on: `NSA_USE_WIN_MASK=1`, `NSA_USE_CMP_MASK=1`, `NSA_USE_SEL_PACK=1`.
- No Triton in M0; FA‑2 optional only in benches (not tests).

## Files of Record
- Core: `nsa/core/{nsa_attention.py, block_index.py, compress_pool.py, selection_scorer.py, rope.py}`
- Tests: `nsa/tests/*` listed above.

## Outcome
All M0 criteria are implemented and validated; batched prefill and selection packing deliver parity with better performance while keeping SDPA-only semantics and CPU fallback intact.

