# M0 — Steel Thread Workplan (NSA)

- **Task ID**: M0-Steel-Thread
- **Status**: Completed
- **Driver**: Lead Engineer (you)
- **Scope**: Minimal, verifiable end-to-end NSA path using SDPA everywhere, avg-pool ϕ, prefill + 1-step decode, with tests and counters from day 1.

## Problem Statement
Deliver a working NSA module that can replace full attention in a LLaMA-style block for prefill and 1-step decode. It must implement the three branches (Compressed, Selected, Sliding), GQA-consistent selection, gating, causal safety, and the compressed→selection scoring pathway from the paper (Eqs. 7–12), without custom kernels yet. It must be validated against correctness properties and a deterministic equivalence configuration.

## Proposed Solution (concrete and executable)
- Implement `NSAAttention` with shared Q (RoPE) and per-branch K/V projections.
- Branch behavior:
  - Compressed (cmp): ϕ = average pooling over overlapped blocks (l, d). Apply RoPE to K before ϕ.
  - Selected (slc): reuse p_cmp; map with sparse CSR `M_csl` (fractional overlap weights); group-reduce within GQA; deterministic top‑k with forced blocks; de‑dup + merge adjacent into contiguous [start,end).
  - Sliding (win): last w tokens using distinct K/V projections and separate cache.
- Gating: group-pooled Q → MLP(Dk→Dk/2→3), last layer zero-init; output gates via softmax(g/τ), default τ=1.0.
- Caches: per-layer ownership; layout `[B,G,S,D*]` (contiguous in D, group‑major).
  - `K_sel,V_sel` for selection; `K_win,V_win` for sliding; `K_cmp,V_cmp` for compressed stream; `win_ptr`, `cmp_emit_next` metadata.
- Compute:
  - SDPA everywhere (cmp/sel/win). No Triton in M0. FA‑2 not required (M1).
  - p_cmp computed in tiles over compressed time for prefill; deterministic flags in tests.
  - Batched prefill (no per-token loop): compute `p_cmp` for all tokens, map to `p_slc` via CSR in batch, top‑n & range merge in batch, run batched SDPA per branch, then gate.
- Causality: strictly clamp indices ≤ t; apply branch-level causal masks.
- Fixed-seq batches in M0 (no varlen). Varlen deferred to M6.

## Automated Test Plan
- Unit / Property (fast):
  - Causality: no branch reads indices > t.
  - Eq. 9 mapping: for random (l,d,l′) obeying d|l and d|l′, verify `p_slc == p_cmp @ M_csl` against a reference overlap sum. CSR weights are fractional overlaps.
  - Eq. 10 group consistency: selected ranges identical across heads within a GQA group.
  - Shapes/dtypes across BF16/FP32; FP32 path for reference.
- Equivalence (deterministic):
  - Full-coverage ≈ full attention: with `w ≥ S` and `n*l′ ≥ S`, MAE < 1e-5 FP32 (after gating merge).
- Long-context functional (smoke in M0):
  - Decode token-reads counter: compare actual per-step reads to exact formula:
    - `num_cmp(S) = 0 if S < l else floor((S - l)/d) + 1`
    - `reads(S) = num_cmp(S) + n*l′ + min(w, S)`
    - Document behavior when `S < l` or `S < w`.
- Observability (debug): heatmaps for `p_cmp`/`p_slc`, selected ranges overlay, per-branch gate means/std, selection distance histogram, per-branch tokens-read counters.

## Components Involved
- `nsa/core/nsa_attention.py` (M0 implementation)
- `nsa/core/compress_pool.py` (avg pooling ϕ)
- `nsa/core/block_index.py` (build blocks, precompute CSR `M_csl`)
- `nsa/core/selection_scorer.py` (p_cmp→p_slc→group-reduce→top‑n with rules)
- `nsa/cache/kv_cache.py` (selection/sliding/compressed caches)
- `nsa/core/rope.py` (RoPE utilities)
- `nsa/tests/*` (new tests listed above)
- `nsa/cli/demo_infer.py` (optional M0 demo)

## Dependencies
- PyTorch ≥ 2.3 (SDPA, RoPE, BF16/FP32); Triton not required in M0.
- No FA‑2 required in M0 (introduced in M1).
- Determinism: set seeds; disable `cudnn.benchmark`; no dropout in tests.

## Implementation Checklist
- [ ] Block meta and mapping
  - [ ] Build compression blocks (overlap) and selection blocks (size l′), causal indices per t.
  - [ ] Build CSR `M_csl` (cmp→sel) with fractional overlap weights; device FP32; rebuild only when (l,d,l′) change.
  - [ ] Unit tests for Eq. 9 mapping properties and divisibility guards (default enforce `d|l` and `d|l′`).
- [ ] Compression ϕ (avg pooling)
  - [ ] Apply RoPE to K per branch before ϕ.
  - [ ] Implement overlapped avg pooling over `[i*d : i*d+l)`; prefill and decode emission rule: emit every d once warmed by l.
  - [ ] Unit tests: stream length; masking; determinism vs simple reference.
- [ ] Selection scorer
  - [ ] Compute p_cmp = softmax(Q·K_cmp^T) (tiled for prefill).
  - [ ] Map via CSR `M_csl`; group-reduce across heads in GQA group.
  - [ ] Top‑n rules: deterministic tie-break (lower index), force block 0 + 2 local, de‑dup, merge adjacent to contiguous `[start,end)`; clamp ≤ t.
  - [ ] Unit tests: Eq. 10 consistency; top‑n stability and invariants.
- [ ] Sliding branch
  - [ ] Maintain `K_win,V_win` via ring buffer; window `[max(0, t-w+1), t]`.
  - [ ] SDPA over window; independent K/V projections.
- [ ] NSAAttention wiring
  - [ ] Shared Q with RoPE; per-branch K/V.
  - [ ] Three SDPA calls (cmp/sel/win) and gate merge.
  - [ ] Batched prefill path without token loop.
  - [ ] Gate MLP (Dk→Dk/2→3), last layer zero-init, softmax(g/τ) with τ=1.0.
  - [ ] Causal masks across branches.
  - [ ] CPU fallback: selection gathers then SDPA; cmp/win use SDPA.
  - [ ] Decode step: single-token mode with compressed emission schedule (every d after warmup l), sliding ring buffer, and per-step reads counters matching formula.
- [ ] Tests & harnesses
  - [ ] `test_masks.py`, `test_block_math.py`, `test_group_consistency.py`, `test_equiv_small.py`, `test_decode_counters.py` (M0 subset).
  - [ ] Deterministic configuration fixtures; golden seeds.
  - [ ] Optional: minimal `demo_infer.py` printing gates and selected ranges.

## Verification Steps
- Run fast tests: `pytest -q` (unit/property + equivalence small‑S).
- Run counters smoke: `pytest -q -k decode_counters` and verify exact integer formula.
- Visual sanity: dump heatmaps/histograms for a small S; confirm selection is causal and sparse; gates near uniform at init.

## Completion Evidence
- Core implementation: `nsa/core/nsa_attention.py`, `nsa/core/selection_scorer.py`, `nsa/cache/kv_cache.py`.
- Tests passing in CI (CPU):
  - `nsa/tests/test_equiv_small.py` (small‑S equivalence)
  - `nsa/tests/test_decode_counters.py`, `nsa/tests/test_decode_step.py` (reads and decode ordering)
  - `nsa/tests/test_block_math.py`, `nsa/tests/test_masks.py` (Eq. 9 mapping, causality)
  - `nsa/tests/test_group_consistency.py`, `nsa/tests/test_group_consistency_sel.py` (GQA consistency)

## Decision Authority
- You (Lead Engineer) may decide independently:
  - Gate MLP dims, zero-init, temperature default (τ=1.0).
  - CSR `M_csl` storage format and numeric dtype; enforcing `d|l` and `d|l′` by default.
  - Tiling sizes for prefill p_cmp computation; determinism flags.
  - Cache memory layout and per-layer ownership.
- Requires product/UX lead input before changing:
  - Gating activation family (softmax ↔ sigmoid) or temperature schedule.
  - Var‑length batch support priority (bringing forward from M6).
  - Any deviation from paper defaults (l,d,l′,n,w) for baseline benchmarks.

## Questions / Uncertainties
- Blocking: None for M0.
- Non‑blocking (documented assumptions):
  - Deterministic tie-breaker: lowest index wins on equal scores.
  - CSR mapping fractional weights defined by token-count overlap; this matches Eq. 9 intent.
  - CPU fallback performance is acceptable for CI/correctness.

## Acceptable Tradeoffs (M0)
- Use SDPA instead of FA‑2 (perf is not a goal yet).
- Compute p_cmp in tiles for determinism and memory safety, even if slightly slower.
- Fixed sequence length batches only (varlen deferred to M6).

## Notes
- Follow PRD defaults: `l=32, d=16, l′=64, n=16, w=512`; GQA `G=4`, `H=64`, `d_k=192`, `d_v=128`.
- Keep code paths explicit and configuration-driven; avoid hidden global state.
- Log per-branch tokens-read and gate statistics to catch early collapse.

---

### Execution Commands (M0)
- Env: `uv venv -p 3.11 .venv && uv pip sync -r requirements.txt`
- Fast tests: `uv run -q pytest`
- Counters (long-context smoke): `uv run -q pytest -k decode_counters`
- Optional demo: `uv run python cli/demo_infer.py --config configs/base.yaml`
