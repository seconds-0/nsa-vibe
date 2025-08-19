### Task ID
M3-Full-Decode-Caches

### Problem Statement
PRD milestone M3 requires completing the full decode cache execution model so decode matches the paper’s order of operations and acceptance math. We must solidify branch‑specific caches, compressed emission scheduling, selection of raw tokens at decode, gating, and exact token‑read counters per step. Implementation must remain deterministic and safe on CPU (SDPA‑only) while engaging FA‑2 on GPU where supported.

### Goals (What M3 Delivers)
- Correct and complete decode step for NSA with three branches (cmp/sel/win) and learned gates.
- Proper cache ownership/layout per branch: `K_sel,V_sel` (selection raw), `K_win,V_win` (sliding), `K_cmp,V_cmp` (compressed stream).
- Compressed emission schedule: emit one compressed token from the last `l` raw tokens every `d` steps once warmed by `l`.
- Selection at decode: compute `p_cmp` against available compressed tokens, map via Eq. 9 to selection blocks, group‑reduce via Eq. 10, choose deterministic top‑n with forced block 0 and two local blocks, de‑dup and merge to contiguous `[start,end)` ranges, clamp ≤ t.
- Gating and output: compute branch outputs and gated sum (Eq. 5), returning `[B,1,dim]` per decode step.
- Exact token‑reads counters: `num_cmp(S) = 0 if S < l else floor((S - l)/d) + 1` and `reads(S) = num_cmp(S) + n*l′ + min(w, S)`. Track predicted and actual reads per step.
- Observability: optional logs for branch tokens read, gate stats, selection distances, and per‑branch timings.

### Non‑Goals (M3)
- Selection Triton kernel (forward/backward) — M4/M5.
- Multi‑token decode batching beyond 1‑step — future.
- Distributed decode server — future.

### Proposed Solution

1) Decode Step Ordering (align PRD §5.2)
- RoPE: apply RoPE to Q and K before attention; for compressed, apply RoPE to K before ϕ; ϕ runs on RoPE‑K/V (PRD RoPE & ϕ rule).
- Sliding: read last `min(w, t)` tokens from `K_win,V_win`.
- Compressed: if warmed (`S_raw ≥ l`) and `(S_raw - l) % d == 0`, emit one compressed token from last `l` raw tokens using ϕ (M0 avg‑pool; later variants per config). Append to `K_cmp,V_cmp`.
- Selection: compute `p_cmp` using Q against available `K_cmp`, map via CSR `M_csl` (fractional‑overlap weights, Eq. 9; rebuild only when `(l,d,l′)` change) to selection scores, group‑reduce (Eq. 10), deterministically select top‑n (forced block 0 + 2 local), de‑dup and merge to contiguous ranges, clamp ranges ≤ t, then gather raw `K_sel,V_sel` within those ranges for SDPA.
- Gate & sum: apply Gate MLP on group‑pooled Q, softmax(g/τ), and sum branch outputs.

2) Caches & Ownership (PRD §6.2)
- Selection raw cache `K_sel,V_sel` append raw token projections each step.
- Sliding cache `K_win,V_win` append and slide to last `w` tokens.
- Compressed stream `K_cmp,V_cmp` append upon emission schedule; track raw compressed sequence indices to drive the schedule.
- Layout: `[B,G,S,D*]`, contiguous in D, group‑major; per‑layer ownership.

3) Counters & Acceptance (PRD §11C; Always Rules)
- Implement predicted and actual per‑step reads with the exact integer formula; track per‑branch actual reads (sel/cmp/win) and totals.
- Test early‑step handling where `S_raw < l` and `S_raw < w`.

4) FA‑2 decode (GPU) & CPU fallback
- Use existing `sliding_window_attention_fa2_decode` and `compressed_attention_fa2_decode` with thresholds and SDPA fallback for tiny lengths or unsupported devices.
- On CPU: SDPA‑only reference path with `is_causal=True` must be used; all decode tests pass.

5) Determinism & Safety
- Clamp all selection ranges ≤ t; no branch may read future indices.
- Deterministic tie‑break for top‑n (lower index on ties).
- Fixed seeds; no dropout in tests; disable `cudnn.benchmark` in CI.

6) Observability
- When `NSA_DEBUG_LOG=1`, log per‑step: tokens read per branch, gate means/std, selection distances (mean, max), and optional per‑branch timings. Optionally emit heatmaps of `p_cmp`/`p_slc` in debug modes.

7) Edge Cases & Guards
- `S_raw < l`: `num_cmp=0`; compressed attention returns zeros.
- `w=0` or `S_raw=0`: sliding returns zeros.
- Empty selection: return zeros.
- Empty buckets in packing paths: no‑ops.

### Automated Test Plan
- Unit/Property:
  - Decode counters: predicted vs actual equals per‑step and cumulative; verify early‑step cases (`S_raw < l`, `S_raw < w`).
  - Selection correctness at decode: forced block 0 + 2 local, de‑dup/merge, deterministic ties.
  - Group consistency at decode (Eq. 10): identical selected ranges across heads within a GQA group.
  - Causality: assert no reads > t across branches; use masked SDPA checks on tiny shapes.
- Integration:
  - Small‑S decode equivalence: NSA with `w ≥ S`, `n*l′ ≥ S` ≈ full attention (MAE < tol FP32).
  - Compressed emission correctness: exact steps when new compressed tokens appear; index math validated.
  - CPU fallback: all decode tests pass with SDPA‑only.
- Optional GPU (opt‑in):
  - FA‑2 decode parity on tiny grids with tolerances (FP32 ≤ 5e‑5; BF16/FP16 ≤ 2e‑4 where run).

### Components Involved
- `nsa/core/nsa_attention.py`: finalize `_forward_decode_step` & ensure cache updates; counters; logging hooks; FA‑2 decode helpers wiring.
- `nsa/cache/kv_cache.py`: verify append APIs; ensure invariant checks.
- `nsa/core/selection_scorer.py`: reuse batched selection logic for single‑step; ensure deterministic tie‑breaks and merges.
- `nsa/core/attention_kernels.py`: confirm decode helpers safe thresholds and fallbacks; causal behavior.
- `nsa/tests/`: strengthen `test_decode_step.py`, `test_decode_counters.py`, `test_equiv_small.py`, add decode‑specific group‑consistency and causality tests.

### Dependencies
- M1 thresholds (FA‑2) pending GPU benches — proceed with current conservative defaults; keep `NSA_FORCE_PARITY` to disable FA‑2 in CI.

### Implementation Checklist
- [ ] Decode step cache updates: selection append, sliding window slide, compressed emission schedule (+ϕ on last `l`).
- [ ] Selection at decode: `p_cmp → p_slc (CSR) → group‑reduce → top‑n (det, forced) → de‑dup/merge → ranges`.
- [ ] Branch attentions at decode: SDPA on CPU; FA‑2 decode helpers on GPU with small‑length thresholds.
- [ ] Gate & output assembly; return `[B,1,dim]` with residual wiring left to caller.
- [ ] Counters: predicted and actual per‑step reads; per‑branch breakdown; unit asserts.
- [ ] Causality & group‑consistency asserts (tests): no reads > t; identical selected ranges across heads in a GQA group.
- [ ] Small‑S equivalence: configure coverage to match full attention; MAE ≤ tol FP32.
- [ ] Observability: env‑gated logs (`NSA_DEBUG_LOG`/`NSA_DEBUG_TIMING`).
- [ ] CPU fallback and deterministic seeds in CI; ensure tests green.

### Verification Steps
- Run default suite (CPU) → green.
- Run optional GPU decode parity: `NSA_TEST_FA2=1` with tiny grids; confirm tolerances.
- Inspect logs for read counters and selection distances on a small scripted decode run.

### Decision Authority
- Engineering owns decode cache integration and acceptance math; Product signs off when counters/tests match PRD and small‑S decode equivalence passes.

### Questions / Uncertainties
- None blocking; FA‑2 thresholds to be finalized via benches (dangling thread from M1/M2).

### Acceptable Tradeoffs
- Prefer clarity and correctness over micro‑perf at this stage; keep SDPA or per‑token reference path available for parity.

### Status
Not Started

### Notes
- Keep `NSA_FORCE_PARITY=1` as an escape hatch in CI until FA‑2 decode parity is validated on GPU.
