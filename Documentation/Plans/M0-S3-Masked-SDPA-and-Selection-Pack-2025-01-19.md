### Task ID
M0-S3-Masked-SDPA-and-Selection-Pack

### Problem Statement
- Masked SDPA paths for sliding and compressed branches currently delegate to per-token reference implementations to guarantee numerics. We need true masked varlen SDPA that exactly matches truncation numerics without delegation.
- Selection attention has a parity-first gather path; we want a real varlen packing path that consolidates SDPA calls while preserving numerics.

### Proposed Solution

Phased approach with parity-first guardrails. Implement masked varlen SDPA without packing first (simple, reliable), verify strict parity, then add packing (perf) while keeping the simple version as a fallback.

1) Sliding: true masked SDPA (no packing)
- Build a row-wise allowed mask A[t, j] over the original sequence: allowed if (j ≤ t) and (t − j < w).
- Shapes:
  - Q: [B,S,G,h,Dk] → Qf: [B*G*h, S, Dk].
  - K,V: [B,G,S,D*] → Kf,Vf: [B*G*h, S, D*].
  - For exact per-row truncation numerics, run queries as length-1 rows: Qrf: [B*G*h*S, 1, Dk], replicate Kf/Vf per row to Kpf/Vpf: [B*G*h*S, S, D*].
  - Build additive mask Mf: [B*G*h*S, 1, S] with 0 for allowed and −inf otherwise. Do NOT pass is_causal=True.
  - Call SDPA once on packed rows; reshape back to [B,S,G,h,Dv].
- Edge cases: w ≥ S (full lower-tri), w ≤ 0 (zeros), empty rows (map NaNs to 0 via nan_to_num), dtype/device alignment.

2) Compressed: true masked SDPA (no packing)
- Per-row valid compressed length num_cmp(t) = 0 if t+1 < l else floor((t+1−l)/d)+1, clamped ≤ S_cmp.
- Build allowed[t, j] = (j < num_cmp(t)) over compressed axis.
- Use the same BH*S row packing as sliding with Mf: [B*G*h*S, 1, S_cmp], SDPA without causal flag, reshape to [B,S,G,h,Dv].
- Edge cases: S_cmp=0 (zeros), early rows with num_cmp=0, dtype/device, NaNs→0.

3) Selection: real varlen packing (bucketed by length)
- Input ranges: [B,S,G,n,2] contiguous, de-duped, clamped ≤ t+1.
- For each (b,t,g): build a flat index list; compute L(b,t,g). Bucket rows by identical L, pack K/V for a bucket into [N_bucket, L, D*].
- Queries: [N_bucket, 1, Dk]. Run one SDPA per bucket; scatter outputs back to [B,S,G,h,Dv]. Keep the current gather path as an always-on parity fallback.
- Edge cases: L=0 → zeros, heterogeneous per-head lengths disallowed by Eq.10 (we already enforce group consistency).

4) Wiring and flags
- Keep existing env flags: NSA_USE_WIN_MASK, NSA_USE_CMP_MASK, NSA_USE_SEL_PACK.
- Implement masked sliding/compressed behind their flags. Start disabled by default, enable by default only after parity passes.
- Keep parity-first delegates as a fallback path if NSA_FORCE_PARITY=1.

### Automated Test Plan
- Unit parity tests (masked):
  - Expand `nsa/tests/test_batched_parity.py` so masked functions call the new masked varlen paths (not delegates) and assert max error < 1e−6 across grids of (S, w) and (S, l, d).
  - Add micro-tests with tiny shapes (S∈{1,2,3,4}, w∈{1..S}) to catch edge masks, including w≥S and w=1.
- Property tests:
  - Random (B,S,G,h,Dk,Dv) small sizes, check parity vs per-token reference.
- Selection packing tests:
  - Extend `test_selection_packed.py` to generate random ranges, bucket by L, run packed path, compare to gather path with MAE < 1e−6.
  - Include degenerate cases: 0 ranges, fully overlapping contiguous ranges (merge correctness is upstream), and max range coverage.
- Determinism: reuse global seed fixture; run tests twice and compare outputs.

### Components Involved
- `nsa/core/attention_kernels.py` (masked SDPA implementations and selection packing path)
- `nsa/core/nsa_attention.py` (flag wiring)
- `nsa/tests/test_batched_parity.py`, `nsa/tests/test_selection_packed.py`

### Dependencies
- SDPA mask semantics (PyTorch). No Triton or FA-2 for M0 per PRD.
- Existing selection range builder and group consistency invariants.

### Implementation Checklist
- [ ] Sliding masked SDPA (no packing) with BH*S row packing and additive mask
- [ ] Compressed masked SDPA (no packing) with BH*S row packing and additive mask
- [ ] Refactor common mask builders into small helpers (shapes, dtypes, devices)
- [ ] Parity tests for sliding/compressed now target the masked implementations (no delegates)
- [ ] Env flags wire-up and safe fallback via NSA_FORCE_PARITY
- [ ] Selection packing via length bucketing with scatter-restore
- [ ] Parity tests for selection packing; add a small perf smoke
- [ ] Docstrings and shape assertions in public APIs (`attention_kernels.py`, `nsa_attention.py`)

### Verification Steps
- Run default test suite (should pass).
- Run masked parity suite: `NSA_TEST_MASKED=1 pytest -q -k test_batched_parity` (must pass with max_err < 1e−6).
- Run selection packed parity suite: `NSA_TEST_SEL_PACK=1 pytest -q -k test_selection_packed` (must pass, MAE < 1e−6).
- Optional: quick local perf smoke for prefill with masks vs delegates (ensure no regressions >10% for small S).

### Decision Authority
- Technical approach (mask vs packing strategy) owned by engineering. Enable-by-default gates require Product sign-off after parity and basic perf checks.

### Questions / Uncertainties
- Mask construction memory: BH*S×S can be large for big S; acceptable for M0 test sizes. We keep a row-packed [N,1,S] formulation to reduce memory.
- Selection packing granularity: simple length bucketing first; more advanced varlen pack (prefix-sum into single mega-buffer) can be a follow-up.

### Acceptable Tradeoffs
- Prioritize numerical parity and clarity over peak performance for M0. Keep delegates as fallback while iterating.

### Status
Completed

### Notes
- Keep `torch.nan_to_num` on outputs to ensure rows with zero allowed keys produce zeros, matching truncation semantics.
- Do not pass `is_causal=True` when using additive masks; causality must be encoded in the mask.

