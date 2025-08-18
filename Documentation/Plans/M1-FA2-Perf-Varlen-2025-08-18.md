### Task ID
M1-FA2-Perf-Varlen

### Problem Statement
M0 establishes correctness with SDPA-only paths (batched prefill, masked varlen SDPA for sliding/compressed, selection packing). To reach paper-aligned performance, we must integrate FlashAttention‑2 (FA‑2) for compressed and sliding branches, preserve numerics within a tight tolerance, retain CPU fallback and determinism in tests, and prepare for future selection kernel work.

### Goals (What M1 Delivers)
- Integrate FA‑2 for compressed (cmp) and sliding (win) branches in prefill and decode with varlen support.
- Provide packing paths that feed FA‑2 efficiently (avoid per-row replication; bucket by lengths; use varlen APIs where available).
- Maintain exact functional semantics; allow a small numeric tolerance vs SDPA (e.g., MAE ≤ 5e‑5 FP32).
- Keep SDPA reference paths available and opt-in parity suite green.
- Preserve CPU fallback: SDPA-only reference path remains the default on CPU.

### Non-Goals (M1)
- Triton selection kernel (forward/backward) — deferred to M4/M5.
- Training/perf tuning beyond FA‑2 adoption (e.g., mixed precision policies) — only minimal hooks.

### Proposed Solution

1) FA‑2 Integration Strategy
- Add a feature flag matrix (Hydra + env):
  - `runtime.use_flash` (global), `NSA_USE_FA2_WIN`, `NSA_USE_FA2_CMP` (branch-specific), and `NSA_TEST_FA2` (tests).
- Implement FA‑2 wrappers that accept packed/varlen layouts:
  - Sliding: bucket rows by window length L′ = min(w, t+1); pack Q (rows) and K/V segments; call FA‑2 varlen if present, else dense FA‑2, else SDPA.
  - Compressed: bucket rows by `num_cmp(t)`; pack Q and the first `L′` compressed tokens; call FA‑2 varlen/dense accordingly.
- Preserve masked SDPA fallbacks: if FA‑2 unavailable or `use_flash=False`, use existing masked row-packed SDPA.

2) FA‑2 API & Constraints
- Head dim alignment: assert and xfail tests when FA‑2 head_dim requirements are not met (e.g., multiples of 8/16 depending on dtype/device).
- Separate sequence lengths for Q vs KV: if available, pass `cu_seqlens_q` and `cu_seqlens_kv` plus `max_seqlen_{q,kv}`; otherwise pad to bucket max.
- CPU-only environments: force SDPA; ensure feature flags are ignored on CPU; parity suite remains green.

3) Packing Layouts (Varlen)
- Build per-(B,G) buckets with identical lengths; allocate contiguous buffers for Q, K, V, and prefix-sum offsets.
- For FA‑2 varlen APIs: provide qkv pointers + cu_seqlens (or q/k/v separate with cu_seqlens_q/kv) plus max_seqlen.
- For dense FA‑2: pad to max length per bucket; provide an additive mask for disallowed tokens if needed.
- Maintain head grouping (GQA): expand heads inside each group only at the last moment to reduce bandwidth.
- Two-level bucketing: first by (B,G), then by effective length; optionally merge tiny buckets to amortize launch overhead.
- Workspace reuse: preallocate/reuse packed buffers and cu_seqlens across steps; grow-on-demand.

4) Decode Path
- Sliding: window length per step is `min(w, S_raw)` — always contiguous tail; use FA‑2 dense with a triangular causal mask or a fixed small length if varlen not supported per-step.
- Compressed: `num_cmp(t)` compressed tokens available; pass that length to FA‑2; if zero, return zeros.
- Kernel auto-select: for very small lengths (e.g., <16), prefer SDPA to avoid FA‑2 launch overhead; otherwise FA‑2.

5) Determinism/Parity Policy
- Reference: SDPA FP32 is the oracle; FA‑2 parity tests accept MAE ≤ 5e‑5 (FP32), and ≤ 2e‑4 (BF16/FP16) if tested.
- Ensure identical softmax scale (1/sqrt(Dk)) and no dropout/attention bias; tests assert these conditions.
- Keep `NSA_FORCE_PARITY=1` to force SDPA/row-packed masked paths in CI; FA‑2 tests are opt-in via `NSA_TEST_FA2=1`.

6) Observability
- Log selected FA‑2 path (varlen vs dense vs fallback), bucket size distribution, and per-branch MAE vs SDPA on demand (`NSA_DEBUG_COMPARE=1`).
- Record max/min lengths, kernel calls per branch, and time per branch when `NSA_DEBUG_TIMING=1` (optional).

### Automated Test Plan
- Parity (opt-in, `NSA_TEST_FA2=1`):
  - Sliding FA‑2 vs SDPA (small S grid; multiple w values) MAE ≤ 5e‑5 FP32 (≤ 2e‑4 BF16/FP16).
  - Compressed FA‑2 vs SDPA (small S; (l,d) grid) MAE ≤ 5e‑5 FP32 (≤ 2e‑4 BF16/FP16).
- Determinism:
  - Two successive FA‑2 runs (same seed) produce identical outputs on CPU fallback; on GPU, verify statistical stability if bitwise non-determinism exists (document).
- Packing correctness:
  - Row vs bucket packing produce identical outputs (within tolerance) against the same kernel.
- CPU fallback:
  - Force fallback: `runtime.use_flash=false` → all parity tests still green.
- Perf smoke (bench only):
  - Sliding FA‑2 vs SDPA row-packed masked; Compressed FA‑2 vs SDPA; print speedups.
 - Xfail coverage:
   - Head_dim constraints or device constraints not met → tests xfail with clear message.

### Components Involved
- `nsa/kernels/flash_wrappers.py`: add FA‑2 varlen/dense wrappers; capability detection.
- `nsa/core/attention_kernels.py`: new FA‑2 attention fns: `sliding_window_attention_fa2`, `compressed_attention_fa2`; packing helpers.
- `nsa/core/nsa_attention.py`: wire flags, choose FA‑2 vs masked SDPA vs per-token reference (decode).
- `nsa/tests/`: `test_fa2_parity.py`, extend tiny grids; determinism/packing tests.
- `bench/`: `bench_fa2.py` to compare FA‑2 vs SDPA for cmp/win.

### Precision & Constraints
- DType policy: inputs may be BF16/FP16, accumulators FP32 in tests; SDPA oracle in FP32.
- Enforce/xfail on FA‑2 head_dim alignment and unsupported devices; fall back to SDPA.

### Dependencies
- PyTorch ≥ 2.3, FlashAttention‑2 ≥ 2.x (optional, guarded imports).
- CPU-only environments must pass tests using SDPA-only paths.

### Implementation Checklist
- [x] Capability gates in `flash_wrappers.py` (varlen probe, dense probe); device/dtype checks via `fa2_supported`; errorless fallbacks.
- [x] Packing helpers: build buckets and `cu_seqlens` (`nsa/core/packing.py`).
- [x] Sliding FA‑2 forward — prefill path using dense FA‑2 per-bucket with try/except fallback to masked SDPA.
  - [ ] Sliding FA‑2 — decode path (per-step packing + FA‑2 call; fallback to SDPA for tiny L). 
- [x] Compressed FA‑2 forward — prefill path using dense FA‑2 per-bucket with try/except fallback to masked SDPA.
  - [ ] Compressed FA‑2 — decode path (per-step `num_cmp(t)` packing + FA‑2 call; fallback to SDPA for tiny L).
- [ ] Varlen FA‑2 wrappers: implement `flash_attn_varlen_*` calls (QKV‑packed or separate Q/K/V) with `cu_seqlens_{q,kv}`.
- [ ] Flags and wiring in `nsa_attention.py`:
  - [x] Global `NSA_USE_FA2`, `NSA_FORCE_PARITY` gates.
  - [ ] Branch toggles `NSA_USE_FA2_WIN`/`NSA_USE_FA2_CMP` and config `runtime.use_flash`.
- [ ] Parity tests (FA‑2 vs SDPA) for sliding/compressed (small grids), MAE thresholds; xfail device/head_dim unsupported.
- [ ] Determinism and packing tests (row vs bucket equivalence within tolerance); document any non‑determinism on GPU.
- [x] Perf smokes: `bench/bench_fa2.py` added; gated to skip on unsupported devices.
- [x] Docs: PRD M1 notes and `.cursor/rules/20-m0-execution.mdc` updated with FA‑2 toggles and guidance.
- [ ] Small-length auto‑switch thresholds exposed via config/env; enforce SDPA when below threshold.
- [ ] Optional: two‑level bucketing + tiny‑bucket merge; workspace reuse across iterations.

### Verification Steps
- Run default suite (CPU) → green.
- Run FA‑2 parity: `NSA_TEST_FA2=1 uv run -q pytest -k test_fa2_parity` → MAE ≤ 5e‑5.
- Run packing determinism/identity tests.
- Run `PYTHONPATH=. uv run python bench/bench_fa2.py` and capture speedups.

### Decision Authority
- Engineering owns integration and acceptance thresholds; Product signs off enabling FA‑2 by default post parity/perf review.

### Questions / Uncertainties
- FA‑2 varlen API coverage for our exact shapes on CPU? If not, keep SDPA-only on CPU.
- Tolerances for MAE vs SDPA on GPU may require slightly looser bounds (document if needed).

### Acceptable Tradeoffs
- Prefer clarity and robust fallbacks over peak perf; keep SDPA and masked SDPA paths callable.
- Allow small numeric drift within tight FP32 tolerance.

### Status
In Progress

### Notes
- Keep `NSA_FORCE_PARITY=1` in CI until FA‑2 paths mature; `NSA_TEST_FA2=1` gates parity suite.

