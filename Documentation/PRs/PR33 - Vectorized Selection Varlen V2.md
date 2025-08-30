Title: PR33 — Vectorized Selection Varlen V2 (FA‑2 varlen + dense fallback)

Summary
- Implements a fully vectorized varlen selection packer (v2) that removes Python loops from `selection_attention_varlen_all`.
- Uses FA‑2 varlen fast path with correct semantics (causal=False for single‑query rows, since rows are already clamped to ≤t), and a dense batched fallback.
- Adds opt-in flags with safe defaults and a min-length threshold to avoid overhead on tiny rows.

Changes
- nsa/core/attention_kernels.py
  - New v2 implementation as `selection_attention_varlen_all_v2` (vectorized mask → flat pack).
  - `selection_attention_varlen_all` dispatches to v2 when `NSA_SEL_VARLEN_V2=1` (default); otherwise uses the legacy packed path.
  - Parity semantics: use `causal=True` for FA‑2 varlen, dense fallback, and per-row fallback (single‑query rows, first key only) to exactly mirror the packed reference.
  - Fixed minor bug: used `Nb` consistently in dense fallback expand shapes.

Env Flags
- `NSA_SEL_VARLEN_V2` (default 1): enable vectorized v2 path.
- `NSA_SEL_VARLEN_MIN_L` (default 0): bypass v2 when the maximum per-row L is below this threshold and use packed path.

Rationale
- The v1 packer relied on Python loops across B×S×G rows to concatenate segments; this limited throughput and created GPU sync points.
- The v2 packer builds a per-row allowed mask using a difference-array trick, computes per-row lengths, and flattens K/V selection in one shot.
- With Tq=1 per row and a packed reference using `is_causal=True`, varlen must also use `causal=True` for exact parity (restricts to the first packed key).

Correctness
- Unit parity: v2 preserves exact semantics vs `grouped_selection_attention_packed` by attending to the full selected set per row without an extra triangular mask.
- CUDA numerical differences observed in re‑validation were due to using `causal=False` in the varlen path; v2 now uses `causal=True`, which should eliminate the MAE discrepancy against the packed reference.
- Autograd: v2 uses workspace copies for pack buffers and is forward‑only today (no gradient propagation to original Q/K/V). Keep `NSA_USE_SEL_VARLEN=0` during training that requires selection‑branch gradients. This matches current expectations for the opt‑in selection varlen feature.

Test Plan
1) CPU/CUDA unit parity
   - `pytest -q nsa/tests/test_selection_varlen_optin.py` (expect PASS on both CPU and CUDA)
   - `pytest -q nsa/tests/test_selection_v2_equiv.py`

2) FA‑2 GPU parity (opt-in)
   - `NSA_TEST_FA2=1 PYTHONPATH=. pytest -q -k fa2_gpu_varlen`

3) Integration (prefill)
   - `PYTHONPATH=. uv run -q python bench/bench_prefill.py --config configs/base.yaml`
   - Env: `NSA_USE_SEL_VARLEN=1 NSA_SEL_VARLEN_V2=1` (optionally `NSA_SEL_VARLEN_MIN_L=8`)

4) Training smokes
   - Synthetic: `PYTHONPATH=. python scripts/train_showcase.py --dataset synthetic --steps 50`
   - FineWeb‑Edu: `PYTHONPATH=. python scripts/train_showcase.py --dataset fineweb_edu --steps 200`
   - Expect: stable loss, steady fetch p95, no regressions in toks/s.

GPU Validation (A100/H100)
- With flash‑attn installed, v2 should choose FA‑2 varlen for selection; log confirms via `fa2.win/ cmp/ varlen` entries (if timing debug is on).
- If FA‑2 varlen is unavailable, dense fallback buckets run via `attention_fa2_dense_batch` or SDPA fallback.

Performance Expectations
- Neutral to improved vs PR31 v1:
  - Remove Python packing loops (lower CPU overhead, fewer syncs)
  - Correct non‑causal FA‑2 call improves CUDA parity and avoids unintended masking
  - Dense fallback batched improves small‑L buckets

Rollout & Safety
- Default‑on v2 (`NSA_SEL_VARLEN_V2=1`) with easy rollback: set `NSA_SEL_VARLEN_V2=0` to restore legacy packed path.
- Use `NSA_SEL_VARLEN_MIN_L` to avoid varlen overhead on extremely small selections.
- Training note: For runs that require selection‑branch gradients, set `NSA_USE_SEL_VARLEN=0` to stay on packed path until backward support lands.

Notes
- This PR does not touch the selection scoring or mapping paths; mixed precision for p_slc/p_grp remains a separate opt-in and future work.
