Title: PR34 — Mixed Precision Selection Mapping (p_slc/p_grp)

Summary
- Adds opt-in mixed precision to the selection mapping pipeline to reduce bandwidth and math cost while preserving stable selection ordering.
- Two flags:
  - `NSA_P_SLC_MIXED=1`: compute `map_pcmp_to_pslc_batched` under CUDA autocast(bfloat16), then upcast the result to the original dtype.
  - `NSA_P_GRP_FP32=1`: cast grouped scores `p_grp` to float32 before top‑k selection to stabilize tie‑breaks.

Changes
- nsa/core/selection_scorer.py
  - `map_pcmp_to_pslc_batched`: optional autocast(bfloat16) on CUDA guarded by `NSA_P_SLC_MIXED`; results are upcast back to the original dtype for downstream code.
  - Safe fallback to precise path if autocast isn’t available or raises.
- nsa/core/nsa_attention.py
  - In batched and sequential prefill paths, optionally cast `p_grp_all` to float32 when `NSA_P_GRP_FP32=1` before calling `select_topn_ranges_batched`.

Rationale
- `p_cmp_all` (B,S,G,h,S_cmp) → `p_slc_all` (B,S,G,h,S_sel) mapping is bandwidth-heavy. On A100/H100, bfloat16 autocast meaningfully reduces memory traffic while maintaining adequate precision for the weighted scatter-add.
- Performing the top‑k in float32 preserves ranking stability and tie‑break behavior; this is especially helpful when upstream steps use mixed precision.

Correctness
- By upcasting `p_slc` back to the original dtype and performing top‑k in float32 (when enabled), we preserve existing selection behavior and tie‑break ordering.
- The feature is opt-in and defaults OFF; existing tests and numerics remain unchanged by default.

Test Plan
1) Parity (defaults OFF)
   - `pytest -q nsa/tests/test_selection_v2_equiv.py`
   - `pytest -q nsa/tests/test_selection_tiebreak.py`
2) CUDA opt-in checks (A100/H100)
   - `NSA_P_SLC_MIXED=1 NSA_P_GRP_FP32=1 pytest -q nsa/tests/test_selection_v2_equiv.py`
   - Verify `test_selection_tiebreak` still passes and tie-break warnings are absent.
3) Microbench (optional)
   - Prefill bench: `PYTHONPATH=. uv run -q python bench/bench_prefill.py --config configs/base.yaml`
   - Env: `NSA_USE_SEL_PACK=1 NSA_P_SLC_MIXED=1 NSA_P_GRP_FP32=1`

Performance Expectations
- Moderate improvement on larger shapes due to reduced memory traffic in `map_pcmp_to_pslc_batched`.
- Tie‑break stability maintained by float32 top‑k.

Rollout & Safety
- Both flags default OFF. Enable selectively in performance runs (`NSA_P_SLC_MIXED=1` and `NSA_P_GRP_FP32=1`).
- No changes to behavior when flags are unset.
