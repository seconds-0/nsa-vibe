Title: PR37 — p_cmp Mixed Precision Finalization (compute_pcmp_all)

Summary
- Finalizes the mixed-precision pathway for selection scoring (p_cmp) with tests and a micro-benchmark.
- Keeps behavior opt-in and numerically safe by default.

Flags
- `NSA_P_CMP_MIXED=1`: Enable CUDA autocast(bfloat16) in `compute_pcmp_all`; result upcasts to original dtype.

Changes
- nsa/tests/test_pcmp_mixed_parity.py: Parity test comparing mixed vs precise outputs on CPU/CUDA with small tolerance for CUDA bf16.
- bench/bench_pcmp.py: Micro-benchmark to measure speedup across several shapes.

Rationale
- `p_cmp_all` is often bandwidth-bound; bf16 autocast on A100/H100 reduces traffic and can yield measurable speedups.
- Upcasting back to the original dtype preserves downstream numerics; top‑k stability is maintained especially when combined with `NSA_P_GRP_FP32` (from PR 34).

Test Plan
1) Default precision (sanity)
   - `pytest -q nsa/tests/test_selection_v2_equiv.py nsa/tests/test_selection_tiebreak.py`
2) Mixed precision parity (CUDA)
   - `pytest -q nsa/tests/test_pcmp_mixed_parity.py -k cuda` (skips when no CUDA)
3) Micro-bench (optional)
   - `PYTHONPATH=. python bench/bench_pcmp.py --device auto --iters 100`

Rollout & Safety
- Default OFF. Enable `NSA_P_CMP_MIXED=1` in performance runs.
- Revert by unsetting the flag.
