M5 Backward Debugging Guide

Overview
- This guide explains how to validate and debug selection backward computations locally using the reference CPU implementation.

Reference Backward API
- Function: `nsa.kernels.triton_sel_kernel.selection_attention_backward_reference(Q,K,V,ranges,dO)`
- Inputs:
  - `Q`: `[B,S,G,h,Dk]`
  - `K`: `[B,G,S_kv,Dk]`, `V`: `[B,G,S_kv,Dv]`
  - `ranges`: `[B,S,G,n,2]` selection spans (clamped internally to `[0, S_kv]`)
  - `dO`: upstream gradient `[B,S,G,h,Dv]`
- Returns: `(dQ, dK, dV)` matching shapes of `Q`, `K`, `V`.
- Behavior mirrors the packed SDPA selection path and enforces the same causal semantics used when `is_causal=True` with `Tq=1`.

Local Validation
1) Run targeted tests:
   - `PYTHONPATH=. pytest -q nsa/tests/test_selection_backward_reference.py`
   - `PYTHONPATH=. pytest -q nsa/tests/test_selection_backward_edges.py`

2) Run CPU gradchecks (tiny shapes):
   - `PYTHONPATH=. pytest -q nsa/tests/test_gradcheck_cpu.py`

3) Inspect edge behavior:
   - Empty ranges produce zero outputs and zero grads.
   - Adjacent spans and long-L buckets are covered by tests and parity checks.

Tips
- If you suspect a mismatch, set `NSA_DEBUG_LOG=1 NSA_LOG_LIMIT=5` and repeat a failing test to print structured shape logs.
- To compare backward against a pure gather path, use `nsa.core.attention_kernels.grouped_selection_attention` for forward and derive grads via a scalar loss (see tests for examples).

