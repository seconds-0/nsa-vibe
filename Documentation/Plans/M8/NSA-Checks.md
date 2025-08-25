# NSA Semantics Checks

Goals: enforce causal semantics, GQA consistency, and deterministic selection.

What to implement
- Deterministic tie-breaks in top‑k; forced block 0 and 2 locals; clamp ≤ t; de‑dup/merge.
- No‑peek tests on all branches; RoPE on Q and per‑branch K before ϕ.
- Eq.9 verifier (slow path) for selection mapping; compare vs fast path in tests.
- GQA group consistency asserts; expose counters for reads per branch.

Deliverables
- Tests tightened in `nsa/tests/*` where applicable — PENDING
- Debug asserts/flags in `nsa/core/*` — PENDING

