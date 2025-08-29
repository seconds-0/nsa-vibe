Title: PR38 — Fused Gate Combine (Bench Opt-in)

Summary
- Adds an opt-in fused gate+combine path to reduce kernel launches in the gating hotspot.
- Uses a small fused function to compute GateMLP forward and the weighted sum in one pass; optionally compiled with `torch.compile`.

Flag
- `NSA_GATE_COMPILE=1`: Enable fused gate+combine path (falls back to existing path on any failure).

Changes
- nsa/core/nsa_attention.py
  - Adds `_fused_gate_combine_bsg` and `_fused_gate_combine_bg` helpers for prefill batched and decode shapes.
  - Lazy compiles fused helpers with `torch.compile(..., mode="reduce-overhead")` when enabled.
  - Integrates fused path in both decode and batched prefill; otherwise preserves existing behavior and monitoring.
- bench/bench_gate.py: Simple benchmark to measure gate+combine overhead.

Rationale
- Gate path computes two linear layers + softmax and three weighted sums each step; fusing reduces Python overhead and can reduce kernel launches.
- Opt-in and safe: falls back to standard path if compilation or fused exec fails.

Behavior
- No change by default (flag OFF). When enabled and `torch.compile` is available, uses the fused helper.
- Gate statistics remain available when unfused; in the fused path we prioritize performance (stats capture is skipped in the fast path).

Test/Bench Plan
1) Functional (defaults): ensure training smokes still pass.
2) Enable fused path: `NSA_GATE_COMPILE=1` and run a short training smoke to check stability.
3) Micro-bench: `PYTHONPATH=. python bench/bench_gate.py --iters 500` (compare with and without `NSA_GATE_COMPILE`).

Rollout & Safety
- Keep OFF by default. Benchmark in perf runs; if beneficial and stable, we can keep as an opt-in knob for production.
