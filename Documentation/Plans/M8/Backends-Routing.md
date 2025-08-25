# Backends & Routing Policy

Goals: conservative defaults, safe fallbacks, parity gating before opt-ins.

Policy
- Baseline: SDPA for all branches; FA‑2 opt‑in for cmp/win only after parity.
- Triton selection: off by default; allow only under explicit flags + parity smokes on A100.
- Backward/edge cases parity criteria with thresholds for acceptance.
- Automatic fallback with counters on failure; warmup and regression detection.

Deliverables
- Routing docs in `Documentation/Guides/Execution-Routing.md` updated as needed — PENDING

