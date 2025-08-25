# Training Loop Telemetry

Goals: rich telemetry, early aborts on collapse, reproducible runs.

What to implement
- Metrics: loss/train+val, lr, grad_norm, tokens/s, dt_fetch_s, clip_fraction.
- Gate health: entropy, temperature (τ) anneal schedule, per-branch share.
- Safety stops: NaN loss, zero grad streaks, throughput stalls.
- Gradient accumulation support with accurate tokens accounting.
- Batch dump on first N steps for debugging.

Deliverables
- Instrument trainer (`scripts/train_showcase.py` or trainer module) — DONE
- CSV/JSONL schema: `artifacts/m7c_125m/training.csv`, `heartbeat_*.jsonl` — DONE

