Title: PR36 — Loader Warmup and Telemetry (FineWeb‑Edu)

Summary
- Adds opt-in warmup for the data loader to reduce first-step stalls and stabilize throughput.
- Emits warmup telemetry to the heartbeat for observability.

Flags
- `NSA_FWE_WARMUP_BATCHES` (int, default 0): minimum number of batches to prefill before step 1.
- `NSA_FWE_WARMUP_TIMEOUT` (float seconds, default 0): maximum time to wait for warmup.

Changes
- scripts/train_showcase.py
  - After optional pinned-CPU prefetch, warm up by pulling up to `NSA_FWE_WARMUP_BATCHES` from `fwe_train` in a background thread, bounded by `NSA_FWE_WARMUP_TIMEOUT`.
  - Heartbeat event: `fineweb_loader_warmup {requested, filled, wait_ms}`.
- scripts/train_showcase_fsdp.py
  - After first-batch smoke and prepend, pull remaining warmup batches similarly and emit the same heartbeat.
- Documentation/Plans/Loading-Performance-Enhancement-Plan.md
  - Status updated to reflect Phase 2 warmup implementation.

Behavior
- Warmup is non-invasive: it preloads batches and chains them ahead of the iterator, without changing dataset semantics or sharding.
- With pinned-CPU prefetch enabled, warmup leverages the same iterator and simply waits for additional queued batches.

Test Plan
1) Single GPU (A100 or local):
   - `NSA_FWE_WARMUP_BATCHES=16 NSA_FWE_WARMUP_TIMEOUT=30 PYTHONPATH=. python scripts/train_showcase.py --dataset fineweb_edu --ddp 0 --steps 50`
   - Expect: heartbeat event `fineweb_loader_warmup`, faster first step, steady fetch_p95.
2) FSDP (2 GPUs):
   - Same flags; confirm `fineweb_loader_warmup` appears and throughput is stable.

Rollout
- Defaults OFF, safe to merge.
- Run A/B smoke with and without warmup on production nodes; keep ON for strict SLA starts.
