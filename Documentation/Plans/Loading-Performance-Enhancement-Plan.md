# Loading Performance Enhancement Plan (FineWeb‑Edu)

## Objectives
- Reduce cold start time to first batch for `fineweb_edu` to < 5–7s (warm cache) and < 10s (cold cache) on A100/H100 nodes.
- Keep steady‑state fetch latency p95 < 80–100 ms per step.
- Avoid GPU idle due to data stalls by overlapping remote I/O, tokenization, and H2D copy.
- Maintain deterministic behavior, correctness, and fallback safety.

## Current Observations
- First batch latency can exceed 16s on cold starts due to HF handshake, metadata, first reads, and per‑doc tokenization.
- Streaming iterator currently relies on single‑threaded production unless prefetch is enabled.
- Remote I/O variability can dominate cold start; steady state is sensitive to tokenization granularity and prefetch depth.

## KPIs & Targets
- Cold start (first batch):
  - Warm cache: < 5–7s
  - Cold cache: < 10s
- Steady state: fetch p95 < 80–100 ms
- Training impact: no regressions in tokens/sec vs baseline with identical model/settings

## Constraints & Invariants
- No increase in GPU memory footprint for data caching (use local NVMe + pinned CPU buffers).
- Preserve dataset streaming semantics and sharding behavior with `Shard(mod, rem)`.
- Determinism: unchanged ordering within a shard.

## Status
- [x] Phase 1 toggles wired and documented: `NSA_FWE_DOC_BATCH`, `NSA_FWE_PREFETCH`, `NSA_FWE_Q`, `NSA_FWE_REPORT_DOCS` are consumed in `scripts/train_showcase.py`/`scripts/train_showcase_fsdp.py`; prefetcher uses pinned CPU when available. Launch/runbooks set sane defaults.
- [x] Instrumentation: Per‑step `dt_fetch_s` and rolling `fetch_p50_ms`/`fetch_p95_ms` emitted to heartbeat; loader readiness event logged with first‑batch latency. Smoke/run scripts consume heartbeat to gate health.
- [x] Local bootstrap read path supported: `--dataset fineweb_edu_local --local-path /data/fwe_bootstrap.jsonl` loads JSONL/TXT via `nsa.data_pipeline`.
- [x] Phase 2 warmup controls: `NSA_FWE_WARMUP_BATCHES`/`NSA_FWE_WARMUP_TIMEOUT` implemented with heartbeat event `fineweb_loader_warmup{requested,filled,wait_ms}` in both `train_showcase.py` and `train_showcase_fsdp.py`.
- [ ] Phase 0 measurements captured as artifacts (cold vs warm) — execute and attach logs.
- [ ] Automation: small helper `scripts/automation/fwe_bootstrap.py` to pre‑stage ~5GB JSONL.
- [ ] Rollout: enable Phase 1 toggles in CI/prod defaults and tune per‑node values (doc batch, queue depth).

## Next Actions
- Implement Phase 2 warmup in `train_showcase.py` and `train_showcase_fsdp.py` (env flags, optional wait on prefetch queue, heartbeat metrics).
- Add `scripts/automation/fwe_bootstrap.py` and wire to runbooks; keep remote streaming as primary with local bootstrap opt‑in for strict SLA starts.
- Run Phase 0 measurements on A100/H100: capture cold/warm first‑batch latencies and steady‑state fetch p95; include heartbeat and console logs under `artifacts/`.
- Apply Phase 1 toggles in CI and production runbooks; validate no tokens/sec regressions; tune `NSA_FWE_DOC_BATCH`/`NSA_FWE_Q` per node.

## Phase 0 — Baseline & Measurement (No code changes)
1) Configure caches on fast local storage.
   - `export HF_HOME=/mnt/hf_home`
   - `export HF_DATASETS_CACHE=/mnt/hf_home/datasets`
2) Measure cold vs warm start:
   - Cold: `PYTHONPATH=. uv run -q python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 0 --steps 5`
   - Warm: rerun the same command immediately.
   - Look for: `[train] first FineWeb‑Edu batch fetched in X.XXs`
3) Loader smoke (independent):
   - `python scripts/automation/fwe_smoke.py --seq-len 1024 --batch 1 --timeout 60 --tokenizer byte`

Artifacts: console logs + time to first batch; optionally capture `artifacts/train_showcase/heartbeat_rank*.jsonl` if enabled.

## Phase 1 — Toggle‑Only Improvements (Existing features)
- Enable batched tokenization and prefetch (already supported):
  - `export NSA_FWE_DOC_BATCH=128`  # try 64→128→256
  - `export NSA_FWE_PREFETCH=1`
  - `export NSA_FWE_Q=8`            # prefetch queue depth
  - `export NSA_FWE_REPORT_DOCS=2000`  # cut logging overhead
- Keep first‑batch guardrails:
  - `--loader-timeout 60 --synthetic-on-fail 1`
- Test commands (single GPU):
  - `PYTHONPATH=. uv run -q python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 0 --steps 200`

Acceptance:
- Cold start < target after warm cache; steady‑state p95 fetch within guideline; no tokens/sec regression.

## Phase 2 — Warmup & Prefill (Small code additions)
Add env‑configurable warmup before entering the training loop:
- `NSA_FWE_WARMUP_BATCHES` (e.g., 32–128): minimum number of batches to have queued in pinned CPU before step 1.
- `NSA_FWE_WARMUP_TIMEOUT` (e.g., 60s): cap warmup duration.

Behavior:
- Start loader thread(s) immediately, tokenize in batches (`NSA_FWE_DOC_BATCH`), fill a pinned CPU queue of size `NSA_FWE_Q`.
- Optionally wait until `NSA_FWE_WARMUP_BATCHES` are available or timeout, then enter the training loop to avoid GPU idle at step 1.

Implementation sketch (train_showcase.py):
- Gate existing `_prefetch_iter` with warmup counters; no change to dataset semantics.
- Emit heartbeat fields `{ "fetch_warmup_batches": N, "fetch_warmup_wait_ms": T }` for diagnostics.

Acceptance:
- First batch latency reduced to sub‑second on warm cache; no training stalls at step 1.

## Phase 3 — Local Bootstrap (Optional but effective)
Pre‑stage ~5 GB of raw texts to local JSONL on the node to minimize cold starts:

```bash
python - <<'PY'
from datasets import load_dataset
import json, os
os.makedirs('/data', exist_ok=True)
ds = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)
bytes_goal = 5*1024**3
bs = 0
with open('/data/fwe_bootstrap.jsonl','w') as w:
    for ex in ds:
        t = ex.get('text') or ''
        if not t: continue
        s = json.dumps({'text': t}, ensure_ascii=False)
        w.write(s+'\n')
        bs += len(s.encode('utf-8'))+1
        if bs >= bytes_goal: break
print('wrote bytes:', bs)
PY

# Train from local file
PYTHONPATH=. uv run -q python -u scripts/train_showcase.py \
  --dataset fineweb_edu_local --local-path /data/fwe_bootstrap.jsonl --ddp 0 --steps 200
```

Notes:
- Keep remote streaming for continued training; local file accelerates startup.
- Optionally refresh the local file in background to extend capacity.

## Instrumentation & Telemetry (Optional)
- Emit `dt_fetch_last` and warmup metrics to heartbeat (`artifacts/train_showcase/heartbeat_rank*.jsonl`).
- Track p50/p95 fetch latency and correlate with tokens/sec over windows.

## Risks & Fallbacks
- Network volatility or slow disk caches can dominate cold starts. Use local bootstrap for production starts.
- Excessive `NSA_FWE_DOC_BATCH` can raise CPU load; tune per node.
- Keep `--synthetic-on-fail 1` to guarantee progress if loader stalls.

## Rollout
1) Apply Phase 1 toggles in CI and production runbooks; record cold/warm starts and p95 fetch.
2) Add Phase 2 warmup controls (small patch), re‑measure.
3) Adopt Phase 3 bootstrap for deployments where cold start SLA is strict.

## Acceptance Summary
- Cold start: < 5–7s (warm cache), < 10s (cold cache) to first batch.
- Steady‑state: fetch p95 < 80–100 ms; no tokens/sec regression.
- Zero training stalls; safe fallbacks on loader timeout.
