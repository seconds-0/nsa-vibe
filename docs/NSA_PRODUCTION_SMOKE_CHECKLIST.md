# NSA Production Smoke Checklist (12L @ 2048)

This checklist lets a test engineer validate readiness on A100 80GB with clear success gates.

## Prereqs
- Hardware: 1× or 2× A100 80GB (per‑GPU memory is key)
- Driver: 575.64.03 (as tested)
- PyTorch: 2.5.1+cu121 (or 2.3.1 confirmed OK)
- Env: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256`

## Quick Verify: Backward Scaling
```
CUDA_VISIBLE_DEVICES=0 bash scripts/run_backward_suite.sh
```
- Expect: S128/S512 pass; S1024/S2048 pass or complete with ~12/20 GB mem (no hangs)
- See summary: `artifacts/nsa_suite/summary_*.json`

## Production‑like Smoke (Synthetic)
```
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 \
python -u scripts/prod_smoke_tinylm.py \
  --dim 768 --layers 12 --heads 12 --groups 12 --d-k 64 --d-v 64 \
  --l 16 --d 16 --l-sel 64 --n-sel 16 --w 512 \
  --seq-len 2048 --batch 1 --steps 100 \
  --dtype bf16 --grad-checkpointing \
  --out-dir artifacts/prod_smoke --tag 12L_2k
```
- Success gates (in `heartbeat.jsonl`):
  - `reserved_mib` ≤ 70,000 (approx ≤ 70 GB)
  - Stable or decreasing trend over steps
  - No anomaly exceptions

## Backend Contrast (Optional)
- Selection masked vs packed vs gather (S=2048): use `scripts/nsa_backward_repro.py` with `NSA_FORCE_BRANCH=sel`.
- Expect near‑identical peak memory at S=2048; choose backend based on perf/stability.

## Artifact Summary
```
python scripts/summarize_backward_runs.py artifacts --glob "**/run_*" --save-json artifacts/summary.json
```
- Attach `artifacts/summary.json` and the `artifacts/` subtree for review.

## Go/No‑Go
- Go if:
  - Smoke passes for 100 steps
  - Peak reserved ≤ 70 GB
  - No anomalies; profiler (if enabled) shows no pathological op growth
- No‑Go if:
  - OOM/abort
  - Peak reserved > 75 GB or rising steadily >10% over 50 steps

## Notes
- DDP/FSDP do not raise per‑GPU memory headroom; use single GPU for this smoke.
- Gradient checkpointing must be enabled for 12L at 2k; expect extra compute overhead.
- For longer runs, consider periodic job restarts to avoid long‑lived allocator fragmentation.

