# NSA Backward Artifacts Guide

Use this guide to run, collect, and summarize artifacts from the NSA backward hang investigations.

## Running Experiments

- Single repro with profiling:
```
NSA_FORCE_BRANCH=sel \
python -u scripts/nsa_backward_repro.py \
  --layers 5 --seq-len 128 --sel masked \
  --profile --hooks \
  --out-dir artifacts/nsa_isolate --tag sel_masked
```

- Compact matrix across branches/backends:
```
CUDA_VISIBLE_DEVICES=0 bash scripts/nsa_backward_matrix.sh
```

## Expected Files per Run

- `env.json`: NSA_* env and key CUDA/allocator flags used.
- `pre_backward_mem.json` / `post_backward_mem.json`: allocated/reserved MiB snapshots.
- `profiler_table.txt`: Top ops by `cuda_memory_usage` (if `--profile`).
- `trace.json`: Chrome trace (if export succeeds).
- `memory_summary.txt`: Full CUDA memory summary.
- `memory_stats.json`: `torch.cuda.memory_stats()` dump.

## Summarizing Artifacts

Generate a quick, readable summary across all runs:

```
python scripts/summarize_backward_runs.py artifacts --glob "**/run_*" --save-json artifacts/summary.json
```

This prints per-run status (completed vs hung_or_killed), key env toggles, pre/post alloc/reserved MiB, and top profiler rows. It also saves a machine-readable `summary.json` for further analysis.

## What to Return

- The entire `artifacts/` subtree from failing and passing runs, or:
  - `env.json`, `pre_backward_mem.json`, `post_backward_mem.json`, `profiler_table.txt`, `memory_summary.txt`, `memory_stats.json`, and `nvidia-smi` snapshots.

## Tips

- For allocator fragmentation tests, set:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
```
- For error surfacing and tracing:
```
CUDA_LAUNCH_BLOCKING=1  # surface CUDA errors
--anomaly                # enable autograd anomaly detection
--profile                # capture memory-heavy ops
```

