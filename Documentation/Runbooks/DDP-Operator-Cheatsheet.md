# DDP Operator Cheatsheet

Quick reference for launching, hardening, and triaging DDP runs on 2×A100 (PCIe).

## Launch (Production)

```
bash scripts/run_2xa100_production.sh
```

- Validated flags baked in:
  - `NSA_PREFILL_BATCHED=1`, `NSA_SEL_RANGES_V2=1`
  - `NSA_DDP_COMPRESS=bf16` (set `off` to disable)
  - `NSA_DDP_BUCKET_MB=25` (try `100` if variance)
  - `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`, `NCCL_IB_DISABLE=1`
  - `NCCL_ASYNC_ERROR_HANDLING=1`, `NCCL_BLOCKING_WAIT=1`, `TORCH_DISTRIBUTED_DEBUG=DETAIL`
  - `NSA_DDP_FIND_UNUSED=0`, `NSA_DDP_STATIC_GRAPH=1`, `NSA_DDP_DISABLE_GC=1`
  - `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True`

## Manual (Copy/Paste)

```
CONFIG=configs/m7c_125m_2xa100_production.yaml \
NSA_PREFILL_BATCHED=1 NSA_SEL_RANGES_V2=1 \
NSA_DDP_COMPRESS=bf16 NSA_DDP_BUCKET_MB=25 \
NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_IB_DISABLE=1 \
NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1 \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
NSA_DDP_DISABLE_GC=1 NSA_DDP_FIND_UNUSED=0 NSA_DDP_STATIC_GRAPH=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True \
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu --ddp 1
```

## Triage (Deadlocks/Timeouts)

- Step 1: Disable compression and increase bucket size
```
NSA_DDP_COMPRESS=off NSA_DDP_BUCKET_MB=100 bash scripts/run_2xa100_production.sh
```
- Step 2: Toggle one graph control at a time
  - `NSA_DDP_STATIC_GRAPH=0` (keep `FIND_UNUSED=0`), or
  - `NSA_DDP_FIND_UNUSED=1` (keep `STATIC_GRAPH=1`)
- Step 3: Keep fail‑fast NCCL flags on. Add `NCCL_DEBUG=INFO` for short windows.
- Step 4: Fall back to single GPU to continue progress
```
CONFIG=configs/m7c_125m_2xa100_production.yaml \
NSA_PREFILL_BATCHED=1 NSA_SEL_RANGES_V2=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True \
PYTHONPATH=. python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 0
```

## Monitoring

- `tail -f artifacts/m7c_125m_2xa100_prod/training.csv`
- `tail -f artifacts/m7c_125m_2xa100_prod/heartbeat_rank0.jsonl`
- `watch -n 1 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv`

## Notes

- Per-rank sharding returns full per-rank batches; do not slice again in-loop.
- Accumulation: `no_sync()` is active when `accumulate_grad_batches>1`.
- Process group timeout is set via `NSA_PG_TIMEOUT_MIN` (default 15).

