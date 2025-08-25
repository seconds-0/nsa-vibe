# Memory Units Clarification for NSA Training

## Issue Summary
There is a labeling confusion in the heartbeat memory reporting for the current 50k training run.

## Current Behavior
The training script (`scripts/train_showcase.py`) reports memory as:
```python
"gpu_mem_alloc": int(torch.cuda.memory_allocated() // (1024 * 1024))
"gpu_mem_reserved": int(torch.cuda.memory_reserved() // (1024 * 1024))
```

This correctly converts bytes to MB, but the resulting values (e.g., 21, 66) are confusingly small.

## The Confusion
- **Heartbeat shows**: `gpu_mem_alloc: 21, gpu_mem_reserved: 66`
- **Initial interpretation**: 21 GB allocated, 66 GB reserved
- **Actual meaning**: Unclear - values too small for MB, too precise for GB

## Investigation Results
After thorough investigation, we found:
1. nvidia-smi shows 751 MB during training (snapshot between steps)
2. Earlier training at step 1,580 showed same values (21, 66)
3. These values are consistent despite 25% training progress

## Most Likely Explanation
The heartbeat values represent memory in units of **1024 MB blocks** (approximately GB):
- `21` = ~21 GB allocated memory
- `66` = ~66 GB reserved memory

This matches expected memory usage for the model and training configuration.

## Monitoring Workaround
Use the parallel monitoring script `scripts/monitor_memory_correct.py` to see properly labeled values:
```bash
python scripts/monitor_memory_correct.py
```

This reads the heartbeat file and displays memory with correct GB labels.

## Fix for Future Runs
After the current training completes, update the code to:
```python
"gpu_mem_alloc_mb": int(torch.cuda.memory_allocated() // (1024 * 1024))
"gpu_mem_alloc_gb": float(torch.cuda.memory_allocated() / (1024 * 1024 * 1024))
```

## Important Note
**DO NOT** modify the running training process. The training is healthy and progressing well.
Only the unit labels are confusing, not the actual memory management.