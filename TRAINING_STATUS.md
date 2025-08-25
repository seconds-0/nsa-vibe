# NSA 50K Training Status Report

## Current Training Status (as of Sun Aug 24 05:53 UTC 2025)
- **Current Step**: 1,580 / 50,000 (3.16% complete)
- **Loss**: 1.37 (decreasing from initial 5.75)
- **Learning Rate**: 2.99e-04 (stable, cosine schedule)
- **Throughput**: ~410 tokens/second
- **GPU Memory**: 21 GB allocated
- **Estimated Completion**: ~30 hours from start

## Key Log Files on Remote GPU

### Real-time Monitoring Files
1. **`artifacts/train_showcase/heartbeat_rank0.jsonl`**
   - JSONL format with per-step metrics
   - Fields: step, loss, toks_per_s, gpu_mem_alloc, gate metrics, etc.
   - Updated every 20 steps

2. **`artifacts/train_showcase/training.csv`**
   - CSV format: step, loss, learning_rate, tokens_per_second
   - Easier to parse for plotting
   - Example last entry: `1580,1.370123,2.992615e-04,412`

3. **`artifacts/train_showcase/train_50k.log`**
   - Human-readable training log
   - Shows progress like: `step 1580 | loss 1.3701 | lr 2.99e-04 | toks/s 412`

### TensorBoard Data
- **Location**: `artifacts/train_showcase/tb/`
- **Access**: http://216.81.248.67:6006
- Contains event files with full metrics history

## How to Monitor Without SSH

### Option 1: Parse CSV for Metrics
```python
# Read the last line of training.csv
# Format: step,loss,learning_rate,tokens_per_second
last_line = "1580,1.370123,2.992615e-04,412"
step, loss, lr, toks = last_line.split(',')
```

### Option 2: Parse JSONL Heartbeat
```python
# Each line is a JSON object
import json
line = '{"step": 1580, "loss": 1.3701, "toks_per_s": 411.69, ...}'
data = json.loads(line)
print(f"Step {data['step']}: Loss={data['loss']:.4f}")
```

## Training Configuration
- **Config File**: `configs/train_50k.yaml`
- **Model**: 8 heads, dim=128, NSA with l=16, d=8, w=64
- **Dataset**: FineWeb-Edu (streaming)
- **Batch Size**: 8 global (4 per GPU)
- **Sequence Length**: 128 tokens
- **Optimizer**: AdamW with cosine LR schedule

## Previous Issue Resolution
- **Problem**: Training was stopping at 200 steps
- **Root Cause**: Script uses `CONFIG` env var, not `--config` flag
- **Solution**: Set `CONFIG=configs/train_50k.yaml` when launching
- **Verification**: Training successfully passed step 200 and continuing

## Expected Milestones
- Step 5,000: First checkpoint save (~3 hours)
- Step 10,000: Second checkpoint (~6 hours)
- Step 25,000: Mid-training checkpoint (~15 hours)
- Step 50,000: Training complete (~30 hours)

## Health Indicators
✅ **Healthy Signs**:
- Loss steadily decreasing (5.75 → 1.37)
- Learning rate following cosine schedule
- Consistent ~400 tokens/second throughput
- Memory usage stable at 21GB
- No fallback counters increasing
- Gate entropy healthy (no collapse)

⚠️ **Watch for**:
- Loss plateauing or increasing
- Throughput dropping below 300 tok/s
- Memory spikes above 70GB
- Training stopping unexpectedly

## Commands for Other Agent (if needed)
If the other agent needs to check status locally:
```bash
# Check latest metrics from local copy
tail -10 artifacts/train_showcase/training.csv

# Parse heartbeat for detailed metrics
tail -5 artifacts/train_showcase/heartbeat_rank0.jsonl | jq .

# Check training log
tail -20 artifacts/train_showcase/train_50k.log
```