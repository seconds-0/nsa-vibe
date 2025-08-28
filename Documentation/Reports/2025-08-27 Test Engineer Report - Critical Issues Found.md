# Critical Issues Found - Production 50K Launch

## Executive Summary

**Status: BLOCKED** ❌

Multiple critical issues prevent successful production training launch:

1. **Initial False Success**: Training appeared to run at 970+ toks/s but was using wrong config (toy model)
2. **Config Loading Bug**: CONFIG env var not being passed correctly to training script
3. **DDP Data Sharding Bug**: batch_size=1 causes IndexError on rank 1 with FineWeb-Edu

## Issues Identified

### Issue 1: Wrong Model Config (RESOLVED)
- **Problem**: Training used default `train_showcase.yaml` (128 dim, 128 seq_len) instead of production config
- **Root Cause**: CONFIG environment variable not exported in launch script
- **Fix**: Added `export CONFIG=configs/m7c_125m_2xa100_production.yaml`
- **Status**: ✅ Fixed

### Issue 2: DDP Data Sharding with batch_size=1 (BLOCKING)
- **Problem**: IndexError on rank 1: "too many indices for tensor of dimension 1"
- **Occurs with**: FineWeb-Edu dataset + batch_size=1 + DDP
- **Root Cause**: Data loader distributes batch_size=1 across 2 GPUs, leaving rank 1 with empty tensor
- **Attempted Fixes**:
  - Using batch_size=2: Still fails with same error
  - Synthetic data: Works correctly
- **Status**: ❌ BLOCKING - FineWeb-Edu data loading incompatible with DDP at batch_size=1

## What Works vs What Doesn't

### ✅ Working Configurations:
1. **Synthetic data + batch_size=1 + DDP**: ~41 toks/s with correct 125M model
2. **Single GPU + any dataset**: Works but not distributed

### ❌ Failing Configurations:
1. **FineWeb-Edu + batch_size=1 + DDP**: IndexError on rank 1
2. **FineWeb-Edu + batch_size=2 + DDP**: Same IndexError

## Evidence

### Correct Model Now Loading:
```
[boot] loading config configs/m7c_125m_2xa100_production.yaml
[debug] step 1: input shape torch.Size([1, 2048]), seq_len 2048
```

### GPU Memory Usage (Correct):
- GPU 0: 16.7 GB (expected for 125M model)
- GPU 1: 1.2 GB (minimal, likely just NCCL buffers)

### Error on Rank 1:
```python
[rank1]: File "/home/ubuntu/nsa-vibe/scripts/train_showcase.py", line 925, in main
[rank1]:     y = x[:, 1:].contiguous()
[rank1]: IndexError: too many indices for tensor of dimension 1
```

## Root Cause Analysis

The FineWeb-Edu data loader's sharding logic doesn't handle batch_size=1 correctly with DDP:
- Rank 0 gets shape [1, 2048] ✅
- Rank 1 gets shape [2048] or empty ❌

This suggests the data loader is dividing batch_size=1 by world_size=2, resulting in rank 1 getting no complete samples.

## Recommendations

### Immediate Options:

1. **Use batch_size=2 minimum** for FineWeb-Edu with DDP
   - Modify production config to use batch_size=2
   - Accept slightly different training dynamics

2. **Fix data loader sharding logic**
   - Ensure each rank gets at least 1 sample
   - May require changes to fineweb_stream_batches function

3. **Use synthetic data for initial validation**
   - Complete 1000 steps with synthetic to verify model
   - Then debug FineWeb-Edu separately

4. **Fall back to single GPU**
   - Use the single-GPU command from core engineer
   - Slower but functional

## Current State

- Training processes still running but stuck
- GPU 0 at 100% utilization (spinning on error recovery?)
- No progress past step 1 with FineWeb-Edu

## Next Steps

Need guidance from core engineering team on:
1. Is batch_size=2 acceptable for production?
2. Can the FineWeb-Edu data loader be fixed for batch_size=1?
3. Should we proceed with synthetic data validation first?

---

*Test Engineer Report - Critical Issues Found*
*Status: BLOCKED on data loader issue*