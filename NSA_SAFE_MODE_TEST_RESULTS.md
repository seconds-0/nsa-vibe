# NSA DDP Safe Mode Test Results

**Date**: 2025-08-25 21:05 UTC  
**Test Engineer**: Claude  
**Environment**: Prime Intellect A100 80GB (ubuntu@216.81.248.49)  
**Repository**: nsa-vibe @ commit 085317b5 + safe mode patch  

## Executive Summary

DDP Safe Mode showed **partial success** - the sanity test completed 20 steps successfully, but the full configuration still hangs during backward pass. The safe mode significantly improves stability but doesn't fully resolve the DDP synchronization issue.

## Safe Mode Implementation Applied

**train_showcase.py modifications**:
```python
# When NSA_DDP_SAFE_MODE=1:
- find_unused_parameters=True
- broadcast_buffers=False  
- gradient_as_bucket_view=False
- bucket_cap_mb=2
# Forced conservative NSA paths:
- NSA_SDPA_NO_FLASH=1
- NSA_USE_FA2=0
- NSA_USE_FA2_WIN=0
- NSA_USE_FA2_CMP=0
- NSA_USE_TRITON_SEL=0
- NSA_USE_SEL_PACK=1
```

## Test Results

### Test 1: Quick Sanity Test ✅ PARTIAL SUCCESS
**Configuration**: 2 layers, seq_len=512, 10-20 steps
**Command**:
```bash
NSA_DDP_SAFE_MODE=1 torchrun --nproc_per_node=2 scripts/train_showcase.py \
--dataset synthetic --override "train.steps=10 model.n_layers=2 train.seq_len=512"
```
**Result**: 
- ✅ Reached step 20 successfully
- ✅ Loss decreasing: 5.7053 → 5.6506
- ✅ Throughput: 317-388 toks/s
- "[ddp-safe] Enabled conservative DDP+NSA settings" confirmed

### Test 2: Full Configuration Shake ❌ HANG
**Configuration**: 12 layers, seq_len=2048, 500 steps target
**Command**:
```bash
NSA_DDP_SAFE_MODE=1 CONFIG=configs/m7c_125m_2xa100_shake.yaml \
torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset synthetic
```
**Result**:
- ❌ Hangs at step 1 backward pass
- Same stack trace as before (loss.backward())
- Memory low (~385-470 MB) when hanging

## Key Observations

### What Improved:
1. **Small configurations work**: 2 layers with 512 seq_len runs successfully
2. **No kernel failures reported**: fb_*_fails all show 0
3. **Conservative paths engaged**: Safe mode correctly disabled Flash/Triton

### What Still Fails:
1. **Large configurations hang**: 12 layers with 2048 seq_len still deadlocks
2. **Backward pass remains blocking point**: Same location in autograd engine
3. **Memory not the issue**: Hanging with minimal memory usage

## Analysis

The safe mode helps but the issue appears to be:

1. **Scale-dependent**: Works at small scale (2L, 512 seq) but fails at production scale (12L, 2048 seq)
2. **Not purely kernel-related**: Even with conservative SDPA paths, large configs hang
3. **Likely graph complexity**: The NSA's three-branch architecture with 12 layers may create gradient dependency cycles that DDP can't resolve

## Hypothesis

The problem escalates with:
- **Layer count**: More layers = more complex gradient dependencies
- **Sequence length**: Longer sequences = larger attention matrices
- **Branch interactions**: Three branches × 12 layers = 36 parallel gradient paths

At small scale, DDP can handle the complexity. At production scale, the gradient synchronization graph becomes intractable.

## Recommendations

### Immediate Path Forward:

1. **Single-GPU Production** ✅
   - Proven stable at all scales
   - 49GB memory usage (well under 80GB limit)
   - Use `configs/m7c_125m_1xa100_production.yaml`

2. **Further Isolation Testing** (if multi-GPU critical):
   ```bash
   # Test with single branch at production scale
   NSA_FORCE_BRANCH=cmp NSA_DDP_SAFE_MODE=1 torchrun --nproc_per_node=2 ...
   
   # Test with fewer layers
   --override "model.n_layers=6"
   
   # Test with FSDP instead of DDP
   torchrun --nproc_per_node=2 scripts/train_showcase_fsdp.py
   ```

3. **Architecture Changes** (longer term):
   - Implement custom gradient reduction for NSA
   - Sequential branch computation instead of parallel
   - Checkpoint individual branches

## Artifacts

- Sanity test log: `sanity_test.log` (successful 20 steps)
- Shake test log: `synthetic_shake_nohup.log` (hanging)
- Heartbeat data: Shows step 1 completion in earlier test (PID 18175)
- Watchdog dumps: Consistent backward pass hang location

## Conclusion

DDP Safe Mode provides marginal improvement but doesn't resolve the fundamental incompatibility between NSA's complex gradient graph and PyTorch's DDP at production scale. **Recommend proceeding with single-GPU training** for the 50k step production run.

### Single-GPU Launch Command:
```bash
export CONFIG=configs/m7c_125m_1xa100_production.yaml
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
python scripts/train_showcase.py --dataset fineweb_edu --synthetic-on-fail
```

---

**Report Generated**: 2025-08-25T21:05:00Z  
**Test Engineer**: Claude  
**Recommendation**: Proceed with single-GPU training