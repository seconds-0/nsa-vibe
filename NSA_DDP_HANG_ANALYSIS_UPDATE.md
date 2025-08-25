# NSA DDP Hang Analysis - Update After Patch

**Date**: 2025-08-25 20:15 UTC  
**Test Engineer**: Claude  
**Environment**: Prime Intellect A100 80GB (ubuntu@216.81.248.49)  
**Repository**: nsa-vibe @ commit 085317b5 + patched train_showcase.py  

## Executive Summary

The DDP backward pass hang persists even after applying the `find_unused_parameters=True` patch. The issue appears deeper than parameter tracking and likely involves the NSA architecture's interaction with DDP's gradient synchronization mechanism.

## Patch Applied

**Changes made to `scripts/train_showcase.py`**:
```python
# Line 378: Changed from False to True
find_unused_parameters=True,

# Lines 381-386: Wrapped static graph behind env check
_use_static = os.getenv("NSA_DDP_STATIC_GRAPH", "0").lower() in ("1", "true", "yes")
if _use_static:
    try:
        model._set_static_graph()
    except Exception:
        pass
```

## Test Results After Patch

### Test 1: DDP Synthetic Shake
**Command**:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset synthetic
```

**Environment**:
- PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
- TORCH_DISTRIBUTED_DEBUG=DETAIL
- NCCL_DEBUG=INFO  
- NCCL_ASYNC_ERROR_HANDLING=1

**Result**: ❌ HANG
- NCCL initialization successful
- Warning: "find_unused_parameters=True was specified but did not find any unused parameters"
- Hang at step 1 during `loss.backward()`
- Timeout after 5 minutes, required SIGKILL

### Test 2: DDP with P2P Disabled
**Command**:
```bash
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset synthetic
```

**Result**: ❌ HANG
- Same behavior as Test 1
- P2P communication not the root cause

## Stack Trace Analysis

**Consistent hang location across all attempts**:
```python
File "scripts/train_showcase.py", line 661, in main
    loss.backward()
File "torch/autograd/graph.py", line 825
    return Variable._execution_engine.run_backward()  # <-- Hangs here
```

## Key Observations

1. **NCCL Communication**:
   - Successfully establishes (4 coll channels, 4 p2p channels)
   - No "Reducer got error" messages
   - Rank synchronization appears normal initially

2. **Memory Usage**:
   - Very low (~353-451 MB) when hanging
   - Not an OOM issue

3. **Warning Message**:
   - "find_unused_parameters=True...but did not find any unused parameters"
   - Suggests all parameters are used, but DDP still can't synchronize

4. **Single vs Multi-GPU**:
   - Single-GPU: Works perfectly (49GB memory, completes in ~233s)
   - Multi-GPU: Consistent hang at first backward pass

## Root Cause Hypothesis

The issue is likely one of:

1. **NSA Dynamic Graph Structure**
   - Selection mechanism creates dynamic computation graph
   - DDP expects consistent graph structure across ranks
   - Mismatch causes deadlock during gradient reduction

2. **Custom Backward Implementation**
   - NSA may have custom autograd functions
   - These might not properly signal completion to DDP's reducer

3. **Gradient Ready Order**
   - DDP expects gradients in specific order
   - NSA's three-branch architecture might violate this expectation

## Recommendations for Core Engineer

### Immediate Debugging Steps

1. **Test Vanilla Model**:
```python
# Replace NSA blocks with standard attention temporarily
# If this works, confirms NSA-specific issue
```

2. **Add Gradient Hooks**:
```python
for name, param in model.named_parameters():
    param.register_hook(lambda grad, name=name: print(f"Grad computed: {name}"))
```

3. **Force Synchronous Execution**:
```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
```

4. **Test with Gradient Accumulation Disabled**:
```python
# In NSA backward, ensure all gradients are produced
# Check for any conditional gradient computation
```

### Potential Fixes

1. **Explicit Synchronization**:
```python
# After loss.backward()
if dist.is_initialized():
    dist.barrier()
```

2. **Disable DDP During Debug**:
```python
# Train without DDP wrapper to isolate issue
# Use manual gradient averaging
```

3. **Check NSA Backward Implementation**:
- Ensure all autograd Functions properly implement backward
- Verify no conditional execution in backward pass
- Check for any in-place operations

## Artifacts

- Patched file: `scripts/train_showcase.py`
- Shake config: `configs/m7c_125m_2xa100_shake.yaml`
- Test logs: `synthetic_shake.log`
- Watchdog dumps: `artifacts/m7c_125m_2xa100_prod/watchdog_stackdump_*.txt`

## Conclusion

The `find_unused_parameters=True` patch did not resolve the DDP hang. The issue appears to be a fundamental incompatibility between NSA's dynamic selection mechanism and PyTorch DDP's gradient synchronization expectations. Single-GPU training remains the only working configuration.

**Next Steps**: 
1. Core engineer should investigate NSA's backward implementation
2. Consider implementing custom DDP wrapper for NSA
3. As fallback: proceed with single-GPU training (slower but functional)

---

**Report Generated**: 2025-08-25T20:15:00Z  
**Test Engineer**: Claude  
**Status**: DDP still blocked, single-GPU remains viable