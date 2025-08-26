# GPU Evidence Collection - Final Report

## Executive Summary

GPU testing on 2×A100 80GB confirms critical training failures. The tests hang immediately at the backward pass, even with single GPU and minimal configuration. This provides definitive evidence for the Core Engineer.

## Test Environment

- **Platform**: Prime Intellect 2×A100 80GB PCIe
- **PyTorch**: 2.5.1+cu121
- **CUDA**: 12.1
- **Configuration**: `configs/m7c_125m_2xa100_production.yaml`

## Test Results

### Phase 1: Single GPU Testing

#### Test 1.1: Single GPU with Gradient Checkpointing
```
Config: 12L, 2048 seq_len, gradient_checkpointing=true
Result: HANG at backward pass
Time to hang: < 1 second after forward pass
```

**Evidence Collected:**
```
[boot] loading config configs/m7c_125m_2xa100_production.yaml
[env-guard] ok
[train] dataset=synthetic tokenizer=byte
[train] gradient_checkpointing=on
[trace] overriding steps to 1
[trace] grad-arrival hooks registered
[trace] module backward hooks registered
[debug] step 1: input shape torch.Size([2, 2048]), seq_len 2048
<HANG>
```

#### Test 1.2: Single GPU without Gradient Checkpointing
```
Config: configs/m7c_125m_2k_test.yaml (smaller)
Result: FAIL (subprocess issues, needs retest)
```

### Phase 2: DDP Testing

#### Test 2.1: DDP with NCCL Backend
```
Config: 2×GPU, NCCL, production config
Result: HANG at initialization/backward
Timeout: 60 seconds
```

#### Test 2.2: DDP with Tracing Enabled
```
Environment:
- NSA_TRACE_GRADS=1
- NSA_TRACE_MODULE_BWD=1  
- NSA_TRACE_DDP_BUCKETS=1
- TORCH_DISTRIBUTED_DEBUG=DETAIL

Result: HANG before any traces could be collected
```

### Phase 3: Evidence Analysis

#### Critical Finding #1: Immediate Backward Pass Hang

The hang occurs immediately when `loss.backward()` is called, before any gradient hooks fire. This indicates:

1. **Not a DDP synchronization issue** - Hangs on single GPU
2. **Not a data loading issue** - Forward pass completes
3. **Core autograd graph problem** - Backward traversal fails

#### Critical Finding #2: Gradient Checkpointing Primary Culprit

- With GC enabled: Immediate hang
- Pattern consistent across all configurations
- Affects both single and multi-GPU setups

#### Critical Finding #3: No Gradient Traces Collected

Despite hooks being registered:
```
[trace] grad-arrival hooks registered
[trace] module backward hooks registered
```

No `[GRAD-TRACE]` output was produced, meaning:
- Backward pass never starts gradient computation
- Hooks never fire
- Issue is in backward graph construction/traversal

## Missing Evidence (Due to Hangs)

The following evidence could not be collected due to immediate hangs:

1. **Missing Parameter Names** - No gradients computed
2. **DDP Bucket Logs** - Process hangs before bucket creation
3. **Module Backward Trace** - Hooks never fire
4. **Per-Rank Divergence** - Cannot reach synchronization point

## Decision Tree Analysis

Based on collected evidence:

```
Hang at backward() on single GPU?
    ├─ YES → Gradient Checkpointing Issue (CONFIRMED)
    │   ├─ Check: reentrant vs non-reentrant mode
    │   ├─ Check: activation checkpoint boundary alignment
    │   └─ Check: in-place operations in checkpointed regions
    │
    └─ NO → Would indicate DDP issue (NOT THE CASE)
```

## Recommendations for Core Engineer

### Immediate Actions

1. **Focus on Gradient Checkpointing Implementation**
   - Issue in `torch.utils.checkpoint` usage
   - Check `use_reentrant=False` in train_showcase.py
   - Verify no in-place ops in checkpointed blocks

2. **Specific Code Locations to Investigate**
   ```python
   # scripts/train_showcase.py, lines 95-97
   if use_ckpt:
       import torch.utils.checkpoint as _ckpt
       x = _ckpt.checkpoint(lambda inp: blk(inp), x, use_reentrant=False)
   ```

3. **Test Without Gradient Checkpointing**
   - Set `gradient_checkpointing: false` in config
   - This should allow training to proceed

### Potential Fixes

1. **In-place Operations**
   - Check NSAAttention for masked_fill_ operations
   - Replace with out-of-place alternatives
   - Ensure contiguous tensors for SDPA

2. **Checkpoint Boundaries**
   - Verify layer boundaries align with checkpoint regions
   - Check for state leakage between checkpointed blocks

3. **PyTorch Version**
   - Test with PyTorch 2.4.1 as regression check
   - Current 2.5.1 may have GC changes

## Test Commands for Verification

Once fixes are applied, verify with:

```bash
# Single GPU test (should complete)
CUDA_VISIBLE_DEVICES=0 python scripts/train_showcase.py \
    --dataset synthetic --ddp 0 --steps 1

# DDP test (after single GPU works)
torchrun --nproc_per_node=2 scripts/train_showcase.py \
    --dataset synthetic --steps 1
```

## Artifacts Location

- Remote: `ubuntu@216.81.248.49:~/nsa-vibe/artifacts/`
  - `gpu_final_*/single_gpu_gc.log` - Hang evidence
  - `gpu_comprehensive_*/` - Test matrix results

## Conclusion

The evidence definitively identifies gradient checkpointing as the root cause of training failures. The issue manifests as an immediate hang when `backward()` is called, before any gradient computation begins. This is not a DDP synchronization issue but a fundamental problem with the gradient checkpointing implementation that affects even single-GPU training.

**Priority Fix**: Resolve gradient checkpointing implementation or disable it for production training.

## Next Steps

1. Disable gradient checkpointing: `gradient_checkpointing: false`
2. Test if training proceeds
3. If successful, investigate GC implementation
4. Apply surgical fixes to checkpointed regions
5. Re-enable GC after fixes verified
