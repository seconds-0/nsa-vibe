# NSA Performance Critical Settings

## The 1000x Performance Bug

The NSA model has a critical routing bug that can cause training to run at **<10 tok/s instead of 9,200+ tok/s** - a 1000x performance degradation.

### Root Cause

NSA has multiple attention implementations for the selection branch:
1. **Fast path** (`grouped_selection_attention_masked`): Fully vectorized, 9,200+ tok/s
2. **Slow path** (`_sdpa_over_ranges`): Nested Python loops, <10 tok/s

The slow path uses Python loops iterating over batches and groups:
```python
for b in range(B):
    for g in range(G):
        # Process each batch/group individually - catastrophically slow
```

### The Fix

**ALWAYS set this environment variable:**
```bash
export NSA_FORCE_SEL_MASK=1
```

This forces the fast masked attention path, bypassing the routing logic that may choose the slow path.

## Critical Performance Checklist

### 1. Selection Path (1000x impact)
```bash
# MANDATORY - without this, training is unusable
export NSA_FORCE_SEL_MASK=1
```

### 2. Gradient Checkpointing (3-4x impact)
```yaml
# In your config file
runtime:
  gradient_checkpointing: false  # MUST be false
```

### 3. FA-2 Settings (2.4-3.4x impact)
```bash
# FA-2 is SLOWER than SDPA - keep it disabled
export NSA_USE_FA2=0  # Default, but explicit is better
```

### 4. Batched Prefill (10-15% impact)
```bash
export NSA_PREFILL_BATCHED=1
```

## Complete Production Launch

```bash
# All critical performance flags
export NSA_FORCE_SEL_MASK=1      # MANDATORY: 1000x speedup
export NSA_PREFILL_BATCHED=1     # 10-15% improvement
export NSA_USE_FA2=0              # Avoid FA-2 slowdown

# Launch training
CONFIG=configs/m7c_125m_2xa100_production.yaml \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 1
```

## Performance Expectations

With correct settings:
- **A100 80GB**: 9,200-9,400 tok/s
- **H100 80GB**: 15,000-18,000 tok/s
- **RTX 4090**: 4,000-6,000 tok/s

Without `NSA_FORCE_SEL_MASK=1`:
- **All GPUs**: <10 tok/s (unusable)

## Debugging Performance Issues

### Symptoms of the Bug
1. Training shows 0-10 tok/s
2. GPU utilization is low (<30%)
3. Python CPU usage is high
4. Warning in logs: "Using slow _sdpa_over_ranges"

### Quick Diagnostic
```bash
# Check if the flag is set
echo $NSA_FORCE_SEL_MASK  # Should output "1"

# Check training logs for warnings
grep "slow _sdpa_over_ranges" training.log
```

### If Performance is Still Poor
1. Verify `NSA_FORCE_SEL_MASK=1` is exported
2. Check gradient checkpointing is OFF in config
3. Ensure FA-2 is not enabled (`NSA_USE_FA2=0`)
4. Look for the warning about slow path in logs

## Why This Happens

The slow path exists as a compatibility fallback but should never be used in production. Due to a routing bug, the system may choose it even when the fast path is available. Setting `NSA_FORCE_SEL_MASK=1` bypasses the faulty routing logic.

## Prevention

As of the latest update, we've:
1. Changed the default value of `force_sel_mask` to `"1"` in the code
2. Added runtime warnings when the slow path is used
3. Updated all production configs with warning comments
4. Documented the issue prominently

However, **always explicitly set `NSA_FORCE_SEL_MASK=1`** to be safe.