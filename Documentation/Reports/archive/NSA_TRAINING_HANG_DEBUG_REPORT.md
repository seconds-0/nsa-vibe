# NSA Training Hang Debug Report

**Date**: 2025-08-25  
**Environment**: Prime Intellect A100 80GB (ubuntu@216.81.248.67)  
**Branch**: feat/m7c-perf-stability  

## Executive Summary

The production training configuration causes a hang during backward pass when using 5+ NSA layers. The issue is specific to the NSA implementation and not related to BF16, gradient checkpointing, or data loading.

## Critical Finding

**NSA backward pass hangs at ≥5 layers**, while standard transformers work fine with 12 layers.

## Test Results

### Configuration Comparison

**Working Config (train_showcase.yaml)**:
- Model: dim=128, n_layers=1, n_heads=8
- Runtime: FP32, no flash, no grad checkpointing
- Training: seq_len=128, batch_size=8
- Status: ✅ Completed 35k+ steps successfully

**Failed Config (m7c_125m_2xa100_production.yaml)**:
- Model: dim=768, **n_layers=12**, n_heads=12
- Runtime: BF16, flash enabled, grad checkpointing enabled
- Training: seq_len=2048, batch_size=2
- Status: ❌ Hangs at step 1

### Systematic Testing Results

#### 1. Data Loading: ✅ Works
- FineWeb-Edu streaming: Loads successfully (11.93s first batch)
- Synthetic data: Generates correctly
- Dataset iteration: No issues

#### 2. Forward Pass: ✅ Works
```python
# All forward passes complete successfully:
- Small model (1 layer, dim=128): OK
- Large model (12 layers, dim=768): OK
- With gradient checkpointing: OK
- With BF16 precision: OK
```

#### 3. Backward Pass: ❌ HANGS
Layer threshold testing results:
```
1 layer:  ✓ OK (completes in <1s)
2 layers: ✓ OK (completes in <1s)
3 layers: ✓ OK (completes in <1s)
4 layers: ✓ OK (completes in <2s)
5 layers: ✗ HANG (no completion after 30s)
6+ layers: ✗ HANG (no completion after 30s)

Standard PyTorch Transformer (12 layers): ✓ OK
```

#### 4. GPU State During Hang
```
GPU 0: 100% utilization, 73,260 MiB / 81,920 MiB memory used
GPU 1: 0% utilization, 4 MiB / 81,920 MiB memory used (idle)
Process: Unresponsive to SIGUSR1, requires SIGKILL
```

### Root Cause Analysis

#### The issue is NOT caused by:
- ❌ **BF16 precision** - A100 supports BF16, forward pass works
- ❌ **Gradient checkpointing** - Hangs without it too
- ❌ **Data loading** - Synthetic data also hangs
- ❌ **DDP/multi-GPU** - Single GPU also hangs
- ❌ **Flash attention** - Disabled doesn't help
- ❌ **Large model size** - Small dim=128 also hangs with many layers
- ❌ **Sequence length** - Short seq_len=128 also hangs
- ❌ **PyTorch/CUDA** - Standard transformers work fine

#### The issue IS caused by:
- ✅ **NSA-specific backward pass implementation**
- ✅ **Triggered when n_layers ≥ 5**
- ✅ **Accumulates with multiple NSA layers**
- ✅ **Extreme memory usage** (73GB for small model!)
- ✅ **Likely bug in gradient computation** for selection/compression branches

### Minimal Reproduction

```python
# This code reproduces the hang:
import torch
from scripts.train_showcase import TinyLM

# Works with 4 layers
model = TinyLM(256, 128, 4, 8, 2, 16, 16, 16, 8, 32, 8, 64).cuda()
x = torch.randint(0, 256, (1, 128)).cuda()
loss = model(x).mean()
loss.backward()  # Completes

# Hangs with 5 layers
model = TinyLM(256, 128, 5, 8, 2, 16, 16, 16, 8, 32, 8, 64).cuda()
x = torch.randint(0, 256, (1, 128)).cuda()
loss = model(x).mean()
loss.backward()  # HANGS HERE
```

## Recommendations for Core Engineer

### 1. Immediate Workaround
- **Limit to 4 layers maximum** for any training runs
- Use smaller models until backward pass is fixed
- Consider running multiple 4-layer models in parallel

### 2. Debug Focus Areas

#### Primary Suspects:
1. **Selection scorer backward** (`nsa/core/selection_scorer.py`)
   - Complex indexing operations
   - Possible gradient retention across layers
   
2. **Compressed branch** (`nsa/core/compress_pool.py`)
   - Pooling operations gradient flow
   - Block overlap handling

3. **Block index operations** (`nsa/core/block_index.py`)
   - CSR matrix gradient computation
   - Index gathering operations

#### Debugging Strategy:
```python
# Add gradient hooks to identify where backward stalls:
def debug_backward(module, grad_input, grad_output):
    print(f"{module.__class__.__name__}: grad_output shape {grad_output[0].shape if grad_output[0] is not None else None}")
    return grad_input

# Register on each NSA layer
for i, layer in enumerate(model.layers):
    layer.nsa_attn.register_backward_hook(debug_backward)
```

### 3. Memory Leak Investigation
The 73GB memory usage for a tiny model suggests severe memory leak:
- Check for gradient accumulation without release
- Look for circular references in autograd graph
- Verify proper tensor detachment in selection logic
- Review custom autograd functions if any

### 4. Specific Code Areas to Review

```python
# Files requiring immediate attention:
nsa/core/nsa_attention.py        # Main NSA attention, line 200-400 (backward flow)
nsa/core/selection_scorer.py     # Lines 150-250 (score computation gradients)
nsa/core/compress_pool.py        # Lines 50-150 (pooling backward)
scripts/train_showcase.py        # Lines 100-150 (TinyLM layer stacking)

# Potential issues to check:
- Gradient flow through discrete selection operations
- Proper detachment of non-differentiable operations
- Memory retention in cached computations
- Interaction between multiple NSA layers' gradients
```

### 5. Testing Commands for Debugging

```bash
# Test with gradient anomaly detection:
PYTHONPATH=. python -c "
import torch
torch.autograd.set_detect_anomaly(True)
from scripts.train_showcase import TinyLM
model = TinyLM(256, 128, 5, 8, 2, 16, 16, 16, 8, 32, 8, 64).cuda()
x = torch.randint(0, 256, (1, 128)).cuda()
loss = model(x).mean()
loss.backward()
"

# Profile memory usage:
PYTHONPATH=. python -c "
import torch
from torch.profiler import profile, ProfilerActivity
from scripts.train_showcase import TinyLM

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             profile_memory=True, record_shapes=True) as prof:
    model = TinyLM(256, 128, 5, 8, 2, 16, 16, 16, 8, 32, 8, 64).cuda()
    x = torch.randint(0, 256, (1, 128)).cuda()
    loss = model(x).mean()
    loss.backward()
print(prof.key_averages().table(sort_by='cuda_memory_usage', row_limit=10))
"
```

## Impact

This is a **CRITICAL BLOCKER** for production training:
- Cannot train the intended 12-layer model
- Prevents using the production configuration
- Blocks 50k step training run
- Wastes GPU resources (A100s sitting idle)

## Temporary Solution

Until fixed, the only option is to:
1. Use maximum 4 layers
2. Reduce model capacity significantly
3. Consider alternative attention mechanisms
4. Wait for backward pass bug fix

---

**Report Generated**: 2025-08-25T06:30:00Z  
**Priority**: P0 - Critical Production Blocker  
**Estimated Impact**: ~$1000+/day in idle GPU costs