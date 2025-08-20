# M4 Triton Kernel Debug & Fix Documentation

## Executive Summary
The Triton selection kernels fail with an MLIR compilation error. Multiple fix attempts were made, with partial success achieved for simple cases but complete failure for complex multi-range patterns.

## Problem Timeline & Fix Attempts

### Initial Problem Discovery
**Error**: `ValueError('Cannot make_shape_compatible: incompatible dimensions at index 1: 128 and 64')`
**Location**: `nsa/kernels/triton_sel_kernel/sel_fwd.py` lines 59, 84, 150, 173
**Root Cause**: Triton 3.1.0 rejects implicit broadcasting pattern

Original failing code:
```python
k_tile = tl.load(
    K_ptr + n * stride_kn + (l0 + offs_L) * stride_kL + (d0 + offs_D)[None, :] * stride_kd,
    mask=Lmask[:, None] & Dmask[None, :],
    other=0.0,
)
```

### Fix Attempt #1: Explicit Pointer Arithmetic (FAILED)
**Approach**: Use explicit 2D pointer construction with `tl.broadcast_to`
**Commit**: 0b315ea5 (cherry-picked from feat/m5-plan)

Changed code:
```python
rows = (l0 + offs_L).to(tl.int32)[:, None]
cols = (d0 + offs_D).to(tl.int32)[None, :]
k_ptrs = K_ptr + n * stride_kn + rows * stride_kL + cols * stride_kd
k_tile = tl.load(
    k_ptrs,
    mask=tl.broadcast_to(Lmask[:, None], k_ptrs.shape) & tl.broadcast_to(Dmask[None, :], k_ptrs.shape),
    other=0.0,
)
```

**Result**: WORSE - MLIR crash with `Assertion 'isIntOrFloat() && "only integers and floats have a bitwidth"' failed`
**Why it failed**: `k_ptrs` is a pointer expression, not a tensor - has no `.shape` attribute

### Fix Attempt #2: Remove k_ptrs.shape Usage (PARTIAL SUCCESS)
**Approach**: Replace `k_ptrs.shape` with direct mask expressions

Final working code (for simple cases):
```python
rows = (l0 + offs_L).to(tl.int32)[:, None]
cols = (d0 + offs_D).to(tl.int32)[None, :]
k_ptrs = K_ptr + n * stride_kn + rows * stride_kL + cols * stride_kd
k_tile = tl.load(
    k_ptrs,
    mask=Lmask[:, None] & Dmask[None, :],
    other=0.0,
)
```

**Result**: 
- ✅ Works for dense kernel with single contiguous ranges
- ❌ Fails for multi-range patterns with same MLIR error
- ❌ Varlen kernel still completely broken

### Triton Version Testing
**Test**: Downgraded from Triton 3.1.0 to 3.0.0
**Result**: SAME MLIR error in both versions
**Conclusion**: Not a Triton regression - our kernel pattern is fundamentally problematic

## Current Kernel Status

### What Works
1. **Dense kernel**: Single contiguous range per row (dist="few")
   - Example: `ranges = [[[[0, 128]]]]`
   - Successfully compiles and runs
   - Performance: ~3000x slower than SDPA

### What Fails
1. **Dense kernel**: Multiple ranges per row (dist="many" or "mixed")
   - Example: `ranges = [[[[0,16], [32,48], [64,80]]]]`
   - MLIR crash during compilation
   
2. **Varlen kernel**: All cases fail
   - Uses similar problematic pattern
   - MLIR crash even for simple cases

## Error Logs & Diagnostics

### MLIR Compilation Error (Full)
```
python: /source/llvm-project/mlir/lib/IR/Types.cpp:126: 
unsigned int mlir::Type::getIntOrFloatBitWidth() const: 
Assertion `isIntOrFloat() && "only integers and floats have a bitwidth"' failed.
Aborted (core dumped)
```

### Environment Details
- **GPU**: NVIDIA GeForce RTX 4090
- **CUDA**: 12.1
- **PyTorch**: 2.5.1+cu121
- **Triton**: 3.0.0 and 3.1.0 (both fail)
- **OS**: Ubuntu 22.04

### Performance Measurements (When Working)
```
N=256, H=8, D=128, L=128, dist=few:
- SDPA: 0.018ms
- Triton: 54.727ms
- Speedup: 0.0003x (3000x slower)

N=1024, H=8, D=128, L=256, dist=few:
- SDPA: 0.018ms  
- Triton: 218.896ms
- Speedup: 0.00008x
```

## Root Cause Analysis

### Why The Current Approach Fails
1. **2D Pointer Arithmetic**: Creating 2D pointer expressions with broadcasting is not well-supported in Triton
2. **Type System Issues**: The `.to(tl.int32)[:, None]` pattern creates type confusion in MLIR
3. **Shape Broadcasting**: Complex shape manipulations trigger MLIR's type checker failures

### Why Partial Fix Works (Sometimes)
- Simple single-range cases avoid some code paths that trigger the MLIR issue
- The mask `Lmask[:, None] & Dmask[None, :]` works when shapes are simple
- Multi-range cases likely trigger different compilation paths that expose the underlying issue

## Recommended Solutions

### Option 1: Complete Kernel Rewrite (Recommended)
Instead of 2D broadcasting, use explicit loops:
```python
# Pseudo-code for safer approach
for l_idx in range(BLOCK_L):
    for d_idx in range(BLOCK_D):
        if (l0 + l_idx < L) and (d0 + d_idx < D):
            addr = K_ptr + n * stride_kn + (l0 + l_idx) * stride_kL + (d0 + d_idx) * stride_kd
            k_val = tl.load(addr)
            # Process k_val
```

### Option 2: Simplified 1D Approach
Process L and D dimensions separately to avoid 2D complications:
```python
# Load full L vector for each D element
for d_idx in range(0, D, BLOCK_D):
    k_vec = tl.load(K_ptr + offs_L * stride_kL + d_idx * stride_kd, mask=Lmask)
    # Process without 2D broadcasting
```

### Option 3: Wait for Triton Maturity
- Keep high threshold (`NSA_SEL_TRITON_MIN_L=2048`)
- Use SDPA fallback for all production cases
- Revisit when Triton has better 2D support

## Files Modified

### Primary Kernel File
`nsa/kernels/triton_sel_kernel/sel_fwd.py`:
- Lines 58-65: Dense kernel K load (partially fixed)
- Lines 83-91: Dense kernel K reload for pass 2
- Lines 94-102: Dense kernel V load  
- Lines 150-157: Varlen kernel K load (still broken)
- Lines 172-180: Varlen kernel K reload
- Lines 183-191: Varlen kernel V load

### Test Infrastructure
- Created: `test_triton_issue.py`, `fixed_kernel.py` (on pod, not committed)
- Modified: Kernel launch parameters (removed conflicting num_warps/num_stages)

## Production Settings

### Current Safe Configuration
```bash
export NSA_USE_TRITON_SEL=1        # Enable wrapper with fallback
export NSA_SEL_TRITON_MIN_L=2048   # High threshold forces SDPA fallback
export NSA_DEBUG_TIMING=1          # Monitor performance
```

### Verification Commands
```python
# Test if dense kernel works (simple case)
import torch
from nsa.kernels.triton_sel_kernel.sel_fwd import sel_attn_fwd_dense
N,H,D,Dv,L = 1,1,64,64,64
Q = torch.randn(N,H,D,device='cuda',dtype=torch.float16)
K = torch.randn(N,L,D,device='cuda',dtype=torch.float16)
V = torch.randn(N,L,Dv,device='cuda',dtype=torch.float16)
O = sel_attn_fwd_dense(Q,K,V)  # Should work

# Test wrapper with fallback
from nsa.kernels.triton_sel_kernel import selection_attention_triton
ranges = torch.tensor([[[[0,64]]]], device='cuda')
O = selection_attention_triton(Q.unsqueeze(0).unsqueeze(0), 
                               K.unsqueeze(0), 
                               V.unsqueeze(0), ranges)
```

## Key Learnings

1. **Triton Limitations**: 2D broadcasting with pointer arithmetic is fragile
2. **MLIR Type System**: Strict about type consistency - mixing int32 conversions with broadcasting fails
3. **Testing Gap**: Need separate test suites for simple vs complex patterns
4. **Fallback Value**: Robust error handling more important than brittle optimizations

## Next Agent Actions

1. **If continuing kernel fix**: Focus on complete rewrite with simpler patterns
2. **If accepting current state**: Document NSA_SEL_TRITON_MIN_L=2048 as permanent setting
3. **For debugging**: Set TRITON_DEBUG=1 and MLIR_ENABLE_DUMP=1 for deeper diagnostics
4. **Alternative approach**: Consider pure CUDA kernel implementation

## Summary for Next Agent

The Triton kernel is partially fixed but only works for the simplest cases. The fundamental issue is that Triton's MLIR backend cannot handle our 2D pointer arithmetic pattern. The safe path forward is either a complete rewrite using simpler patterns or accepting SDPA fallback as the permanent solution. The current wrapper with high threshold (NSA_SEL_TRITON_MIN_L=2048) ensures production safety.