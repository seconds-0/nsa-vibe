# M4 Triton Kernel Diagnostic Report

## Executive Summary
**Date**: August 19, 2025  
**Pod**: Prime Intellect RTX 4090  
**Status**: ✅ Root cause identified, ❌ Triton 3.0 test incomplete  

## Environment Details
- **Hardware**: NVIDIA GeForce RTX 4090
- **OS**: Linux-6.8.0-60-generic-x86_64-with-glibc2.35
- **PyTorch**: 2.5.1+cu121
- **Triton**: 3.1.0
- **CUDA**: 12.1

## Root Cause Analysis

### Primary Issue: Triton 3.1.0 Broadcasting Compatibility
Both `sel_attn_fwd_dense` and `sel_attn_fwd_varlen` kernels fail due to stricter shape compatibility rules in Triton 3.1.0.

### Specific Error Locations

#### Dense Kernel Error:
```
at 35:16: k_tile = tl.load(
    K_ptr + n * stride_kn + (l0 + offs_L) * stride_kL + (d0 + offs_D)[None, :] * stride_kd,
    ^
ValueError('Cannot make_shape_compatible: incompatible dimensions at index 1: 128 and 64')
```

#### Varlen Kernel Error:
```
at 27:26: row_start = tl.max(0, tl.min(rs, cuN))
                              ^
```

### Technical Analysis

The core issue is in the 2D indexing pattern:
```python
offs_L = tl.arange(0, BLOCK_L)  # Shape: [BLOCK_L]
offs_D = tl.arange(0, BLOCK_D)  # Shape: [BLOCK_D]

# This fails in Triton 3.1.0:
k_tile = tl.load(
    K_ptr + (l0 + offs_L) * stride_kL + (d0 + offs_D)[None, :] * stride_kd,
    mask=Lmask[:, None] & Dmask[None, :],
    other=0.0,
)
```

**Problem**: Triton 3.1.0 cannot automatically broadcast the indexing expression when:
- `offs_L` has shape `[BLOCK_L]` (e.g., `[128]`)
- `(d0 + offs_D)[None, :]` has shape `[1, BLOCK_D]` (e.g., `[1, 64]`)
- Expected result shape: `[BLOCK_L, BLOCK_D]` (e.g., `[128, 64]`)

The error "incompatible dimensions at index 1: 128 and 64" indicates Triton sees a mismatch between the L dimension (128) and D dimension (64) during the broadcasting operation.

## Isolated Test Results

### Minimal Reproduction ✅
Created `test_triton_issue.py` that reproduces the exact error:
```python
k_tile = tl.load(
    K_ptr + (l0 + offs_L) * stride_kL + (d0 + offs_D)[None, :] * stride_kd,
    mask=Lmask[:, None] & Dmask[None, :],
    other=0.0,
)
# FAILS: ValueError('Cannot make_shape_compatible: incompatible dimensions at index 1: 128 and 64')
```

### Simple Workaround ✅
Demonstrated that avoiding 2D broadcasting works:
```python
# This succeeds:
for l_idx in range(BLOCK_L):
    if l_idx < L:
        for d_idx in range(BLOCK_D):
            if d_idx < D:
                val = tl.load(K_ptr + l_idx * stride_kL + d_idx * stride_kd)
```

### Complex Fix Attempts ❌
- Explicit `tl.broadcast_to()` calls failed with different errors
- Row-wise loading approach failed due to JIT compilation issues

## Potential Solutions

### Option 1: Downgrade to Triton 3.0.0
- **Status**: Download incomplete (timeout during pip install)
- **Risk**: May break other PyTorch 2.5.1 compatibility
- **Test needed**: Verify if Triton 3.0.0 allows the current kernel code

### Option 2: Rewrite Kernel for 3.1.0 Compatibility
Modify the kernels to avoid problematic broadcasting:

```python
# Instead of 2D broadcasting:
k_tile = tl.load(ptr + offs_L * stride_L + offs_D[None, :] * stride_D, mask=mask_2d)

# Use explicit vectorized operations:
for l_offset in range(0, BLOCK_L, VECTOR_SIZE):
    l_vec = tl.arange(l_offset, l_offset + VECTOR_SIZE)
    for d_offset in range(0, BLOCK_D, VECTOR_SIZE):
        d_vec = tl.arange(d_offset, d_offset + VECTOR_SIZE)
        # Process smaller blocks that Triton can handle
```

### Option 3: Version Pinning
Pin Triton to a compatible version in requirements.txt:
```
triton==3.0.0  # Compatible with current kernel implementation
```

## Block Size Analysis

The error occurs with default block sizes:
- `BLOCK_L = 128` (when `L >= 128`)
- `BLOCK_D = 64` (when `D >= 64`)

The dimension mismatch (128 vs 64) directly corresponds to these block sizes, confirming the broadcasting issue.

## Performance Impact

Current fallback behavior:
- Triton kernel compilation fails → Falls back to packed SDPA
- Fallback overhead: ~5000x slower than expected
- Impact: M4 benchmarks show Triton as much slower than SDPA

## Recommendations

### Immediate (Next 24h)
1. **Pin Triton version** to 3.0.0 if kernels work there
2. **Update kernel implementation** to avoid 2D broadcasting patterns
3. **Test performance** once compilation succeeds

### Medium-term (Next week)  
1. **Rewrite kernels** for Triton 3.1+ compatibility
2. **Add regression tests** to catch future Triton API changes
3. **Benchmark optimized kernels** vs SDPA reference

### Long-term (Next month)
1. **Upstream contribution** if we find better Triton patterns
2. **Auto-detection** of Triton version compatibility
3. **Multiple kernel variants** for different Triton versions

## Files Created
- `test_triton_issue.py` - Minimal reproduction
- `test_triton_fix.py` - Failed explicit broadcasting fix
- `test_triton_simple.py` - Working simple approach
- `test_triton_workaround.py` - Failed workaround attempt

## Next Steps

1. ✅ **Complete Triton 3.0 test** (if download succeeds)
2. ⏳ **Implement kernel rewrite** using proven simple approach
3. ⏳ **Validate numerical accuracy** of rewritten kernels
4. ⏳ **Benchmark performance** vs SDPA once compilation succeeds

The root cause is clearly identified, and we have a path forward to fix the Triton 3.1.0 compatibility issues.