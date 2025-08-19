# M4 Triton Selection Benchmark Results

## Executive Summary
**Date**: August 19, 2025  
**Hardware**: NVIDIA GeForce RTX 4090 (Prime Intellect Pod)  
**Software**: PyTorch 2.5.1+cu121, Triton 3.1.0  
**Status**: ⚠️ Triton kernel has compatibility issues with Triton 3.1.0

## Key Findings

### 1. Triton Kernel Compilation Error
The direct Triton selection kernel (`sel_attn_fwd_dense`) fails to compile with Triton 3.1.0:
```
ValueError: Cannot broadcast, the expanded size of the tensor (1) must match the existing size (64) 
at non-singleton dimension 0: [64, 64], [1, 64]
```

This appears to be a breaking change in Triton 3.1.0's broadcasting semantics that affects the kernel's load operations.

### 2. Fallback Performance
When the Triton kernel fails, the wrapper correctly falls back to packed SDPA, but with significant overhead:
- **SDPA Direct**: ~0.019ms per operation
- **Triton Wrapper (falling back)**: ~100ms per operation
- **Overhead**: ~5000x slower due to fallback path overhead

### 3. Recommendations

#### Immediate Actions
1. **Keep `sel_triton_min_L` high (≥1024)** to avoid Triton path until kernel is fixed
2. **Fix Triton kernel** for compatibility with Triton 3.1.0
3. **Consider pinning Triton version** to 3.0.x if kernel worked there

#### Kernel Fix Required
The issue is in `nsa/kernels/triton_sel_kernel/sel_fwd.py` around line 34:
```python
k_tile = tl.load(
    k_base + (l0 + offs_L[:, None]) * stride_kl + (d0 + offs_D[None, :]) * stride_kd,
    mask=Lmask[:, None] & Dmask[None, :],  # Issue with broadcasting here
    other=0.0
)
```

## Test Configuration

### Environment
- **Pod**: Prime Intellect RTX 4090 (Ubuntu 22, CUDA 12)
- **Python**: 3.10.12
- **PyTorch**: 2.5.1+cu121
- **Triton**: 3.1.0
- **Flash-Attention**: 2.8.3

### Benchmark Parameters Tested
- **N** (batch): 64, 256
- **H** (heads): 4, 8  
- **D** (dim): 64, 128
- **L** (selected length): 16, 32, 64, 128, 256, 512
- **Distribution**: few (single contiguous range)

## Fallback Behavior

The wrapper (`selection_attention_triton`) correctly implements fallback logic:
1. Checks if Triton is enabled via `NSA_USE_TRITON_SEL`
2. Checks if selected length ≥ `NSA_SEL_TRITON_MIN_L`
3. Attempts Triton kernel execution
4. Falls back to packed SDPA on failure

This ensures correctness but with performance penalty when Triton kernel fails.

## Next Steps

1. **Fix Triton Kernel**: Update broadcasting logic for Triton 3.1.0 compatibility
2. **Re-benchmark**: Once fixed, run full benchmark matrix:
   - Small (N=64), Medium (N=256), Large (N=1024) configurations
   - Various distributions (few, many, mixed)
   - Determine optimal `sel_triton_min_L` threshold
3. **Update Configuration**: Set production defaults based on benchmarks

## Appendix: Successful Wrapper Test

Despite kernel issues, the high-level wrapper works correctly:
```python
# This works (uses fallback):
O = selection_attention_triton(Q, K, V, ranges)  # ✅

# This fails (direct kernel):
O = sel_attn_fwd_dense(Q, K, V)  # ❌ Broadcasting error
```

This confirms the wrapper's defensive programming is working as intended.