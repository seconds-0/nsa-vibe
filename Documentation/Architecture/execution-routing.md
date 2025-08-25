# NSA Execution Routing Guide

This document explains how the NSA (Native Sparse Attention) system chooses between different attention implementations based on environment flags, hardware capabilities, and execution context.

## Overview

NSA implements a hierarchical routing system that selects the optimal attention kernel for each branch (Compressed, Selected, Sliding) based on:

1. **Environment flags** - User preferences and feature toggles
2. **Hardware detection** - GPU architecture and capability checks  
3. **Execution mode** - Prefill vs decode, sequence length, batch size
4. **Fallback safety** - Graceful degradation when preferred methods fail

## Branch-Specific Routing

### Selection Branch Routing

The selection branch has the most complex routing due to multiple implementation options:

```
Selection Attention Flow:
┌─────────────────┐
│ Environment     │
│ Checks          │──┐
└─────────────────┘  │
                     ▼
┌─────────────────┐  ┌─────────────────┐
│ Hardware        │  │ Force Parity    │
│ Detection       │  │ NSA_FORCE_PARITY│──► SDPA Gather
└─────────────────┘  └─────────────────┘
         │                    │
         ▼                    │
┌─────────────────┐           │
│ Implementation  │           │
│ Priority        │           │
└─────────────────┘           │
         │                    │
         ▼                    │
┌─────────────────┐           │
│ 1. Triton       │◄──────────┘
│ 2. CUDA         │
│ 3. Packed SDPA  │
│ 4. Masked SDPA  │
│ 5. Gather SDPA  │
└─────────────────┘
```

#### Selection Implementation Priority

1. **Triton Selection** (`NSA_USE_TRITON_SEL=1`)
   - **When**: `NSA_USE_TRITON_SEL=1` or module `use_triton_sel=True`
   - **Hardware**: GPU with SM ≥ 7.0 (Volta+), not SM 8.9 (Ada/RTX 4090)
   - **Performance**: Fastest for L ≥ `runtime.sel_triton_min_L` (default 4096)
   - **Module**: `nsa.kernels.triton_sel_kernel.selection_attention_triton`

2. **CUDA Selection** (`NSA_SEL_CUDA=1`)
   - **When**: `NSA_SEL_CUDA=1` (experimental)
   - **Status**: Forward-only, falls back to packed SDPA for backward
   - **Module**: `nsa.kernels.cuda_sel_kernel.selection_attention_cuda`

3. **Packed SDPA** (`NSA_USE_SEL_PACK=1`, default)
   - **When**: Default selection method when Triton/CUDA unavailable
   - **Performance**: Good balance of speed and memory efficiency
   - **Module**: `nsa.core.attention_kernels.grouped_selection_attention_packed`

4. **Masked SDPA** (`NSA_USE_SEL_MASK=1`)
   - **When**: Explicitly enabled, research/debugging use
   - **Performance**: Memory intensive but numerically precise
   - **Module**: `nsa.core.attention_kernels.grouped_selection_attention_masked`

5. **Gather SDPA** (fallback)
   - **When**: All other methods disabled or failed
   - **Performance**: Slowest but most compatible
   - **Module**: `NSAAttention._sdpa_over_ranges`

### Compressed Branch Routing

```
Compressed Attention Flow:
┌─────────────────┐
│ Force Parity?   │──Yes──► Per-token SDPA Loop
└─────────────────┘
         │ No
         ▼
┌─────────────────┐
│ Flash Enabled?  │──No───► Masked SDPA or
│ FA2_ALL/FA2_CMP │        Per-token Loop
└─────────────────┘
         │ Yes
         ▼
┌─────────────────┐
│ FlashAttention-2│
│ Kernel          │
└─────────────────┘
```

#### Compressed Implementation Selection

1. **FlashAttention-2** (`NSA_USE_FA2=1` or `NSA_USE_FA2_CMP=1`)
   - **Prefill**: `compressed_attention_fa2()`
   - **Decode**: `compressed_attention_fa2_decode()`
   - **Performance**: Optimal memory and speed for most scenarios

2. **Masked SDPA** (`NSA_USE_CMP_MASK=1`, default in prefill)
   - **Method**: `batched_causal_attention_compressed_masked()`
   - **Use case**: When FlashAttention unavailable, good memory efficiency

3. **Per-token SDPA Loop** (fallback)
   - **Method**: Manual loop with `attention_bgh()` per position
   - **Use case**: Debugging, testing, fallback when batch methods fail

### Sliding Window Branch Routing

```
Sliding Window Flow:
┌─────────────────┐
│ Force Parity?   │──Yes──► Per-token SDPA Loop  
└─────────────────┘
         │ No
         ▼
┌─────────────────┐
│ Flash Enabled?  │──No───► Masked SDPA or
│ FA2_ALL/FA2_WIN │        Per-token Loop
└─────────────────┘
         │ Yes
         ▼
┌─────────────────┐
│ FlashAttention-2│
│ Kernel          │
└─────────────────┘
```

#### Sliding Window Implementation Selection

1. **FlashAttention-2** (`NSA_USE_FA2=1` or `NSA_USE_FA2_WIN=1`)
   - **Prefill**: `sliding_window_attention_fa2()`
   - **Decode**: `sliding_window_attention_fa2_decode()`

2. **Masked SDPA** (`NSA_USE_WIN_MASK=1`, default in prefill)
   - **Method**: `sliding_window_attention_masked()`

3. **Per-token SDPA Loop** (fallback)
   - **Method**: Manual loop with causal masking

## Environment Flag Reference

### Core Routing Flags

| Flag | Default | Effect |
|------|---------|--------|
| `NSA_FORCE_PARITY` | `0` | Forces SDPA everywhere for testing |
| `NSA_USE_FA2` | `0` | Enables FlashAttention-2 for all branches |
| `NSA_PREFILL_BATCHED` | `0` | Uses batched prefill (vs sequential) |

### Selection Branch Flags  

| Flag | Default | Effect |
|------|---------|--------|
| `NSA_USE_TRITON_SEL` | `0` | Enables Triton selection kernel |
| `NSA_SEL_CUDA` | `0` | Enables experimental CUDA selection |
| `NSA_USE_SEL_PACK` | `1` | Enables packed SDPA selection |
| `NSA_USE_SEL_MASK` | `0` | Enables masked SDPA selection |
| `NSA_TRITON_SEL_FORCE` | `0` | Forces Triton on incompatible HW |

### Branch-Specific Flash Flags

| Flag | Default | Effect |
|------|---------|--------|
| `NSA_USE_FA2_CMP` | `0` | FlashAttention-2 for compressed only |
| `NSA_USE_FA2_WIN` | `0` | FlashAttention-2 for sliding only |
| `NSA_USE_FA2_SEL` | `0` | FlashAttention-2 for selection (future) |

### Masking and Fallback Flags

| Flag | Default | Effect |
|------|---------|--------|
| `NSA_USE_CMP_MASK` | `1` | Masked SDPA for compressed (prefill) |
| `NSA_USE_WIN_MASK` | `1` | Masked SDPA for sliding (prefill) |

### Debug and Validation Flags

| Flag | Default | Effect |
|------|---------|--------|
| `NSA_STRICT_ASSERTS` | `0` | Enables GPU-sync causality checks |
| `NSA_DEBUG_COMPARE` | `0` | Compares batched vs sequential outputs |
| `NSA_ENV_STATIC` | `0` | Caches env flags for decode performance |

## Hardware Detection

### GPU Architecture Support

```python
# Triton Selection Hardware Detection
def _detect_triton_selection_support():
    if not torch.cuda.is_available():
        return False
    
    # Get compute capability
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major * 10 + minor
    
    # SM 8.9 (RTX 4090/Ada) not supported
    if compute_capability == 89:
        return False if not NSA_TRITON_SEL_FORCE else True
    
    # Require SM 7.0+ (Volta+)
    return compute_capability >= 70
```

**Supported Architectures:**
- ✅ **Volta (SM 7.0+)**: V100, Titan V
- ✅ **Turing (SM 7.5)**: RTX 20-series, T4, GTX 16-series  
- ✅ **Ampere (SM 8.0, 8.6)**: A100, RTX 30-series, A40
- ❌ **Ada Lovelace (SM 8.9)**: RTX 4090, RTX 40-series (fallback to packed)
- ✅ **Hopper (SM 9.0)**: H100 (when available)

## Performance Characteristics

### Selection Branch Performance

| Implementation | Memory | Speed | Compatibility |
|---------------|--------|-------|---------------|
| Triton | ⭐⭐⭐ | ⭐⭐⭐ | GPU-specific |
| CUDA | ⭐⭐ | ⭐⭐⭐ | Forward-only |
| Packed SDPA | ⭐⭐ | ⭐⭐ | Universal |
| Masked SDPA | ⭐ | ⭐⭐ | Memory intensive |
| Gather SDPA | ⭐⭐ | ⭐ | Slowest |

### Sequence Length Recommendations

| Sequence Length | Recommended Selection | Rationale |
|----------------|----------------------|-----------|
| S < 1K | Packed SDPA | Setup overhead dominates |
| 1K ≤ S < 4K | Packed SDPA | Good balance |
| 4K ≤ S < 16K | Triton (if available) | Kernel efficiency matters |
| S ≥ 16K | Triton (if available) | Maximum performance critical |

## Common Routing Patterns

### Development/Testing
```bash
# Force simple SDPA everywhere for debugging
export NSA_FORCE_PARITY=1

# Enable strict causality checks 
export NSA_STRICT_ASSERTS=1

# Compare batched vs sequential outputs
export NSA_DEBUG_COMPARE=1
```

### Production Training
```bash
# Optimal performance (default settings work well)
export NSA_USE_FA2=1                # Enable FlashAttention-2
export NSA_PREFILL_BATCHED=1         # Use batched prefill
export NSA_USE_SEL_PACK=1            # Default packed selection
```

### High-Performance Inference
```bash
# Maximum speed for supported hardware
export NSA_USE_TRITON_SEL=1          # Triton selection if available
export NSA_USE_FA2=1                 # FlashAttention-2 everywhere
export NSA_ENV_STATIC=1              # Cache env parsing in decode
```

### Conservative/Compatibility Mode
```bash
# Most compatible settings
export NSA_FORCE_PARITY=1            # Pure SDPA
export NSA_PREFILL_BATCHED=0         # Sequential prefill
export NSA_USE_SEL_PACK=0            # Fallback to gather
```

## Troubleshooting

### Selection Kernel Issues

**Problem**: `RuntimeError: Triton selection failed`
```bash
# Solution: Disable Triton and use fallback
export NSA_USE_TRITON_SEL=0
export NSA_USE_SEL_PACK=1
```

**Problem**: Memory errors with packed selection
```bash
# Solution: Use gather method
export NSA_USE_SEL_PACK=0
```

### FlashAttention Issues

**Problem**: FlashAttention import/compilation errors
```bash
# Solution: Disable and use masked SDPA
export NSA_USE_FA2=0
export NSA_USE_CMP_MASK=1
export NSA_USE_WIN_MASK=1
```

### Performance Debugging

**Problem**: Unexpectedly slow training
```bash
# Check if strict asserts are enabled
echo $NSA_STRICT_ASSERTS  # Should be 0 in production

# Verify optimal routing
export NSA_DEBUG_COMPARE=1  # Check for perf differences
export NSA_USE_FA2=1        # Ensure FA2 is used
```

## Future Routing Extensions

### Planned Implementations
- **Selection FlashAttention**: Native FA2 kernel for selection branch
- **Multi-GPU routing**: Automatic kernel selection per device
- **Dynamic switching**: Runtime adaptation based on sequence length
- **Autotuning**: Empirical kernel selection based on benchmarking

### Extension Points
- `NSAAttention._get_optimal_kernel()`: Central routing logic
- `runtime.py`: Global configuration and capability detection
- `attention_kernels.py`: New kernel implementations

This routing system ensures NSA can adapt to diverse hardware and workload requirements while maintaining correctness and providing clear performance characteristics.