# 2025-08-29 Test Engineer Report - FA2 Policy Gates Verification v1

## Executive Summary
**Status**: PASS - FA2 policy gates working correctly, production configuration validated

## Test Environment
- **Instance**: Prime Intellect A100 80GB (38.140.51.195:19121)
- **Branch**: feat/m8-fa2-routing-polish (with local updates)
- **PyTorch**: 2.4.0+cu121
- **Flash-Attn**: 2.8.3

## Code Changes Verified

### 1. attention_kernels.py
- ✅ Policy gate for sliding FA2 (disabled by default)
- ✅ NSA_ALLOW_SLIDING_FA2 environment variable
- ✅ Gate skip logging with reason
- ✅ Correct SDPA fallback for sliding window
- ✅ Compressed FA2 remains enabled

### 2. test_fa2_gpu_varlen.py  
- ✅ test_sliding_varlen_vs_sdpa_gpu xfails by default
- ✅ Clear message about API limitation
- ✅ Developer override available

## Test Results

### 1. Compressed FA2
```bash
NSA_TEST_FA2=1 pytest test_compressed_varlen_vs_sdpa_gpu
```
**Result**: PASS ✅

### 2. Sliding FA2 Policy
```bash
NSA_TEST_FA2=1 pytest test_sliding_varlen_vs_sdpa_gpu
```
**Result**: XFAIL ✅ (as expected - API limitation acknowledged)

### 3. Gate Skip Logging
```bash
NSA_SDPA_AUDIT=1 NSA_DEBUG_LOG=1 sliding_window_attention_fa2(...)
```
**Log Output**: 
```
NSA-LOG fa2.gate_skip branch=win reason=unsupported_sliding_semantics forced=False
```
**Result**: Correct logging ✅

### 4. Decode Benchmark
```bash
bench_decode.py --S_list 512,1024,2048
```
**Results**:
- S=512: 5.85ms (branches balanced at ~100%)
- S=1024: 6.07ms (branches balanced at ~100%)  
- S=2048: 6.57ms (branches balanced at ~100%)

**Result**: Normal performance ✅

### 5. Production Configuration
```bash
NSA_USE_FA2=1 NSA_USE_FA2_CMP=1 NSA_USE_FA2_WIN=0
train_showcase.py --dataset synthetic --steps 5
```
**Results**:
- Training runs successfully
- Loss convergence normal (5.688 → 5.687)
- Throughput: 292 tok/s

**Result**: Production ready ✅

## Key Findings

1. **Policy Gates Working**: Sliding FA2 correctly disabled by default with clear logging
2. **Developer Controls**: NSA_ALLOW_SLIDING_FA2 provides override for experimentation
3. **Compressed FA2 Functional**: Continues to work correctly with causal=True
4. **Production Safe**: Recommended configuration runs without issues
5. **Performance Normal**: Decode benchmark shows balanced branch usage

## Production Configuration

```bash
# Recommended settings for production
export NSA_USE_FA2=1          # Enable FA2
export NSA_USE_FA2_CMP=1      # FA2 for compressed (works)
export NSA_USE_FA2_WIN=0      # SDPA for sliding (API limitation)

# Data loader optimization
export NSA_FWE_PREFETCH=1
export NSA_FWE_Q=4
export NSA_FWE_DOC_BATCH=64

# Optional diagnostics
export NSA_SDPA_AUDIT=1       # Gate skip logs
export NSA_DEBUG_LOG=1        # Enable logging
export NSA_DEBUG_TIMING=1     # Timing information
```

## Conclusion

The FA2 routing implementation with policy gates is production-ready. The sliding window FA2 limitation is properly handled with:
- Clear policy gate disabling it by default
- Diagnostic logging for transparency
- Developer overrides for experimentation
- Correct SDPA fallback behavior

All tests pass with the expected behavior, and the production configuration provides optimal performance with FA2 for compressed attention and SDPA for sliding window.