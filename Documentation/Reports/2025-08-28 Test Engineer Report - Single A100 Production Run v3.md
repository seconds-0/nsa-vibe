# Test Engineer Report - Single A100 Production Run v3

**Date**: 2025-08-28  
**Author**: Test Engineer  
**Branch**: feat/single-a100-prod  
**Commit**: b2d5ed73 (on remote)  
**Status**: ❌ **Critical Code Bug - CUDA Memory Access Violation**

## Executive Summary

Following a systematic debugging approach, identified that the NSA implementation has a critical CUDA memory access violation bug. While simple forward passes work at S=128, actual training triggers an illegal memory access in the sliding window attention code. This is a code bug, not a configuration or environment issue.

## Test Environment

- **Hardware**: 1×A100 80GB PCIe (Prime Intellect, 38.140.51.195:18884)
- **CUDA**: 12.2, Driver 535.86.05
- **PyTorch**: 2.4.0+cu121
- **Python**: 3.11.13
- **FlashAttention 2**: 2.8.3 (successfully installed and verified)
- **Config**: configs/m7c_125m_2xa100_production.yaml
  - gradient_checkpointing: false (corrected)
  - batch_size: 1 (corrected)
  - precision: bf16
  - seq_len: 2048

## Systematic Testing Approach (Corrected Method)

### Phase 1: Pre-Flight Verification ✅
1. Reset config to clean state
2. Verified gradient_checkpointing=false
3. Verified batch_size=1
4. Cleaned up HALT files
5. FA2 installation confirmed

### Phase 2: Progressive Testing Results

| Test | Command | Result | Notes |
|------|---------|--------|-------|
| Forward Pass S=128 | `test_model_forward.py` | ✅ Success (1.28s) | Model initialization OK |
| Progressive Timing | `test_forward_timing.py` | ⏱️ Timeout | Hung after 5+ minutes |
| Training S=512 w/ FA2 | `train_showcase.py --seq-len 512` | ❌ CUDA Error | Illegal memory access |
| Training S=512 no FA2 | `NSA_USE_FA2=0` | ❌ CUDA Error | Same error, FA2 flag ignored |

## Critical Bug Identified

### Error Details:
```
RuntimeError: CUDA error: an illegal memory access was encountered
Location: nsa/core/attention_kernels.py, line 469
Function: sliding_window_attention_fa2
```

### Key Findings:
1. **Bug Location**: `sliding_window_attention_fa2` function
2. **Trigger**: Occurs during training forward pass, not during simple model forward
3. **Not Config Related**: Happens regardless of FA2 settings
4. **Code Issue**: The NSA_USE_FA2=0 flag is not properly disabling FA2 code path

### Stack Trace Analysis:
```python
# The error path:
1. nsa_attention.py:1007: O_win = sliding_window_attention_fa2(...)
2. attention_kernels.py:469: if not torch.isfinite(o_pack).all()
3. CUDA illegal memory access
```

## Comparison to Previous Attempts

| Attempt | Issue | Root Cause | 
|---------|-------|------------|
| v1 | 27 tok/s performance | No FA2, gradient checkpointing on |
| v2 | NaN loss | Unknown, possibly same bug manifesting differently |
| v3 | CUDA error | Code bug in sliding window attention |

## Why This Approach Was Better

1. **Started Small**: Tested S=128 first, confirming model init works
2. **Isolated Components**: Used standalone test scripts
3. **Progressive Complexity**: Built up from simple forward to training
4. **Proper Configuration**: Fixed all settings before testing

## Code Bug Analysis

The issue appears to be:
1. FA2 code path is being called even when NSA_USE_FA2=0
2. The sliding_window_attention_fa2 function has a memory access bug
3. The error checking line `torch.isfinite(o_pack).all()` triggers the crash
4. This suggests `o_pack` tensor has corrupted memory

## Recommendations

### Immediate Actions:
1. **Fix the Code Bug**: 
   - Check why NSA_USE_FA2=0 doesn't disable FA2 path
   - Debug the memory access in sliding_window_attention_fa2
   - Verify tensor dimensions and memory allocations

2. **Code Review Needed**:
   - Review recent changes to attention_kernels.py
   - Check if sliding window code works with current FA2 version
   - Verify CUDA kernel compatibility

3. **Testing Strategy**:
   - Cannot proceed with performance testing until bug is fixed
   - Need to verify code works locally first
   - Consider rolling back to a known working commit

## Conclusion

### Status: **BLOCKED BY CODE BUG**

The systematic testing approach successfully identified a critical CUDA memory access violation in the NSA sliding window attention implementation. This is not a configuration, environment, or training stability issue, but a fundamental code bug that prevents any training from running.

### Key Differences from Previous Attempts:
- **v1/v2**: I incorrectly assumed configuration issues
- **v3**: Properly identified the actual code bug through systematic testing

### Next Steps:
1. Fix the CUDA memory access bug in attention_kernels.py
2. Ensure NSA_USE_FA2 flag properly controls code paths
3. Test locally before attempting remote execution
4. Only proceed with performance testing after code is fixed

### Performance Targets:
Cannot be evaluated until the code bug is resolved. The target of 300-800 tok/s remains unachievable with broken code.

## Lessons Learned

1. **Always start with minimal tests** (S=128 forward pass)
2. **Test components in isolation** before full training
3. **CUDA errors often indicate code bugs**, not config issues
4. **Systematic debugging** reveals root causes faster than trial-and-error
5. **Configuration fixes don't help** when there's a fundamental code bug