# 2025-08-29 Test Engineer Report - PR31 Varlen Selection Attention Validation v1

**Date**: 2025-08-29  
**GPU**: NVIDIA A100 80GB PCIe  
**Branch**: perf/sel-varlen (commit 2c819484)  
**PR**: #31  
**Test Engineer**: Claude

## Executive Summary

**PARTIAL PASS WITH ISSUES** - The varlen selection attention feature passes all correctness tests but has implementation bugs that prevent it from working in practice. The feature requires fixes before merging.

## Test Configuration

### Environment
- **GPU**: NVIDIA A100 80GB PCIe
- **CUDA**: Available, PyTorch 2.4.0+cu121
- **Python**: 3.11
- **FlashAttention-2**: Available with varlen support
- **Branch**: perf/sel-varlen (2c819484)

### Core Environment Variables
```bash
NSA_PREFILL_BATCHED=1
NSA_USE_SEL_PACK=1
NSA_SEL_RANGES_V2=1
NSA_USE_FA2=1
NSA_USE_FA2_CMP=1
NSA_USE_FA2_WIN=0
NSA_USE_SEL_VARLEN=1  # Feature flag for this PR
NSA_VARLEN_RESERVE_N=8192
NSA_VARLEN_RESERVE_K=33554432
```

## Test Results

### 1. Correctness Tests ✅

| Test Suite | Baseline (Flag OFF) | Varlen (Flag ON) | Status |
|------------|---------------------|------------------|---------|
| Selection v2 equivalence (52 tests) | PASS (52/52) | PASS (52/52) | ✅ |
| Sliding NaN CUDA (7 tests) | PASS (7/7) | PASS (7/7) | ✅ |
| Core invariants (6 tests) | PASS (6/6) | Not tested¹ | - |
| FA-2 varlen parity (3 tests) | N/A | PASS (3/3) | ✅ |

¹ Core invariants not re-tested with flag ON as they don't exercise the varlen path

### 2. Performance Benchmarks 📊

#### Decode Microbenchmark (ms per decode step)

| Context | Baseline | Varlen Enabled | Delta | Status |
|---------|----------|----------------|-------|---------|
| 512 | 5.75 ms | 6.00 ms | +4.3% | ⚠️ Slower |
| 1024 | 5.98 ms | 6.29 ms | +5.2% | ⚠️ Slower |
| 2048 | 6.46 ms | 6.71 ms | +3.9% | ⚠️ Slower |

**Note**: Varlen path shows slight performance degradation instead of expected improvement. This suggests the varlen path may not be activating correctly or has overhead issues.

### 3. Implementation Issues Found 🐛

#### Critical Bug in `selection_attention_varlen_all()`

**Location**: `nsa/core/attention_kernels.py`, lines 461-462

**Issue**: Tensor dimension mismatch in expand operation
```python
# Current (broken):
k_pack[write_pos : write_pos + Lseg] = seg_k.unsqueeze(1).expand(Lseg, h, Dk)
v_pack[write_pos : write_pos + Lseg] = seg_v.unsqueeze(1).expand(Lseg, h, Dv)
```

**Error**: 
```
expand(torch.cuda.FloatTensor{[Lseg, 1, Dk]}, size=[Lseg, h, Dk]): 
the number of sizes provided (3) must match tensor dimensions (3)
```

**Root Cause**: The code incorrectly tries to expand already 2D tensors `[Lseg, Dk]` after unsqueezing to 3D.

#### Fallback Path Bug

**Location**: `nsa/core/attention_kernels.py`, line 352

**Issue**: UnboundLocalError in `grouped_selection_attention_packed()`
```python
UnboundLocalError: cannot access local variable 'need_new' where it is not associated with a value
```

**Impact**: When varlen path fails and falls back to packed attention, it crashes due to uninitialized variable.

### 4. Training Test Results

| Test Type | Baseline | Varlen Enabled | Status |
|-----------|----------|----------------|---------|
| Synthetic data (10 steps) | ✅ Works | ❌ Crashes | Failed |
| Training throughput | ~456 tok/s | N/A | Cannot test |

**Failure Details**: Training with `NSA_USE_SEL_VARLEN=1` fails immediately on forward pass due to the tensor dimension bug.

## Code Review Findings

### Strengths ✅
- Well-structured opt-in design with proper feature gating
- Comprehensive fallback strategy (FA-2 → dense batch → packed)
- Good workspace pre-allocation mechanism
- Proper causal masking preservation

### Critical Issues ❌

1. **Tensor Shape Bug** (Lines 461-462)
   - Incorrect expand operation causes runtime failure
   - Prevents varlen path from executing

2. **Fallback Path Bug** (Line 352)
   - Uninitialized variable in fallback function
   - Makes graceful degradation impossible

3. **Missing Direct Tests**
   - No unit test specifically for `selection_attention_varlen_all()`
   - Existing tests don't exercise the varlen code path

### Recommendations for Fix

1. **Fix tensor expansion** (Priority: Critical)
```python
# Suggested fix:
k_pack[write_pos : write_pos + Lseg] = seg_k.repeat(1, h).reshape(Lseg * h, Dk)
v_pack[write_pos : write_pos + Lseg] = seg_v.repeat(1, h).reshape(Lseg * h, Dv)
```

2. **Fix undefined variable** (Priority: Critical)
   - Initialize `need_new` properly in `grouped_selection_attention_packed()`

3. **Add direct unit test** (Priority: High)
   - Test `selection_attention_varlen_all()` in isolation
   - Verify against packed attention output

## Conclusion

The PR introduces a promising optimization but contains critical implementation bugs that prevent it from working:

1. **Tensor dimension bug** prevents varlen path from executing
2. **Fallback bug** prevents graceful degradation
3. **Performance regression** when enabled (likely due to fallback overhead)

### Recommendation

**DO NOT MERGE** - The feature requires the following before approval:

1. Fix the tensor expansion bug in lines 461-462
2. Fix the undefined variable in the fallback path
3. Add direct unit tests for the varlen function
4. Re-validate performance after fixes

The architectural approach is sound, but the implementation needs debugging before it can be merged safely.

## Test Artifacts

- Test execution logs available on GPU instance
- All correctness tests pass because they don't actually exercise the broken varlen path
- Training fails immediately when varlen is enabled

---
*Test completed on 2025-08-29 on Prime Intellect A100 80GB GPU instance*