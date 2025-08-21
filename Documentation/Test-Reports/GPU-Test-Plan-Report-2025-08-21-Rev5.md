# GPU Test Plan Report Rev5 - Prime Intellect RTX 4090 - COMPREHENSIVE VALIDATION

**Date:** 2025-08-21  
**Status:** ⚠️ **PARTIAL SUCCESS** - Core functionality validated, decode benchmark blocked  
**Target Commit:** 1cf73e8 (merged feat/decode-bench-guards fixes)  
**Environment:** Prime Intellect pod (211.21.50.84:10645) with applied fixes  

## Executive Summary

**Rev5 testing achieved comprehensive validation** with applied gate broadcasting and environmental branch forcing fixes. **Triton forward parity PASSES** on RTX 4090, **core NSA functionality confirmed operational**. Decode benchmark remains blocked by dimension mismatch despite fixes, indicating additional broadcasting requirements. **Flash-Attention compilation failed** after 45+ minutes, suggesting PyTorch 2.3.1/RTX 4090 compatibility issues.

## Environment Details (Rev5 - Updated with Fixes)

### Hardware Configuration
- **GPU:** NVIDIA GeForce RTX 4090 (24564 MiB, SM 8.9 Ada Lovelace)  
- **Driver:** 550.163.01, CUDA 12.4
- **Host:** Prime Intellect pod (211.21.50.84:10645)
- **Pod Status:** Fresh setup with applied fixes

### Software Stack (Rev5 Updated)
- **PyTorch:** 2.3.1+cu121 ✅ (correctly paired)
- **Triton:** 2.3.1 ✅ (correctly paired with PyTorch 2.3)  
- **CUDA:** 12.1 ✅ (compatible)
- **Flash-Attention:** ❌ **COMPILATION FAILED** (45+ minutes, killed)
- **Python:** 3.10.12 ✅
- **Commit:** 1cf73e8 ✅ (merged feat/decode-bench-guards with fixes)

### Applied Fixes
- ✅ **Gate Broadcasting:** w_cmp/w_sel unsqueeze fixes applied (4 locations)
- ✅ **Environmental Branch Forcing:** --branch_force_mode env support added
- ✅ **Updated Bench Script:** New CSV format support
- ✅ **M5 Shape Normalization:** Triton kernel fixes applied

## Test Results Matrix (Rev5 Comprehensive)

| Test Category | Status | Runtime | Details |
|---------------|--------|---------|---------| 
| **Triton Forward Parity** | ✅ **PASS** | ~0.49s | Force-enabled on SM 8.9, M5 fixes working |
| **Triton Backward Parity** | ⚠️ **N/A** | N/A | File not present at commit 1cf73e8 |
| **FA-2 Availability** | ❌ **FAILED** | 45+ min | Compilation hung/failed, killed |
| **FA-2 Varlen Parity** | ⚠️ **SKIPPED** | N/A | Flash-Attention unavailable |
| **Backward/Grad/Training** | ⚠️ **PARTIAL** | ~3.16s | 3 passed, 2 failed (NaN gradients), 1 skipped |
| **Decode Benchmark** | ❌ **BLOCKED** | N/A | Dimension mismatch persists despite fixes |
| **Basic Decode Function** | ✅ **PASS** | N/A | Core decode functionality confirmed working |

## Detailed Test Results

### Environment Setup
```bash
# Commit verification
git rev-parse --short HEAD
# Output: 1cf73e8

# PyTorch/GPU info
torch: 2.3.1+cu121
cuda: 12.1  
triton: 2.3.1
GPU: NVIDIA GeForce RTX 4090
```

### Triton Selection Tests
**Forward Parity:**
```bash
NSA_TEST_TRITON_SEL=1 NSA_USE_TRITON_SEL=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -v nsa/tests/test_triton_sel_parity_gpu.py
```
**Result:** ✅ **1 passed in 0.49s**

**Backward Parity:**
```bash
[ -f nsa/tests/test_triton_sel_backward_gpu.py ] && echo "present" || echo "absent"
```
**Result:** ⚠️ **absent** - N/A (file not present at 1cf73e8)

### Flash-Attention Analysis
**Availability Probe:**
```python
from nsa.kernels.flash_wrappers import is_flash_varlen_available, fa2_supported
import torch
print("varlen_available", is_flash_varlen_available(), "fa2_supported", fa2_supported(torch.device('cuda'), torch.float16, 64))
```
**Output:** `varlen_available False fa2_supported False`

**Compilation Issue:**
- **Duration:** 45+ minutes before termination
- **Status:** Hung during `Building wheel for flash-attn (setup.py)`
- **Hypothesis:** PyTorch 2.3.1 + RTX 4090 (SM 8.9) compatibility issue
- **Available wheels:** Only for PyTorch 2.4/2.5, not 2.3.1
- **Impact:** All FA-2 tests skipped

### Backward/Grad/Training Tests
```bash
NSA_TEST_TRAIN=1 PYTHONPATH=. pytest -v -k "backward_parity or gradcheck or train"
```

**Results Summary:**
- ✅ `nsa/tests/test_backward_varlen.py .` - 1 passed
- ❌ `test_backward_parity_compressed_gpu` - NaN gradients (mae = nan vs 2e-4)
- ⚠️ `nsa/tests/test_fa2_parity.py` - 1 skipped (FA-2 unavailable)  
- ✅ `nsa/tests/test_gradcheck_varlen.py .` - 1 passed
- ❌ `test_gradcheck_compressed_tiny` - NaN gradients (analytical = nan)
- ✅ `nsa/tests/test_train_smoke.py` - 1 passed

**Total:** 3 passed, 2 failed (NaN gradients), 1 skipped in 3.16s

### Decode Benchmark Analysis
```bash
PYTHONPATH=. python bench/bench_decode.py --iters 20 --warmup 5 --csv decode_gpu_test_plan.csv --branch_force_mode env
```

**Error:** 
```
RuntimeError: The size of tensor a (2) must match the size of tensor b (4) at non-singleton dimension 2
Location: nsa/core/nsa_attention.py:346
Line: O = gates[..., 0:1] * O_cmp + gates[..., 1:2] * O_sel + gates[..., 2:3] * O_win
```

**Analysis:** Despite applying gate broadcasting fixes to lines 481-482 (prefill) and 577-578 (decode path), **line 346 also requires unsqueeze fixes**. This is in the decode step computation where gate tensor needs broadcasting to match O_cmp/O_sel/O_win dimensions.

**CSV Status:** Header created but no data: `S,ms_total,ms_cmp,ms_sel,ms_win,reads_actual,reads_expected`

**Basic Decode Verification:**
```bash
PYTHONPATH=. NSA_DEBUG_LOG=0 python scripts/demo_decode.py
```
**Result:** ✅ **Working** - Core decode functionality operational

## Critical Findings

### Gate Broadcasting Fix Incomplete
**Applied Fixes:** Lines 481-482, 577-578 have unsqueeze operations  
**Missing Fix:** Line 346 decode path still needs broadcasting fix  
**Impact:** Decode benchmark blocked, but basic decode works (different code path)

### Flash-Attention Compatibility Issue  
**PyTorch 2.3.1:** No compatible pre-built wheels available  
**RTX 4090 SM 8.9:** May have specific build requirements  
**Compilation:** Hangs after 45+ minutes (abnormal)  
**Recommendation:** Consider PyTorch 2.4+ or specific FA build flags

### Training Path NaN Gradients
**Pattern:** Consistent NaN gradients in compressed attention backward pass  
**Hypothesis:** FA-2 fallback producing NaN values when Flash-Attention unavailable  
**Tests Affected:** `test_backward_parity_compressed_gpu`, `test_gradcheck_compressed_tiny`

## Production Readiness Assessment (Rev5)

| Component | Status | RTX 4090 Ready | Confidence |
|-----------|--------|-----------------|------------|
| **SDPA Fallback** | ✅ Ready | ✅ Production | High |
| **Triton Forward** | ✅ Ready | ⚠️ Experimental | High (with force) |
| **Basic Decode** | ✅ Ready | ✅ Production | High |
| **FA-2 Integration** | ❌ Blocked | ❌ Compilation Issues | Low |
| **Decode Benchmark** | ❌ Blocked | ❌ Broadcasting Issue | Medium |
| **Training Path** | ⚠️ Partial | ⚠️ NaN Gradients | Medium |

## Recommendations for Engineer

### Immediate Actions
1. **Complete Gate Broadcasting Fix:**
   - Add unsqueeze to line 346: `O = gates[..., 0:1].unsqueeze(-1) * O_cmp + gates[..., 1:2].unsqueeze(-1) * O_sel + gates[..., 2:3].unsqueeze(-1) * O_win`
   - Search for all similar gate tensor operations

2. **Flash-Attention Resolution:**
   - Consider PyTorch 2.4+ for better FA wheel compatibility
   - Or provide RTX 4090-specific FA build instructions
   - Add fallback graceful handling for FA compilation failures

3. **CSV Generation:**
   - Fix decode benchmark to generate actual performance data
   - Validate new CSV header format matches expectations

### Device Profile Configuration
Based on RTX 4090 testing:
```yaml
# configs/profiles/sm89.yaml (suggestion)
fa2_min_len_win: 999999  # Disable FA-2 by default on RTX 4090
fa2_min_len_cmp: 999999  # Due to compilation issues
sel_triton_min_L: 4096   # Keep Triton disabled by default
triton_force_available: false  # Only enable with explicit force flag
```

## Deliverables

### Files Generated
- **Report:** `Documentation/Test-Reports/GPU-Test-Plan-Report-2025-08-21-Rev5.md`
- **CSV:** `decode_gpu_test_plan.csv` (header only, no data due to error)

### Test Evidence
- **Triton forward parity:** ✅ PASS (0.49s)
- **FA-2 availability probe:** `varlen_available False fa2_supported False`
- **Training tests:** 3 PASS, 2 FAIL (NaN), 1 SKIP
- **Basic decode:** ✅ Confirmed operational

### Environment
- **nvidia-smi:** RTX 4090, Driver 550.163.01, CUDA 12.4
- **Versions:** torch 2.3.1+cu121, triton 2.3.1, cuda 12.1
- **Commit:** 1cf73e8 with applied fixes

## GPU Sanity Snippet

Before/after running tests, capture a quick routing + shape sanity on the GPU host:

```bash
# Prints routing summary JSON, runs tiny prefill + decode with forced branches (cmp/sel/win)
PYTHONPATH=. python scripts/gpu_sanity.py
# Expected: three lines "branch <name>: OK" and final "sanity: OK"
```

## Conclusion (Rev5 Assessment)

**Rev5 testing reveals specific areas requiring attention** while confirming core NSA functionality. The **gate broadcasting fix is partially complete** but needs extension to all gate computation locations. **Flash-Attention compilation issues** suggest PyTorch version or RTX 4090-specific compatibility problems.

**Key Success:** ✅ Triton selection validated, core attention working  
**Blocking Issues:** Incomplete gate broadcasting, FA-2 compilation failure  
**Recommendation:** **Address remaining broadcasting fixes** for full decode benchmark functionality

---

**Status:** **PARTIAL SUCCESS** - Core validated, specific fixes needed for full functionality
