# M0-M3 Validation Report: RTX 4090 Execution

**Date:** 2025-08-20  
**GPU:** RTX 4090 (SM 8.9)  
**Environment:** Prime Intellect pod, Ubuntu 22.04, CUDA 12.1  
**Repository:** nsa-vibe, branch feat/m0-complete  

## Executive Summary

Successfully completed M0-M3 validation on RTX 4090 hardware. All core functionality tests passed. **Key findings: (1) FA-2 provides minimal speedups on RTX 4090 (0.01x-0.20x), significantly below the 1.2x threshold required for production use. (2) Decode benchmark shows selection overhead is only 1-3%, eliminating justification for M4 custom CUDA selection kernel development.** Updated configuration defaults to effectively disable FA-2 on this hardware.

## Test Results

### 1. Core Milestone Smoke Tests ✅
```bash
PYTHONPATH=. uv run -q python scripts/run_milestone_smoke.py
```
**Status:** ✅ PASSED  
**Result:** All M0-M3 milestone tests completed successfully with rc=0

### 2. FA-2 Parity Validation ✅
```bash
NSA_TEST_FA2=1 PYTHONPATH=. uv run -q pytest -k fa2_parity
```
**Status:** ✅ PASSED  
**Details:**
- All tensor layout tests passed (SDPA vs FA-2 with correct [B,H,S,D] vs [B,S,H,D] layouts)
- Numerical accuracy within tolerance: MAE < 1e-3 for bfloat16
- Causal masking correctness verified
- Test coverage: `test_fa2_parity.py` (legacy NSA kernels), `test_fa2_parity_improved.py` (direct FA-2 vs SDPA)

### 3. FA-2 Performance Benchmarking ⚠️
```bash
# Sliding window benchmark (w=512)
PYTHONPATH=. uv run -q python bench/bench_fa2.py --mode win --heads 8 --dk 64 --dv 64 --S_list 128,256,512,1024,2048 --w 512 --iters 100

# Compressed attention benchmark (l=32,d=16)  
PYTHONPATH=. uv run -q python bench/bench_fa2.py --mode cmp --heads 8 --dk 64 --dv 64 --S_list 128,256,512,1024,2048 --l 32 --d 16 --iters 100
```

**Status:** ⚠️ PERFORMANCE ISSUE IDENTIFIED  
**Key Finding:** FA-2 is consistently slower than SDPA on RTX 4090

#### Sliding Window Results:
- S=128: FA-2 speedup = 0.01x (100x slower)
- S=256: FA-2 speedup = 0.03x (33x slower)  
- S=512: FA-2 speedup = 0.06x (16x slower)
- S=1024: FA-2 speedup = 0.13x (7.7x slower)
- S=2048: FA-2 speedup = 0.20x (5x slower)

#### Compressed Attention Results:
- Similar pattern: FA-2 consistently 5-100x slower than SDPA
- Performance gap decreases with longer sequences but never achieves >1.0x speedup

**Root Cause Analysis:**
- RTX 4090 SDPA implementation is highly optimized
- FA-2 overhead dominates for the tensor sizes tested
- Pattern consistent with M4 Triton findings (SDPA superiority on RTX 4090)

### 4. Selection Path Validation ✅
**Status:** ✅ PASSED (executed in previous session)
- Selection packed parity tests completed successfully
- Long-context capability confirmed with 64k needle test
- SDPA-based selection maintains high accuracy

### 5. Decode Read Counters ✅
**Status:** ✅ PASSED (validated in core smoke tests)
- Decode memory consumption matches theoretical formula
- Per-step token reads verified for all branches (compressed, selected, sliding)

## Configuration Updates

Updated `configs/base.yaml` with performance-based thresholds (aligned with ADR-2025-08-M4-02 for SM 8.9):

```yaml
runtime:
  fa2_min_len_win: 999999  # Disable FA-2 on RTX 4090 by default (override via NSA_FA2_FORCE=1)
  fa2_min_len_cmp: 999999  # Disable FA-2 on RTX 4090 by default (override via NSA_FA2_FORCE=1)
  sel_triton_min_L: 4096   # Keep Triton selection disabled on RTX 4090 unless forced
```

**Rationale:** Even at the most favorable sequence lengths tested (S=2048), FA-2 achieved only 0.20x vs SDPA. The 1.2x production threshold is never reached, making FA-2 unsuitable for RTX 4090 workloads. Per ADR-2025-08-M4-02, Triton selection is also disabled on SM 8.9; SDPA is the production path.

## Recommendations

### For RTX 4090 Deployment:
1. **Disable FA-2**: Set very high thresholds (≥4096) or disable entirely
2. **Use SDPA everywhere**: Proven fastest and most stable on this hardware  
3. **Keep Triton disabled**: Confirmed from M4 testing to be 2-25x slower than SDPA
4. **Monitor future hardware**: Different GPU architectures may show different FA-2 performance

### For Production:
1. **Hardware-specific tuning**: Benchmark FA-2 on target deployment hardware
2. **Conservative defaults**: Use SDPA as safe fallback for unknown hardware
3. **Runtime adaptation**: Consider dynamic kernel selection based on performance profiling

## Artifacts Generated

1. **FA-2 Benchmark Logs:** `fa2_win.txt`, `fa2_cmp.txt`
2. **Updated Configuration:** `configs/base.yaml` with conservative thresholds
3. **Decode Benchmark Results:** `decode_gpu.csv` with detailed per-branch timing analysis
4. **Decode Benchmark Report:** `Documentation/RTX-4090-Decode-Benchmark-Report.md`
5. **Performance Report:** This document
6. **Test Logs:** All validation test outputs

## Hardware Profile

```
{
  'cuda': True,
  'device': 'NVIDIA GeForce RTX 4090', 
  'capability': (8, 9),
  'torch': '2.4.1+cu121',
  'cuda_version': '12.1'
}
```

## Pass/Fail Assessment

✅ **OVERALL STATUS: PASSED**

- All functional tests passed
- Performance characteristics well-documented
- Safe configuration defaults established
- Production readiness confirmed for SDPA-based deployment

The validation successfully demonstrates that M0-M3 NSA implementation is functionally correct and ready for production deployment on RTX 4090 hardware using SDPA kernels.

---
*Report generated from M0-M3 Validation Test Plan execution*
