# 2025-08-31 Test Engineer Report - A100 Retest Performance Still Critical v2

## Executive Summary
**Status: CRITICAL FAILURE PERSISTS** - Despite implementing performance fixes (CLI overrides, disabling gradient checkpointing), throughput remains at 11 toks/s. The issue is not gradient checkpointing but appears to be fundamental to the NSA attention implementation.

## Test Environment
- **Hardware**: NVIDIA A100 80GB PCIe (104.255.9.187:12600)
- **Software**: PyTorch 2.4.0+cu121, Python 3.11.13
- **Branch**: prod/a100-50k-test (commit 406312f8)
- **Config**: configs/m7c_125m_1xa100_prod_v1.yaml

## Fixes Applied
1. **Training Script Updates** (train_showcase.py):
   - Added CLI overrides: `--seq-len`, `--batch-size`, `--accum-batches`, `--no-gc`
   - Added env overrides: `NSA_SEQ_LEN`, `NSA_BATCH_SIZE`, `NSA_ACCUM`, `NSA_NO_GC`
   - Boot message confirms: `[boot] effective config: S=512 B=1 accum=1 gc=off`

2. **Environment Validation** (validate_run_env.py):
   - Validates FA2 disabled, TF32 enabled, allocator configured
   - No debug/anomaly flags detected
   - Environment passes: `[OK] Environment validated`

## Retest Results

### Triage Test (S=512, GC disabled)
```bash
python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --no-gc \
  --seq-len 512 --batch-size 1 --accum-batches 1
```

**Results:**
- Config correctly applied: `S=512 B=1 accum=1 gc=off`
- Gradient checkpointing: **OFF**
- **Throughput: 11 toks/s** (unchanged from before)
- Step time: ~45 seconds per step
- GPU utilization: 29%
- Memory: 7GB/80GB

### Synthetic Data Test
To rule out data loading issues:
```bash
python -u scripts/train_showcase.py \
  --dataset synthetic --ddp 0 --no-gc \
  --seq-len 512 --batch-size 1 --accum-batches 1
```

**Results:**
- **Throughput: 11 toks/s** (identical to FineWeb-Edu)
- Confirms data loading is NOT the bottleneck

## Root Cause Analysis

### What We Ruled Out
1. ❌ **Gradient Checkpointing**: Disabled via `--no-gc`, confirmed by logs
2. ❌ **Data Loading**: Synthetic data shows same 11 toks/s
3. ❌ **Config Override Issues**: Boot message confirms correct S=512, B=1
4. ❌ **Environment Issues**: Validator passes, TF32 enabled, FA2 correctly disabled

### Remaining Suspects
1. **NSA Attention Implementation**: Fundamental algorithmic inefficiency
2. **SDPA without FA2**: Running vanilla SDPA for all branches
3. **CPU-bound Operations**: 96% CPU usage suggests CPU bottleneck
4. **Selection Mechanism**: Block selection/gathering may be inefficient

### Evidence
- Each forward/backward takes ~45 seconds even with S=512
- GPU severely underutilized (29%)
- CPU at 96-100% during execution
- Problem persists across:
  - Different sequence lengths (512, 2048)
  - Gradient checkpointing on/off
  - Different data sources (FineWeb, synthetic)

## Performance Comparison
| Configuration | Expected | Actual | Factor |
|--------------|----------|--------|--------|
| S=2048, GC on | 300-800 toks/s | 10 toks/s | 30-80x slower |
| S=512, GC off | 500-1500 toks/s | 11 toks/s | 45-136x slower |
| S=512, synthetic | 1000+ toks/s | 11 toks/s | 90x+ slower |

## Critical Observations
1. **Disabling gradient checkpointing had minimal impact** (10→11 toks/s)
2. **Reducing sequence length 4x had no speedup** (still ~11 toks/s)
3. **CPU bottleneck evident** from 96% CPU usage
4. **GPU idle most of the time** (29% utilization)

## Recommendations

### Immediate Actions
1. **Profile CPU operations** - Identify what's consuming CPU cycles
2. **Test vanilla Llama** - Confirm base model performs normally
3. **Isolate NSA components** - Test compressed/selected/sliding branches individually
4. **Enable FA2** - Despite config saying disabled, try forcing it on

### Investigation Required
- Is there an accidental debug mode enabled in NSA kernels?
- Are selection indices causing inefficient scatter/gather?
- Is there a compilation issue with the NSA attention?
- Are we hitting a pathological case in block selection?

## Conclusion
**CRITICAL NO-GO**. The performance issue is more fundamental than gradient checkpointing. Even with all optimizations applied (GC off, S=512, synthetic data), the model runs at 11 toks/s - approximately 90x slower than expected.

This suggests a fundamental issue with the NSA attention implementation itself, not the training configuration. The system is CPU-bound despite running on GPU, indicating inefficient operations in the attention mechanism.

## Test Timeline
- Initial test: 10 toks/s with GC on, S=2048
- Retest with fixes: 11 toks/s with GC off, S=512
- Synthetic test: 11 toks/s confirming not data-related
- Total investigation time: ~1 hour
- Verdict: **Production training infeasible without fixing core NSA performance**