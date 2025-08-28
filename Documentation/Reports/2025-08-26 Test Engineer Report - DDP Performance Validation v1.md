# DDP Performance Validation Report

## Executive Summary

**Status: NO-GO** ❌

The DDP throughput testing revealed a critical performance regression with production configurations achieving only **6-17 toks/s** instead of the target 45-55 toks/s. While the v2 selection optimization is confirmed working, there is a severe scaling issue with larger sequence lengths and production configurations.

## Test Environment

- **Instance**: Prime Intellect 2×A100 80GB PCIe 
- **Branch**: feat/nsa-training-breakthrough-stable-a100
- **Git SHA**: 840303b8eaea7221e93fab53d52ba352ba68817a
- **PyTorch**: 2.7.1+cu118
- **Python**: 3.10.12
- **Date**: 2025-08-26

## Test Results Summary

### ✅ Passed Tests

1. **Sanity Tests**
   - test_equiv_small.py: PASS
   - test_group_consistency.py: PASS

2. **DDP Gradient Compression**
   - BF16 compression confirmed active
   - Log shows: `[ddp] gradient compression enabled: bf16`

3. **V2 Selection Path**
   - Confirmed enabled via NSA_SEL_RANGES_V2=1
   - Dispatch correctly routes to v2 implementation

4. **Intermediate Configurations Work Well**
   - seq_len=1024, batch_size=1: **102 toks/s** ✅
   - seq_len=2048, batch_size=1: **165 toks/s** ✅

### ❌ Failed Tests

1. **Production Configuration Performance**
   - Config: m7c_125m_2xa100_production.yaml
   - Settings: seq_len=2048, batch_size=2, gradient_accumulation=2
   - Result: **6-17 toks/s** (Target: 45-55 toks/s)
   - Status: CRITICAL FAILURE

2. **NVTX Profiling**
   - Hangs/times out when enabled
   - Unable to profile the bottleneck

3. **V2 Parity Tests**  
   - "gaps" pattern fails due to test data generation issue
   - Other patterns pass

## DDP Bucket Sweep Results

Tested with seq_len=1024, batch_size=1, 20 steps:

| Bucket Size | Initial toks/s | Final toks/s | Recommendation |
|------------|---------------|--------------|----------------|
| 25 MB | 216 | 232 | Good |
| 50 MB | 203 | 228 | Good |
| 100 MB | 244 | 229 | **Best warmup** |

**Recommendation**: Use 50-100 MB bucket size for PCIe systems

## Performance Analysis

### Scaling Breakdown

| Configuration | Seq Length | Batch Size | Throughput | Status |
|--------------|------------|------------|------------|---------|
| Minimal | 512 | 1 | 260 toks/s | ✅ Fast |
| Intermediate | 1024 | 1 | 102 toks/s | ✅ Good |
| Production-like | 2048 | 1 | 165 toks/s | ✅ Acceptable |
| Full Production | 2048 | 2 | 6-17 toks/s | ❌ CRITICAL |

### Root Cause Analysis

1. **Not the v2 Selection**: V2 is confirmed working and dispatching correctly
2. **Critical Issue**: Performance degrades catastrophically with batch_size=2
3. **Likely Culprits**:
   - Gradient accumulation (accumulate_grad_batches=2)
   - Gradient checkpointing interaction with DDP
   - Model initialization overhead with larger configs
   - Possible O(n²) complexity in attention or other components

## Artifacts Generated

- `artifacts/git_sha.txt` - Commit verification
- `artifacts/collect_env.txt` - PyTorch environment
- `artifacts/nvidia_smi.xml` - GPU configuration 
- `artifacts/baseline_ddp.log` - Training logs
- `artifacts/m7c_125m_2xa100_prod/heartbeat_rank0.jsonl` - Telemetry data
- `artifacts/m7c_125m_2xa100_prod/training.csv` - Step metrics

## Recommendations

### Immediate Actions Required

1. **DO NOT DEPLOY** to production with current performance
2. **Profile without DDP** to isolate the issue:
   ```bash
   python scripts/train_showcase.py --config configs/m7c_125m_2xa100_production.yaml --ddp 0
   ```
3. **Test without gradient accumulation**:
   ```bash
   # Edit config to set accumulate_grad_batches: 1
   ```
4. **Binary search on batch size** to find performance cliff

### Configuration for Testing

```bash
# Working configuration (for reference)
NSA_SEL_RANGES_V2=1 \
NSA_DDP_COMPRESS=bf16 \
NSA_DDP_BUCKET_MB=50 \
PYTHONPATH=. python scripts/train_showcase.py \
  --dataset synthetic \
  --seq-len 1024 \
  --batch-size 1 \
  --steps 100
```

## Conclusion

The system exhibits a severe performance regression with production configurations. While individual optimizations (v2 selection, DDP compression) are working, the combination of seq_len=2048 and batch_size=2 causes throughput to collapse to 6-17 toks/s, making the system unsuitable for production deployment.

**Next Critical Step**: Profile the single-GPU case with production config to identify the bottleneck before attempting any DDP optimizations.

---

*Test Engineer Report - DDP Performance Validation*  
*Status: NO-GO due to critical performance regression*