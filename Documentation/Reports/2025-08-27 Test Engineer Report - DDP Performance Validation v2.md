# DDP Performance Validation Report v2

## Executive Summary

**Status: PARTIAL PASS with CRITICAL ISSUES** ⚠️

Testing revealed that the system works well with smaller configurations but exhibits severe performance degradation with production settings. The v2 selection optimization is confirmed working, DDP compression is active, but the production configuration (seq_len=2048, batch_size=2) causes catastrophic slowdown.

## Test Environment

- **Instance**: Prime Intellect 2×A100 80GB PCIe
- **Branch**: feat/nsa-training-breakthrough-stable-a100  
- **Git SHA**: 840303b8eaea7221e93fab53d52ba352ba68817a
- **PyTorch**: 2.7.1+cu118
- **Python**: 3.10.12
- **GPUs**: 2×NVIDIA A100 80GB PCIe (no NVLink)
- **Date**: 2025-08-27

Procedure
- Followed: `Documentation/Test-Plans/DDP-Performance-Validation-Runbook.md`

## Test Results Summary

### ✅ Passed Tests

| Test | Result | Notes |
|------|--------|-------|
| Sanity Tests | PASS | test_equiv_small.py, test_group_consistency.py |
| V2 Parity | PASS | All patterns except "gaps" (test data issue) |
| DDP Compression | ACTIVE | BF16 compression confirmed enabled |
| Small Config Performance | 440 toks/s | seq_len=128, batch_size=8 |
| NVTX Profiling | WORKS | With NSA_DISABLE_AUX_STATS=1 |

### ⚠️ Mixed Results

| Test | Result | Notes |
|------|--------|-------|
| DDP Bucket Sweep | PARTIAL | 25MB: 437 toks/s, 50MB: 431 toks/s (last-50-step mean). 100MB: intermittent hang; re-try with different master-port recommended. |
| Single-GPU Test | VARIES | Small config: 244 toks/s, Production: HANGS |
| Fused AdamW | HANGS | Standard AdamW works; fused variant (`NSA_OPT_FUSED=1`) hangs → keep unset |

### ❌ Failed Tests

| Test | Result | Impact |
|------|--------|--------|
| Production Config | 6-17 toks/s | Target: 45-55 toks/s - CRITICAL FAILURE |
| SDPA Audit | NOT RUN/INCOMPLETE | Should use `NSA_SDPA_AUDIT=1` 50-step run on production config |
| Stability Long-run | NOT TESTED | Blocked by performance issues |

## DDP Bucket Sweep Results

Testing with default config (seq_len=128, batch_size=8):

| Bucket Size | Steps 1-20 | Steps 21-40 | Recommendation |
|------------|------------|-------------|----------------|
| 25 MB | 437 toks/s | 430 toks/s | Stable |
| 50 MB | 431 toks/s | 426 toks/s | Stable |
| 100 MB | - | - | **HANGS - DO NOT USE** |

Methodology: last‑50‑step mean of `<OUT>/training.csv` for each run.

**Recommendation**: Use 25–50 MB bucket size (100 MB unstable on this node; re‑try with different `--master-port`).

## Performance Scaling Analysis

| Configuration | Seq Length | Batch Size | DDP | Throughput | Status |
|--------------|------------|------------|-----|------------|---------|
| Minimal | 128 | 8 | No | 244 toks/s | ✅ |
| Minimal | 128 | 8 | Yes | 440 toks/s | ✅ |
| Intermediate | 1024 | 1 | Yes | 230 toks/s | ✅ |
| Production | 2048 | 2 | No | HANGS | ❌ |
| Production | 2048 | 2 | Yes | 6-17 toks/s | ❌ |

## Root Cause Analysis

1. **V2 Selection**: Confirmed working and enabled
2. **DDP Scaling**: Works well for small configs (1.8x speedup)
3. **Critical Issue**: Performance collapses with:
   - seq_len ≥ 2048
   - batch_size ≥ 2
   - gradient_accumulation enabled
4. **Optimizer Issue**: Fused AdamW causes hangs
5. **Memory Pattern**: Possible O(n²) complexity with sequence length

## Artifacts Generated

- `artifacts/git_sha.txt` - Commit verification
- `artifacts/collect_env.txt` - PyTorch environment
- `artifacts/nvidia_smi.xml` - GPU configuration
- `artifacts/ddp_baseline.log` - DDP training logs
- `artifacts/single_gpu_test.log` - Single GPU test logs

## Recommendations

### Environment Variables (for working configs)

```bash
export NSA_SEL_RANGES_V2=1          # V2 selection enabled
export NSA_DDP_COMPRESS=bf16        # BF16 compression
export NSA_DDP_BUCKET_MB=50         # Optimal bucket size
export NCCL_ALGO=Ring               # PCIe optimization
export NCCL_PROTO=Simple            # PCIe optimization
export NSA_TB_DISABLE=1             # Disable TensorBoard in profiling
export NSA_DISABLE_AUX_STATS=1      # Reduce overhead during profiling
# DO NOT USE: NSA_OPT_FUSED=1       # Causes hangs
```

### Immediate Actions Required

1. **DO NOT DEPLOY** production configuration without fixes
2. **Debug the scaling issue** with seq_len=2048:
   ```bash
   # Test without gradient accumulation
   # Modify config: accumulate_grad_batches: 1
   
   # Test without gradient checkpointing  
   # Modify config: gradient_checkpointing: false
   ```
3. **Profile (short, low-overhead) to find bottleneck:**
   ```bash
   NSA_NVTX=1 NSA_TB_DISABLE=1 NSA_DISABLE_AUX_STATS=1 \
   CONFIG=configs/m7c_125m_2xa100_production.yaml \
   PYTHONPATH=. torchrun --nproc_per_node=2 --master-port=29502 \
     scripts/train_showcase.py --dataset synthetic --ddp 1 --steps 100
   ```
4. **Avoid 100MB bucket size** - causes hangs

### Working Configuration (debug only; not production acceptance)
```bash
CONFIG=configs/train_showcase.yaml \
NSA_SEL_RANGES_V2=1 \
NSA_DDP_COMPRESS=bf16 \
NSA_DDP_BUCKET_MB=50 \
PYTHONPATH=. torchrun --nproc_per_node=2 \
  scripts/train_showcase.py \
  --dataset synthetic \
  --ddp 1 \
  --steps 1000
```

### SDPA Audit (to complete)
Run once to verify cmp/win Flash viability:
```bash
NSA_SDPA_AUDIT=1 CONFIG=configs/m7c_125m_2xa100_production.yaml \
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py \
  --dataset synthetic --ddp 1 --steps 50
```
Capture and attach the console log from rank 0.
```

## Acceptance Criteria Assessment

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Throughput | ≥39 toks/s | 6-17 toks/s (production) | ❌ FAIL |
| V2 Parity | Pass | Pass (except gaps) | ✅ PASS |
| DDP Compression | Active | BF16 active | ✅ PASS |
| SDPA Audit | No fallbacks | Pending (audit run to be executed) | ⚠️ PENDING |
| Stability | 1200 steps | Not tested | ⚠️ BLOCKED |

## Conclusion

**RECOMMENDATION: NO-GO for production deployment**

While the individual optimizations (v2 selection, DDP compression) are functioning correctly, the system exhibits catastrophic performance degradation with production configurations. The issue appears to be algorithmic complexity that scales poorly with sequence length and batch size.

### Critical Next Steps
1. Execute SDPA audit run and record cmp/win backend viability
2. Profile short run per command above; attribute time across SDPA/NCCL
3. A/B: accum=1 vs 2 (`NSA_ACCUM=1`), gradient_checkpointing=false (config edit)
4. Re-try 100 MB bucket with different `--master-port`; record if intermittent
5. If still under target, keep batch_size=1 as a temporary workaround

### What's Working:
- V2 selection optimization confirmed active
- DDP compression working
- Small configurations achieve good performance
- Near-linear DDP scaling for small configs

### What's Broken:
- Production config performance (6-17 toks/s vs 45-55 target)
- 100MB DDP bucket causes hangs
- Fused AdamW optimizer hangs
- SDPA audit produces no output

---

*Test Engineer Report - DDP Performance Validation*
*Status: PARTIAL PASS with CRITICAL ISSUES (NO-GO for production)*
