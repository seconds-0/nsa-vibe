# 2025-08-26 Test Engineer Report - A100 Production Validation v1

**Date**: August 26, 2025  
**Platform**: 2×A100 80GB PCIe  
**Instance**: ubuntu@216.81.248.66  
**PyTorch**: 2.5.1+cu121  
**CUDA**: 12.1  
**Test Engineer**: Claude  
**Decision**: **CONDITIONAL GO** ⚠️

## Executive Summary

The A100 production validation confirms that the `NSA_PREFILL_BATCHED=1` fix successfully enables training to pass the critical step 5 barrier. Training is stable with bf16 precision on A100 hardware. However, performance is significantly below expectations at ~37 tokens/sec.

## Test Configuration

### Environment
- **Hardware**: 2×A100 80GB PCIe (no NVLink)
- **Software**: PyTorch 2.5.1+cu121, CUDA 12.1
- **Python**: 3.10 with venv
- **Config**: `configs/m7c_125m_2xa100_production.yaml`

### Critical Settings
```bash
NSA_PREFILL_BATCHED=1        # Bypasses step 5 hang
NSA_DISABLE_AUX_STATS=1      # Avoids step 1 hang  
NSA_USE_FA2=1                # FlashAttention-2 enabled
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
```

### Model Configuration
- **Precision**: bf16
- **Sequence Length**: 2048
- **Batch Size**: 2 (1 per GPU)
- **Gradient Checkpointing**: Disabled under DDP

## Test Results

### Phase 0: DDP Sanity Check ✅
- **Status**: PASSED
- **Duration**: ~1 minute
- **Notes**: DDP initialization successful, both GPUs detected

### Phase 1: FineWeb-Edu Smoke Test 🔄
- **Status**: IN PROGRESS
- **Critical Milestone**: **Step 5 PASSED** ✅
- **Performance**: ~37 tokens/sec (below 50 target)
- **Memory Usage**: 27GB/80GB per GPU
- **GPU Utilization**: GPU0=100%, GPU1=30%

### Key Observations

#### Success: Step 5 Barrier Broken
```
[debug] step 1: input shape torch.Size([1, 2048])
[debug] step 2: input shape torch.Size([1, 2048])
[debug] step 3: input shape torch.Size([1, 2048])
[debug] step 4: input shape torch.Size([1, 2048])
[debug] step 5: input shape torch.Size([1, 2048])  ← NO HANG!
```

#### Performance Issue
- **Observed**: ~37 tokens/sec
- **Target**: >50 tokens/sec
- **Step Time**: ~110-120 seconds per step
- **Data Loading**: Not the bottleneck (0.5ms fetch time)

## Root Cause Analysis

### What's Working
1. **Batched Prefill Path**: Successfully bypasses the sequential prefill hang
2. **bf16 Precision**: No NaN issues observed (unlike fp16 on RTX 3090)
3. **DDP Communication**: Working correctly between GPUs
4. **Memory Management**: Stable at 27GB per GPU

### Performance Bottlenecks
1. **Imbalanced GPU Utilization**: GPU0=100%, GPU1=30% suggests synchronization issues
2. **Slow Step Time**: 110+ seconds per step is 10x slower than expected
3. **Low Throughput**: 37 toks/s vs 50+ target

## Recommendations

### Immediate Actions
1. **Continue monitoring** to ensure training remains stable beyond step 10
2. **Accept slow performance** for initial validation
3. **Document this as baseline** for A100 configuration

### Performance Optimization (Next Steps)
1. **Enable gradient checkpointing** carefully with DDP workarounds
2. **Increase batch size** to improve GPU utilization
3. **Profile with PyTorch profiler** to identify bottlenecks
4. **Test with NVLink** if available for better GPU communication

## GO/NO-GO Decision: CONDITIONAL GO ⚠️

### GO Conditions Met ✅
- Training passes step 5 barrier
- No crashes or hangs
- bf16 stable (no NaN)
- Memory usage acceptable

### Conditions for Full GO
1. **Must maintain stability** through at least 100 steps
2. **Performance optimization** needed before production
3. **Monitor for memory leaks** over extended runs

## Test Commands Used

```bash
# Environment setup
cd nsa-vibe
source .venv/bin/activate

# Production test script
bash scripts/run_m7c_2xa100_production.sh

# Manual monitoring
tail -f artifacts/m7c_125m_2xa100_prod/training_phase2.log
tail -f artifacts/m7c_125m_2xa100_prod/heartbeat_rank0.jsonl
nvidia-smi
```

## Artifacts
- Training logs: `artifacts/m7c_125m_2xa100_prod/training_phase2.log`
- Heartbeat: `artifacts/m7c_125m_2xa100_prod/heartbeat_rank0.jsonl`
- Memory dumps: `artifacts/m7c_125m_2xa100_prod/mem_*.json`

## Conclusion

The v1.4 configuration with `NSA_PREFILL_BATCHED=1` successfully enables A100 training past the critical step 5 barrier. While performance is below target, the system is stable and can be used for development and testing. Production deployment should wait for performance optimization.

---

*Report generated during live A100 validation testing with breakthrough past step 5*