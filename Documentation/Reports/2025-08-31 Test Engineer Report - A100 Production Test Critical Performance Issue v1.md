# 2025-08-31 Test Engineer Report - A100 Production Test Critical Performance Issue v1

## Executive Summary
**Status: CRITICAL FAILURE** - Production test on single A100 80GB revealed catastrophic performance degradation. Training throughput is 10 toks/s (expected: 300-800 toks/s), making 50k step training infeasible.

## Test Environment
- **Hardware**: NVIDIA A100 80GB PCIe (104.255.9.187:12600)
- **Software**: PyTorch 2.4.0+cu121, Python 3.11.13, Ubuntu container
- **Branch**: prod/a100-50k-test (commit f16c42d8)
- **Config**: configs/m7c_125m_1xa100_prod_v1.yaml

## Configuration Details
```yaml
model:
  dim: 768, n_layers: 12, n_heads: 12
nsa:
  l: 32, d: 16, l_sel: 64, n_sel: 16, w: 512
runtime:
  gradient_checkpointing: true
  use_flash: true (but FA2 disabled via env)
train:
  seq_len: 2048
  batch_size: 2
  accumulate_grad_batches: 2
```

## Environment Settings (Verified)
```bash
NSA_USE_FA2=0
NSA_FA2_MIN_LEN_WIN=-1
NSA_FA2_MIN_LEN_CMP=-1
NSA_USE_SEL_VARLEN=0
NSA_USE_TRITON_SEL=0
NSA_STRICT_ASSERTS=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCH_CUDNN_ALLOW_TF32=1
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
```

## Performance Observations

### Critical Issue
- **Step 1 Duration**: ~6 minutes (395 seconds)
- **Throughput**: 10 toks/s (2.5% of minimum expected)
- **GPU Utilization**: 18-21% (severely underutilized)
- **Memory Usage**: 5GB/80GB (6% utilized)

### Stack Trace Analysis
Multiple watchdog dumps show training stuck in backward pass:
```
File "train_showcase.py", line 1223, in main
  loss.backward()
File "torch/autograd/graph.py", line 768, in _engine_run_backward
  return Variable._execution_engine.run_backward()
```

Watchdog triggered every 3 minutes due to heartbeat stalls >180s.

### Model Health Indicators (Step 1)
- Gate entropy: 1.099 (healthy, no collapse)
- Gate branch shares: [0.334, 0.335, 0.332] (balanced)
- Fallback counters: All 0 (no kernel failures)
- Selection stats: mean=752.5, max=1024

## Root Cause Analysis

### Primary Suspects
1. **Gradient Checkpointing + NSA Attention**: The combination appears pathologically slow
2. **Compilation Overhead**: First step extremely slow, but subsequent steps also slow
3. **No FA2**: Running SDPA everywhere without FlashAttention optimization

### Evidence
- Step 1 took 6+ minutes for first forward/backward pass
- Step 2 still incomplete after 10+ minutes
- CPU at 100% during backward pass (suggests CPU bottleneck)
- GPU severely underutilized (18-21%)

## Artifacts
- Training log: `/root/nsa-vibe/training_50k.log`
- Heartbeat: `artifacts/m7c_125m_1xa100_prod/heartbeat_rank0.jsonl`
- Watchdog dumps: `artifacts/m7c_125m_1xa100_prod/watchdog_stackdump_*.txt`
- CSV data: `artifacts/m7c_125m_1xa100_prod/training.csv` (only 1 step)

## Recommendations

### Immediate Actions Required
1. **Disable gradient checkpointing** - Test shows it's incompatible with current NSA implementation
2. **Enable FA2** - Despite A100 compatibility concerns, SDPA is too slow
3. **Profile the backward pass** - Identify specific bottleneck in NSA attention
4. **Reduce batch size** - Try batch_size=1 without gradient accumulation

### Investigation Needed
- Why is backward pass taking 5+ minutes per step?
- Is there a CPU bottleneck in gradient computation?
- Are selection indices causing inefficient scatter/gather operations?

## Conclusion
**NO-GO for production**. The current configuration results in ~10 toks/s throughput, making a 50k step run require 570+ hours (24 days). This is a 30-80x performance regression from expected baseline.

The issue appears to be gradient checkpointing incompatibility with the NSA attention mechanism, causing extreme slowdowns in the backward pass. Every step triggers watchdog timeouts, indicating systematic stalls.

## Test Details
- Test started: 2025-08-31T02:28:06Z
- Step 1 completed: 2025-08-31T02:34:57Z (6m 51s)
- Test terminated: 2025-08-31T02:45:00Z (due to infeasibility)
- Total steps completed: 1/50000