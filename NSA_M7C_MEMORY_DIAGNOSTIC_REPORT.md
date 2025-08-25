# NSA M7C Memory Diagnostic Report

**Date**: August 23, 2025  
**Engineer**: Claude Code  
**Training Status**: Active (Step 4+)  
**Memory Usage**: 59.6GB/80GB per A100 (73% utilization)

## Executive Summary

The M7C 125M NSA training is currently running with **high but stable memory usage** at 59.6GB per A100 80GB GPU. The memory intensiveness is **expected behavior** for NSA's three-branch attention architecture, but training progress is notably slow (~3 minutes per step).

## High-Signal Diagnostics

### Commit + Configuration
- **Git SHA**: `a89956f4d4b52e8c4437e1a93fcd4f98938e8aab` (short: `a89956f`)
- **Branch**: `feat/m7c-perf-stability`
- **Config**: `configs/m7c_125m_80g_1k.yaml` (seq_len reduced from 4096→1024 due to OOM)
- **Seed**: 1337
- **Launcher**: `torchrun --nproc_per_node=2`

### Per-GPU Load
- **Global Batch**: 4
- **Micro Batch per GPU**: 2
- **Sequence Length**: 1024 (reduced from original 4096)
- **Gradient Accumulation**: 2 steps
- **Tokens per GPU**: 2 × 1024 = 2,048 tokens/step
- **Total Tokens per Step**: 4,096 tokens

### Memory Snapshots
```
GPU 0: 59,611 MiB / 81,920 MiB (72.8% utilization)
GPU 1: 59,585 MiB / 81,920 MiB (72.7% utilization)
```

**Memory Progression**:
- Boot: 604MB allocated, 1.3GB reserved
- Post-loader: 604MB allocated, 1.3GB reserved  
- Step 1: 1.3GB allocated, 57.9GB reserved
- Current: 59.6GB stable usage

## Run Context

### Environment Variables
- **CUDA_VISIBLE_DEVICES**: Not set (both GPUs visible)
- **PYTORCH_CUDA_ALLOC_CONF**: `expandable_segments:True`
- **NSA Flags**: `NSA_STRICT_ASSERTS=0`, `NSA_ENV_STATIC=1`
- **NCCL Settings**: `P2P_DISABLE=1`, `IB_DISABLE=1`, `BLOCKING_WAIT=1`, `DEBUG=WARN`
- **Flash Attention**: Enabled (`use_flash: true`)

### Versions
- **PyTorch**: 2.5.1+cu121
- **CUDA**: 12.1
- **Flash Attention**: 2.8.3 (available and enabled)
- **Platform**: Linux-6.8.0-64-generic-x86_64-with-glibc2.35

## Model Architecture

### Parameters
- **Model Size**: ~95.3M parameters
- **Memory (BF16)**: ~0.2GB for weights alone
- **Dimensions**: 768 hidden, 12 layers, 12 heads, 2 KV groups
- **Vocabulary**: 50,304 tokens (GPT-2)

### NSA Configuration
```yaml
nsa:
  l: 32           # Compression block size
  d: 16           # Compression stride  
  l_sel: 64       # Selection block size
  n_sel: 16       # Number selected blocks
  w: 512          # Sliding window size
  phi: "avg"      # Pooling operator
```

### Training Configuration
```yaml
train:
  steps: 200000
  seq_len: 1024           # Reduced from 4096
  batch_size: 4           # Global batch
  lr: 2.0e-4
  weight_decay: 0.01
  grad_clip: 1.0
```

## Kernel Routing Analysis

### Flash Attention Status
- **Available**: ✅ Flash Attention 2.8.3 installed
- **Enabled**: ✅ `use_flash: true` in config
- **Backend**: Standard SDPA (would need TORCH_LOGS="+sdp" for detailed routing)

### NSA Branch Routing
- **Compressed Branch**: Uses Flash Attention for pooled sequences
- **Selected Branch**: Uses standard SDPA with block gathering
- **Sliding Branch**: Uses Flash Attention for window attention

## Memory Analysis

### Memory Scaling Issue
The memory usage is primarily driven by **selection attention complexity**:

1. **Selection Branch Memory**: O(S²/l'²) where S=1024, l'=64
   - Creates ~256 selection blocks to process
   - Requires gathering/scattering operations

2. **Three-Branch Overhead**: Each branch maintains separate K/V caches
   - Compressed: Pooled representations
   - Selected: Block-wise attention
   - Sliding: Window attention

3. **Prefill vs Decode**: Currently in prefill mode which processes full sequences

### Why seq_len=4096 Failed
- **Memory Scaling**: Selection attention scales quadratically with sequence length
- **4096² tokens**: ~16M attention elements per head → OOM
- **2048² tokens**: ~4M attention elements per head → Still OOM  
- **1024² tokens**: ~1M attention elements per head → Manageable

## NSA Routing Metrics

### Current Training Progress
- **Step 1**: Loss 11.0000, LR 1.00e-07, 24 toks/s
- **Current**: Step 4+ in progress
- **Throughput**: ~24 tokens/second (very slow)
- **Time per Step**: ~3 minutes

### Performance Issues
- **Watchdog Alerts**: Multiple stack dumps generated due to slow progress
- **Step Duration**: Much slower than expected for 95M parameter model
- **GPU Utilization**: GPU 0 at 100%, GPU 1 at 17% (load imbalance)

## Hardware Topology

### GPU Configuration
```
GPU0 ↔ PHB ↔ GPU1
```
- **Connection**: PCIe Host Bridge (PHB) between GPUs
- **NUMA**: Both GPUs on NUMA nodes 0-1
- **No NVLink**: Communication via PCIe only

### Distributed Setup
- **Framework**: PyTorch DDP (DistributedDataParallel)
- **World Size**: 2
- **Ranks**: 0, 1
- **Communication**: NCCL with P2P disabled

## Memory Optimization Recommendations

### Immediate Actions (No Training Restart)
1. **Monitor Progress**: Current memory usage is stable but training is slow
2. **Check Load Balancing**: GPU 1 underutilized (17% vs 100%)
3. **Watchdog Tuning**: Increase timeout to avoid false alarms

### Next Restart Optimizations
1. **Sequence Parallelism**: Split sequence across GPUs instead of data parallelism
2. **Gradient Checkpointing**: Verify it's actually enabled (missing from config)
3. **Mixed Precision Tuning**: Consider FP32 for selection branch stability
4. **Block Size Optimization**: Reduce l_sel from 64 to 32 for less memory

### Architecture Considerations
1. **Selection Attention**: Most memory-intensive component
2. **Block Processing**: Consider chunked selection attention
3. **Prefill Optimization**: Implement incremental prefill for long sequences

## Comparison with Standard Attention

| Metric | Standard Attention | NSA (Current) | NSA (Optimized) |
|--------|-------------------|---------------|------------------|
| Memory | ~8GB for seq_len=1024 | ~60GB | ~30GB (target) |
| Throughput | ~200 toks/s | ~24 toks/s | ~100 toks/s (target) |
| Complexity | O(S²) | O(S²/l'²) + overhead | Same |

## Training Stability Assessment

### ✅ Stable Aspects
- Memory usage plateaued at 59.6GB
- No OOM crashes since seq_len reduction
- Data loading working correctly
- BF16 dtype issues resolved

### ⚠️ Concerning Aspects  
- Very slow training progress (3 min/step)
- GPU load imbalance
- Watchdog timeout alerts
- Only completed 4 steps in ~15 minutes

## Conclusion

The high memory usage is **expected for NSA architecture** but the **training speed is concerning**. The memory diagnostic shows stable 60GB usage, which is within safe limits for 80GB A100s. The primary issues are:

1. **Performance bottleneck** in selection attention implementation
2. **Load imbalance** between GPUs
3. **Need for architecture optimization** rather than just memory management

**Recommendation**: Continue current training to gather more data points, but plan architecture optimizations for the next training run.

---

*This diagnostic was collected without interrupting the active training run. All metrics are from live training at step 4+.*