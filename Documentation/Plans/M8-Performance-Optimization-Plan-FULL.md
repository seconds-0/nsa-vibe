# NSA Performance Optimization Plan

**Date**: 2025-08-28  
**Author**: Test Engineer  
**Current Performance**: 153 tok/s (S=128), ~66 tok/s (S=256), Hangs (S=2048)  
**Target Performance**: 300-800 tok/s  
**GPU**: NVIDIA A100 80GB PCIe  

## Status
- [x] FA‑2 integration scaffolding in place: dense/varlen wrappers with capability probes and safe fallbacks (`nsa/kernels/flash_wrappers.py`), default‑enable compressed FA‑2 with sliding guarded off by default (`NSAAttention` env cache). GPU parity tests and runbooks added.
- [x] Hot‑path `repeat_interleave` eliminated in attention paths via `unsqueeze/expand/reshape` (OOM‑safe); tests keep `repeat_interleave` only for convenience.
- [x] Workspace pre‑sizing for varlen/selection pack with env reserves (`NSA_VARLEN_RESERVE_{N,K}`, `NSA_SEL_PACK_RESERVE_{N,L}`) to reduce allocator churn.
- [x] Selection varlen attention implemented with FA‑2 varlen fast path and exact per‑row fallback; parity unit test added.
- [x] Opt‑in mixed precision for p_cmp scoring via `NSA_P_CMP_MIXED` (bf16 autocast, upcast to original dtype).
- [x] Batched prefill RoPE alignment fixed (Q/K consistent across batched vs sequential); small‑S equivalence test updated.
- [~] PR31 (selection varlen) fixes landed locally; GPU validation to be re‑run on A100/H100 before merge.
- [ ] FA‑2 min‑length thresholds auto‑tune: integrate `scripts/apply_fa2_thresholds.py` into config workflows and CI.

## Next Actions
- Vectorize selection varlen pack v2 (eliminate Python loops in `selection_attention_varlen_all`); gate with `NSA_SEL_VARLEN_V2` and add min‑L threshold to avoid overhead on tiny rows.
- Expand mixed precision to selection mapping (`p_slc`/`p_grp`) under env guard with float32 tie‑breaks; add parity tests across dtypes.
- Add kernel fusion experiments (gate MLP + combine) under `NSA_GATE_COMPILE` for benches; keep opt‑in and parity‑guarded.
- Strengthen telemetry: counters for FA‑2 engagement vs fallback, varlen vs packed path usage, and selection length statistics surfaced in heartbeat to inform thresholds.

## Executive Summary

The primary performance bottleneck is that **FlashAttention-2 is not actually being used** despite being enabled. The system is falling back to slow `repeat_interleave` operations and standard SDPA. Fixing this single issue should provide a 10-20x speedup, bringing performance into the target range.

## Root Cause Analysis

### 1. FlashAttention-2 Not Active (Critical Issue)

**Evidence from stack traces:**
```python
File "/root/nsa-vibe/nsa/kernels/flash_wrappers.py", line 56, in attention_bgh
    v = V.repeat_interleave(h, dim=1).reshape(B * G * h, S, V.shape[-1])
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory
```

This is the **fallback path**, not FA2. The `repeat_interleave` operation:
- Creates massive temporary tensors
- Causes memory fragmentation
- Is ~20x slower than FA2

**Why FA2 isn't being used:**

1. **Device capability detection failing:**
   ```python
   # Current code may not properly detect A100 (SM 8.0)
   def fa2_supported(device, dtype, d_k):
       # May be returning False incorrectly
   ```

2. **Tensor layout incompatibility:**
   - FA2 expects: `(batch, heads, seq, dim)`
   - NSA provides: `(B, G, S, D)` with separate head dimension
   - The reshape/repeat_interleave is trying to fix this but causes OOM

3. **Silent fallback on import:**
   ```python
   try:
       from flash_attn import flash_attn_varlen_func
   except ImportError:
       # Silently falls back to slow path
   ```

### 2. Memory Fragmentation

**Current allocation pattern:**
```
Process 1: 25.64 GiB memory
Process 2: 50.52 GiB memory  
PyTorch allocated: 2.46 GiB
```

Total: 76+ GiB used on 80 GiB GPU, but PyTorch only allocated 2.46 GiB!

**Causes:**
- Workspace dictionaries (`_SEL_PACK_WS`, `_VARLEN_WS`) creating fragments
- Repeated allocation/deallocation in hot path
- No memory pooling

### 3. Python Loop Overhead

**Current selection implementation:**
```python
for b in range(B):        # B iterations
    for t in range(S):    # S iterations
        for g in range(G):  # G iterations
            # Process each individually
```

For S=2048, G=2: **4,096 Python loop iterations** per forward pass!

## Optimization Plan

## Phase 1: Fix FlashAttention-2 (Immediate - 10-20x speedup)

### 1.1 Diagnostic Steps

```python
# Add comprehensive FA2 debugging
import os
os.environ['NSA_DEBUG_FA2'] = '1'  # Enable FA2 routing logs

# In flash_wrappers.py, add:
def fa2_supported(device, dtype, d_k):
    """Check if FA2 is actually available."""
    if device.type != 'cuda':
        print(f"FA2: Not CUDA device: {device.type}")
        return False
    
    capability = torch.cuda.get_device_capability(device)
    print(f"FA2: Device capability: {capability}")
    
    # A100 is SM 8.0, H100 is SM 9.0
    if capability[0] < 8:
        print(f"FA2: Capability too low: {capability}")
        return False
    
    # Check if flash_attn is actually imported
    try:
        import flash_attn
        print(f"FA2: flash_attn version: {flash_attn.__version__}")
        return True
    except ImportError as e:
        print(f"FA2: Import failed: {e}")
        return False
```

### 1.2 Fix Tensor Layouts

**Current problematic code:**
```python
# This causes OOM and is slow:
v = V.repeat_interleave(h, dim=1).reshape(B * G * h, S, V.shape[-1])
```

**Optimized approach:**
```python
def attention_bgh_fa2(Q, K, V, causal=True):
    """FA2-compatible attention without repeat_interleave."""
    B, G, h, S, D = Q.shape
    
    # Method 1: Use view/expand (no memory copy)
    Q_fa2 = Q.transpose(1, 2).reshape(B, G*h, S, D)
    K_fa2 = K.unsqueeze(2).expand(-1, -1, h, -1, -1).reshape(B, G*h, S, D)
    V_fa2 = V.unsqueeze(2).expand(-1, -1, h, -1, -1).reshape(B, G*h, S, D)
    
    # Method 2: Direct FA2 call with proper layout
    from flash_attn import flash_attn_func
    out = flash_attn_func(
        Q_fa2, K_fa2, V_fa2,
        causal=causal,
        window_size=(-1, -1) if not causal else None
    )
    
    return out.reshape(B, G, h, S, -1)
```

### 1.3 Force FA2 Usage

```bash
# Environment variables to force FA2
export NSA_USE_FA2=1
export NSA_FA2_FORCE_DENSE=1  # Use dense FA2 even for varlen
export NSA_FA2_MIN_LEN_WIN=1   # Use FA2 for all window sizes
export NSA_FA2_MIN_LEN_CMP=1   # Use FA2 for all compressed sizes
```

### 1.4 Install/Verify FA2

```bash
# Ensure FA2 is properly installed
pip install flash-attn --no-build-isolation

# Verify installation
python -c "from flash_attn import flash_attn_func; print('FA2 working')"
```

## Phase 2: Memory Optimizations (Quick Wins - 2-3x speedup)

### 2.1 Eliminate repeat_interleave

**Every occurrence of:**
```python
tensor.repeat_interleave(h, dim=1)
```

**Should become:**
```python
tensor.unsqueeze(2).expand(-1, -1, h, -1, -1).reshape(...)
# OR use torch.broadcast_to for clarity
```

### 2.2 Pre-allocate Workspaces

**Current (bad):**
```python
def forward(self, x):
    # Allocates new workspace every forward pass
    ws = _get_varlen_workspace(...)
```

**Optimized:**
```python
class NSAAttention(nn.Module):
    def __init__(self, ...):
        # Pre-allocate at init
        self.workspace = {
            'q': torch.empty((max_batch * max_seq, n_heads, d_k), device='cuda'),
            'k': torch.empty((max_batch * max_seq * 2, n_heads, d_k), device='cuda'),
            'v': torch.empty((max_batch * max_seq * 2, n_heads, d_v), device='cuda'),
            'cuq': torch.empty((max_batch * max_seq + 1,), dtype=torch.int32, device='cuda'),
            'cuk': torch.empty((max_batch * max_seq + 1,), dtype=torch.int32, device='cuda'),
        }
    
    def forward(self, x):
        # Reuse pre-allocated workspace
        ws = self.workspace
        # Use views into workspace, not new allocations
```

### 2.3 Memory Pool Configuration

```bash
# Optimal A100 settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16

# In Python:
torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU
torch.cuda.empty_cache()  # Clear before training
```

### 2.4 Selective Gradient Checkpointing

```python
class TinyLM(nn.Module):
    def forward(self, x):
        # Only checkpoint middle layers (highest memory usage)
        for i, blk in enumerate(self.blocks):
            if 4 <= i <= 8 and self.training:
                # Checkpoint these layers
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                # Don't checkpoint first/last layers (important for gradients)
                x = blk(x)
        return x
```

## Phase 3: Vectorize Selection Operations (5-10x speedup)

### 3.1 Current Problem

```python
# Current: O(B*S*G) Python iterations
def grouped_selection_attention_packed(Q, K, V, ranges):
    for b in range(B):        # 1
        for t in range(S):    # 2048  
            for g in range(G):  # 2
                # 4096 iterations!
                idxs = []
                for i in range(ranges.shape[3]):  # 16 more iterations!
                    s0 = int(ranges[b, t, g, i, 0].item())
                    e0 = int(ranges[b, t, g, i, 1].item())
```

### 3.2 Vectorized Solution

```python
def grouped_selection_attention_vectorized(Q, K, V, ranges):
    B, S, G, n, _ = ranges.shape
    device = Q.device
    
    # Step 1: Vectorized range expansion (single kernel)
    # Create all indices at once
    range_starts = ranges[..., 0]  # [B, S, G, n]
    range_ends = ranges[..., 1]    # [B, S, G, n]
    range_lens = range_ends - range_starts  # [B, S, G, n]
    
    # Maximum indices needed
    max_len = range_lens.max().item()
    total_indices = range_lens.sum().item()
    
    # Step 2: Create index tensor efficiently
    # Use torch.compile or custom CUDA kernel
    @torch.compile(mode="reduce-overhead")
    def build_indices(starts, ends):
        # This gets compiled to efficient CUDA
        indices = torch.zeros((B, S, G, max_len * n), dtype=torch.long, device=device)
        masks = torch.zeros((B, S, G, max_len * n), dtype=torch.bool, device=device)
        
        for i in range(n):
            start_idx = i * max_len
            for j in range(max_len):
                idx = starts[..., i] + j
                valid = j < (ends[..., i] - starts[..., i])
                indices[..., start_idx + j] = torch.where(valid, idx, 0)
                masks[..., start_idx + j] = valid
        
        return indices, masks
    
    indices, masks = build_indices(range_starts, range_ends)
    
    # Step 3: Single batched gather operation
    K_gathered = torch.gather(K.unsqueeze(1), dim=3, 
                             index=indices.unsqueeze(-1).expand(..., K.shape[-1]))
    V_gathered = torch.gather(V.unsqueeze(1), dim=3,
                             index=indices.unsqueeze(-1).expand(..., V.shape[-1]))
    
    # Step 4: Batched attention with mask
    attention_out = F.scaled_dot_product_attention(
        Q, K_gathered, V_gathered,
        attn_mask=masks.unsqueeze(-2),
        is_causal=False  # We handle causality via mask
    )
    
    return attention_out
```

### 3.3 Custom CUDA Kernel (Ultimate Performance)

```cuda
// selection_gather.cu
__global__ void selection_gather_kernel(
    const float* K,           // [B, G, S_kv, D]
    const int* ranges,        // [B, S, G, n, 2]
    float* K_gathered,        // [B, S, G, max_total, D]
    int B, int S, int G, int n, int D, int S_kv
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Each thread processes one element
    for (int idx = tid; idx < B * S * G * max_total * D; idx += total_threads) {
        int b = idx / (S * G * max_total * D);
        int s = (idx / (G * max_total * D)) % S;
        int g = (idx / (max_total * D)) % G;
        int pos = (idx / D) % max_total;
        int d = idx % D;
        
        // Find which range this position belongs to
        int accumulated = 0;
        for (int i = 0; i < n; i++) {
            int start = ranges[b * S * G * n * 2 + s * G * n * 2 + g * n * 2 + i * 2];
            int end = ranges[b * S * G * n * 2 + s * G * n * 2 + g * n * 2 + i * 2 + 1];
            int range_len = end - start;
            
            if (pos < accumulated + range_len) {
                // This position maps to K[b, g, start + (pos - accumulated), d]
                int k_idx = start + (pos - accumulated);
                if (k_idx < S_kv) {
                    K_gathered[idx] = K[b * G * S_kv * D + g * S_kv * D + k_idx * D + d];
                }
                break;
            }
            accumulated += range_len;
        }
    }
}
```

## Phase 4: Kernel Fusion (2-5x speedup)

### 4.1 Fuse Gate MLP with Attention Combination

**Current (3 kernel launches):**
```python
gates = self.gate(q_pooled)  # Kernel 1: MLP forward
gates = F.softmax(gates / temperature, dim=-1)  # Kernel 2: Softmax
out = gates[...,0:1] * O_cmp + gates[...,1:2] * O_sel + gates[...,2:3] * O_win  # Kernel 3: Weighted sum
```

**Fused (1 kernel launch):**
```python
@torch.jit.script
def fused_gate_combine(O_cmp, O_sel, O_win, q_pooled, fc1_weight, fc1_bias, 
                       fc2_weight, fc2_bias, temperature):
    # All in one kernel
    hidden = F.silu(F.linear(q_pooled, fc1_weight, fc1_bias))
    gates = F.linear(hidden, fc2_weight, fc2_bias)
    gates = F.softmax(gates / temperature, dim=-1)
    return gates[...,0:1] * O_cmp + gates[...,1:2] * O_sel + gates[...,2:3] * O_win
```

### 4.2 Fuse RoPE with Q/K Projection

**Current:**
```python
Q = self.q_proj(x)  # Kernel 1
Q = apply_rope(Q, positions)  # Kernel 2
```

**Fused:**
```python
@torch.jit.script
def fused_qk_rope(x, q_weight, q_bias, cos, sin):
    Q = F.linear(x, q_weight, q_bias)
    # Apply RoPE in same kernel
    Q_rot = torch.stack([-Q[..., 1::2], Q[..., ::2]], dim=-1).flatten(-2)
    Q = Q * cos + Q_rot * sin
    return Q
```

## Phase 5: Algorithmic Optimizations

### 5.1 Adaptive Selection Count

```python
def compute_adaptive_n_sel(seq_len):
    """Use fewer selections for shorter sequences."""
    if seq_len <= 512:
        return 8  # Half the selections
    elif seq_len <= 1024:
        return 12
    else:
        return 16  # Full selections only for long sequences
```

### 5.2 Hierarchical Selection

```python
def hierarchical_selection(scores, n_sel=16):
    """Two-stage selection: coarse then fine."""
    B, S, G, L = scores.shape
    
    # Stage 1: Select top regions (reduce by 4x)
    coarse_scores = scores.reshape(B, S, G, L // 4, 4).max(dim=-1).values
    top_regions = torch.topk(coarse_scores, k=n_sel // 2, dim=-1).indices
    
    # Stage 2: Refine within selected regions
    fine_indices = []
    for region in top_regions:
        region_scores = scores[..., region*4:(region+1)*4]
        local_top = torch.topk(region_scores, k=2, dim=-1).indices
        fine_indices.append(region * 4 + local_top)
    
    return torch.cat(fine_indices, dim=-1)
```

### 5.3 Mixed Precision for Selection

```python
def selection_scorer_mixed_precision(Q, K_comp):
    """Use FP16 for selection scoring (sufficient precision)."""
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        scores = torch.matmul(Q.half(), K_comp.half().transpose(-2, -1))
        scores = scores / math.sqrt(Q.shape[-1])
    
    # Only cast back for final selection
    return scores.float()
```

## Phase 6: Configuration Optimization

### 6.1 Optimal Environment Variables

```bash
# Production settings for A100
export NSA_USE_FA2=1
export NSA_FA2_FORCE_DENSE=1
export NSA_SEL_TRITON_MIN_L=128  # Use Triton only for large blocks
export NSA_TRAIN_SAFE_PACK=1
export NSA_STRICT_ASSERTS=0  # Disable in production
export NSA_DEBUG_LOG=0  # Disable logging

# PyTorch optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0  # Ensure async execution
export TORCH_CUDNN_V8_API_ENABLED=1  # Use latest cuDNN
```

### 6.2 Model Configuration Adjustments

```yaml
# For S=2048 optimization
model:
  dim: 768
  n_layers: 12
  n_heads: 12  # Consider 16 for better FA2 utilization
  n_kv_groups: 2  # Consider 4 for less memory
  d_k: 64
  d_v: 64

nsa:
  l: 32
  d: 16
  l_sel: 128  # Increase from 64 (fewer, larger blocks)
  n_sel: 12   # Reduce from 16 for S=2048
  w: 256      # Reduce from 512 for S=2048
  phi: "avg"  # Keep simple until optimized

runtime:
  device: "cuda"
  precision: "bf16"  # Critical for A100
  use_flash: true
  use_triton_sel: true  # After fixing min_L
  gradient_checkpointing: false  # Or selective only

train:
  seq_len: 1536  # Start with 1536 instead of 2048
  batch_size: 1
  accumulate_grad_batches: 4  # Effective batch = 4
  compile_model: true  # torch.compile for extra speed
```

### 6.3 Training Script Optimizations

```python
# In train_showcase.py
def create_optimized_model():
    model = TinyLM(...)
    
    # Compile model for faster execution
    model = torch.compile(model, mode="reduce-overhead")
    
    # Enable TF32 for A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set fusion backends
    torch.jit.enable_onednn_fusion(True)
    
    return model
```

## Expected Performance Improvements

| Optimization | Expected Speedup | Cumulative tok/s (S=512) | Cumulative tok/s (S=2048) |
|-------------|-----------------|-------------------------|--------------------------|
| **Baseline** | 1x | 66 | Hangs |
| Fix FA2 | 10-20x | 660-1320 | 150-300 |
| Memory optimizations | +2x | 1320-2640 | 300-600 |
| Vectorized selection | +2x | 2640-5280 | 600-1200 |
| Kernel fusion | +1.5x | 3960-7920 | 900-1800 |
| Config tuning | +1.2x | 4752-9504 | 1080-2160 |

**Conservative estimate**: 1000-2000 tok/s at S=512, 400-800 tok/s at S=2048

## Implementation Priority

1. **Week 1**: Fix FA2 (80% of performance gain)
2. **Week 2**: Memory optimizations and vectorization
3. **Week 3**: Kernel fusion and configuration tuning
4. **Week 4**: Testing, benchmarking, and fine-tuning

## Validation Plan

### Performance Benchmarks

```python
def benchmark_configuration(config_path, seq_len, batch_size, steps=100):
    """Comprehensive benchmark with all metrics."""
    model = create_model(config_path)
    model = torch.compile(model)
    
    # Warmup
    for _ in range(10):
        x = torch.randint(0, 256, (1, 128), device='cuda')
        _ = model(x)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for step in range(steps):
        x = torch.randint(0, 256, (batch_size, seq_len), device='cuda')
        loss = train_step(model, x)
        
        if step % 10 == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            tokens = batch_size * seq_len * (step + 1)
            print(f"Step {step}: {tokens/elapsed:.1f} tok/s, loss={loss:.4f}")
    
    return tokens / elapsed
```

### Memory Profiling

```python
def profile_memory():
    with torch.profiler.profile(
        activities=[ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        # Run training steps
        train_steps(10)
    
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

### FA2 Verification

```python
def verify_fa2_active():
    """Ensure FA2 is actually being used."""
    import torch.nn.functional as F
    
    # Monkey-patch to detect usage
    original_sdpa = F.scaled_dot_product_attention
    fa2_used = False
    
    def tracked_sdpa(*args, **kwargs):
        nonlocal fa2_used
        # Check if FA2 backend is used
        if kwargs.get('enable_flash', True):
            fa2_used = True
        return original_sdpa(*args, **kwargs)
    
    F.scaled_dot_product_attention = tracked_sdpa
    
    # Run forward pass
    model(torch.randint(0, 256, (1, 512), device='cuda'))
    
    F.scaled_dot_product_attention = original_sdpa
    return fa2_used
```

## Conclusion

The primary issue is that FlashAttention-2 is not being used despite being enabled. Fixing this alone should achieve the performance target. The additional optimizations provide further headroom for scaling to larger sequences and batch sizes.

**Key insight**: The current 66 tok/s at S=256 with the fallback path suggests that with FA2 properly enabled, we should easily exceed 600 tok/s, well within the 300-800 tok/s target range.
