# NSA Backward Pass Research Report

**Date**: 2025-08-25  
**Purpose**: Research findings on PyTorch backward pass issues relevant to NSA training hang  
**Context**: Supporting core engineer's analysis of ≥5 layer backward pass hang

## Executive Summary

Research reveals multiple known PyTorch issues that align with NSA's backward pass symptoms:
- **SDPA with dense masks** can cause extreme memory usage and hangs
- **Advanced indexing with duplicates** causes 2x performance degradation
- **Scatter operations** have memory-inefficient backward passes
- **No automatic graph pruning** means large intermediates persist until backward completes
- **CUDA fragmentation** on A100 can manifest as hangs when reserved memory climbs

## 1. PyTorch SDPA Backward Failure Modes

### Critical Findings
- **Illegal memory access** occurs in backward pass when sequence length ≥65,536 tokens with custom attention masks (GitHub #145040)
- **Dense masks with -inf values** cause numerical instabilities during backward, especially with large sequences
- **Memory-efficient backend** has known issues with mask support on certain GPUs
- **Multiple tiny SDPA calls** in loops vs single batched call significantly increases memory overhead

### Relevance to NSA
The NSA selection branch uses SDPA with dense masks in the masked path:
- Creates [B,S,G,S_kv] boolean masks converted to attention bias
- Multiple per-token SDPA calls in gather path
- Dense masks with many -inf entries for causal masking

### Recommended Mitigations
1. Use sparse masking or nested tensors instead of dense masks
2. Batch SDPA calls when possible rather than looping
3. Explicitly control backend selection to avoid problematic implementations
4. Consider xformers as alternative for better mask support

## 2. Index Operations Backward Complexity

### Key Issues
- **Backward pass limitation**: Only implemented for `src.shape == index.shape` in scatter operations
- **Memory inefficiency**: Scatter operations require intermediate tensors that persist through backward
- **Non-deterministic behavior**: When indices have duplicates, gradient propagation is arbitrary
- **Atomic operations on GPU**: Lead to non-deterministic accumulation order

### Performance Impact
- Research shows memory bottlenecks from scatter operations in graph neural networks
- Alternative implementations (edge-parallel) exist to avoid intermediate tensor computation
- Standard implementation scales poorly with many non-zero indices

### Relevance to NSA
The selection scorer uses:
- `index_add_` with COO indices in `map_pcmp_to_pslc_batched`
- Multiple scatter operations for varlen packing/unpacking
- Advanced indexing for K/V gathering

## 3. Advanced Indexing Gradient Performance

### Documented Issues
- **2x slowdown** when index tensor contains many duplicate indices (GitHub #41162)
- **Memory bloat** in loop-heavy simulations with backward passes using 14GB+ unexpectedly
- **Order of magnitude slower** than NumPy for basic indexing operations
- **Graph bloat** from recreating computation graph at each iteration

### Autograd Dynamics
- Graph recreated from scratch after each `.backward()` call
- No automatic dead code elimination - zero multiplications still computed
- In-place operations require rewriting computational graph
- Saved tensors for backward can't be freed until gradient computation

### Implications for NSA
Multiple layers compound the issue:
- Each layer's K[b,g,idx] indexing adds to graph
- Selection indices may have duplicates (forced blocks)
- Tight loops in gather path multiply overhead

## 4. Flash Attention 2 Autograd Interactions

### Current Status
- **Memory linear in sequence length** vs quadratic for standard attention
- **Variable length support** through `flash_attn_varlen_func` with cumulative sequence lengths
- **Backward pass requirements**: Head dim >192 requires A100/A800/H100
- **Consumer GPU support**: Head dim 256 works on RTX 4090 (without dropout) as of v2.5.5

### Memory Savings
- 10x memory savings at sequence length 2K
- 20x memory savings at sequence length 4K
- Chunk-based calculations reduce peak memory requirements

### Integration Notes
- Requires PyTorch 2.2+
- Available through `F.scaled_dot_product_attention` with backend settings
- FlexAttention (PyTorch 2.5.0) provides automatic backward generation

### NSA Considerations
FA2 is already conditionally used in NSA, but:
- Varlen support could improve packed paths
- Current implementation may not leverage all optimizations
- Backward pass savings not realized if other branches dominate memory

## 5. Autograd Graph Pruning Best Practices

### Key Limitations
- **No automatic pruning**: PyTorch doesn't eliminate dead code or unused computations
- **No eager pruning**: Zero multiplications still computed at full cost
- **Graph persistence**: Intermediates saved until backward completes
- **Dynamic recreation**: Graph rebuilt at each iteration, no cross-iteration optimization

### Memory Management Strategies

#### torch.no_grad()
- Prevents graph creation for operations not needing gradients
- Essential for inference and non-differentiable operations
- Should wrap selection scoring if indices-only output

#### detach()
- Creates gradient-free copy while preserving values
- Useful for breaking gradient flow at specific points
- Can prevent reference cycles reducing memory

#### saved_tensors_hooks
- Customize tensor saving/retrieval during forward
- Pack hook can use `x.detach()` to reduce memory
- Allows offloading to CPU or compression

### NSA Optimization Opportunities
1. Wrap integer selection operations in `torch.no_grad()`
2. Detach intermediate scores after computing indices
3. Use hooks to manage large intermediates (p_cmp_all, p_slc_all)
4. Explicitly free tensors not needed for backward

## 6. CUDA Allocator Behavior and Fragmentation

### A100-Specific Issues
- **Fragmentation manifests as hang**: When reserved >> allocated memory
- **cudaMallocAsync issues**: Reports of 20GB+ "missing" memory on A100
- **Allocator stalls**: Can appear as hang when waiting for contiguous block

### expandable_segments Solution
- **Reduces fragmentation**: Uses 2MiB pages (small) and 20MiB pages (large)
- **Proven impact**: Reduced VRAM from 16.39GB to 10.83GB in benchmarks
- **Trade-off**: Slower initial allocation (milliseconds) but better long-term behavior
- **Gap filling**: Attempts to keep segments contiguous by filling unmapped blocks

### Configuration Options
```bash
# Recommended for NSA training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

# Alternative for debugging
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
```

### NSA Memory Pattern
The 73GB reserved (vs much lower allocated) exactly matches fragmentation symptoms:
- Multiple layers allocate/free in complex patterns
- Peak reserved grows with each layer
- Allocator can't find contiguous blocks for new allocations

## 7. Actionable Recommendations for NSA

### Immediate Experiments (No Code Changes)

#### Force Single Branch Testing
```bash
# Isolate which branch causes hang
NSA_FORCE_BRANCH=win  # Test sliding only
NSA_FORCE_BRANCH=cmp  # Test compressed only  
NSA_FORCE_BRANCH=sel  # Test selection only
```

#### Backend Configuration Testing
```bash
# Try different selection backends
NSA_USE_SEL_PACK=1 NSA_USE_SEL_MASK=0  # Packed only
NSA_USE_SEL_PACK=0 NSA_USE_SEL_MASK=1  # Masked only
NSA_FORCE_PARITY=1  # Force gather/parity path
```

#### Memory Diagnostics
```bash
# Enhanced memory debugging
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
CUDA_LAUNCH_BLOCKING=1  # Surface CUDA errors immediately
torch.autograd.set_detect_anomaly(True)  # Detailed error messages
```

### Code Optimization Priorities

1. **Wrap selection scoring in torch.no_grad()**
   - Scores only produce indices, no gradient needed
   - Would free p_cmp_all, p_slc_all after forward

2. **Batch SDPA calls in selection gather**
   - Current: Multiple per-(b,t,g) calls
   - Better: Single batched call with masking

3. **Use sparse operations for COO mapping**
   - Current: Dense index_add_
   - Better: torch.sparse operations

4. **Implement saved_tensors_hooks**
   - Compress or offload large intermediates
   - Particularly for p_cmp_all [B,S,G,h,S_cmp]

5. **Profile memory per operation**
   ```python
   with torch.profiler.profile(
       activities=[ProfilerActivity.CUDA],
       profile_memory=True
   ) as prof:
       # Run forward/backward
   print(prof.key_averages().table(
       sort_by="cuda_memory_usage", row_limit=10
   ))
   ```

### Hypothesis Validation Tests

Based on research, the most likely causes in order:

1. **Dense mask SDPA backward** (H1)
   - Test: Force packed/gather to avoid masked path
   - Expected: Hang disappears

2. **Scatter operations scaling** (H2)
   - Test: Reduce n_sel to minimize scatter ops
   - Expected: Higher layer threshold before hang

3. **Index_add_ COO backward** (H3)
   - Test: Mock with dense operations temporarily
   - Expected: Different memory pattern

4. **Autograd graph accumulation** (H4)
   - Test: Add torch.no_grad() around scoring
   - Expected: Reduced memory, possible fix

## 8. Paper Alignment Considerations

The research suggests NSA's implementation aligns with standard practices, but:

1. **Selection as non-differentiable**: Paper treats selection indices as discrete
   - Current: Gradients flow through attended values only ✓
   - Optimization: Aggressively detach scoring paths to save memory

2. **Training gradient design**: No soft selection/STE mentioned
   - Current: Hard selection ✓
   - Consider: Explicitly document gradient flow assumptions

3. **Memory expectations**: Paper doesn't discuss training memory
   - Reality: O(S²) patterns emerge from implementation details
   - Need: Architecture changes or training-specific optimizations

## Conclusion

The research strongly indicates NSA's backward hang stems from well-known PyTorch patterns:
- Dense attention masks creating memory pressure
- Advanced indexing with duplicates degrading performance  
- Lack of automatic graph pruning retaining large intermediates
- CUDA fragmentation on A100 manifesting as allocation stalls

The "≥5 layers" threshold aligns with cumulative memory pressure crossing a fragmentation cliff. The 73GB reserved memory exactly matches known fragmentation symptoms.

**Most promising immediate fix**: Configure backends to avoid dense masks and enable expandable_segments. Longer term, implement torch.no_grad() wrapping and saved_tensors_hooks for memory efficiency.

---

**Research compiled**: 2025-08-25  
**Sources**: PyTorch GitHub issues, documentation, forums, and research papers  
**For**: Core engineering team debugging NSA backward pass
