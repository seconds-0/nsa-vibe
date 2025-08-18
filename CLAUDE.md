# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an implementation of Native Sparse Attention (NSA), a drop-in attention module for decoder-only Transformers with trainable, hardware-aligned sparse attention. The implementation follows the paper's architecture combining three branches (Compressed, Selected, Sliding) with learned gates.

## Build and Test Commands

### Environment Setup
```bash
# Use uv for Python environment management
uv venv -p 3.10 .venv
uv pip sync -r requirements.txt
```

### Testing
```bash
# Run fast unit tests
uv run -q pytest

# Run long-context tests (needle & counters)
uv run -q pytest -m long

# Run specific test categories
uv run -q pytest -k decode_counters  # Test decode memory counters
uv run -q pytest -k equiv_small      # Test equivalence with full attention
uv run -q pytest -k group_consistency # Test GQA group consistency
```

### Benchmarking
```bash
# Benchmark prefill performance
uv run python bench/bench_prefill.py --config configs/base.yaml

# Benchmark decode performance and token-reads
uv run python bench/bench_decode.py --config configs/base.yaml
```

### Demo
```bash
# Run demo inference with visualization of selected blocks
uv run python cli/demo_infer.py --config configs/base.yaml
```

## Architecture & Code Structure

### Core Components

1. **NSAAttention Module** (`nsa/core/nsa_attention.py`)
   - Main module implementing three-branch architecture
   - Shared Q projection with RoPE, per-branch K/V projections
   - Gate MLP for branch combination via softmax
   - Supports both prefill and decode modes

2. **Branch Implementations**
   - **Compressed** (`compress_pool.py`): Overlapping blocks with learnable ϕ operator
     - M0: Average pooling
     - M2+: Conv1d + MLP
   - **Selected** (`selection_scorer.py`): Blockwise selection via compressed scores
     - Implements Equations 8-12 from paper
     - GQA group-consistent selection
   - **Sliding**: Last w tokens with separate K/V projections

3. **Block Index Management** (`block_index.py`)
   - Compression blocks: overlapping, size l, stride d
   - Selection blocks: non-overlapping, size l'
   - CSR mapping matrix M for Eq. 9

4. **KV Cache** (`cache/kv_cache.py`)
   - Per-branch caches: K_sel/V_sel, K_win/V_win, K_cmp/V_cmp
   - Rolling window for sliding, compressed stream emission

### Key Design Decisions

1. **GQA Group Consistency**: All heads in a group share selected blocks (mandatory for decode efficiency)
2. **Gate Normalization**: Using softmax instead of sigmoid for stability
3. **No Auxiliary Losses**: End-to-end trainable via compressed score reuse
4. **CPU Fallback**: SDPA gather for selection when Triton unavailable

### Development Milestones

- **M0 (Current)**: Steel thread with SDPA everywhere, average pooling ϕ
- **M1**: FlashAttention-2 for compressed/sliding branches
- **M2**: Learnable ϕ (Conv1d+MLP) and trainable gates
- **M3**: Full decode caching
- **M4**: Triton selection kernel (forward)
- **M5**: Triton backward pass
- **M6**: Performance optimization and robustness

## Important Constraints

1. **Default Hyperparameters** (from paper Section 4.1):
   - Blocks: l=32, d=16, l'=64, n=16 (including forced initial + 2 local), w=512
   - GQA: G=4 groups, H=64 total heads, d_k=192, d_v=128

2. **Divisibility Requirements**: 
   - Must enforce d|l and d|l' for correct block mapping

3. **Causality**: 
   - All branches must strictly respect causal masking (no future tokens)

4. **Decode Memory Formula**:
   - Tokens per step: ((S-l)/d) + n*l' + w
   - Must match Table 4 expectations

## Testing Priorities

1. **Correctness First**: 
   - Causal masking invariants
   - Block math (Eq. 9 mapping)
   - GQA consistency (Eq. 10)
   - Small-S equivalence with full attention

2. **Long Context**:
   - Decode counter verification
   - 64k needle-in-haystack retrieval (target: 100%)

3. **Trainability** (M5+):
   - Gradcheck on small dimensions
   - Loss convergence on toy tasks

## Common Issues & Solutions

1. **Selection Failures**: Check Eq. 9 mapping matrix M, ensure forced blocks included
2. **Gate Collapse**: Monitor gate histograms, adjust temperature if needed
3. **Decode Counter Mismatch**: Verify compressed emission schedule (every d steps after l warmup)
4. **Future Leakage**: Check all range clamping to ≤ t

## Paper References

Key figures and equations to reference during implementation:
- Figure 2 (p.3): Three-branch architecture
- Equations 7-12 (pp.6-8): Core algorithms
- Figure 3 (p.9): Kernel execution model
- Table 4 (p.14): Decode memory economics
- Figure 5 (p.12): 64k needle retrieval target