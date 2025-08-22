# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an implementation of Native Sparse Attention (NSA), a drop-in attention module for decoder-only Transformers with trainable, hardware-aligned sparse attention. The implementation follows the paper's architecture combining three branches (Compressed, Selected, Sliding) with learned gates.

## Prime Intellect GPU Pod Access

### SSH Connection Setup
1. **SSH Key Location**: Prime Intellect ED25519 key is at `~/.ssh/primeintellect_ed25519`
2. **Current Prime Intellect Host**: `ssh -i ~/.ssh/primeintellect_ed25519 ubuntu@216.81.248.82`
3. **Old Host Reference (legacy)**: `ssh root@47.47.180.127 -p 12181`
3. **Recommended ~/.ssh/config** (easier usage):
   ```
   Host prime-4090
     HostName 47.47.180.127
     Port 12181
     User root
     IdentityFile ~/.ssh/primeintellect_ed25519
     IdentitiesOnly yes
     ServerAliveInterval 30
     ServerAliveCountMax 6
   ```
   Then connect with: `ssh prime-4090`
4. **Pod Template**: Use "UBUNTU 22, CUDA 12" base image for cleanest environment
5. **Note**: The private key MUST be in your local SSH directory with 600 permissions for authentication to work

### Pod Setup Script
```bash
# After connecting to pod
cd /root
apt-get update && apt-get install -y git python3-pip python3-venv ninja-build
git clone https://github.com/seconds-0/nsa-vibe.git
cd nsa-vibe
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton packaging ninja
pip install flash-attn --no-build-isolation
pip install numpy hydra-core pydantic pytest hypothesis ruff mypy
```

### Quick: Run Selection Benches + Threshold
```bash
# Enable group kernels and lower min-L for bench only
NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_GROUP=1 NSA_SEL_TRITON_MIN_L=64 \
PYTHONPATH=. .venv/bin/python bench/bench_sel_triton.py \
  --N 1024 --H 8 --D 128 --Dv 128 --L_list 64,128,256,512,1024 \
  --dist few --iters 50 --warmup 5 --streams 1 --csv sel_dense.csv

NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_GROUP=1 NSA_SEL_TRITON_MIN_L=64 \
PYTHONPATH=. .venv/bin/python bench/bench_sel_triton.py \
  --N 1024 --H 8 --D 128 --Dv 128 --L_list 128,256,512,1024 \
  --dist many --iters 50 --warmup 5 --streams 2 --csv sel_varlen.csv

# Compute a recommended sel_triton_min_L at margin 1.2x and write a report
PYTHONPATH=. .venv/bin/python bench/sel_threshold_from_csv.py \
  --dense sel_dense.csv --varlen sel_varlen.csv --margin 1.2 --out selection_report.md
```

### Important: Production Guardrails (ADR-2025-08-M4-02)
- Triton selection is non‑viable on RTX 4090 (Ada, SM 8.9). The wrapper detects SM 8.9 and forces fallback to packed SDPA unless `NSA_TRITON_SEL_FORCE=1` is set for experiments.
- Default `runtime.sel_triton_min_L=4096` keeps Triton effectively off. Do not enable in production on consumer GPUs.

### Experimental: CUDA Selection (forward)
- Flag: `NSA_SEL_CUDA=1` to route selection through the CUDA wrapper (currently falls back to packed SDPA until the kernel is implemented).
- Bench: `PYTHONPATH=. uv run -q python bench/bench_sel_cuda.py --N 1024 --H 8 --D 128 --Dv 128 --L_list 128,256,512`

## Build and Test Commands

### Environment Setup
```bash
# Use uv for Python environment management
uv venv -p 3.11 .venv
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

### Milestone Smokes (CPU default)
```bash
# Quick pass over key M0–M3 tests
uv run -q python scripts/run_milestone_smoke.py

# Long-context selection smoke (optional, large S)
PYTHONPATH=. uv run -q python bench/needle_64k_smoke.py --S 65536 --device cpu
```

### Experimental CUDA Selection (GPU)
```bash
# Build and route selection via CUDA ATen implementation (still behind flags)
NSA_SEL_CUDA_BUILD=1 NSA_SEL_CUDA=1 \
PYTHONPATH=. uv run -q python bench/bench_sel_cuda.py --N 1024 --H 8 --D 128 --Dv 128 --L_list 128,256,512

# In-model decode path will also honor NSA_SEL_CUDA=1 (falls back safely otherwise)
```

### Demo
```bash
# Run demo inference with visualization of selected blocks
uv run python cli/demo_infer.py --config configs/base.yaml
```

## M7 Training Debug Runbook

These commands help diagnose training hangs and dataset streaming issues introduced in M7C.

### Dependencies
```bash
pip install -U datasets transformers  # transformers only if tokenizer=gpt2
```

### Unbuffered Training with Diagnostics
```bash
export CONFIG=configs/m7c_125m_fast_log.yaml
export PYTHONUNBUFFERED=1

# Optional: tune loader verbosity and timeout
#   --fwe-report-docs 500     prints loader progress every 500 docs
#   --loader-timeout 120      waits up to 120s for first batch
#   --synthetic-on-fail       falls back to synthetic data on loader stall

python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 \
  --fwe-report-docs 500 --loader-timeout 120 \
  2>&1 | tee training.log

# Artifacts written to: artifacts/train_showcase/
# - env.json                            environment snapshot
# - heartbeat_rank0.jsonl               JSONL with loss/toks_per_s/GPU mem
# - stackdump_*.txt                     on SIGUSR1 or SIGTERM
# - watchdog_stackdump_*.txt            on >180s heartbeat stall
```

### Trigger a Live Stack Dump
```bash
kill -USR1 <TRAINING_PID>
# Inspect artifacts/train_showcase/stackdump_*.txt
```

### Loader-Only Smoke Test
```bash
# Byte tokenizer
python scripts/automation/fwe_smoke.py --seq-len 1024 --batch 1 --timeout 60 --report-docs 500 --tokenizer byte

# GPT-2 tokenizer
python scripts/automation/fwe_smoke.py --seq-len 1024 --batch 1 --timeout 60 --report-docs 500 --tokenizer gpt2
```

### HF Streaming Sanity (direct)
```bash
python - <<'PY'
from datasets import load_dataset
s=load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)
print('ok, first text head:', next(iter(s))['text'][:80])
PY
```

### Local Fallback (offline JSONL/Text)
```bash
# Prepare /data/local.jsonl with {"text": "..."} per line or a .txt file with raw lines
python -u scripts/train_showcase.py --dataset fineweb_edu_local --local-path /data/local.jsonl --ddp 0
```

### Synthetic Sanity (rule out training loop issues)
```bash
python -u scripts/train_showcase.py --dataset synthetic --ddp 0
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
