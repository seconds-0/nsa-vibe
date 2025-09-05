# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Reports & Naming Rules (MUST FOLLOW)

- Role: Claude is the Test Engineer. All Test Engineer authored reports MUST:
  - Live under `Documentation/Reports/`
  - Be named: `<yyyy-mm-dd> Test Engineer Report - <Subject> <vX>.md`
  - Example: `2025-08-26 Test Engineer Report - DDP One-Step Trace v1.md`
- Content: Summarize objectives, environment, exact commands, results, evidence paths, and a clear go/no‑go.
- Scope: Prefer one primary report per subject per day; increment `vX` for iterations.
- Anti-pattern: Do not drop ad‑hoc `.md` reports in the repo root; do not create multiple similarly named files across directories.

## Project Overview

This is an implementation of Native Sparse Attention (NSA), a drop-in attention module for decoder-only Transformers with trainable, hardware-aligned sparse attention. The implementation follows the paper's architecture combining three branches (Compressed, Selected, Sliding) with learned gates.

## M8 Roadmap

See Documentation/Plans/M8/Roadmap.md for the stabilization plan, execution order, and links to subsystem subplans.

## M8 Data Pipeline Migration

The training system now uses the new `nsa.data_pipeline` module for robust data handling:

### Streaming FineWeb-Edu (New Default)
```bash
# Basic streaming with timeout and progress reporting
python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 0 \
  --fwe-report-docs 500 --loader-timeout 120 --synthetic-on-fail \
  2>&1 | tee training.log

# Multi-GPU with deterministic sharding
CONFIG=configs/train_showcase.yaml python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 1 --fwe-report-docs 1000
```

### Local Data Fallback
```bash
# Local JSONL or text file
python -u scripts/train_showcase.py --dataset fineweb_edu_local \
  --local-path /path/to/data.jsonl --ddp 0

# Works with both byte and GPT-2 tokenizers
NSA_TOKENIZER=gpt2 python -u scripts/train_showcase.py \
  --dataset fineweb_edu_local --local-path /path/to/data.txt
```

### Monitoring and Watchdog
```bash
# Start watchdog for anomaly detection
python scripts/_watchdog.py --dir artifacts/train_showcase --halt 1

# Watchdog creates .HALT file to gracefully stop training
# Trainer polls for .HALT every step and exits cleanly
```

### Key Features
- **Deterministic sharding**: `Shard(mod=world_size, rem=rank)` for multi-GPU
- **Graceful fallbacks**: Synthetic data if streaming fails with `--synthetic-on-fail`
- **Rich telemetry**: Heartbeat JSONL includes `dt_fetch_s` for data stall diagnosis
- **Environment validation**: Automatic PyTorch/CUDA/device capability checks
- **Safe tokenization**: S-1 truncation preserves sequence length contracts

## Prime Intellect GPU Training Setup

**All GPU training is conducted through Prime Intellect cloud GPUs.** Claude has SSH access to Prime Intellect instances for executing training runs, performance tests, and production validation.

### SSH Access Configuration
- **SSH Keys**: Available in the repository (`.ssh/primeintellect_ed25519`)
- **Host Access**: Ask user for current SSH address when needed
- **Environment**: Ubuntu 22.04, CUDA 12.x, 2×A100 80GB typical configuration
- **Access Method**: Direct SSH with provided keys

### GPU Training Commands
```bash
# Ask user for SSH address, then connect directly
ssh -i ~/.ssh/primeintellect_ed25519 ubuntu@<provided-address>

# Standard training commands on remote GPU
cd nsa-vibe && source .venv/bin/activate
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu

# FSDP training (post-DDP compatibility fix)
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase_fsdp.py --dataset fineweb_edu
```

## Training Config Hygiene (Critical)

Always pin and verify the training config before launching multi‑hour GPU runs.

### CRITICAL PERFORMANCE FLAGS (MUST SET)
```bash
# MANDATORY: Without this, training runs at <10 tok/s instead of 9,200+ tok/s
export NSA_FORCE_SEL_MASK=1  # Forces fast masked attention path (1000x speedup)

# IMPORTANT: Gradient checkpointing MUST be OFF
# With gradient_checkpointing: true -> <10 tok/s
# With gradient_checkpointing: false -> 1000+ tok/s
```

### Common Performance Issues and Fixes
1. **Training at <10 tok/s**: Set `NSA_FORCE_SEL_MASK=1` (fixes slow _sdpa_over_ranges path)
2. **Training at 10-50 tok/s**: Disable gradient checkpointing in config
3. **FA-2 slowdown**: FA-2 is 2.4-3.4x SLOWER than SDPA. Keep `NSA_USE_FA2=0` (default)

### Production Configuration
- Use production profiles on A100/H100: `configs/m7c_125m_2xa100_production.yaml`.
- Required settings for long runs:
  - `runtime.precision: bf16`
  - `runtime.gradient_checkpointing: false`  # MUST be false
  - `runtime.use_flash: true`  # SDPA flash, not FA-2
  - `train.save_every: >= 1000` (recommend 5000)
  - Distinct `train.out_dir` per run

- Preflight checklist (must be green before launch):
  - Printed `CONFIG=...` path matches intended file
  - Precision shows bf16 in logs; gradient checkpointing "off"
  - `NSA_FORCE_SEL_MASK=1` is set
  - `save_every` > 0; `seq_len`, `batch_size`, `steps` match the plan
  - `out_dir` is empty or a new folder

One‑liner example:
```bash
CONFIG=configs/m7c_125m_2xa100_production.yaml \
NSA_FORCE_SEL_MASK=1 NSA_PREFILL_BATCHED=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True \
NCCL_P2P_DISABLE=0 IB_DISABLE=0 \
python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 1
```

## Memory Units Guidance

- Heartbeat fields are recorded in MiB (base‑2 mebibytes):
  - `gpu_mem_alloc = memory_allocated() // (1024*1024)`
  - `gpu_mem_reserved = memory_reserved() // (1024*1024)`
- When reporting in docs/dashboards, label units explicitly. Optionally add derived GiB (`MiB/1024`) for readability.
- `nvidia-smi` snapshots may capture inter‑step lows; reconcile by sampling over several seconds during steady compute.

### Monitoring and Diagnostics
```bash
# Real-time monitoring on Prime Intellect GPUs
nvidia-smi dmon -s mu -d 1  # GPU utilization
tail -f artifacts/*/heartbeat_rank0.jsonl  # Training telemetry
```

### Environment Setup
- **Python Environment**: Pre-configured venv with PyTorch CUDA
- **Dependencies**: All NSA requirements installed
- **Data Pipeline**: FineWeb-Edu streaming and local data support
- **Diagnostics**: Full telemetry and artifact collection enabled

### Automated Pod Bootstrap
The `make train-prime` command automatically runs this setup on the remote host:
```bash
# Creates parameterized setup script and executes it
# - Installs system dependencies (git, python3-venv, ninja-build, tmux)  
# - Clones repo and switches to correct branch
# - Creates Python venv with GPU dependencies
# - Validates environment and data loader
# - Starts training in tmux session 'nsa-training'
```

### Manual Pod Setup (if needed)
```bash
# Connect to pod
ssh $REMOTE_HOST

# Run bootstrap script
cd nsa-vibe
bash scripts/prime_bootstrap.sh

# Validate environment
python scripts/_env_guard.py
```

### Training Monitoring Commands

#### Real-Time Status Checking
```bash
# Quick status check (GPU, training process, tmux session)
make status-prime

# Comprehensive training monitor with alerts
bash scripts/monitor_training.sh

# Live training logs
make logs-prime

# TensorBoard tunnel (auto-opens browser)
make monitor-prime
```

#### Heartbeat and Telemetry Monitoring
```bash
# Watch live heartbeat telemetry (loss, throughput, memory, gate health)
ssh $REMOTE_HOST 'cd nsa-vibe && tail -f artifacts/train_showcase/heartbeat_rank0.jsonl'

# Parse specific metrics from heartbeat
ssh $REMOTE_HOST 'cd nsa-vibe && grep "progress" artifacts/train_showcase/heartbeat_rank0.jsonl | tail -10 | jq ".loss, .toks_per_s"'

# Check for data loader stalls
ssh $REMOTE_HOST 'cd nsa-vibe && grep "dt_fetch_s" artifacts/train_showcase/heartbeat_rank0.jsonl | tail -20'
```

#### Watchdog and Anomaly Detection
```bash
# Start watchdog (monitors heartbeat, creates .HALT on stall)
ssh $REMOTE_HOST 'cd nsa-vibe && python scripts/_watchdog.py --dir artifacts/train_showcase --halt 1 &'

# Manual halt (graceful training stop)
ssh $REMOTE_HOST 'cd nsa-vibe && touch artifacts/train_showcase/.HALT'

# Check watchdog status
ssh $REMOTE_HOST 'cd nsa-vibe && ps aux | grep _watchdog'
```

### Emergency Procedures and Troubleshooting

## FA-2 Policy (2025-09-04)

**Status: FA-2 is deprecated as the primary attention backend.**

Comprehensive testing shows FA-2 underperforms SDPA by 2.4-3.4× across all configurations on H100 and A100.
See `Documentation/Decisions/ADR-2025-09-04-FA2-Deprecation.md` for full details.

### Production Configuration (Required)
```bash
# SDPA flash backend is default - no env vars needed
# Or explicitly disable FA-2:
export NSA_USE_FA2=0
export NSA_USE_FA2_WIN=0
export NSA_USE_FA2_CMP=0
```

### Experimental FA-2 Testing (Not Recommended)
If you need to test FA-2 for research purposes:
```bash
# Enable FA-2 (expect 2-3× performance degradation)
export NSA_USE_FA2=1
export NSA_USE_FA2_WIN=1
export NSA_USE_FA2_CMP=1
# Use high thresholds to minimize activation
export NSA_FA2_MIN_LEN_WIN=8192
export NSA_FA2_MIN_LEN_CMP=8192
# Required for A100/SM80 stability
export NSA_FA2_ALLOW_D_GT_128=0
```

### Performance Analysis Tools
- CSV sweep: `python bench/bench_fa2.py --csv-sweep`
- Knee point analysis: `python scripts/fa2_knee_analyzer.py`
- Test artifacts: `artifacts/fa2_vs_sdpa_*.csv`

# Single A100 (80GB) Production Runbook — NSA Selection Varlen Packed

This runbook provides exact, reproducible steps to run the 50k training on a single NVIDIA A100 80GB instance, with the optimized selection varlen packing and improved data pipeline. It includes branch/commit capture, environment setup, configs, commands, monitoring, expectations, and toggles.

## 0) Prepare Code (on your dev machine)

- Create a feature branch and commit the latest changes:

```bash
cd /path/to/nsa-vibe
git checkout -b feat/single-a100-prod
# Stage updated files (selection packing, data pipeline, FA probes, runbooks)
git add \
  nsa/core/attention_kernels.py \
  nsa/core/selection_scorer.py \
  nsa/kernels/flash_wrappers.py \
  scripts/train_showcase.py \
  nsa/data_pipeline.py \
  launch_production_50k_single_gpu.sh \
  Documentation/Runbooks/Single-A100-80GB-Production.md \
  a.md \
  "Documentation/Reports/2025-08-27 Core Engineer Report - NSA Selection Varlen Packing v1.md"

git commit -m "Single A100 prod: selection varlen packing + data loader prefetch + runbook"
COMMIT_SHA=$(git rev-parse --short=12 HEAD); echo "COMMIT=$COMMIT_SHA"

git push origin feat/single-a100-prod
```

Record for the report:
- Branch: `feat/single-a100-prod`
- Commit: `${COMMIT_SHA}`

If you prefer not to push, create a tarball and upload:

```bash
git archive --format=tar.gz --output nsa-vibe_${COMMIT_SHA}.tar.gz HEAD
# scp or upload the tarball to the new instance
```

## 1) Provision A100 80GB Instance

- OS: Ubuntu 22.04 (recommended) or equivalent
- Driver/CUDA: Standard NVIDIA driver (matching PyTorch CU121 wheels); CUDA toolkit not required
- Ensure outbound network access (for Hugging Face streaming) or have a local dataset path

## 2) Environment Setup (on the new instance)

```bash
# System prep
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv build-essential git

# Clone repo and checkout branch/commit
cd ~
git clone https://<your_git_remote>/nsa-vibe.git
cd nsa-vibe
git fetch origin feat/single-a100-prod
git checkout feat/single-a100-prod
# Optionally pin to exact commit
# git checkout <COMMIT_SHA>

# Create venv and install deps (Torch 2.4 CU121)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip uv
uv pip sync -r requirements-gpu-cu121-torch24.txt

# Required: FlashAttention 2 (enables selection varlen FA‑2 fast path)
pip install -U pip wheel ninja packaging
pip install --no-cache-dir --no-build-isolation flash-attn

# Probe FA‑2 (must print 'FA2 varlen OK')
python - << 'PY'
try:
    from flash_attn import flash_attn_varlen_func
    print('FA2 varlen OK')
except Exception as e:
    raise SystemExit(f'FA2 missing: {e}')
PY

# Quick version check
python - << 'PY'
import torch, sys
print('python:', sys.version)
print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
PY
```

## 3) Configs (exact)

We will use `configs/m7c_125m_2xa100_production.yaml` with runtime overrides:
- `train.seq_len = 2048`
- `train.batch_size = 2` (file value; we override to 1 at runtime)
- `train.steps = 50000`

No file edits are required; runtime env sets `NSA_BATCH_SIZE=1` and `NSA_ACCUM=4`.

## 4) Single‑GPU Smoke (200 steps)

This validates kernels and the data pipeline before a long run. FA‑2 is not required; SDPA flash is the default.

```bash
source .venv/bin/activate
export CONFIG=configs/m7c_125m_2xa100_production.yaml

# Training profile
export NSA_BATCH_SIZE=1
export NSA_ACCUM=4

# NSA fast paths
export NSA_PREFILL_BATCHED=1
export NSA_USE_SEL_PACK=1
export NSA_FORCE_PARITY=0
export NSA_SEL_RANGES_V2_MIN_S=1024
export NSA_SEL_RANGES_V2=1
# FA‑2 is OFF by default (SDPA is faster). Only enable for experiments:
# export NSA_USE_FA2=1; export NSA_USE_FA2_WIN=1; export NSA_USE_FA2_CMP=1
# export NSA_FA2_MIN_LEN_WIN=8192; export NSA_FA2_MIN_LEN_CMP=8192

# Data loader perf
export NSA_FWE_DOC_BATCH=64
export NSA_FWE_PREFETCH=1
export NSA_FWE_Q=4

# Debug & allocator
export NSA_SDPA_AUDIT=1
export NSA_DEBUG_TIMING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export PYTHONUNBUFFERED=1

# Run 200 steps
PYTHONPATH=. python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 0 --steps 200 | tee artifacts/smoke_single_a100.log

# Acceptance gate: require >= 300 toks/s average (rough check)
awk '/toks\/s/{sum+=$(NF); n++} END{if(n>0){avg=sum/n; printf("avg_toks_per_s=%.1f\n",avg); exit (avg>=300)?0:1} else {exit 1}}' artifacts/smoke_single_a100.log
```

Expected in logs:
- Regular step lines: `toks/s XXX | fetch YYms | fetch_p50 ZZms | fetch_p95 WWms`
- Heartbeat files under `artifacts/train_showcase/heartbeat_rank0.jsonl`
- No OOM; `fetch_p95` typically < 50–100 ms; fallback counters near zero

## 5) 50k Production Run (single GPU)

Use the provided launcher script (already configured with recommended env):

```bash
source .venv/bin/activate
bash ./launch_production_50k_single_gpu.sh
```

It launches:
- Batch size 1, grad accumulation 4
- Batched tokenization + prefetch (queue size 4)
- Selection varlen packing with v2 ranges and FA-2 enabled (if installed)
- Debug timing and a one-time SDPA audit

Artifacts:
- `artifacts/m7c_125m_2xa100_prod/production_single_gpu.log`
- `artifacts/train_showcase/*` (heartbeats, CSVs, mem dumps)

## 6) Monitoring & Expectations

- Throughput (toks/s): depend on hardware and config; SDPA flash is typical baseline. Use threshold bands from production runbook for acceptance.
- Fetch times: aim for `fetch_p95 < 80–100 ms`; if higher, increase `NSA_FWE_DOC_BATCH=128` and/or `NSA_FWE_Q=8`.
- Memory: 80GB is sufficient at batch=1; if near OOM, unset `NSA_SDPA_AUDIT` and lower `NSA_ACCUM=2`.
- Stability: no rising fallback counters; gate stats not collapsed (entropy_mean > 0.5 early on).

## 7) Common Toggles

- IO faster: `export NSA_FWE_DOC_BATCH=128; export NSA_FWE_Q=8`
- Disable prefetch: `export NSA_FWE_PREFETCH=0`
- Force selection v1: `export NSA_SEL_RANGES_V2=0` (debug only)
- FA-2 is OFF by default (no action needed). To explicitly disable: `export NSA_USE_FA2=0; export NSA_USE_FA2_WIN=0; export NSA_USE_FA2_CMP=0`
- Byte tokenizer smoke: `export NSA_TOKENIZER=byte`

## 8) Troubleshooting

- Loader stalls early: ensure `pip install datasets` (comes via requirements), and instance has outbound network. Try `--dataset fineweb_edu_local --local-path /data/fineweb.jsonl` as a fallback.
- OOM: confirm batch=1, reduce `NSA_ACCUM` to 2, unset `NSA_SDPA_AUDIT`, keep `PYTORCH_CUDA_ALLOC_CONF` as provided.
- Low toks/s with small fetch times: compute‑bound; confirm FA-2 is installed; otherwise acceptable.

## 9) Reporting

- Save a daily report under `Documentation/Reports/`:
  - `<yyyy-mm-dd> Test Engineer Report - Single A100 50K v1.md`
  - Include Branch, Commit, GPU, Torch/CUDA, env, config path, toks/s trend, fetch p50/p95, gate/counters, and artifacts links.

---

This runbook is designed to be copy‑paste ready. Deviations (e.g., no FA‑2 wheel) are acceptable; note them in the report.

#### Stack Dumps and Process Analysis
```bash
# Trigger live stack dump (sends SIGUSR1 to training process)
TRAINING_PID=$(ssh $REMOTE_HOST 'ps aux | grep train_showcase.py | grep -v grep | awk "{print \$2}"')
ssh $REMOTE_HOST "kill -USR1 $TRAINING_PID"

# Inspect stack dump
ssh $REMOTE_HOST 'cd nsa-vibe && ls -la artifacts/train_showcase/stackdump_*.txt'
ssh $REMOTE_HOST 'cd nsa-vibe && tail -50 artifacts/train_showcase/stackdump_*.txt'

# Check for watchdog-generated dumps (>180s heartbeat stall)
ssh $REMOTE_HOST 'cd nsa-vibe && ls -la artifacts/train_showcase/watchdog_stackdump_*.txt'
```

#### Training Recovery and Resume
```bash
# Check for existing checkpoints
ssh $REMOTE_HOST 'cd nsa-vibe && ls -la artifacts/*/checkpoint_step*.pt'

# Resume training from latest checkpoint (automatic in train_m7c_prime.sh)
ssh $REMOTE_HOST 'cd nsa-vibe && bash scripts/train_m7c_prime.sh'

# Manual resume from specific checkpoint
ssh $REMOTE_HOST 'cd nsa-vibe && python scripts/train_showcase.py --dataset fineweb_edu --resume artifacts/m7c_125m/checkpoint_step1000.pt'
```

#### Data Loader Troubleshooting
```bash
# Test HuggingFace streaming directly
ssh $REMOTE_HOST 'cd nsa-vibe && python - <<EOF
from datasets import load_dataset
s=load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
print("OK, first text head:", next(iter(s))["text"][:80])
EOF'

# Loader-only smoke test
ssh $REMOTE_HOST 'cd nsa-vibe && python scripts/automation/fwe_smoke.py --seq-len 1024 --batch 1 --timeout 60 --tokenizer byte'

# Fallback to synthetic data
ssh $REMOTE_HOST 'cd nsa-vibe && python scripts/train_showcase.py --dataset synthetic --ddp 0'

# Fallback to local data
ssh $REMOTE_HOST 'cd nsa-vibe && python scripts/train_showcase.py --dataset fineweb_edu_local --local-path /path/to/data.jsonl --ddp 0'
```

#### GPU and CUDA Troubleshooting
```bash
# Check GPU status
ssh $REMOTE_HOST 'nvidia-smi'

# Detailed GPU query
ssh $REMOTE_HOST 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv'

# Test CUDA availability
ssh $REMOTE_HOST 'cd nsa-vibe && python -c "import torch; print(f\"CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}, GPU: {torch.cuda.get_device_name()}\")"'

# Check for CUDA errors in training logs
ssh $REMOTE_HOST 'cd nsa-vibe && grep -i "cuda\|RuntimeError\|NCCL" artifacts/train_runs/*/train.log'
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
# Use uv for Python environment management (CPU default)
uv venv -p 3.11 .venv
uv pip sync requirements-cpu.txt
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

### Milestone Smokes and Validation

#### Local Testing (CPU default)
```bash
# Quick pass over key M0–M8 tests
python scripts/run_milestone_smoke.py

# M8 comprehensive suite with baseline capture
python scripts/run_smoke_tests.py --run-synthetic --smoke-steps 500 --save-baseline baselines/local_cpu.json

# Compare against saved baseline (5% tolerance)
python scripts/run_smoke_tests.py --csv artifacts/train_showcase/training.csv --compare-baseline baselines/local_cpu.json

# Long-context selection smoke (optional, large S)
PYTHONPATH=. python bench/needle_64k_smoke.py --S 65536 --device cpu
```

#### Remote GPU Validation
```bash
# Run comprehensive smoke tests on remote GPU
ssh $REMOTE_HOST 'cd nsa-vibe && python scripts/run_smoke_tests.py --run-synthetic --run-fineweb --smoke-steps 1000'

# Validate existing training run
ssh $REMOTE_HOST 'cd nsa-vibe && python scripts/run_smoke_tests.py --csv artifacts/train_showcase/training.csv --heartbeat artifacts/train_showcase/heartbeat_rank0.jsonl'

# Create baseline for this GPU configuration
ssh $REMOTE_HOST 'cd nsa-vibe && python scripts/run_smoke_tests.py --csv artifacts/train_showcase/training.csv --save-baseline baselines/gpu_a100.json'

# Regression detection (compare against baseline)
ssh $REMOTE_HOST 'cd nsa-vibe && python scripts/run_smoke_tests.py --csv artifacts/train_showcase/training.csv --compare-baseline baselines/gpu_a100.json --baseline-tolerance 3.0'
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

## Complete Testing Workflow Summary

### Phase 1: Local Development Testing
```bash
# Environment setup
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Core correctness validation
python scripts/run_milestone_smoke.py
python scripts/run_smoke_tests.py --run-synthetic --smoke-steps 500

# Create local baseline
python scripts/run_smoke_tests.py --run-synthetic --smoke-steps 500 --save-baseline baselines/local_cpu.json
```

### Phase 2: Prime Intellect GPU Testing
```bash
# Setup remote host (ask user for current address)
# SSH keys available in repo: ~/.ssh/primeintellect_ed25519

# One-command launch (automated setup + training)
make train-prime

# Monitor in separate terminal
make monitor-prime  # TensorBoard at http://localhost:6006
make status-prime   # Quick status checks
```

### Phase 3: Production Monitoring
```bash
# Live telemetry monitoring
ssh $REMOTE_HOST 'cd nsa-vibe && tail -f artifacts/train_showcase/heartbeat_rank0.jsonl'

# Comprehensive status with alerts
bash scripts/monitor_training.sh

# Smoke test validation with baseline comparison
ssh $REMOTE_HOST 'cd nsa-vibe && python scripts/run_smoke_tests.py --csv artifacts/train_showcase/training.csv --compare-baseline baselines/gpu_a100.json'
```

### Emergency Response Procedures
1. **Training Stall**: Trigger stack dump with `kill -USR1 <PID>`
2. **Data Issues**: Check loader with `fwe_smoke.py`, fallback to synthetic
3. **GPU Issues**: Check `nvidia-smi`, validate CUDA availability  
4. **Graceful Halt**: Create `.HALT` file or run watchdog
5. **Resume**: Automatic checkpoint detection in `train_m7c_prime.sh`

### Key Artifacts Locations
- Training logs: `artifacts/train_runs/m7c_*/train.log`
- Heartbeat telemetry: `artifacts/train_showcase/heartbeat_rank0.jsonl`
- TensorBoard events: `artifacts/m7c_125m/tb/`
- Checkpoints: `artifacts/m7c_125m/checkpoint_step*.pt`
- Stack dumps: `artifacts/train_showcase/stackdump_*.txt`
