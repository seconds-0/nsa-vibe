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

# Optional: FlashAttention 2 (enables selection varlen FA-2 fast path)
pip install flash-attn --no-build-isolation || true

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

This validates kernels and the data pipeline before a long run.

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
export NSA_USE_FA2=1; export NSA_USE_FA2_WIN=1; export NSA_USE_FA2_CMP=1

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

- Throughput (toks/s): depends on FA-2 availability. Reasonable brackets for 125M @ S=2048, batch=1:
  - Conservative: 300–500 toks/s
  - With FA-2/dataloader solid: 500–800 toks/s
- Fetch times: aim for `fetch_p95 < 80–100 ms`; if higher, increase `NSA_FWE_DOC_BATCH=128` and/or `NSA_FWE_Q=8`.
- Memory: 80GB is sufficient at batch=1; if near OOM, unset `NSA_SDPA_AUDIT` and lower `NSA_ACCUM` to 2.
- Stability: no rising fallback counters; gate stats not collapsed (entropy_mean > 0.5 early on).

## 7) Common Toggles

- IO faster: `export NSA_FWE_DOC_BATCH=128; export NSA_FWE_Q=8`
- Disable prefetch: `export NSA_FWE_PREFETCH=0`
- Force selection v1: `export NSA_SEL_RANGES_V2=0` (debug only)
- Disable FA-2: `export NSA_USE_FA2=0; export NSA_USE_FA2_WIN=0; export NSA_USE_FA2_CMP=0`
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
