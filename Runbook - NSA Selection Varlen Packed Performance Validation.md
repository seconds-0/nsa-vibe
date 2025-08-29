# NSA Selection Varlen Packed Performance Runbook (Exact, Reproducible)

This runbook pins the branch, commit capture, environment, configs, and commands to reproduce the selection varlen packed performance tests on A100 GPUs.

## 0) Repo State and Branch

- Working repository: this repo (nsa-vibe)
- Base branch observed: `feat/nsa-training-breakthrough-stable-a100`
- You will create and commit a new feature branch to pin the exact code state used for tests.

Commands:

```bash
# Ensure clean working tree or commit/stash local edits before proceeding
git status

# Create feature branch for this test
git checkout -b feat/nsa-selection-varlen-packing

# Stage the relevant changes (selection varlen, adaptive v2, FA probes, report)
git add \
  nsa/core/attention_kernels.py \
  nsa/core/selection_scorer.py \
  nsa/kernels/flash_wrappers.py \
  "Documentation/Reports/2025-08-27 Core Engineer Report - NSA Selection Varlen Packing v1.md"

# Commit with clear message
git commit -m "M0: selection varlen packing + adaptive v2 + FA probes; runbook perf test"

# Capture the exact commit used for testing
COMMIT_SHA=$(git rev-parse --short=12 HEAD)
echo "Test commit: ${COMMIT_SHA}"
```

Record in your test report:
- Branch: `feat/nsa-selection-varlen-packing`
- Commit: `${COMMIT_SHA}` (resolved by command above)

## 1) Environment (Exact)

- GPUs: 1× or 2× NVIDIA A100 80GB (PCIe or SXM)
- Python: 3.11.x
- CUDA/Torch stack: PyTorch 2.4.x with CU121, from repo requirements
- Package manager: uv

Commands (create venv and install):

```bash
# Create and enter venv
uv venv -p 3.11 .venv
source .venv/bin/activate

# Install GPU stack from repo requirements (Torch 2.4 CU121)
uv pip sync -r requirements-gpu-cu121-torch24.txt

# Optional: install FlashAttention 2 to enable FA‑2 varlen fast path for selection
# If unavailable on the node, you can skip; the dense-batch fallback will be used.
pip install flash-attn --no-build-isolation || true

# Verify core versions
python - << 'PY'
import torch, sys
print('python:', sys.version)
print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
PY
```

## 2) Configs (Exact YAMLs for S=512/1024/2048)

Base config path: `configs/m7c_125m_2xa100_production.yaml`
- Model (confirm): dim=768, n_layers=12, n_heads=12, n_kv_groups=2, d_k=64, d_v=64
- NSA (confirm): l=32, d=16, l_sel=64, n_sel=16, w=512

Create three runbook configs with exact `train.seq_len`:

```bash
# Create copies for S=512/1024/2048
cp configs/m7c_125m_2xa100_production.yaml configs/runbook_s512.yaml
cp configs/m7c_125m_2xa100_production.yaml configs/runbook_s1024.yaml
cp configs/m7c_125m_2xa100_production.yaml configs/runbook_s2048.yaml

# Edit train.seq_len in each using a tiny Python snippet (no external tools needed)
python - << 'PY'
from omegaconf import OmegaConf
for path, S in (
  ('configs/runbook_s512.yaml', 512),
  ('configs/runbook_s1024.yaml', 1024),
  ('configs/runbook_s2048.yaml', 2048),
):
    cfg = OmegaConf.load(path)
    cfg.train.seq_len = int(S)
    # Keep batch_size=2 in the file; adjust DDP/single-GPU usage below
    OmegaConf.save(config=cfg, f=path)
    print('wrote', path, 'seq_len=', S)
PY
```

## 3) Sanity Tests (Fast)

```bash
# Quick guards and selection equivalence
uv run -q pytest -k "performance_guards or selection_v2_equiv or selection_packed"

# Core correctness smoke
uv run -q pytest -k "equiv_small or group_consistency or decode_counters or masks"
```

## 4) Environment Variables (Exact)

Use these for performance tests (selection varlen path, adaptive v2, FA where available):

```bash
# Batched prefill and selection packed (varlen)
export NSA_PREFILL_BATCHED=1
export NSA_USE_SEL_PACK=1
export NSA_FORCE_PARITY=0

# Selection ranges: adaptive default (v2 at S>=1024) unless overridden per test
unset NSA_SEL_RANGES_V2
export NSA_SEL_RANGES_V2_MIN_S=1024

# Enable FA‑2 for sliding/compressed (selection uses FA‑2 varlen if installed)
export NSA_USE_FA2=1
export NSA_USE_FA2_WIN=1
export NSA_USE_FA2_CMP=1

# Optional: short probe SDPA logs; disable for long runs
export TORCH_LOGS=+sdp
```

To force specific range mode per test:
- Force v1: `export NSA_SEL_RANGES_V2=0`
- Force v2: `export NSA_SEL_RANGES_V2=1`
- Adaptive (default): `unset NSA_SEL_RANGES_V2; export NSA_SEL_RANGES_V2_MIN_S=1024`

## 5) Single‑GPU Perf Runs (Exact Commands)

Short 200‑step runs at S∈{512,1024,2048} under three range modes.

Common pieces:
- Script: `scripts/train_showcase.py`
- Dataset: `fineweb_edu` (streamed)
- Steps: `--steps 200`
- DDP: `--ddp 0`

Commands:

```bash
# S=512, adaptive v2 (defaults to v1 at this S due to threshold)
unset NSA_SEL_RANGES_V2
CONFIG=configs/runbook_s512.yaml PYTHONPATH=. uv run -q python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s512_adaptive.log

# S=512, force v1
export NSA_SEL_RANGES_V2=0
CONFIG=configs/runbook_s512.yaml PYTHONPATH=. uv run -q python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s512_v1.log

# S=512, force v2
export NSA_SEL_RANGES_V2=1
CONFIG=configs/runbook_s512.yaml PYTHONPATH=. uv run -q python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s512_v2.log

# S=1024, adaptive (will pick v2)
unset NSA_SEL_RANGES_V2
CONFIG=configs/runbook_s1024.yaml PYTHONPATH=. uv run -q python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s1024_adaptive.log

# S=1024, force v1
export NSA_SEL_RANGES_V2=0
CONFIG=configs/runbook_s1024.yaml PYTHONPATH=. uv run -q python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s1024_v1.log

# S=1024, force v2
export NSA_SEL_RANGES_V2=1
CONFIG=configs/runbook_s1024.yaml PYTHONPATH=. uv run -q python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s1024_v2.log

# S=2048, adaptive (v2)
unset NSA_SEL_RANGES_V2
CONFIG=configs/runbook_s2048.yaml PYTHONPATH=. uv run -q python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s2048_adaptive.log

# S=2048, force v1
export NSA_SEL_RANGES_V2=0
CONFIG=configs/runbook_s2048.yaml PYTHONPATH=. uv run -q python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s2048_v1.log

# S=2048, force v2
export NSA_SEL_RANGES_V2=1
CONFIG=configs/runbook_s2048.yaml PYTHONPATH=. uv run -q python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 200 | tee run_s2048_v2.log
```

Notes:
- If FA‑2 is not installed, selection still benefits from varlen packed dense‑batch fallback.
- For clean timing, you may temporarily drop `TORCH_LOGS=+sdp` after the first run.

## 6) 2×A100 DDP Short Probe (Exact)

- Use batch_size=2 in config; if you see OOM at S=2048, switch to batch_size=1.

```bash
# Adaptive v2, S=2048 (adjust batch_size in configs/runbook_s2048.yaml if needed)
unset NSA_SEL_RANGES_V2
CONFIG=configs/runbook_s2048.yaml \
  torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu --ddp 1 --steps 200 | tee run_ddp2_s2048_adaptive.log

# If initial DDP instability: enable ddp-safe (conservative settings)
DDP_SAFE=1 CONFIG=configs/runbook_s2048.yaml \
  torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu --ddp 1 --steps 200 | tee run_ddp2_s2048_safe.log
```

## 7) Metrics to Record (Exact)

- From stdout/logs for each run:
  - Average step wall time and tok/s (compute as total tokens per step / step time)
  - Whether FA‑2 varlen engaged (based on `flash-attn` import success)
- Artifacts:
  - `artifacts/train_showcase/heartbeat_rank*.jsonl` (mem, steps)
  - Save `run_*.log` alongside a CSV summary under `artifacts/` (optional)
- GPU utilization snapshot during steady state (optional):

```bash
nvidia-smi dmon -s pucmq -o DT -d 1 -f artifacts/nv_s2048_adaptive_dmon.csv -i 0 -c 120
```

## 8) Report Template (Exact)

Save as:
`Documentation/Reports/<yyyy-mm-dd> Test Engineer Report - NSA Selection Varlen Packed Perf v2.md`

Include:
- Branch: `feat/nsa-selection-varlen-packing`
- Commit: `${COMMIT_SHA}`
- Env: Python/Torch/CUDA, GPU type/count
- Config path and key fields (train.seq_len, batch_size)
- Table: S×(adaptive/v1/v2) with avg step time and tok/s
- Whether FA‑2 varlen path was used
- Memory: peak `gpu_mem_reserved` per run
- Links to artifacts logs/CSVs

## 9) Production 50K Launch (Optional)

- Recommended (single GPU per worker):

```yaml
# configs/runbook_prod_2048.yaml
# copy from configs/runbook_s2048.yaml and set:
train:
  seq_len: 2048
  batch_size: 1      # per GPU if 80GB; increase only if safe
  steps: 50000
  accumulate_grad_batches: 4   # effective batch 4
```

```bash
unset NSA_SEL_RANGES_V2
export NSA_SEL_RANGES_V2_MIN_S=1024
export NSA_PREFILL_BATCHED=1
export NSA_USE_SEL_PACK=1
export NSA_USE_FA2=1
export NSA_USE_FA2_WIN=1
export NSA_USE_FA2_CMP=1

CONFIG=configs/runbook_prod_2048.yaml PYTHONPATH=. uv run -q python -u scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 0 --steps 50000 | tee run_prod50k_single.log
```

---

This runbook is designed to be copy‑paste reproducible. If any step diverges on your node (e.g., FA‑2 wheel unavailable), note the deviation in your report and proceed—the dense‑batch fallback remains valid for performance comparison.

