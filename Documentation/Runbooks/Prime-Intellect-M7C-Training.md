# Prime Intellect — M7C Training Runbook (Hands‑Off)

This document is self‑contained. Follow it exactly to bootstrap, run, monitor, validate, and hand off artifacts for the M7C (~125M) training pilot on Prime Intellect (single node, 1–2 GPUs).

## Branches
- Base (stable): `master`
- Active PR: `test-plan/m7-training-readiness` (PR #13)
- NSA long‑context/dev ref: `m7c-64k-demo` (do not use for training; for demos only)

Use `master` once PR #13 merges; until then, use `test-plan/m7-training-readiness`.

## Deliverables (Outputs)
All outputs land under `artifacts/`:
- `artifacts/train_runs/m7c_<timestamp>/` — per‑run logs
  - `run.info` — selected config, GPU count, VRAM, run dir
  - `smoke.log` — 50‑step synthetic smoke test output
  - `train.log` — full streaming training logs
- `artifacts/m7c_125m/` — canonical training outputs
  - `training.csv` — `step,loss,lr,toks_per_s`
  - `val.csv` — optional eval `step,loss,ppl`
  - `metrics.json` — metadata dump
  - `checkpoint_step{N}.pt` — rank‑0 checkpoints (resume from latest)

Optional diagnostics (when running CI or runner):
- `artifacts/test-reports/` — routing/long‑context logs
- `artifacts/bench/` — decode bench + summary

## One‑Time Setup (SSH)
1) Open a resilient shell:
- `tmux new -s m7c` (recommended; use screen if preferred)

2) Clone repo and checkout branch:
- `git clone <repo> && cd nsa-vibe`
- `git checkout test-plan/m7-training-readiness` (switch to `master` after PR #13 merges)

3) Bootstrap GPU environment:
- `bash scripts/prime_bootstrap.sh`
  - Creates `.venv`, installs CUDA‑12.1/Torch‑2.4 GPU deps
  - Installs `transformers` + `datasets`
  - Prints GPU model(s) and VRAM

4) Dataset pre‑flight (sanity):
- `PYTHONPATH=. python scripts/datasets/check_fwe_stream.py`
  - Expect a few `ok batch[...] lens: [...]` lines; if it errors, verify network and `HF_DATASETS_CACHE`

## Start Training (Single Command)
- `bash scripts/train_m7c_prime.sh`

What it does automatically:
- Sets robust NCCL env for SSH single‑node: `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 TORCH_NCCL_BLOCKING_WAIT=1 NCCL_DEBUG=WARN`
- Detects max GPU VRAM and selects config:
  - `configs/m7c_125m_80g.yaml` (≥80 GB)
  - `configs/m7c_125m_40g.yaml` (≥35 GB)
  - `configs/m7c_125m_24g.yaml` (<35 GB)
- Detects GPU count and launches:
  - 2 GPUs → `torchrun --nproc_per_node=2`
  - 1 GPU → `python`
- Smoke check: runs `configs/train_showcase.yaml` for ~50 steps (synthetic) to validate env/logging
- Real training: streams FineWeb‑Edu with GPT‑2 BPE
- Auto‑resume: picks the latest `artifacts/m7c_125m/checkpoint_step*.pt` if present
- Logs and run info to `artifacts/train_runs/m7c_<timestamp>/`

### Live Loss Graph (TensorBoard)
- Start on Prime Intellect:
  - `bash scripts/run_tensorboard.sh` (defaults to `artifacts/m7c_125m/tb`, port 6006)
- From your laptop, SSH port‑forward:
  - `ssh -L 6006:localhost:6006 <user>@<prime-host>`
- Open: http://localhost:6006 and select the run directory

## Health Checks (Operator)
- Training CSV tails:
  - `tail -f artifacts/m7c_125m/training.csv`
  - Expect loss trending down; `toks_per_s` stable within a run
- GPU monitoring:
  - `watch -n2 nvidia-smi` (VRAM and utilization)
- Checkpoint cadence:
  - Files appear in `artifacts/m7c_125m/` at `save_every` steps
- Eval (optional in config):
  - `val.csv` with loss/PPL every `eval_every` steps

## Success Criteria (Pilot)
- Trainer runs for at least 2–4 hours without OOM/hangs
- `training.csv` shows decreasing loss; throughput reported (`toks_per_s` non‑zero and stable)
- At least one checkpoint saved; resume from it succeeds

## Resume Procedure
- The runner auto‑resumes. To resume explicitly:
- `CONFIG=<same YAML> PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu --resume artifacts/m7c_125m/checkpoint_step{N}.pt`

## Troubleshooting
- SSH disconnections: always run inside `tmux`; reattach via `tmux attach -t m7c`
- NCCL hangs: already mitigated via env; if needed, also set `export CUDA_LAUNCH_BLOCKING=1`
- OOM on 24 GB GPUs:
  - Use the 24G config (auto‑selected). If still OOM:
    - Decrease `train.batch_size` by 1 and increase `train.accumulate_grad_batches` to keep tokens/step consistent
- Slow or stalled data:
  - Verify internet and HF cache: `echo $HF_DATASETS_CACHE`; set a fast local path
  - As a fallback, remove `--dataset fineweb_edu` (synthetic data) to validate trainer stability
  - If tokenizer not installed, install: `. ./.venv/bin/activate && pip install transformers datasets`
 - TensorBoard not showing data:
   - Confirm `artifacts/m7c_125m/tb` has event files (written every log interval on rank 0)
   - Ensure you started TB in the same working directory and forwarded the port

## Exact Files and Commands (for Auditing)
- Bootstrap: `scripts/prime_bootstrap.sh`
- Training runner: `scripts/train_m7c_prime.sh`
- VRAM‑tuned configs:
  - `configs/m7c_125m_24g.yaml`
  - `configs/m7c_125m_40g.yaml`
  - `configs/m7c_125m_80g.yaml`

## Expected Throughput (Rule‑of‑Thumb)
- 2× RTX 4090 (24 GB): low tens of k toks/s (bf16, seq_len=4096, batch=2 global, accum=8)
- 2× A100‑40G: higher; reduce accumulation (config auto‑select does this)
- 2× A100‑80G: significantly higher; fewer accum steps

## Handoff Checklist
- Attach the following artifacts:
  - `artifacts/train_runs/m7c_<timestamp>/run.info`
  - `artifacts/train_runs/m7c_<timestamp>/train.log`
  - `artifacts/m7c_125m/training.csv`, latest `checkpoint_step*.pt`
- Provide GPU model, VRAM, and `torchrun` launch details (captured in `run.info`)
- Confirm Success Criteria met or note any deviations with timestamps from logs

## Optional: Long‑Context Demo (Post‑Pilot)
- Inference only (not training). Run:
  - `PYTHONPATH=. python scripts/demo_64k.py --S 65536 --prefill_tile 4096 --rope_scale 8.0 --use_fa2 1`
- Record timing and `reads_total` for sanity; do not run during training sessions

---

This runbook is authoritative for Prime Intellect M7C pilots. Do not change configs or scripts without recording the change in `run.info` and including the altered file(s) in your handoff.
