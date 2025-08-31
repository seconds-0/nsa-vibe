# Single A100 (80GB) Production Runbook — SDPA‑First, FA‑2 Off

This runbook provides exact, reproducible steps to run the 50k training on a single NVIDIA A100 80GB instance using SDPA‑first defaults with FA‑2 and selection‑varlen disabled (per current A100 policy). It includes env setup, config, commands, monitoring, expectations, and toggles.

## 1) Provision A100 80GB Instance

- OS: Ubuntu 22.04 (recommended)
- Driver: Recent NVIDIA driver compatible with CUDA 12.x
- Network: outbound access for dataset streaming; otherwise prepare a local dataset path

## 2) Environment Setup

```bash
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv build-essential git

cd ~ && git clone <your-remote>/nsa-vibe.git && cd nsa-vibe

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip uv
uv pip sync -r requirements-gpu-cu121-torch24.txt

# Quick version & GPU check
python - << 'PY'
import torch, sys
print('python:', sys.version)
print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
PY
```

## 3) Config (single‑GPU)

Use the provided file `configs/m7c_125m_1xa100_prod_v1.yaml` (added in this repo) — it sets BF16, gradient checkpointing, and disables FA‑2 via thresholds and boolean policy.

Validate:
```bash
python -c "import yaml; yaml.safe_load(open('configs/m7c_125m_1xa100_prod_v1.yaml')); print('config ok')"
```

## 4) Core Environment Flags

```bash
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_ALLOW_TF32=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# SDPA-first policy on A100
export NSA_USE_FA2=0
export NSA_FA2_MIN_LEN_WIN=-1
export NSA_FA2_MIN_LEN_CMP=-1
export NSA_USE_SEL_VARLEN=0
export NSA_USE_TRITON_SEL=0
export NSA_STRICT_ASSERTS=0

export CONFIG=configs/m7c_125m_1xa100_prod_v1.yaml
```

Validate environment (recommended):
```bash
python scripts/validate_run_env.py --strict
```

## 5) Data Loader Smoke (Important)

Ensure first batch arrives quickly (catches dataset / connectivity issues early):
```bash
python scripts/automation/fwe_smoke.py --seq-len 1024 --batch 1 --timeout 60 --tokenizer byte
```
Expect: `[smoke][OK] first batch in X.XXs shape=(1, 1024)`

## 6) Launch 50k Training (Single GPU)

Easiest path is the provided wrapper script:
```bash
bash scripts/run_m7c_1xa100_production.sh
```

Manual command (equivalent):
```bash
python -u scripts/train_showcase.py \
  --dataset fineweb_edu \
  --ddp 0 \
  --fwe-report-docs 1000 \
  --loader-timeout 120 \
  --synthetic-on-fail \
  2>&1 | tee training.log
```

Artifacts: `artifacts/m7c_125m_1xa100_prod/` (CSV, heartbeat, TB logs, counters)

## 7) Monitoring & Smoke Validation

- Heartbeat freshness:
  ```bash
  tail -f artifacts/m7c_125m_1xa100_prod/heartbeat_rank0.jsonl
  ```
- Quick CSV trend:
  ```bash
  watch -n 10 "tail -n 3 artifacts/m7c_125m_1xa100_prod/training.csv"
  ```
- Smoke validation on run artifacts:
  ```bash
  python scripts/run_smoke_tests.py \
    --csv artifacts/m7c_125m_1xa100_prod/training.csv \
    --heartbeat artifacts/m7c_125m_1xa100_prod/heartbeat_rank0.jsonl \
    --min-steps 200 --min-tps 10
  ```

- Watchdog (Recommended): proactively halts on stalls/collapses
  ```bash
  # Auto-started by the wrapper; to run manually:
  python scripts/_watchdog.py --dir artifacts/m7c_125m_1xa100_prod --halt 1 --interval 30 &
  ```

## 8) TensorBoard (Recommended)

```bash
tensorboard --logdir artifacts/m7c_125m_1xa100_prod/tb --port 6006 --bind_all
```
Open http://<host>:6006 and track loss/throughput.

## 9) Optional: Bench + Baseline Snapshot (Perf Guard)

Create a simple decode baseline you can compare against later (local only):
```bash
PYTHONPATH=. python scripts/bench_snapshot_baseline.py \
  --csv artifacts/decode_guard.csv \
  --out baselines/a100_decode_guard.json \
  --S_list 512,1024 --iters 16 --warmup 4
cat baselines/a100_decode_guard.json
```

## 10) Expectations

- Heartbeat: frequent updates; `dt_step_s` stabilizes; data fetch times mostly sub‑second.
- CSV: loss gradual decrease (not strictly monotonic); throughput stabilizes after warmup.
- Fallback counters: low/stable; no FA‑2 fallbacks (FA‑2 is off).
- Selection stats: not degenerate; diversity present in `k_stats.csv`.

## 11) Troubleshooting

- Loader stalls: re‑run `fwe_smoke.py`; keep `--synthetic-on-fail`; ensure outbound access.
- Low throughput: ensure debug is off; confirm FA‑2 is off; SDPA will be the path on A100.
- OOM (unlikely at 80GB): lower `train.batch_size` to 1 or `train.seq_len` to 1024; keep checkpointing.

## 12) Graceful Stop / Resume

- Stop:
  ```bash
  touch artifacts/m7c_125m_1xa100_prod/.HALT && echo "halt: manual_stop" > artifacts/m7c_125m_1xa100_prod/.HALT
  ```
- Resume: relaunch with `--resume <checkpoint>` (see artifacts saved every 5000 steps)

---

This runbook reflects SDPA‑first defaults on A100 with FA‑2 and selection‑varlen disabled. It is copy‑paste ready for a single A100 80GB setup.
