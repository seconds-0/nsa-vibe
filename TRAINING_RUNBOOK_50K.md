# NSA 50K Training Run — Test Engineer Handoff / Runbook

This runbook contains everything needed to validate, monitor, and wrap up the current 50k training run, plus the guardrails and commands for the next (production) run. Hand this document directly to the test engineer.

## Scope & Current Context
- Run target: 50,000 steps on 2×A100 80GB (Prime Intellect pod).
- Code branch/tag: `feat/m7c-perf-stability` tagged `m8-prefill-causality-fix`.
- Key fix included: M8 batched prefill causality strict asserts (SHA `12c6ea7c38db20b3581c243fc7c7c199b6e1750f`).
- Current run config (mistake, for awareness): `configs/train_showcase.yaml` (fp32, no grad ckpt, flash off, save_every=0). Run remains healthy but is a toy profile; finish if required, then move to production config for second run.

## Repo & Artifacts
- Repo root: `nsa-vibe`
- Validation helper: `scripts/run_gpu_causality_test.sh` (use `--full` to run extended GPU checks).
- Training script: `scripts/train_showcase.py` (saves `model.pt` at end; periodic checkpoints only if `save_every>0`).
- Default out_dir (showcase): `artifacts/train_showcase/`
  - `training.csv` (step, loss, lr, toks/s)
  - `val.csv` (optional eval, if enabled)
  - `heartbeat_rank*.jsonl` (per‑step telemetry)
  - `tb/` (TensorBoard events)
  - `metrics.json`, `env.json`
  - `checkpoint_step*.pt` (if `save_every>0`)

## Memory Units (Important)
- Heartbeat logs MiB (base‑2 mebibytes):
  - `gpu_mem_alloc = memory_allocated() // (1024*1024)` (MiB)
  - `gpu_mem_reserved = memory_reserved() // (1024*1024)` (MiB)
- When comparing with `nvidia-smi`, expect short‑term mismatches if sampled between steps. Reconcile by sampling over several seconds during steady compute.

## One‑Time Validation (GPU)
Run this once per GPU pod to confirm the environment and the M8 fix are healthy.
- Command (in repo root):
  - `. .venv/bin/activate`
  - `bash scripts/run_gpu_causality_test.sh --full`
- Expected:
  - Causality test: 3/3 passed
  - Core M0 subset: green
  - Extended full mode: selection packed parity, FA‑2 varlen parity (on A100/H100), decode/masked tiny, RoPE dtype, long‑context smoke → green

## Live Monitoring (current run)
- Status:
  - `tail -f artifacts/train_showcase/training.csv`
  - `tail -f artifacts/train_showcase/heartbeat_rank0.jsonl`
- GPU:
  - `watch -n 1 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv`
- TensorBoard (if needed):
  - `tensorboard --logdir artifacts/train_showcase/tb --port 6006`
- Health expectations:
  - Loss: smooth decline; no plateaus > 1k steps
  - Throughput: ~400 toks/s ±10% (for seq_len=128 toy run)
  - Memory (heartbeat MiB): allocated/reserved flat; no creep
  - Fallback counters: near zero; not trending upward
  - Gates: entropy_mean > 0.5; max_gate_mean < 0.9; collapse_fraction ≈ 0

## Checkpoints (current run)
- This run was launched without `save_every`; only final `model.pt` is saved on completion.
- If you need recovery points, stop this run and use the production config below with `save_every`.

## Wrap‑Up & Artifact Validation (current run)
When the 50k run completes (or you stop it intentionally):
1. Verify final artifact exists:
   - `ls -lh artifacts/train_showcase/model.pt`
2. Record final metrics:
   - `cat artifacts/train_showcase/metrics.json`
   - `tail -20 artifacts/train_showcase/training.csv`
3. Quick inference sanity (byte tokenizer):
   - See “Post‑Run Model Tests” below for a ready‑to‑run snippet.
4. Optional tests:
   - `PYTHONPATH=. pytest -q nsa/tests/test_m8_integration.py -k integration`
   - `PYTHONPATH=. pytest -q -k test_long_context_needle`

## Post‑Run Model Tests (Greedy Inference)
- Byte tokenizer:
```
PYTHONPATH=. python - <<'PY'
import torch
from omegaconf import OmegaConf
from scripts.train_showcase import TinyLM
ckpt = torch.load('artifacts/train_showcase/model.pt', map_location='cuda:0')
cfg = OmegaConf.create(ckpt['cfg'])
model = TinyLM(
  vocab=256,
  dim=int(cfg.model.dim), n_layers=int(cfg.model.get('n_layers',1)),
  n_heads=int(cfg.model.n_heads), n_kv_groups=int(cfg.model.n_kv_groups),
  d_k=int(cfg.model.d_k), d_v=int(cfg.model.d_v),
  l=int(cfg.nsa.l), d=int(cfg.nsa.d), l_sel=int(cfg.nsa.l_sel), n_sel=int(cfg.nsa.n_sel), w=int(cfg.nsa.w),
  grad_checkpointing=False,
).to('cuda').eval().to(dtype=torch.bfloat16)
model.load_state_dict(ckpt['state_dict'], strict=True)
ids = torch.tensor([[72,101,108,108,111,32,78,83,65,33,32]], dtype=torch.long, device='cuda')
for _ in range(64):
  logits = model(ids)[:, -1, :]
  next_id = torch.argmax(logits, dim=-1, keepdim=True)
  ids = torch.cat([ids, next_id], dim=1)
print(bytes(int(t) for t in ids[0].tolist()).decode('utf-8', errors='ignore'))
PY
```
- GPT‑2 tokenizer variant: encode/decode with `transformers.GPT2Tokenizer` and set `vocab=tok.vocab_size` when constructing `TinyLM`.

## Production Run (Second Run) — Start Here
Use a production profile with BF16, gradient checkpointing, and frequent checkpoints.

- Recommended config: `configs/m7c_125m_2xa100_production.yaml`
  - `runtime.precision: bf16`
  - `runtime.gradient_checkpointing: true`
  - `runtime.use_flash: true`
  - `train.seq_len: 2048` (validate), consider 3072 if reserved < 40 GB/GPU
  - `train.save_every: 5000` (or 1000 if tighter RPO)
  - Distinct `train.out_dir`
- Launch (2×A100):
```
CONFIG=configs/m7c_125m_2xa100_production.yaml \
NSA_USE_FA2=1 NCCL_P2P_DISABLE=0 IB_DISABLE=0 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True \
python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 1
```
- Success criteria (first 500 steps):
  - No DDP errors; stable throughput; reserved < 30–40 GB/GPU (@2k seq)
  - Low FA‑2 fallback counts; healthy gate entropy; no causality asserts

## Preflight Checklist (Must be GREEN before long runs)
- Config printed by trainer shows the intended file (e.g., `m7c_125m_2xa100_production.yaml`).
- Precision is BF16; gradient checkpointing “on”; flash enabled.
- `save_every` > 0; `steps`, `seq_len`, `batch_size`, `out_dir` as planned.
- Env:
  - `NSA_USE_FA2=1`
  - `NCCL_P2P_DISABLE=0` `IB_DISABLE=0`
  - `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True`
- Short SDPA routing probe (`TORCH_LOGS=+sdp`) only for the first 30–100 steps, then unset.

## Troubleshooting
- Missing checkpoints: Ensure `train.save_every > 0`; check write permissions in `out_dir`.
- Low/volatile GPU mem in `nvidia-smi`: sample during steady compute; reconcile with heartbeat (MiB) values.
- DDP + checkpointing errors: keep `accumulate_grad_batches=1`; use `no_sync` during accumulation; start here before scaling.
- Kernel routing mismatches: don’t force flash‑only globally; use short probes; selection isn’t always flash‑eligible.

## RUN / Tradeoffs (Recommended First, with context)
- Recommend: BF16 + gradient checkpointing ON.
  - Why: maximizes memory headroom with minimal numerical risk on A100; enables longer seq_len.
  - Tradeoff: ~5–15% walltime overhead from recomputation.
- Recommend: DDP with no_sync during accumulation (accumulate_grad_batches ≥ 1). Start with 1.
  - Why: avoids “mark variable ready only once” with checkpointing.
  - Tradeoff: small reduction in comm overlap during accum steps (negligible vs stability gain). Escalate to FSDP if DDP still trips at accum=1.
- Recommend: ENABLE FA‑2 for sliding/compressed (`NSA_USE_FA2=1`).
  - Why: lower memory + better throughput on A100.
  - Tradeoff: if FA‑2 not eligible (shapes/dtype), it auto‑falls back (see counters).
- Recommend: DO NOT force flash‑only SDPA for full runs.
  - Why: selection path often isn’t flash‑eligible; forcing flash can hard‑fail. Use only for a 20–50 step probe; then remove.
- Recommend: NCCL P2P/IB enabled (`NCCL_P2P_DISABLE=0`, `IB_DISABLE=0`).
  - Why: maximizes bandwidth; fixes load imbalance.
  - Tradeoff: requires network fabric; if unavailable, expect lower throughput.
- Recommend: allocator tuning `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True`.
  - Why: reduces fragmentation; smoother peaks.
  - Tradeoff: none meaningful on A100.
- Recommend: short `+sdp` logging window (first 30–100 steps) only.
  - Why: proves kernel routing (flash vs math) without noise.
  - Tradeoff: verbose logs if left on too long.
- Recommend: start DDP with `accumulate_grad_batches=1`, then scale.
  - Why: isolates DDP + ckpt stability first.
  - Tradeoff: may reduce throughput initially; scale after stable.

## After Run — Promotion & Tests
- Artifact to keep: `model.pt` or specific `checkpoint_step*.pt`.
- Tests:
  - Integration: `PYTHONPATH=. pytest -q nsa/tests/test_m8_integration.py -k integration`
  - Long‑context: `PYTHONPATH=. pytest -q -k test_long_context_needle`
- Benches:
  - Prefill: `PYTHONPATH=. NSA_USE_FA2=1 uv run python bench/bench_prefill.py --config configs/base.yaml`
  - Decode: `PYTHONPATH=. NSA_USE_FA2=1 uv run python bench/bench_decode.py --config configs/base.yaml`

---
Prepared for the test engineer. Follow this runbook verbatim for validation, monitoring, and executing the production redo with guardrails.
