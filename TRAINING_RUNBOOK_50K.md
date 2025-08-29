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
  - Throughput acceptance (production profile, seq_len=2048):
    - 2×A100 80GB PCIe Gen3: 35–40 toks/s
    - 2×A100 80GB PCIe Gen4: 45–55 toks/s
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
Use the validated batched prefill path and a conservative DDP profile (no gradient accumulation) for stability and throughput.

- Recommended config: `configs/m7c_125m_2xa100_production.yaml`
  - `runtime.precision: bf16`
  - `runtime.gradient_checkpointing: false` (initially; can revisit after a stable 500–1000 steps)
  - `runtime.use_flash: true`
  - `train.seq_len: 2048` (validate), consider 3072 if reserved < 40 GB/GPU
  - `train.batch_size: 1`
  - `train.accumulate_grad_batches: 1` (avoid accumulation with DDP initially)
  - `train.save_every: 5000` (or 1000 if tighter RPO)
  - Distinct `train.out_dir`

### One‑Command 2×A100 Launch (Preferred)
```
bash scripts/run_2xa100_production.sh
```
- Optional overrides:
  - `NP=2` to change processes per node
  - `DATASET=fineweb_edu_local` and `LOCAL_PATH=/path/to/data.jsonl` if using local data (pass via CLI separately)
  - `NSA_DDP_BUCKET_MB={25|100}` to sweep bucket sizes
  - `NSA_DDP_COMPRESS=off` to temporarily disable gradient compression during triage
  - `TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1` are already set by the launcher for fail‑fast

### Manual Launch (2×A100, PCIe‑aware)
```
CONFIG=configs/m7c_125m_2xa100_production.yaml \
NSA_PREFILL_BATCHED=1 \
NSA_SEL_RANGES_V2=1 \
NSA_DDP_COMPRESS=bf16 NSA_DDP_BUCKET_MB=25 \
NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_IB_DISABLE=1 \
NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1 \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
NSA_DDP_DISABLE_GC=1 NSA_DDP_FIND_UNUSED=0 NSA_DDP_STATIC_GRAPH=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True \
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu --ddp 1
```
- Notes:
  - Keep per‑GPU batch at 1; avoid accumulation initially (`NSA_ACCUM=1`).
  - If overlap is poor or variance high, sweep `NSA_DDP_BUCKET_MB` in {25, 100}.
  - Selection v2 is on; disable only for A/B (`NSA_SEL_RANGES_V2=0`).
  - Optional FA‑2 (A100): `NSA_USE_FA2=1 NSA_USE_FA2_WIN=1 NSA_USE_FA2_CMP=1` (auto‑fallbacks in code).
  - Optional short SDPA probe: `TORCH_LOGS=+sdp` for first ~100 steps, then unset.

### Single‑GPU Launch (Fallback)
```
CONFIG=configs/m7c_125m_2xa100_production.yaml \
NSA_PREFILL_BATCHED=1 NSA_SEL_RANGES_V2=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True \
PYTHONPATH=. python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 0
```

### Success Criteria (first 500 steps)
- Stability: No DDP errors; no causality/assert failures
- Throughput: Stable; meets PCIe expectation (ref: 2×A100 PCIe Gen4 target 45–55 toks/s @ 2k). Validation v4 showed much higher rates with synthetic—use target band for realism.
- Memory: Heartbeat reserved < 30–40 GB/GPU @ 2k seq
- Fallbacks: Near zero; not trending upward
- Gates: entropy_mean > 0.5; collapse_fraction ≈ 0

## DDP Hardening & Deadlock Remediation

- Default hardening (set by `scripts/run_2xa100_production.sh`):
  - `NSA_DDP_FIND_UNUSED=0`: avoid per‑step unused‑param graph scans
  - `NSA_DDP_STATIC_GRAPH=1`: prefer static graph for consistent collective ordering
  - `NSA_DDP_DISABLE_GC=1`: keep gradient checkpointing off initially under DDP
  - `NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1`: convert hangs into errors
  - `TORCH_DISTRIBUTED_DEBUG=DETAIL`: detailed dist logs for triage
  - `NSA_DDP_COMPRESS=bf16`: gradient compression (disable with `NSA_DDP_COMPRESS=off`)

- Fast triage matrix if hangs occur again:
  - Step 1: rerun with `NSA_DDP_COMPRESS=off` and `NSA_DDP_BUCKET_MB=100`
  - Step 2: keep compression off; toggle graph controls one at a time:
    - `NSA_DDP_STATIC_GRAPH=0` (keep `FIND_UNUSED=0`)
    - or `NSA_DDP_FIND_UNUSED=1` (keep `STATIC_GRAPH=1`)
  - Step 3: revert both, keep `STATIC_GRAPH=1, FIND_UNUSED=0`; try single‑GPU to confirm model path stability
  - Step 4: escalate to FSDP alternative if DDP remains unstable on your fabric

- Useful one‑liners:
  - Disable compression and bump buckets:
    - `NSA_DDP_COMPRESS=off NSA_DDP_BUCKET_MB=100 bash scripts/run_2xa100_production.sh`
  - Max verbosity for a short window (remember to unset afterward):
    - `TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO bash scripts/run_2xa100_production.sh`

## Single‑GPU Fallback (Stability First)

If DDP deadlocks recur during a long run, prefer finishing milestones on one GPU:
```
CONFIG=configs/m7c_125m_2xa100_production.yaml \
NSA_PREFILL_BATCHED=1 NSA_SEL_RANGES_V2=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True \
PYTHONPATH=. python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 0
```
Expected throughput: ~50% of the 2× run.

## Preflight Checklist (Must be GREEN before long runs)
- Config printed by trainer shows the intended file (e.g., `m7c_125m_2xa100_production.yaml`).
- Precision is BF16; gradient checkpointing “off” initially under DDP; flash enabled.
- `save_every` > 0; `steps`, `seq_len`, `batch_size`, `out_dir` as planned.
- Env:
  - `NSA_SEL_RANGES_V2=1` (default; v2 GPU range conversion)
  - `NSA_DDP_COMPRESS=bf16` (or `off` for triage); `NSA_DDP_BUCKET_MB` set (25 or 100)
  - `NSA_DDP_FIND_UNUSED=0` and `NSA_DDP_STATIC_GRAPH=1` set
  - `NCCL_ALGO=Ring` `NCCL_PROTO=Simple` (PCIe) and `NCCL_IB_DISABLE=1` if no IB
  - `NCCL_ASYNC_ERROR_HANDLING=1` and `NCCL_BLOCKING_WAIT=1`
  - `NSA_PG_TIMEOUT_MIN=15` (process group timeout in minutes; adjust for fabric)
  - Optional progressive GC under DDP: `NSA_DDP_SAFE_GC_LAYERS=A:B` to enable checkpointing only for layers A..B-1
  - `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True`
  - Optional: `NSA_SDPA_AUDIT=1` for one‑time Flash viability log
- Short SDPA routing probe (`TORCH_LOGS=+sdp`) only for the first 30–100 steps, then unset.

## Troubleshooting
- Missing checkpoints: Ensure `train.save_every > 0`; check write permissions in `out_dir`.
- Low/volatile GPU mem in `nvidia-smi`: sample during steady compute; reconcile with heartbeat (MiB) values.
- DDP deadlock/timeout: re‑run with `NSA_DDP_COMPRESS=off NSA_DDP_BUCKET_MB=100`; ensure `NSA_DDP_FIND_UNUSED=0 NSA_DDP_STATIC_GRAPH=1`; verify fail‑fast flags (`NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1`).
- DDP + checkpointing errors: keep `accumulate_grad_batches=1` and `gradient_checkpointing=false`; re‑enable GC only after a stable 500–1000 steps.
- Non‑finite loss (NaN/Inf): the trainer performs a coherent abort across ranks.
  - Symptoms: `[FATAL] non-finite loss` on rank 0; all ranks exit cleanly; `.anomaly_type` and `.HALT` written to `out_dir`.
  - Action: inspect recent `training.csv`, `heartbeat_rank*.jsonl`; root‑cause (data, LR, dtype); relaunch.
  - During DDP GC tuning, consider `NSA_DDP_SAFE_GC_LAYERS=0:6` to checkpoint only early layers.
- Kernel routing mismatches: don’t force flash‑only globally; use short probes; selection isn’t always flash‑eligible.
 - Selection hotspot regression: ensure `NSA_SEL_RANGES_V2=1`; run `scripts/profiler_comparison.py` for v1 vs v2 A/B.

## RUN / Tradeoffs (Recommended First, with context)
- Recommend: BF16; gradient checkpointing OFF initially under DDP.
  - Why: avoid DDP hook/collective edge‑cases; simplify graph; improve stability.
  - Tradeoff: higher activation memory; revisit GC after stability is proven.
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
 - A/B perf: `python scripts/profiler_comparison.py --steps 100 --warmup 10`

---
Prepared for the test engineer. Follow this runbook verbatim for validation, monitoring, and executing the production redo with guardrails.
# NSA 50K Production Run — Operator Runbook (A100 80GB)

This runbook provides a clean, end‑to‑end path to launch, monitor, and recover a 50k‑step production run. It encodes hard‑won fixes (batched prefill, selection v2, DDP hardening, coherent abort) and a safe triage path.

## 0) Clean Slate (GPU and Workspace)

- Stop prior jobs (shell on the training node):
```
pkill -f 'scripts/train_showcase.py' || true
pkill -f 'torchrun' || true
pkill -f 'tensorboard' || true
```
- Inspect GPU processes and utilization:
```
nvidia-smi
nvidia-smi pmon -c 1
```
- Optional (admin only): reset a stuck GPU (not always permitted):
```
sudo nvidia-smi --gpu-reset -i 0
sudo nvidia-smi --gpu-reset -i 1
```
- Clear stale artifacts/logs for a fresh run dir:
```
rm -rf artifacts/m7c_125m_2xa100_prod/
```

## 1) Repo, Branch, and Commit

- Repo root: `nsa-vibe`
- Branch: `feat/nsa-training-breakthrough-stable-a100`
- Commit: `840303b8eaea7221e93fab53d52ba352ba68817a`
- Commands:
```
cd nsa-vibe
git fetch --all --tags
git checkout feat/nsa-training-breakthrough-stable-a100
git reset --hard 840303b8eaea7221e93fab53d52ba352ba68817a
git submodule update --init --recursive || true
```

## 2) Environment Setup

- Python and deps (GPU):
```
uv venv -p 3.11 .venv
. .venv/bin/activate
uv pip sync -r requirements-gpu-cu121-torch24.txt
```
- Quick CUDA check:
```
python - <<'PY'
import torch; print('cuda_available=', torch.cuda.is_available(), 'device_count=', torch.cuda.device_count())
PY
```

## 3) Pre‑Flight Checks (Must Pass Before Long Run)

- GPU causality and core suite:
```
bash scripts/run_gpu_causality_test.sh --full
```
- Selection v2 equivalence and guards:
```
PYTHONPATH=. pytest -q nsa/tests/test_selection_v2_equiv.py
PYTHONPATH=. pytest -q nsa/tests/test_performance_guards.py -k v2
```
- Sanity: tiny single‑GPU smoke (synthetic):
```
CONFIG=configs/m7c_125m_2xa100_production.yaml \
NSA_PREFILL_BATCHED=1 \
PYTHONPATH=. python -u scripts/train_showcase.py --dataset synthetic --ddp 0 --steps 50
```

## 4) Launch Profiles

### A) Single‑GPU Stability Probe (Recommended First)
```
CONFIG=configs/m7c_125m_2xa100_production.yaml \
NSA_PREFILL_BATCHED=1 NSA_SEL_RANGES_V2=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True \
PYTHONPATH=. python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 0 --steps 1000
```
Expect: clean 1000 steps, no NaN/Inf, steady loss and memory.

### B) Two‑GPU Production (Hardened DDP)
Preferred one‑command launcher:
```
bash scripts/run_2xa100_production.sh
```
Manual (equivalent) command:
```
CONFIG=configs/m7c_125m_2xa100_production.yaml \
NSA_PREFILL_BATCHED=1 NSA_SEL_RANGES_V2=1 \
NSA_DDP_COMPRESS=bf16 NSA_DDP_BUCKET_MB=25 \
NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_IB_DISABLE=1 \
NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1 \
TORCH_DISTRIBUTED_DEBUG=DETAIL NSA_PG_TIMEOUT_MIN=15 \
NSA_DDP_DISABLE_GC=1 NSA_DDP_FIND_UNUSED=0 NSA_DDP_STATIC_GRAPH=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True \
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu --ddp 1
```
Notes:
- Keep `batch_size: 1`, `accumulate_grad_batches: 1` initially.
- Optional FA‑2: `NSA_USE_FA2=1 NSA_USE_FA2_WIN=1 NSA_USE_FA2_CMP=1`.
- Optional selection v1 fallback for triage: `NSA_SEL_RANGES_V2=0`.

### C) Progressive Gradient Checkpointing (After Stability)
Enable layer‑wise GC under DDP (example: layers 0–6):
```
NSA_DDP_SAFE_GC_LAYERS=0:6 bash scripts/run_2xa100_production.sh
```
Scale up cautiously (e.g., 0:8 → 0:12) after 500–1000 clean steps.

## 5) What To Watch (First 500–1000 Steps)

- DDP startup (rank 0 log):
  - `[ddp] process group ready | backend=nccl world_size=2 timeout_min=15`
- Loader readiness:
  - `[train] first FineWeb‑Edu batch fetched ...`
  - Both ranks print `[debug] step 1: input shape torch.Size([1, 2048])`
- Throughput and memory:
  - `tail -f artifacts/m7c_125m_2xa100_prod/training.csv`
  - `tail -f artifacts/m7c_125m_2xa100_prod/heartbeat_rank0.jsonl`
  - `watch -n 1 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv`
- Success band (2× A100 PCIe Gen4 @ seq_len=2048): 45–55 toks/s (production path).
- Fallback counters and gate stats (if enabled): low and stable.
- Coherent abort:
  - If `[FATAL] non-finite loss` appears, all ranks exit; `.anomaly_type` and `.HALT` written.

## 6) Triage Playbook (Hangs/Instability)

- Deadlock/timeout:
```
NSA_DDP_COMPRESS=off NSA_DDP_BUCKET_MB=100 NSA_PG_TIMEOUT_MIN=5 \
bash scripts/run_2xa100_production.sh
```
- Toggle graph controls one at a time if needed:
  - `NSA_DDP_STATIC_GRAPH=0` (keep `FIND_UNUSED=0`), or
  - `NSA_DDP_FIND_UNUSED=1` (keep `STATIC_GRAPH=1`)
- Selection isolation: `NSA_SEL_RANGES_V2=0`
- Persisting issues: run single‑GPU to continue progress; consider FSDP alternative.

## 7) Wrap‑Up & Artifact Validation

- Checkpoints and metrics:
```
ls -lh artifacts/m7c_125m_2xa100_prod/
tail -50 artifacts/m7c_125m_2xa100_prod/training.csv
cat artifacts/m7c_125m_2xa100_prod/metrics.json || true
```
- Optional quick inference (byte tokenizer) — see prior section snippet if needed.

## 8) Reference: Config Defaults

- `configs/m7c_125m_2xa100_production.yaml` (key fields):
  - runtime: `precision=bf16`, `use_flash=true`, `gradient_checkpointing=false`
  - train: `seq_len=2048`, `batch_size=1`, `accumulate_grad_batches=1`, `save_every=5000`, `out_dir=artifacts/m7c_125m_2xa100_prod`

## 9) Memory Units (Heartbeat)

- MiB units: `gpu_mem_alloc` and `gpu_mem_reserved` are mebibytes.
- Reconcile with `nvidia-smi` by sampling over several seconds during steady compute.
