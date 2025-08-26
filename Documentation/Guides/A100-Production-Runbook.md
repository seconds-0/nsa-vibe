Title: 2×A100 Production Runbook (NSA M7C 125M)
Version: v1.3
Date: 2025-08-26

Overview (What & Why)
- Goal: validate and launch a 50k‑step production run on 2×A100 80GB using the stable path identified in 3090 testing.
- Strategy: run short DDP smokes on bf16 with batched prefill to bypass the step‑5 sequential prefill hang, then scale to 50k with guardrails.
- Logging I/O (TB/CSV) starts disabled to avoid cadence stalls; re‑enable TB after stability confirmation.

Environment & Prereqs
- Hardware: 2×A100 80GB (NVLink or PCIe), recent NVIDIA drivers.
- Software: PyTorch 2.5.1 (bf16 supported), CUDA 12.x.
- Network: NCCL default; IB allowed if present (we don’t require it for 2 GPUs on a single host).
- Repo scripts: `scripts/run_m7c_2xa100_production.sh` and `scripts/train_showcase.py`.

Defaults (A100)
- Precision: `bf16` (DDP friendly, stable numerics).
- Prefill: `NSA_PREFILL_BATCHED=1` (bypasses step‑5 sequential hang).
- Aux stats: `NSA_DISABLE_AUX_STATS=1` (avoid end‑of‑step overhead during smokes).
- Logging: `NSA_TB_DISABLE=1`, `NSA_DISABLE_CSV_LOGS=1` for smokes.
- GC: config enables GC, but trainer disables under DDP by default via `NSA_DDP_DISABLE_GC=1` to simplify hooks. Turn GC back on after smokes if needed for memory.

Mandatory Envs (for production runs)
- `NSA_PREFILL_BATCHED=1` (bypass sequential prefill hang)
- `NSA_DISABLE_AUX_STATS=1` (avoid end‑of‑step overhead during smokes)
- `NSA_DDP_STATIC_GRAPH=1` and `NSA_DDP_FIND_UNUSED=0` (static‑graph performance)
- `NSA_DDP_BUCKET_MB=25` (larger DDP buckets)
- On PCIe (no NVLink): set `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`; keep `NCCL_IB_DISABLE=1` if no InfiniBand.

Preflight
- GPU check: `nvidia-smi -L`
- Torch stack: `python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
- Optional TF32: trainer enables TF32 for matmul/cudnn when CUDA is available.

Phase Plan
- Phase 0 (DDP sanity, synthetic, 1 step): checks DDP wiring and bf16 path.
- Phase 1 (smoke, dataset, ~300 steps): FineWeb‑Edu stream to validate real data I/O + training loop.
- Phase 2 (extended smoke, 1000 steps): confidence before 50k launch.
- Phase 3 (production, 50k steps): enable TB, keep CSV off; checkpoint cadence per config.

Commands
- Run scripted (recommended; tuned DDP + batch):
  - `bash scripts/run_m7c_2xa100_production.sh`

- Manual (explicit) 2×A100 smoke (300 steps):
  - `NSA_PREFILL_BATCHED=1 NSA_DISABLE_AUX_STATS=1 NSA_TB_DISABLE=1 NSA_DISABLE_CSV_LOGS=1 NSA_DDP_STATIC_GRAPH=1 NSA_DDP_FIND_UNUSED=0 NSA_DDP_BUCKET_MB=25 \
CONFIG=configs/m7c_125m_2xa100_production.yaml \
torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu --steps 300 --precision bf16`

Performance Tuning (A100)
- DDP static graph and unused params off (if graph is static):
  - `NSA_DDP_STATIC_GRAPH=1 NSA_DDP_FIND_UNUSED=0`
- Larger DDP buckets for better overlap (e.g., 25–50 MB):
  - `NSA_DDP_BUCKET_MB=25`
- Prefer gradient accumulation over larger per‑GPU batches on PCIe systems:
  - Keep `batch_size: 2` and set `accumulate_grad_batches: 2` (effective batch 4) to reduce sync frequency without increasing PCIe traffic.
  - Avoid `batch_size > 2` on PCIe (empirically reduces throughput and utilization).
- Keep `NSA_USE_FA2=1`; if instability suspected, A/B with `NSA_SDPA_NO_FLASH=1`.
 - Quick env overrides (no config edits): `NSA_ACCUM=2`.

Tuned Production Config
- Use `configs/m7c_125m_2xa100_production_tuned.yaml` (batch_size: 4) for better utilization on 2×A100 PCIe.
- Manual tuned run (300 steps):
  - `NSA_PREFILL_BATCHED=1 NSA_DISABLE_AUX_STATS=1 NSA_DDP_STATIC_GRAPH=1 NSA_DDP_FIND_UNUSED=0 NSA_DDP_BUCKET_MB=25 \
CONFIG=configs/m7c_125m_2xa100_production_tuned.yaml \
torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu --steps 300 --precision bf16`

- Re‑enable TensorBoard after stability:
  - `NSA_TB_DISABLE=0 NSA_DISABLE_CSV_LOGS=1 torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu --steps 300 --precision bf16`

Acceptance Criteria
- Phase 0: completes 1 step; heartbeats present.
- Phase 1: completes 300 steps; no stalls; loss finite beyond step 5; steady step times; utilization across both GPUs balanced (>80%); throughput within expected PCIe bounds (Gen3: 35–40 toks/s; Gen4: 45–55 toks/s).
- Phase 2: completes 1000 steps; memory usage stable per GPU (< 60 GB, configurable threshold).
- Heartbeats: rank0 shows step “progress” and “step_end”; subsequent steps emit “step_start”.

Monitoring & Artifacts
- Output dir: `artifacts/m7c_125m_2xa100_prod/` (env.json, heartbeats, optional mem dumps, logs).
- Heartbeats: `heartbeat_rank0.jsonl` (per‑step “progress” and “step_end”).
- Memory snapshots: set `NSA_MEM_DUMP_EVERY=100` (or `1` for dense) to write `mem_pre_stepX.*` / `mem_post_stepX.*`.
- Live stack if stall: `pid=$(pgrep -f scripts/train_showcase.py) && kill -USR1 $pid` → check `stackdump_*.txt`.

Toggles & Tuning
- Required for smokes:
  - `NSA_PREFILL_BATCHED=1` (batched prefill path)
  - `NSA_DISABLE_AUX_STATS=1` (skip end‑of‑step NSA stats)
  - `NSA_TB_DISABLE=1`, `NSA_DISABLE_CSV_LOGS=1` (avoid I/O stalls initially)
- DDP safety (use if needed):
  - `NSA_DDP_SAFE_MODE=1` (conservative DDP settings; disables some kernel optimizations)
  - `NSA_DDP_DISABLE_GC=1` (default: keep GC off under DDP, re‑enable later if memory bounded)
- SDPA A/B (if regressions):
  - `NSA_SDPA_NO_FLASH=1` (force mem_efficient/math kernels)
  - `NSA_SDPA_FLASH_ONLY=1` (diagnostic; flash only)
- fp16 is not recommended on 3090; on A100 stick to bf16. If you must run fp16, trainer supports AMP + GradScaler with env tuning:
  - `NSA_SCALER_INIT_SCALE` (65536), `NSA_SCALER_GROWTH_INTERVAL` (2000), `NSA_SCALER_GROWTH_FACTOR` (2.0), `NSA_SCALER_BACKOFF_FACTOR` (0.5)
  - `NSA_LR` to override learning rate for quick A/B.

Troubleshooting
- Hang at step boundary: verify “step_end” for the last completed step; if missing, disable aux stats, keep TB/CSV off, and check stackdump.
- Data loader stalls: run synthetic to isolate; then re‑enable dataset with modest `--loader-timeout`; ensure local caching.
- Memory pressure: reduce `seq_len` to 1024, increase `accumulate_grad_batches`, or re‑enable GC (unset `NSA_DDP_DISABLE_GC`).
- Imbalanced GPU utilization: enable static graph and disable unused params search (`NSA_DDP_STATIC_GRAPH=1 NSA_DDP_FIND_UNUSED=0`); increase bucket size (`NSA_DDP_BUCKET_MB=25`); increase batch size or gradient accumulation.
- SDPA sensitivity: try `NSA_SDPA_NO_FLASH=1` for diagnostic; keep batched prefill.

Production Launch (50k)
- Enable TB; keep CSV off: `NSA_TB_DISABLE=0 NSA_DISABLE_CSV_LOGS=1`.
- Keep `NSA_PREFILL_BATCHED=1` throughout the 50k run (sequential prefill hang still under investigation).
- Checkpoint per config (`save_every: 5000`); ensure storage throughput and space.
- Monitoring: use `scripts/_watchdog.py` to HALT on heartbeat/throughput anomalies.

Go/No‑Go Checklist
- Go when:
  - 300‑step and 1000‑step bf16 DDP smokes pass; loss finite, no stalls.
  - Throughput and memory within expected bounds; heartbeats healthy.
- No‑Go when:
  - Any backward hang or heartbeat stall with TB/CSV/AUX disabled.
  - Non‑finite loss in bf16, or reproducible SDPA instability under both flash and no‑flash.

Appendix: Scripted Runner Notes
- `scripts/run_m7c_2xa100_production.sh` performs:
  - GPU preflight; env setup with `NSA_PREFILL_BATCHED=1`, `NSA_DISABLE_AUX_STATS=1`.
  - Phase 1 synthetic (200 steps) then dataset run.
  - Basic success metrics summary (memory, throughput, selection health) from artifacts.
