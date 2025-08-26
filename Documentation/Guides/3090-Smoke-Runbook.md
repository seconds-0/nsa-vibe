Title: 3090 Single‑GPU Next Test Runbook (Preamble + Full Steps)
Version: v1.5
Date: 2025-08-26

Overview (What & Why)
- We validate NSA/TinyLM loop stability on a single RTX 3090 (24 GB) at low cost.
- Flow: 1‑step autograd sanity → 200‑step stability. If needed, a 2‑step trace is available to localize step‑boundary stalls.
- TensorBoard and CSV logging are disabled to isolate compute/autograd and avoid periodic I/O stalls.
- This establishes a reliable 3090 baseline before re‑enabling logging and scaling to A100/DDP.

What’s New in v1.1
- Fixed scatter_add index dtype in selection scorer (enforced Long) to prevent subtle stalls on Ampere.
- Added NSA_DISABLE_AUX_STATS to disable auxiliary gate/selection stats collection at step end.
- Added explicit end‑of‑step heartbeat ("step_end") to pinpoint stalls between iterations.
- Updated smoke runner to export NSA_DISABLE_AUX_STATS=1 by default.
\- v1.3: Added batched prefill lane, autograd anomaly lane, per-step memory dump option, and a minimal isolated repro script.
\- v1.4: Default runs now set `NSA_PREFILL_BATCHED=1` and the trainer uses AMP (autocast) + GradScaler for fp16 to address NaNs.

Decision Points (At a Glance)
- Go if 1‑step and 200‑step synthetic runs complete with finite loss and no stalls.
- If TB‑off passes, optionally re‑run with TensorBoard on (CSV still off). If TB‑on stalls while TB‑off passes, keep TB off for longer runs.
- No‑go if any backward hang with GC off and TB/CSV/AUX stats disabled, or NaN loss.
- Heartbeat acceptance: step 1 must emit both a "progress" and a "step_end" heartbeat; step 2 must emit a "step_start" heartbeat. Missing step_end implies stall during end‑of‑step; present step_end but missing step_start implies stall at iteration rollover.

Environment & Defaults
- GPU: RTX 3090 24 GB (Ampere, sm_86)
- PyTorch: 2.5.1 preferred (AB: 2.4.1+cu121 if regressions suspected)
- Precision: fp16 on 3090 (use bf16 on A100)
- Gradient checkpointing: off for baseline
- Config: `configs/m7c_3090_smoke.yaml` (seq_len=1024, accumulate_grad_batches=8)
- Env vars for smokes:
  - `NSA_TB_DISABLE=1` (disable TensorBoard)
  - `NSA_DISABLE_CSV_LOGS=1` (disable CSV writes)
  - `NSA_HEARTBEAT_EVERY=1` (per‑step heartbeat)
  - `NSA_DISABLE_AUX_STATS=1` (skip end‑of‑step NSA gate/selection stats)
  - `NSA_PREFILL_BATCHED=1` (use batched prefill path; bypasses step‑5 hang)
  - `NSA_SEL_RANGES_V2=1` (GPU range conversion; default ON)
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256`
  - `NSA_MEM_DUMP_EVERY=0|N` (optional; dump GPU mem pre/post step every N steps; use 1 for dense debugging)

Preflight
1) Verify GPU and torch stack
   - `nvidia-smi -L`
   - `python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
2) TF32 is enabled during training via `scripts/_env_guard.py` (called in train script).

Primary Commands (3090)
These wrap the config/env defaults and emit artifacts to `artifacts/3090_smoke`.

- Synthetic baseline (recommended first):
  - `bash scripts/run_3090_next_smoke.sh synthetic`

- Re‑run with TensorBoard enabled only (optional, after synthetic passes):
  - `NSA_TB_DISABLE=0 NSA_DISABLE_CSV_LOGS=1 bash scripts/run_3090_next_smoke.sh synthetic`

- FineWeb‑Edu 200‑step (optional; requires datasets/transformers):
  - `bash scripts/run_3090_next_smoke.sh fineweb_edu`

- Two‑step trace (localize step‑boundary issues):
  - `NSA_TB_DISABLE=1 NSA_DISABLE_CSV_LOGS=1 NSA_HEARTBEAT_EVERY=1 NSA_DISABLE_AUX_STATS=1 NSA_TRACE_GRADS=1 NSA_TRACE_MODULE_BWD=1 CONFIG=configs/m7c_3090_smoke.yaml python -u scripts/train_showcase.py --dataset synthetic --steps 2 --ddp 0`

Advanced A/B Lanes (run in order if baseline fails)
- Conservative NSA path (synthetic, 200 steps):
  - `NSA_SDPA_NO_FLASH=1 NSA_USE_TRITON_SEL=0 NSA_USE_FA2=0 NSA_USE_SEL_PACK=0 NSA_USE_SEL_MASK=0 NSA_STOPGRAD_GATES=1 bash scripts/run_3090_next_smoke.sh synthetic`

- SDPA flash‑only (synthetic, 200 steps):
  - `NSA_SDPA_FLASH_ONLY=1 bash scripts/run_3090_next_smoke.sh synthetic`

- Precision A/B (fp32, 2 steps):
  - `sed 's/precision: "fp16"/precision: "fp32"/' configs/m7c_3090_smoke.yaml > configs/m7c_3090_smoke_fp32.yaml && NSA_TB_DISABLE=1 NSA_DISABLE_CSV_LOGS=1 NSA_HEARTBEAT_EVERY=1 NSA_DISABLE_AUX_STATS=1 CONFIG=configs/m7c_3090_smoke_fp32.yaml python -u scripts/train_showcase.py --dataset synthetic --steps 2 --ddp 0`

- Deterministic debug (2 steps with synchronous kernels):
  - `CUDA_LAUNCH_BLOCKING=1 NSA_TB_DISABLE=1 NSA_DISABLE_CSV_LOGS=1 NSA_HEARTBEAT_EVERY=1 NSA_DISABLE_AUX_STATS=1 NSA_TRACE_GRADS=1 NSA_TRACE_MODULE_BWD=1 CONFIG=configs/m7c_3090_smoke.yaml python -u scripts/train_showcase.py --dataset synthetic --steps 2 --ddp 0`

- Batched prefill path (synthetic, 200 steps):
  - `NSA_PREFILL_BATCHED=1 bash scripts/run_3090_next_smoke.sh synthetic`

- Autograd anomaly (2 steps; debug only):
  - `NSA_DETECT_ANOMALY=1 NSA_TB_DISABLE=1 NSA_DISABLE_CSV_LOGS=1 NSA_HEARTBEAT_EVERY=1 NSA_DISABLE_AUX_STATS=1 CONFIG=configs/m7c_3090_smoke.yaml python -u scripts/train_showcase.py --dataset synthetic --steps 2 --ddp 0`

- Minimal isolated repro (NSA block only, 6 steps):
  - `python scripts/repro_step5.py --steps 6 --seq-len 1024 --dim 256 --layers 1 --precision fp16`
  - Optional A/B: `NSA_PREFILL_BATCHED=1 python scripts/repro_step5.py --steps 6 --seq-len 1024 --precision fp16`

What the Runner Does
- Preflight prints GPU and torch info; smokes run with TB/CSV disabled and heartbeat enabled.
- Phase 0: 1‑step sanity validates forward/backward success and heartbeat.
- Phase 1: 200‑step stability verifies no stalls or non‑finite loss under fp16, GC off.
- Logs are captured to `artifacts/3090_smoke/phase*.log`.

Acceptance Criteria
- Phase 0 (1 step): completes quickly, heartbeat present, loss finite.
- Phase 1 (200 steps): no stalls; steady step time; VRAM stable (< about 20–22 GB on 3090); loss remains finite beyond step 5.
- Optional TB‑on rerun: also passes; if not, keep TB off for longer tests.

Troubleshooting
- Suspected backward hang (forward ok, `loss.backward()` stalls):
  - Reproduce on a 1‑step with tracing: `NSA_TRACE_GRADS=1 NSA_TRACE_MODULE_BWD=1 bash scripts/run_3090_next_smoke.sh synthetic`
  - Inspect `artifacts/3090_smoke/phase0_1step_synthetic.log` and heartbeat files for the last successful module/param.
- Memory tight at `seq_len=1024`:
  - Reduce to 768/512 or keep 1024 and increase `accumulate_grad_batches`; enable non‑reentrant GC only if needed.
- fp16 overflow:
  - AMP + GradScaler is active by default for fp16. If instability persists, reduce learning rate or try bf16 (A100+).
- Dataloader stalls or GPU idle spikes:
  - Prefer synthetic to isolate; if using data, reduce `num_workers`, consider `pin_memory=False`, and verify disk throughput.
- Flash SDPA sensitivity on Ampere:
  - If hangs persist, try `NSA_SDPA_NO_FLASH=1 bash scripts/run_3090_next_smoke.sh synthetic` (forces mem_efficient/math SDPA).
- Live stack capture on hang:
  - `pid=$(pgrep -f scripts/train_showcase.py) && kill -USR1 $pid` → inspect `artifacts/3090_smoke/stackdump_*.txt`.

Tuning Knobs (fp16 stability)
- Learning rate override: `NSA_LR=5e-5` (overrides config LR for quick A/B)
- GradScaler parameters (defaults in parentheses):
  - `NSA_SCALER_INIT_SCALE` (65536), `NSA_SCALER_GROWTH_INTERVAL` (2000)
  - `NSA_SCALER_GROWTH_FACTOR` (2.0), `NSA_SCALER_BACKOFF_FACTOR` (0.5)
- Consider smaller `grad_clip` (e.g., 0.5) and/or lower LR if NaNs appear.

Artifacts
- `artifacts/3090_smoke/phase0_1step_*.log`, `phase1_200step_*.log`
- `artifacts/3090_smoke/env.json`, `heartbeat_rank*.jsonl`, memory snapshots

Next Steps (if 3090 passes)
- Keep CSV off and optionally keep TB on if TB‑on rerun was stable.
- Run short A100 DDP smokes (200 steps) with previously validated settings before launching longer runs.

Appendix: Repo Touchpoints
- Config: `configs/m7c_3090_smoke.yaml` (fp16, GC off, accum=8, seq_len=1024)
- Runner: `scripts/run_3090_next_smoke.sh` (preflight + Phase 0/1 orchestration)
- Train script: `scripts/train_showcase.py` (env guard, heartbeat, logging gates)
- Optional simple runner: `scripts/small_gpu_smoke.sh` (ensures artifacts dir exists)
