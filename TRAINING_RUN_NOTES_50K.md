# NSA 50K Training Run Notes (STOPPED at 35,360 steps - 70.7%)

## Scope
- 50k‑step run on 2×A100 80GB. Baseline correctness plus stability/perf smoke for M8.
- **STOPPED EARLY**: Terminated at step 35,360 (70.7%) to avoid wasting compute on toy config
- Code branch/tag: `feat/m7c-perf-stability` tagged `m8-prefill-causality-fix`.
- Fix commit (merged): `12c6ea7c38db20b3581c243fc7c7c199b6e1750f` (batched prefill strict asserts).

## What We Changed
- Restored batched prefill causality asserts by defining `strict_asserts` inside `_forward_prefill_batched`.
- Added `scripts/run_gpu_causality_test.sh` with `--full` mode for A100/H100 validations (strict error surfacing, safe defaults, routing probe).

## Validations Run
- GPU causality tests: 3/3 passed on 2×A100.
- M0 core subset: masks, block math, group consistency, small equivalence, decode counters — green.
- Full mode: selection packed parity, FA‑2 GPU varlen parity (A100 eligible), decode/masked tiny, RoPE dtype, long‑context smoke — green.

## Final Run Status
- **Total steps completed**: 35,360 / 50,000 (70.7%)
- **Duration**: ~24.5 hours before stopping
- **Final loss**: 1.249 (down from 5.75 initial)
- **Throughput**: Stable 393-405 tok/s throughout
- **Memory**: 21 GiB allocated, 66 GiB reserved (stable)
- **Reason for stopping**: Toy config (dim=128, seq_len=128) not worth completing

## Memory Reporting Reconciliation
- Heartbeat fields are logged in MiB (not GB). Report text mislabeled them as “GB”.
  - `gpu_mem_alloc` = `torch.cuda.memory_allocated() // (1024*1024)` (MiB)
  - `gpu_mem_reserved` = `torch.cuda.memory_reserved() // (1024*1024)` (MiB)
- `nvidia-smi` snapshots at 0.7–1.0 GiB likely captured an inter‑step lull or different process.
- Action: Update reporting/notes to state MiB, and optionally add derived GiB in heartbeat for clarity.

## Post‑Mortem (why we wasted compute)
- Root cause: The job used `configs/train_showcase.yaml` (toy fp32 config, no checkpoints) instead of a production profile.
- Contributing factors:
  - No enforced `CONFIG` pinning or preflight validation for long GPU runs.
  - Missing checkpoint cadence (`save_every=0`).
  - Documentation mislabeled heartbeat MiB as GB which masked the config mistake.
  - No automated guardrails (CI/lints) to fail fast on fp32/no‑ckpt for long runs.
- Impact: ~70% of a 50k run on a model not suitable for promotion; intermediate recovery points missing.
- Detection: Mismatch between expected checkpoints and artifacts; memory/unit confusion; manual review of config.
- Resolution: Stopped at 35,360 steps; documented learnings; prepared production config guidance for the second run.

## Config Used vs Recommended
- Observed training config (`configs/train_showcase.yaml`):
  - Precision: FP32
  - Flash: off
  - Gradient checkpointing: off
  - Seq len: 128, Batch (global): 8
  - `save_every`: not set (=> 0). Only final `model.pt` saved.
- Recommended production config for A100 (next run): `configs/m7c_125m_2xa100_production.yaml`
  - Precision: BF16
  - Flash: on
  - Gradient checkpointing: true
  - Seq len: 2048 (validate), then 3072 if reserved < 40 GB
  - Batch: global 2 (micro‑batch 1 per GPU), `accumulate_grad_batches=1`
  - Checkpointing: `save_every: 5000` (or 1000 for tighter recovery points)
  - Out dir: separate from showcase run to avoid mixing artifacts

## Runtime Env (A100, defaults)
- `NSA_USE_FA2=1`
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True`
- `NCCL_P2P_DISABLE=0` `IB_DISABLE=0`
- Short probe only: `TORCH_LOGS="+sdp"` (first 30–100 steps), then unset
- Keep Triton selection off by default: `NSA_USE_TRITON_SEL=0`

## Health Checklist (live)
- Loss: smooth monotonic decline; no >1k‑step plateaus.
- Throughput: ±10% band; no downward drift.
- Memory: reserved/allocated flat; no creep; no OOM retries.
- Fallback counters: near zero; not trending upward.
- Gates: entropy_mean > 0.5; max_gate_mean < 0.9; collapse_fraction ~0.0; balanced branch_shares.
- Selection stats: `k_mean << k_max`; `pct_at_max < 5–10%`.
- Reads: `reads_pred == reads_act_total` per step.

## Milestones & Artifacts
- Checkpoints: configure `save_every` next run (5k/10k/25k/50k). Current run only final `model.pt`.
- Final artifact: `artifacts/train_showcase/model.pt` (state_dict + cfg).
- Logs/metrics: `<out_dir>/training.csv`, `val.csv`, `metrics.json`, `tb/*`, `heartbeat_rank*.jsonl`.

## Post‑Run Use & Tests
- Inference sanity (byte tokenizer): greedy generation snippet using `TinyLM` + `model.pt` (BF16 on A100).
- Integration tests (GPU): `test_m8_integration.py -k integration`, `test_long_context_needle`.
- Benches: `bench_prefill.py`, `bench_decode.py` with `NSA_USE_FA2=1`.

## Risks & Notes
- RTX 4090: selection Triton disabled by ADR — stay on SDPA/packed paths.
- Do not force flash‑only SDPA for full runs; use short probes only.
- DDP + checkpointing: keep `accumulate_grad_batches=1`; use `no_sync` during accumulation.

## Plan For Second Run (staged)
- Soak: 2×A100, BF16, gradient checkpointing ON, seq_len=2048, `save_every=5000`.
  - Success: no DDP errors in 500 steps; reserved < 30–40 GB/GPU; > 50 toks/s global; low fallbacks; healthy gates/selection.
- Scale: consider seq_len=3072 if headroom; hold accum=1.
- Observability: continue heartbeat, counters, gate stats; fix unit labels (MiB vs GiB) in notes/UI.

## Key Learnings from Aborted Run
- **Config mistake**: Used test config (dim=128) instead of production (dim=768)
- **Validated**: M8 fix stable for 35k+ steps, no crashes or memory leaks
- **Wasted compute**: ~$30-40 of A100 time on unusable model
- **Missing**: No checkpoints saved due to missing `save_every` parameter

## Action Items (for second run and beyond)
- Config hygiene:
  - Enforce `CONFIG` must point to a production profile (bf16, gradient_checkpointing: true, flash: true, save_every>0) for multi‑hour GPU runs.
  - Add a preflight validator script to assert critical fields before launch; abort with a clear message if mismatched.
  - Add a canonical `configs/train_50k.yaml` (bf16, ckpt on, flash on, steps=50000, save_every=5000) to avoid ambiguity.
- Checkpointing & recovery:
  - Require `save_every` > 0 when `steps >= 5000`; add a runtime assertion in `train_showcase.py`.
  - Write a `latest.ckpt` symlink after every save to simplify resume.
- Telemetry clarity:
  - Keep heartbeat in MiB but add derived GiB fields; update all docs and dashboards to label units correctly.
- Docs & process:
  - Update CLAUDE.md and AGENTS.md with a Training Preflight Checklist and Memory Units guidance; include RUN/TRADEOFFS defaults.
- Automation:
  - Extend `scripts/run_gpu_causality_test.sh` or add `scripts/preflight_train.sh` to check `CONFIG`, flags, and checkpoint cadence before starting long runs.

## One‑Liners
- GPU validation script: `bash scripts/run_gpu_causality_test.sh --full`
- Production training example:
  - `CONFIG=configs/m7c_125m_2xa100_production.yaml NSA_USE_FA2=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True NCCL_P2P_DISABLE=0 IB_DISABLE=0 python -u scripts/train_showcase.py --dataset fineweb_edu --ddp 1`
