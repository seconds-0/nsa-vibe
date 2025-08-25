# NSA M7C — FSDP Patch Notes (2025‑08‑23)

**Scope:** Fix mislaunch/stability issues, harden diagnostics, and optimize FSDP configuration for NSA M7C on 2×A100.

**Commit Context:** Post‑mortem on FSDP test report showing GPU1 idle and ~4 toks/s.

---

## Summary

- Enforced correct multi‑GPU launch and improved startup observability.
- Tightened FSDP configuration for throughput and correctness.
- Made diagnostics FSDP‑aware (gate/selection/fallback metrics now emit under FSDP).
- Enabled safe performance flags (TF32) and clarified dtype policy.

---

## Modified Files

- `scripts/train_showcase_fsdp.py`

No other files were changed in this patch.

---

## Changes By Category

**Launch Guardrails**
- Startup prints `rank`, `local_rank`, `world_size`, `device` on every rank.
- Guard: If multi‑GPU node has `world_size == 1`, raise with fix hint unless `NSA_ALLOW_SINGLE_RANK=1`.

**FSDP Configuration**
- `ShardingStrategy.FULL_SHARD`: reduces optimizer/state memory and clarifies semantics.
- `limit_all_gathers=True`: bounds parameter all‑gather spikes.
- `forward_prefetch=True`: overlaps comm/compute for better throughput.
- `use_orig_params=True`: cleaner optimizer semantics with sharded params.
- `MixedPrecision`: `param_dtype`, `reduce_dtype`, and `buffer_dtype` now all set to BF16/FP16.
- TF32 enabled for matmul and cuDNN on A100 (`allow_tf32=True`).
  - New: sharding strategy is switchable by env `NSA_FSDP_SHARDING={full|grad_op}`; defaults to `full`.
  - New: prefetch/all‑gather toggles via `NSA_FSDP_FORWARD_PREFETCH` and `NSA_FSDP_LIMIT_ALL_GATHERS`.

**Diagnostics & Telemetry (FSDP‑Safe)**
- Traverses `named_modules()` on the FSDP‑wrapped root to aggregate across all `NSAAttention` modules:
  - Gate stats: entropy mean/min, max of max_gate, collapse fraction, averaged branch shares.
  - Fallback counters: summed across modules; appended to `fallback_counters_fsdp.csv`.
  - Selection stats: averaged `k_mean`, max `k_max`, total `rows`, averaged `pct_at_max`; appended to `k_stats_fsdp.csv`.
- Heartbeat (`heartbeat_rank*.jsonl`) includes above aggregates on rank 0.
- Memory dumps preserved: boot and step1; periodic via `NSA_MEM_DUMP_EVERY`.

**Logging & Audits**
- Dtype audit preserved at startup (`dtypes_report_fsdp.txt`).
- Step logs show global toks/s and optional grad norm.
 - New: CI guardrail fallback — if `CI=1`, single‑rank on multi‑GPU warns instead of raising.

---

## Code Pointers

- World‑size/logging guard: `scripts/train_showcase_fsdp.py` near device/init section.
- FSDP wrapper options: `scripts/train_showcase_fsdp.py` in the `if world_size > 1:` block.
- Diagnostics aggregation: rank‑0 logging branch around training step logging.
- TF32 enablement: right after device selection.

---

## Rationale

- The FSDP test report showed GPU1 idle, tiny memory on rank1, and ~4 toks/s. The most probable cause was single‑process launch. Guardrails prevent accidental single‑rank on multi‑GPU nodes and surface ranks/devices immediately.
- FSDP settings such as `limit_all_gathers`, `forward_prefetch`, and full mixed‑precision policy reduce comm stalls and dtype conversions.
- Diagnostics previously accessed `model.blocks[0].attn`; under FSDP the root is wrapped, so traversal is required to avoid attribute errors and to aggregate multi‑block metrics.

---

## Risks / Tradeoffs

- Guardrail may block intentional single‑GPU runs. Mitigation: set `NSA_ALLOW_SINGLE_RANK=1` to bypass.
- `forward_prefetch` can increase overlap but may hurt on extremely small models; trivial to disable if regressions are observed.
- TF32 subtly changes math for matmul; safe for A100 training and commonly enabled alongside BF16.

---

## Validation Steps

- Launch with two ranks:
  - `CONFIG=configs/m7c_125m_2xa100_production.yaml \
     CUDA_VISIBLE_DEVICES=0,1 NSA_USE_FA2=1 \
     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True \
     NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0 TORCH_LOGS=+sdp NSA_MEM_DUMP_EVERY=100 \
     torchrun --standalone --nproc_per_node=2 scripts/train_showcase_fsdp.py --dataset synthetic`
- Expect both GPUs active, toks/s > 50 global after warmup.
- Verify artifacts exist and populate:
  - `training_fsdp.csv`, `heartbeat_rank{0,1}.jsonl`, `dtypes_report_fsdp.txt`.
  - `k_stats_fsdp.csv`, `fallback_counters_fsdp.csv`, `mem_fsdp_*`, `opt_state_fsdp_mb.txt`.

---

## Rollback Plan

- To revert behavior quickly:
  - Remove env guard by setting `NSA_ALLOW_SINGLE_RANK=1` (for single‑GPU runs).
  - Disable TF32: set `torch.backends.cuda.matmul.allow_tf32=False` and `torch.backends.cudnn.allow_tf32=False`.
  - Relax FSDP options: remove `limit_all_gathers`/`forward_prefetch` or switch sharding if needed.

---

## Review Notes / Questions

- Do we want the guardrail to warn (log) instead of raise for CI smoke tests? If yes, I can downgrade to a warning under `CI=1`.
- Current aggregation emits only on rank 0. If per‑rank CSVs are desired, I can add `*_rank{rank}.csv` emission.
- Auto‑wrap remains at block granularity (`LlamaBlockNSA`). If we aim to benchmark alternative wrap sizes, I can expose a config flag.
