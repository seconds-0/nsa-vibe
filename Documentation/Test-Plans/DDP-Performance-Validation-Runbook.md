# DDP Performance Validation Runbook (2×A100 PCIe)

## Overview
Validates DDP throughput, SDPA backends, bucket sizing, selection v2 parity/perf, and long-run stability. Commands are production-safe and use existing env toggles.

## Branch & Environment
- Branch: `feat/nsa-training-breakthrough-stable-a100`
- Python: 3.10+
- GPUs: 2×A100 80GB PCIe (no NVLink)
- Torch/CUDA: Production stack; record with collect_env

## Preflight (Snapshot & Sync)
```
git fetch && git checkout feat/nsa-training-breakthrough-stable-a100 && git pull
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD > artifacts/git_sha.txt
python -m torch.utils.collect_env > artifacts/collect_env.txt
nvidia-smi -q -x > artifacts/nvidia_smi.xml
```
Optional sanity:
```
PYTHONPATH=. pytest -q nsa/tests/test_equiv_small.py nsa/tests/test_group_consistency.py
```

## Baseline DDP Throughput (Synthetic)
Note: For production configs with long sequences (seq_len ≥ 1024), set `NSA_PREFILL_BATCHED=1` to use the vectorized prefill path and avoid slow sequential prefill.

```
CONFIG=configs/m7c_125m_2xa100_production.yaml \
NSA_PREFILL_BATCHED=1 \
NSA_SEL_RANGES_V2=1 \
NSA_DDP_COMPRESS=bf16 \
NSA_DDP_BUCKET_MB=50 \
NCCL_ALGO=Ring \
NCCL_PROTO=Simple \
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py \
  --dataset synthetic --ddp 1 --steps 300
```
Verify: `[ddp] gradient compression enabled: bf16`; `<OUT>/training.csv`, `<OUT>/heartbeat_rank0.jsonl`, `<OUT>/env.json`; target ≥39 toks/s (aim 45–55).

Tip (OUT dir): The out_dir is defined in the config. To confirm at runtime, check `<OUT>/env.json` after startup or print from YAML:
```
python - <<'PY'
import yaml; print(yaml.safe_load(open('configs/m7c_125m_2xa100_production.yaml'))['train']['out_dir'])
PY
```

## DDP Bucket Sweep
```
for mb in 25 50 100; do 
  NSA_PREFILL_BATCHED=1 NSA_DDP_BUCKET_MB=$mb PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py \
    --dataset synthetic --ddp 1 --steps 200; 
done
```
Verify: last‑50‑step mean/stdev; choose highest mean (tie → lower variance).

Compute last‑50‑step mean/stdev (example):
```
python - <<'PY'
import sys,statistics as st
from pathlib import Path
csv = Path(sys.argv[1]).read_text().strip().splitlines()
rows = [r.split(',') for r in csv if r.strip()]
vals = [float(r[3]) for r in rows[-50:]] if len(rows)>=50 else [float(r[3]) for r in rows]
print(f"mean={st.mean(vals):.1f} stdev={st.pstdev(vals):.1f} n={len(vals)}")
PY artifacts/m7c_125m_2xa100_prod/training.csv
```

## SDPA Backend Audit (One‑Time)
Note: The SDPA audit runs on the batched prefill path. Ensure `NSA_PREFILL_BATCHED=1` is set.
```
NSA_PREFILL_BATCHED=1 NSA_SDPA_AUDIT=1 \
CONFIG=configs/m7c_125m_2xa100_production.yaml \
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py \
  --dataset synthetic --ddp 1 --steps 50
```
Verify: audit logs; no errors. Optional A/B: `NSA_SDPA_FLASH_ONLY=1` or `NSA_SDPA_NO_FLASH=1` for 50 steps.

If you see no audit output, ensure `NSA_SDPA_AUDIT=1` is set and run at least 1 step where cmp/win branches are exercised. Check only rank 0 logs.

## Selection V2 Parity & Performance (CUDA)
```
PYTHONPATH=. NSA_TEST_SEL_PACK=1 pytest -q nsa/tests/test_selection_v2_equiv.py -k equivalence_various_patterns
PYTHONPATH=. pytest -q nsa/tests/test_performance_guards.py -k v2_performance_improvement --disable-warnings
```
Verify: parity passes; perf shows v2 faster on CUDA.

## NVTX Profiling (Attribution)
```
NSA_PREFILL_BATCHED=1 NSA_NVTX=1 NSA_TB_DISABLE=1 NSA_DISABLE_AUX_STATS=1 \
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py \
  --dataset synthetic --ddp 1 --steps 100
```
Verify: `nsa.sel.ranges_v2` present; no Python hotspots dominating.

If profiling appears to hang, shorten to `--steps 50`, disable TensorBoard (`NSA_TB_DISABLE=1`) and aux stats (`NSA_DISABLE_AUX_STATS=1`).

## Single‑GPU Isolation (Production Config)
```
CONFIG=configs/m7c_125m_2xa100_production.yaml \
NSA_PREFILL_BATCHED=1 PYTHONPATH=. python scripts/train_showcase.py \
  --dataset synthetic --ddp 0 --steps 300
```
Verify: per‑GPU throughput higher than DDP per‑GPU; if slow, capture NVTX.

Note: Use the same production config (`CONFIG=...`) to keep seq_len/batch identical.

## Fused AdamW A/B (Optional)
Baseline:
```
NSA_PREFILL_BATCHED=1 PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py \
  --dataset synthetic --ddp 1 --steps 200
```
Variant:
```
NSA_PREFILL_BATCHED=1 NSA_OPT_FUSED=1 PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py \
  --dataset synthetic --ddp 1 --steps 200
```
Verify: keep if stable and ≥ baseline mean toks/s (last 50 steps).

If fused AdamW hangs, leave `NSA_OPT_FUSED` unset and note in results.

## Stability Long‑Run + Watchdog
```
python scripts/_watchdog.py --dir artifacts/m7c_125m_2xa100_prod --halt 1 --interval 30
NSA_PREFILL_BATCHED=1 PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py \
  --dataset synthetic --ddp 1 --steps 1200
```
Verify: no `.HALT`; steady `toks_per_s`; healthy gate stats; memory plateau.

If the run stalls or NCCL hangs:
- Re‑run with a different master port, e.g., `torchrun --master-port=29502 ...`
- Ensure no leftover trainer processes (`ps aux | grep train_showcase`); avoid `pkill` unless necessary.
- Optionally set `NCCL_DEBUG=INFO` to aid diagnosis (avoid in perf runs).

## FineWeb‑Edu (Optional)
```
python scripts/automation/fwe_smoke.py --seq-len 1024 --batch 1 --timeout 60 --tokenizer gpt2
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py \
  --dataset fineweb_edu --ddp 1 --steps 200 --loader-timeout 120
```
Verify: loader responsive; training progresses.

## Deliverables
- Bucket table: `NSA_DDP_BUCKET_MB` vs mean/stdev (last 50 steps)
- Optional: fused AdamW A/B table
- Artifacts: `<OUT>/training.csv`, `<OUT>/heartbeat_rank*.jsonl`, `<OUT>/env.json`, memory dumps; `artifacts/collect_env.txt`, `artifacts/nvidia_smi.xml`, `artifacts/git_sha.txt`

## Acceptance
- Throughput ≥ 39 toks/s; aim 45–55 toks/s (S=2048, global batch=2, BF16)
- v2 parity tests pass; group consistency intact
- DDP BF16 compression active; best bucket selected; near‑linear per‑GPU scaling
- SDPA audit ok; no unexpected fallbacks

## Failure Playbook
- Low throughput: `NSA_DDP_BUCKET_MB=50`; ensure flash not disabled; disable tracing
- Stalls: watchdog `.HALT` → send `SIGUSR1` to trainer; attach stackdump
- v2 issues: `NSA_SEL_RANGES_V2=0`; re‑run parity tests
- Optimizer instability: unset `NSA_OPT_FUSED`

## Notes & Hygiene
- Pin `CUDA_VISIBLE_DEVICES=0,1` to avoid device drift if needed.
- For long sequences (seq_len ≥ 1024), set `NSA_PREFILL_BATCHED=1` to force the vectorized prefill path. Why: prevents sequential prefill’s Python loops (O(B·S·G)) which can appear as timeouts at S=2048.
- Always record which config was used (`CONFIG=...`) and attach the corresponding `<OUT>/env.json`.
- For every throughput number, state the CSV path and window used (e.g., last 50 steps).
