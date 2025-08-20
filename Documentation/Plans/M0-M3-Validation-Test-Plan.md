# M0–M3 Validation Test Plan (GPU)

Purpose: End-to-end validation of NSA’s M0–M3 milestones on a single GPU box. Confirms correctness, FA‑2 gating, long‑context behavior, and safe defaults for production.

## Requirements
- GPU: RTX 4090 (Ada, SM 8.9) or similar. CPU fallback tests are included; GPU focus is FA‑2.
- OS/Tooling:
  - CUDA ≥ 12.1, PyTorch ≥ 2.3, optional FlashAttention‑2 ≥ 2.x
  - Repo checked out at `/root/nsa-vibe` with `.venv` created
- Environment setup (one time):
  ```bash
  cd /root/nsa-vibe
  source .venv/bin/activate
  python - << 'PY'
  import torch
  print({
    'cuda': torch.cuda.is_available(),
    'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'capability': torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    'torch': torch.__version__,
    'cuda_version': torch.version.cuda,
  })
  PY
  ```
  Expected: `cuda: True`, valid `device`, `capability` tuple.

## Global Test Settings
- Determinism: set fixed seeds within tests; no dropout.
- TF32: leave default; for stricter parity, you may export `TORCH_ALLOW_TF32=0`.
- Triton selection: remains disabled on SM 8.9 by guard; do not enable for production testing.

## 1) Core CPU/GPU Smokes (M0–M3)
Run consolidated smoke script (mostly CPU paths; validates causality, mapping, group consistency, decode counters, and training smoke):
```bash
PYTHONPATH=. uv run -q python scripts/run_milestone_smoke.py
```
Expected output:
- All invoked pytest selections exit 0.
- No assertion failures; console shows rc=0 per block.

Artifacts: none.

## 2) FA‑2 Parity (GPU, optional but recommended)
Validates FA‑2 vs SDPA parity on tiny grids for sliding/compressed branches.
```bash
NSA_TEST_FA2=1 PYTHONPATH=. uv run -q pytest -k fa2_parity
```
Expected:
- Tests pass. Implicit tolerances (FP32 ≤ 5e-5; BF16/FP16 ≤ 2e-4) are upheld.
- If FA‑2 unavailable, tests xfail/skip with clear reason.

Artifacts: none.

## 3) FA‑2 Performance Thresholds (GPU)
Benchmark FA‑2 vs masked SDPA and compute safe min-length gates.

Commands:
```bash
# Sliding (window w=512)
PYTHONPATH=. uv run -q python bench/bench_fa2.py \
  --mode win --heads 8 --dk 64 --dv 64 \
  --S_list 128,256,512,1024,2048 --w 512 --iters 100 > fa2_win.txt

# Compressed (l=32,d=16)
PYTHONPATH=. uv run -q python bench/bench_fa2.py \
  --mode cmp --heads 8 --dk 64 --dv 64 \
  --S_list 128,256,512,1024,2048 --l 32 --d 16 --iters 100 > fa2_cmp.txt

# Compute recommended thresholds and update config + report
PYTHONPATH=. uv run -q python bench/threshold_optimizer.py \
  fa2_win.txt --config configs/base.yaml --report fa2_thresholds.md
PYTHONPATH=. uv run -q python bench/threshold_optimizer.py \
  fa2_cmp.txt --config configs/base.yaml --report - --dry-run  # prints summary only
```
Expected console/report lines (examples):
- "Recommended Thresholds:" followed by
  - `fa2_min_len_win: <int>` where speedup ≥ 1.2× across tested S for w=512
  - `fa2_min_len_cmp: <int>` mapping to effective compressed lengths
- "Config updated: configs/base.yaml" for the first optimizer call

Artifacts to attach:
- `fa2_win.txt`, `fa2_cmp.txt`, `fa2_thresholds.md`, updated `configs/base.yaml`

Acceptance:
- Use thresholds only where speedup ≥ 1.2×; otherwise keep conservative defaults (win≥512, cmp≥32).

## 4) Selection Path Checks (Production = SDPA)
Selection remains on packed SDPA; validate selection pack parity and long-context smoke.

Commands:
```bash
# Selection packed parity (CPU/GPU)
PYTHONPATH=. uv run -q pytest -k test_selection_packed

# Long-context needle smoke (selection over ideal range)
PYTHONPATH=. uv run -q python bench/needle_64k_smoke.py --S 65536 --device cuda
```
Expected:
- Tests pass for selection packing.
- Needle smoke prints dict like:
  `{ 'S': 65536, 'pos': <int>, 'cos': ~1.0, 'mae': ~0.0, 'time_ms': <float> }`
  with cosine similarity ≥ 0.99 and MAE ≤ 1e-3.

Artifacts:
- Console output; optional screenshot/log capture.

## 5) Decode Read Counters (GPU optional)
Spot-check decode reads match PRD formula at larger S without OOM.
```bash
PYTHONPATH=. uv run -q pytest -k test_decode_counters
```
Expected:
- Pass: per-step predicted/actual reads match exactly, including early S<w and S<l cases.

Artifacts: none.

## Summary of Deliverables
- Logs: `fa2_win.txt`, `fa2_cmp.txt`
- Report: `fa2_thresholds.md` (includes recommended thresholds and tables)
- Config: updated `configs/base.yaml` with `runtime.fa2_min_len_win` and `runtime.fa2_min_len_cmp`
- Console captures for long-context needle and overall test results

## Pass/Fail Criteria
- All smokes/tests pass (exit code 0), or expected skips/xfails are annotated (FA‑2 not available).
- FA‑2 thresholds only enable when speedup ≥ 1.2×; otherwise keep defaults.
- Long-context needle cosine ≥ 0.99, MAE ≤ 1e-3.
- Selection remains on SDPA; Triton disabled by default on SM 8.9 (no attempts to enable).

*** End Plan ***
