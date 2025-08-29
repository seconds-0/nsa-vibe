Title: PR35 — FA‑2 Min‑Length Auto‑Tune (GPU)

Summary
- Adds a self-hosted GPU workflow and a local script to benchmark FA‑2 vs masked SDPA and auto‑apply tuned thresholds into `configs/base.yaml`.
- Produces a concise markdown report and preserves the raw bench log as an artifact.

Changes
- .github/workflows/fa2-autotune.yml: Manual (workflow_dispatch) job for self-hosted GPU runners. Runs `bench/bench_fa2.py` (sliding + compressed), parses results with `bench/threshold_optimizer.py`, updates `configs/base.yaml`, writes `fa2_thresholds.md`, and opens a PR.
- scripts/automation/fa2_autotune.sh: Local helper to run the same flow outside CI (writes to `artifacts/fa2_autotune/`, updates a specified config path).
- Makefile: Adds `make fa2-autotune` convenience target.

Usage
- Local (GPU node):
  - `make fa2-autotune` (defaults to `configs/base.yaml`; outputs to `artifacts/fa2_autotune/`)
  - Or directly: `bash scripts/automation/fa2_autotune.sh configs/base.yaml artifacts/fa2_autotune`
- CI (self-hosted GPU):
  - Run workflow “FA-2 Auto-Tune (GPU, manual)” with `safety_margin` (default 1.2x).
  - The workflow opens a PR with updated thresholds and a report.

What It Updates
- `runtime.fa2_min_len_win`: minimum sliding window length for FA‑2.
- `runtime.fa2_min_len_cmp`: minimum compressed effective length for FA‑2.
- Report: `fa2_thresholds.md` with device info, results table, and recommended thresholds.

Safety & Overrides
- Env overrides remain honored at runtime: `NSA_FA2_MIN_LEN_WIN`, `NSA_FA2_MIN_LEN_CMP`.
- On unsupported devices or when FA‑2 shows no benefit under the selected safety margin, defaults remain conservative.

Validation
- After merge, confirm on target GPU:
  - `NSA_TEST_FA2=1 PYTHONPATH=. pytest -q -k fa2_gpu_varlen`
  - Smoke: `PYTHONPATH=. NSA_USE_FA2=1 uv run -q python bench/bench_fa2.py`

