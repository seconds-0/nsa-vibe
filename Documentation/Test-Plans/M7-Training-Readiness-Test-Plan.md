# M7 Training Readiness — Test Plan (Runbook)

Owner: Infra/QA agent
Scope: Verify NSA correctness gates and training harness readiness to start M7C (~125M) training and long-context demos.

## Branches
- Base: `master`
- Under test: `test-plan/m7-training-readiness` (this PR)
- Training config branch (optional reference): `m7c-64k-demo` for demo scripts and NSA enhancements

## Environments
- Python 3.10+
- CPU suite: any Linux runner
- GPU suite: CUDA 12.1, PyTorch 2.4.x, Triton 3.x, A100/4090
- Install: `pip install -U pip && pip install -r requirements-cpu.txt` (CPU) or `-r requirements-gpu-cu121-torch24.txt` (GPU). For FineWeb‑Edu + GPT‑2 BPE: `pip install transformers datasets`.

## Test Matrix
1) NSA correctness (CPU-safe)
2) GPU routing sanity and optional Triton/FA‑2 parity (GPU)
3) Long-context probes (64k) (GPU)
4) Trainer readiness: single-process and DDP (CPU/GPU)
5) Bench/telemetry deliverables

## Commands and Expectations

### 1) NSA correctness (CPU-safe)
- Command:
  - `PYTHONPATH=. pytest -q -k "test_equiv_small or test_block_math or test_masks or test_group_consistency or test_decode_counters or test_selection_packed"`
- Expectation:
  - All selected tests PASS.
  - No GPU required; skips are acceptable where marked.
- Deliverables:
  - `artifacts/test-reports/cpu-correctness.txt` (captured output)

### 2) GPU routing + optional Triton/FA‑2 (GPU)
- Commands:
  - Routing snapshot: `PYTHONPATH=. python scripts/print_routing.py > artifacts/test-reports/routing.json`
  - Triton selection forward parity (opt-in; force on 4090):
    - `NSA_USE_TRITON_SEL=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_parity_gpu.py`
  - Triton selection backward parity (opt-in):
    - `NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_ALLOW_GRAD=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_backward_gpu.py`
  - FA‑2 availability probe: `python - << 'PY'\nfrom nsa.kernels.flash_wrappers import is_flash_varlen_available; print('fa2_varlen_available', is_flash_varlen_available())\nPY`
  - FA‑2 varlen parity (if available): `NSA_TEST_FA2=1 NSA_USE_FA2=1 PYTHONPATH=. pytest -q -k fa2_gpu_varlen`
- Expectations:
  - Routing JSON reflects CUDA available; Triton version aligned with Torch if installed.
  - Parity tests PASS when enabled; otherwise cleanly SKIP.
- Deliverables:
  - `artifacts/test-reports/triton_fwd.txt`, `triton_bwd.txt`, `fa2_probe.txt`, `fa2_varlen.txt`

### 3) Long-context probes (64k) (GPU)
- Commands:
  - 64k demo (tile prefill):
    - `PYTHONPATH=. python scripts/demo_64k.py --S 65536 --prefill_tile 4096 --rope_scale 8.0 --use_fa2 0 > artifacts/test-reports/demo_64k.txt`
  - Needle test:
    - `PYTHONPATH=. pytest -q nsa/tests/test_long_context_needle.py -k needle`
- Expectations:
  - Demo prints summary with device, timing, and read counters; no exceptions.
  - Needle test PASS on at least one GPU runner (SKIP acceptable on CPU-only).
- Deliverables:
  - `artifacts/test-reports/demo_64k.txt`, `artifacts/test-reports/needle_64k.txt`

### 4) Trainer readiness (single + DDP)
- Single-process smoke (CPU or GPU):
  - `CONFIG=configs/train_showcase.yaml PYTHONPATH=. python scripts/train_showcase.py > artifacts/train/train_showcase.txt`
  - Expectation: completes configured steps; writes `training.csv`, `val.csv` (optional), `metrics.json` under `artifacts/train_showcase`.
- DDP on 2 GPUs (GPU):
  - M7C: `CONFIG=configs/m7c_125m.yaml PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_showcase.py --dataset fineweb_edu > artifacts/train/m7c_ddp.txt`
  - Expectations:
    - Rank 0 logs contain `toks/s` column; `training.csv` has `step,loss,lr,toks_per_s`.
    - Checkpoints written at configured cadence by rank 0 only.
    - Loss decreases over time on synthetic or real data; exact PPL not gated.
- Deliverables:
  - `artifacts/train/train_showcase.txt`, `artifacts/train/m7c_ddp.txt`
  - `artifacts/m7c_125m/training.csv`, `val.csv` (if enabled), and the latest checkpoint path

### 5) Bench/telemetry
- Decode bench (GPU or CPU fallback if no CUDA):
  - `PYTHONPATH=. python bench/bench_decode.py --S_list 512,1024,2048,4096 --iters 32 --warmup 8 --csv artifacts/bench/decode.csv > artifacts/bench/decode.txt`
  - `python scripts/summarize_bench.py artifacts/bench/decode.csv > artifacts/bench/decode_summary.txt`
- Expectations:
  - CSV produced and summary renders a stable table; no header mismatches.
- Deliverables:
  - `artifacts/bench/decode.csv`, `artifacts/bench/decode_summary.txt`

## Reporting
- Bundle a zip/tar of the `artifacts/` directory with subfolders:
  - `test-reports/`, `train/`, `bench/`, plus any `tracked/` entries
- Post a short summary with:
  - Environment (GPU model, Torch/Triton versions), pass/fail counts, notable throughput numbers, checkpoint path(s).

## Known Skip/Flake Notes
- Triton selection is disabled by policy on SM 8.9 unless forced via `NSA_TRITON_SEL_FORCE=1`.
- FA‑2 parity requires an environment with FA‑2 varlen installed; otherwise SKIP is acceptable.

---

Appendix: Quick Setup
- CPU: `python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements-cpu.txt`
- GPU: `python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements-gpu-cu121-torch24.txt`
- Extras for data/tokenizer: `pip install transformers datasets`

