Contributing to NSA‑Vibe

Thank you for your interest in contributing! This repo aims to be small, clear, and reproducible.

Environment
- Python 3.11+ (3.12 on GPU pods)
- Create venv: `python -m venv .venv && . .venv/bin/activate`
- CPU dev: `pip install -r requirements-cpu.txt`
- GPU dev (CUDA 12.1 + Torch 2.4): `pip install -r requirements-gpu-cu121-torch24.txt`

Quality
- Lint: `ruff check .`
- Type check: `mypy -p nsa`
- Tests (CPU-safe core):
  - `PYTHONPATH=. pytest -q -k "test_masks or test_block_math or test_equiv_small or test_decode_counters or test_group_consistency"`
- GPU opt-in tests (skip cleanly otherwise):
  - FA‑2 parity/backward: `NSA_TEST_TRAIN=1 PYTHONPATH=. pytest -q nsa/tests/test_backward_varlen.py`
  - Triton selection parity/backward (4090 requires force):
    - `NSA_USE_TRITON_SEL=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_parity_gpu.py`
    - `NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_ALLOW_GRAD=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_backward_gpu.py`

Benches
- Decode bench CSV: `PYTHONPATH=. python bench/bench_decode.py --S_list 512,1024 --iters 16 --warmup 4 --csv artifacts/decode_test.csv`
- Summarize: `python scripts/summarize_bench.py artifacts/decode_test.csv`
- Bench summary (screenshot-friendly): `bash scripts/bench_report.sh`

Outputs and cleanup
- Generated files go under `artifacts/` (ignored).
- Keep human-readable reports checked in (e.g., NSA-*-Report-*.md).
- Cleanup helper: `bash scripts/cleanup_repo.sh` (add `--apply` to remove).

Style & scope
- Prefer small, focused changes.
- Preserve tests and deterministic behavior (fixed seeds in CI).
- Avoid introducing Triton in M0 paths; follow PRD/ADR for defaults and flags.

Thank you for helping improve NSA‑Vibe!
