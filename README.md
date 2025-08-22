# NSA-Vibe — Native Sparse Attention

NSA-Vibe is a clean reference implementation of Native Sparse Attention (NSA) for decoder‑only Transformers. NSA combines three causal branches — Compressed (cmp), Selected (sel), and Sliding window (win) — with a learned gate, enforcing strict causality and group‑consistent selection (GQA). It runs everywhere with SDPA; FA‑2 is optional for cmp/win; Triton selection is opt‑in and disabled by policy on RTX 4090.

Highlights
- Three branches + learned gate, strict causality, GQA group consistency (Eq.9/10).
- CPU fallback via SDPA; FA‑2 opt‑in for cmp/win; robust fallbacks and parity tests.
- Per‑branch counters and gate stats for observability; long‑context “needle” test.
- Training‑ready: backward passes for all branches; tiny showcase trainer included.

## Quick Start

Setup (CPU or GPU)
- Python 3.11 recommended (3.12 on CUDA GPU pods)
- CPU: `python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements-cpu.txt`
- GPU (CUDA 12.1 + Torch 2.4): `python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements-gpu-cu121-torch24.txt`

Core tests (CPU-safe)
- `PYTHONPATH=. pytest -q -k "test_masks or test_block_math or test_equiv_small or test_decode_counters or test_group_consistency"`

Bench summary (screenshot‑ready)
- `bash scripts/bench_report.sh`  # prints env, PASS lines, needle checks, decode table, csv head

Decode bench (CSV + summary)
- `PYTHONPATH=. python bench/bench_decode.py --S_list 512,1024,2048,4096 --iters 64 --warmup 8 --csv artifacts/decode_gpu_final.csv`
- `python scripts/summarize_bench.py artifacts/decode_gpu_final.csv`

Prime Intellect Runbook (SSH)
- Bootstrap GPU env: `bash scripts/prime_bootstrap.sh`
- Start training (auto‑picks config by VRAM, handles 1–2 GPUs): `bash scripts/train_m7c_prime.sh`
- Logs: `artifacts/train_runs/` and `artifacts/m7c_125m/` (training.csv, checkpoints)

End-to-end test runner
- `PYTHONPATH=. python scripts/run_m7_readiness.py --out artifacts/run_$(date +%Y%m%d-%H%M)`
- Optional flags: `--enable-triton` (parity tests), `--enable-fa2` (if available), `--skip-long` (omit 64k probes)

Training showcase (tiny byte‑LM)
- `CONFIG=configs/train_showcase.yaml PYTHONPATH=. python scripts/train_showcase.py`

## Execution Routing

How NSA chooses SDPA vs FA‑2 vs Triton, and all device/flag guards:
- Documentation/Guides/Execution-Routing.md

Notes
- On RTX 4090 (SM 8.9), Triton selection is disabled by ADR. Use `NSA_TRITON_SEL_FORCE=1` only for parity tests.
- Torch↔Triton: Torch 2.3 ↔ Triton 2.3; Torch 2.4+ ↔ Triton 3.x.

## What’s Implemented
- NSAAttention: cmp/sel/win branches + GateMLP (zero‑init last layer, τ=1.0), strict causal masks.
- Selection semantics: CSR Eq.9 mapping, group reduction (Eq.10), deterministic top‑n, forced initial + locals, range merging, ≤ t clamping.
- Caches: separate `K_sel,V_sel` and `K_win,V_win`; compressed emission every d after warmup l.
- Backward: autograd paths for all branches; selection has an analytical backward in the wrapper.
- Tests: causality, Eq.9/10 consistency, small‑S equivalence, decode counters, selection backward, FA‑2 parity (opt‑in), long‑context needle.

## Reports (kept in repo)
- NSA-Accuracy-Validation-Report-224091b.md — decode‑only reads accuracy fix validation
- NSA-64k-Needle-Test-Report-224091b5.md — 64k long‑context selection PASS
- Oneshot-Validation-Report-2025-08-21.md — end‑to‑end snapshot
  (Corresponding artifact folders live under `artifacts/tracked/`.)

## Repo Layout
- `nsa/` core modules (attention, selection, packing, rope, flags) and tests under `nsa/tests/`
- `bench/` decode and selection benches; CSV summarizer
- `configs/` base + training showcase config
- `scripts/` helpers: bench_report.sh, train_showcase.py, cleanup_repo.sh, etc.
- `Documentation/` PRD, plans (M0–M6), guides, test plans

## Contributing
- Lint/type check: `ruff check .` and `mypy -p nsa`
- Tests: `PYTHONPATH=. pytest -q` (GPU‑optional suites skip cleanly)
- Clean outputs: `bash scripts/cleanup_repo.sh` (add `--apply` to remove)

## Advanced (GPU)
- FA‑2 install: Documentation/Guides/FA2-Install-Guide.md (optional)
- Triton selection parity (opt‑in; 4090 needs force):
  - `NSA_USE_TRITON_SEL=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_parity_gpu.py`
  - Backward: `NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_ALLOW_GRAD=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_backward_gpu.py`

If you’re new to the repo, start with Documentation/Guides/Start-Here.md and PRD.md.
