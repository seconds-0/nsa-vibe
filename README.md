# NSA-Vibe

This repository implements Native Sparse Attention (NSA) with Triton-accelerated selection kernels and FA-2 integration.

## Kernel Status (M4)
- Triton selection forward is implemented (dense + varlen) with guards and fallbacks.
- Broadcasting issues on Triton 3.1.0 were fixed by explicit 2D pointer loads.
- Backward pass is currently routed through packed SDPA via a custom autograd wrapper; Triton backward will land in M5.
- Until M4 benches land, sel_triton_min_L should remain conservative; SDPA fallback is preserved.

Known limitations:
- Supported dtypes: fp16/bf16. Alignment requirement by default (D and Dv multiples of 8).
- Training uses packed SDPA backward unless NSA_SEL_TRITON_ALLOW_GRAD=1 (which still calls packed backward).

## Diagnostics
- NSA_DEBUG_TIMING=1 per-bucket timing
- NSA_DEBUG_SHAPES=1 shapes/strides logging
- NSA_DEBUG_COMPARE=1 parity MAE logging

See Documentation/M4-Triton-Selection-Test-Plan.md for GPU validation/bench steps.

## Execution Routing

For how NSA chooses SDPA vs FA‑2 vs Triton based on device/flags and safe fallbacks, see:
- `Documentation/Guides/Execution-Routing.md`

Notes:
- On RTX 4090 (SM 8.9), Triton selection is disabled by default per ADR. Use `NSA_TRITON_SEL_FORCE=1` only for parity tests.
- Torch↔Triton versions are coupled: Torch 2.3 ↔ Triton 2.3; Torch 2.4+ ↔ Triton 3.x.

## Runner Quickstart (GPU)
- Create venv and install (choose one):
  - Torch 2.3 baseline (CPU-safe): `python3.10 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements-cpu.txt`
  - Torch 2.4 wheels (CUDA 12.1 Linux): `python3.10 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0+cu121 && pip install "triton>=3.0,<3.2" && pip install flash-attn`
    - Or: `pip install --index-url https://download.pytorch.org/whl/cu121 -r requirements-gpu-cu121-torch24.txt`
- CPU sanity on GPU host: `PYTHONPATH=. pytest -q`
- Routing snapshot: `python scripts/print_routing.py`
- Triton forward parity (4090 requires force): `NSA_USE_TRITON_SEL=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_parity_gpu.py`
- Triton backward parity: `NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_ALLOW_GRAD=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_backward_gpu.py`
- Decode bench (env-based branch forcing): `PYTHONPATH=. python bench/bench_decode.py --S_list 512,1024,2048,4096 --iters 64 --warmup 8 --csv decode_gpu_final.csv --branch_force_mode env`
- Summarize: `python bench/summarize_decode_csv.py decode_gpu_final.csv`

See also:
- Documentation/Guides/Decode-Benchmark-Guide.md
- Documentation/Test-Plans/GPU-Test-Plan.md
- Documentation/Runbooks/Runner-Engineer-Checklist.md
- Training showcase: `CONFIG=configs/train_showcase.yaml python scripts/train_showcase.py`
- Config check: `python scripts/check_config.py configs/base.yaml`
- Start here: Documentation/Guides/Start-Here.md
- Remote GPU (RunPod): Documentation/Guides/Remote-GPU-Runner.md

FA‑2 install (optional, GPU hosts)
- Guide: Documentation/Guides/FA2-Install-Guide.md
- Quick install helper: `bash scripts/install_fa2.sh 2`  # uses MAX_JOBS=2 by default

One-shot runner (GPU host)
- `bash scripts/runner_oneshot.sh`  # writes artifacts under artifacts/runner/<commit>/

Open a PR (optional)
- `bash scripts/open_pr.sh`  # uses gh CLI if available; otherwise prints instructions

Formatting/lint (optional, pre-commit)
- Install hooks: `pip install pre-commit && pre-commit install`
- Hooks include: ruff, black, isort, mypy. If unfamiliar: they auto-format and lint Python to keep quality high.
