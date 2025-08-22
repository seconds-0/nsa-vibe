Runner Engineer Checklist

Scope: Execute GPU validation and benches while devs iterate locally. This checklist consolidates exact commands, env toggles, and deliverables.

Environment
- Python: 3.11+ (3.12 on GPU pods)
- OS: Linux with CUDA (nvidia-smi works)
- GPU: RTX 4090/A100/H100
- Versions: torch 2.3.*, triton 2.3.* (flash-attn 2.x on Linux)

Setup
- git clone <repo> && cd nsa-vibe
- python3.11 -m venv .venv && . .venv/bin/activate
- pip install -U pip wheel setuptools
- Option A (Torch 2.3 baseline, CPU-safe): pip install -r requirements-cpu.txt
- Option B (Torch 2.4 wheels, CUDA 12.1 Linux):
  - pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0+cu121
  - pip install "triton>=3.0,<3.2"
  - pip install flash-attn  # uses prebuilt wheel on Torch 2.4+
- Record: git rev-parse --short HEAD

Quick Sanity (CPU on the GPU box)
- PYTHONPATH=. pytest -q  # many tests will skip; expect PASS/skip
- python scripts/print_routing.py  # snapshot of routing config

GPU Tests (Triton / FA‑2)
- Triton selection forward parity (4090 requires force flag):
  - NSA_USE_TRITON_SEL=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_parity_gpu.py
- Triton selection backward parity:
  - NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_ALLOW_GRAD=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_backward_gpu.py
- FA‑2 varlen parity (optional; requires flash-attn):
  - NSA_TEST_FA2=1 NSA_USE_FA2=1 PYTHONPATH=. pytest -q -k fa2_gpu_varlen

Decode Bench (GPU)
- Baseline, env-based branch forcing for single-branch timings:
  - PYTHONPATH=. python bench/bench_decode.py --B 1 --dim 256 --heads 8 --groups 2 --dk 32 --dv 32 \
    --l 32 --d 16 --l_sel 64 --n_sel 16 --w 512 --S_list 512,1024,2048,4096 --iters 64 --warmup 8 \
    --csv decode_gpu_final.csv --branch_force_mode env
- Summarize:
  - python bench/summarize_decode_csv.py decode_gpu_final.csv

Artifacts to Collect
- Env metadata: nvidia-smi; torch/triton/cuda versions; git short-hash
- Test logs: stdout for each command (with NSA_DEBUG_LOG=1 NSA_LOG_LIMIT=5 for one representative run)
- CSVs: decode_gpu_final.csv (+ any others produced)
- Summaries: output of bench/summarize_decode_csv.py

Expected Outcomes (PASS criteria)
- Triton parity tests: PASS on forward; backward PASS when NSA_SEL_TRITON_ALLOW_GRAD=1
- FA‑2 tests: PASS or SKIPPED when wheels unavailable; note status
- Decode bench: CSV produced with header [S, ms_total, ms_cmp, ms_sel, ms_win, reads_actual, reads_expected]

Notes
- On RTX 4090 (SM 8.9), keep NSA_TRITON_SEL_FORCE=1 scoped to tests only; default runtime falls back to packed SDPA per ADR.
- If FA‑2 not available, note it and proceed; decode bench does not require FA‑2.
