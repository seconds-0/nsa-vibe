GPU Test Plan: NSA Selection + FA‑2 Parity and Benches

Scope
- Validate Triton selection forward/backward parity against packed SDPA.
- Validate FA‑2 sliding/compressed parity and grad paths on GPU.
- Capture decode/prefill performance benches with environment metadata.

Target Environment
- OS: Linux with CUDA (e.g., Ubuntu 22.04).
- GPU: Any CUDA‑capable device. Notes for RTX 4090 (SM 8.9) below.
- Python: 3.10+ recommended.
- Toolchain: gcc/g++, build‑essentials, recent NVIDIA driver.

Prerequisites
- CUDA driver/toolkit installed; `nvidia-smi` returns the GPU.
- Internet access to install wheels (PyPI).
- Git access to the repo.

Version Pairing (Torch ↔ Triton)
- torch 2.2 → triton 2.2
- torch 2.3 → triton 2.3 (recommended pin: `triton>=2.3,<3.1`)
- torch 2.4+ → triton 3.x
FA‑2 Install Notes (RTX 4090)
- For PyTorch 2.3.1, FA‑2 wheels may be unavailable; prefer Torch 2.4+ for prebuilt wheels.
- If you must stay on 2.3.1, ensure Ninja is installed and limit parallelism:
  - `pip install ninja && ninja --version`
  - `MAX_JOBS=2 pip install flash-attn --no-build-isolation`
See Documentation/Guides/FA2-Install-Guide.md for details.
Ensure the installed Triton matches your Torch minor version.

Repo Setup
1) Clone and enter repo
   - `git clone <repo_url> && cd nsa-vibe`
   - `git rev-parse --short HEAD`  # record commit

2) Create venv and install deps
   - `python3.10 -m venv .venv && . .venv/bin/activate`
   - `pip install -U pip wheel setuptools`
   - Option A (Torch 2.3 baseline, CPU-safe): `pip install -r requirements-cpu.txt`
   - Option B (Torch 2.4 wheels, CUDA 12.1 Linux):
     - `pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0+cu121`
     - `pip install "triton>=3.0,<3.2"`
     - `pip install flash-attn`
   - Expect: Torch/Triton paired (2.3↔2.3.x, 2.4↔3.x). FA‑2 available on 2.4+ via wheels.

3) Quick smoke on CPU (on the GPU host)
   - `PYTHONPATH=. pytest -q`  # many tests will skip; ensures imports OK

Environment Toggles (reference)
- Triton selection
  - `NSA_USE_TRITON_SEL=1` enable wrapper
  - `NSA_TRITON_SEL_FORCE=1` override 4090 ADR for tests
  - `NSA_SEL_TRITON_ALLOW_GRAD=1` enable backward via custom autograd
  - `NSA_SEL_TRITON_MIN_L` threshold for using Triton vs packed SDPA (default ~4096)
- FA‑2
  - `NSA_USE_FA2=1` opt‑in
  - `NSA_FA2_MIN_LEN_WIN`, `NSA_FA2_MIN_LEN_CMP` small‑len cutoffs (default 16)
  - `NSA_FA2_FORCE_VARLEN=1` or `NSA_FA2_FORCE_DENSE=1` path forcing (debug)
- Debug/observability
  - `NSA_DEBUG_LOG=1` enable structured logs
  - `NSA_LOG_LIMIT=5` limit repeated tags
  - `NSA_DEBUG_TIMING=1` time buckets and packing

Important Note: RTX 4090 (SM 8.9)
- ADR forces Triton selection fallback by default.
- For test parity only, set `NSA_TRITON_SEL_FORCE=1`. Keep this to test scope; do not enable for production benches unless explicitly requested.

Test Matrix (GPU)
1) Triton selection forward parity (GPU)
   - `NSA_USE_TRITON_SEL=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_parity_gpu.py`

2) Triton selection backward parity (GPU)
   - `NSA_USE_TRITON_SEL=1 NSA_SEL_TRITON_ALLOW_GRAD=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. pytest -q nsa/tests/test_triton_sel_backward_gpu.py`

3) FA‑2 GPU varlen parity (opt‑in)
   - `NSA_TEST_FA2=1 NSA_USE_FA2=1 PYTHONPATH=. pytest -q -k fa2_gpu_varlen`

4) Backward/gradcheck/training smokes (opt‑in)
   - `NSA_TEST_TRAIN=1 PYTHONPATH=. pytest -q -k "backward_parity or gradcheck or train"`
   - Targets include: `test_backward_varlen.py`, `test_gradcheck_varlen.py`, `test_train_smoke.py`

5) CUDA wrapper (experimental, if built)
   - `PYTHONPATH=. pytest -q nsa/tests/test_sel_cuda_gpu.py`

Benches (GPU)
1) Decode bench
   - `PYTHONPATH=. python bench/bench_decode.py --B 1 --dim 256 --heads 8 --groups 2 --dk 32 --dv 32 --l 32 --d 16 --l_sel 64 --n_sel 16 --w 512 --S_list 512,1024,2048,4096 --iters 64 --warmup 8 --csv decode_gpu_final.csv --branch_force_mode env`
   - `python bench/summarize_decode_csv.py decode_gpu_final.csv`

2) FA‑2 bench (optional)
   - `NSA_USE_FA2=1 PYTHONPATH=. python bench/bench_fa2.py`

3) Triton selection bench (forward)
   - `NSA_USE_TRITON_SEL=1 NSA_TRITON_SEL_FORCE=1 PYTHONPATH=. python bench/bench_sel_triton.py`
   - Optional profile flags:
     - `NSA_SEL_TRITON_GROUP=1` group‑centric kernels
     - `NSA_DEBUG_TIMING=1` to log per‑bucket timings

Artifacts to Collect (Deliverables)
- Env metadata (text):
  - `nvidia-smi`
  - `python -c "import torch, triton; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('triton', triton.__version__)"`
  - `git rev-parse --short HEAD`
- Test outputs (text):
  - Logs for each test command above (stdout) with `NSA_DEBUG_LOG=1 NSA_LOG_LIMIT=5` for at least one run.
  - Summary: counts of passed/failed/skipped.
- Parity metrics (text):
  - For Triton selection parity tests, collect mean absolute error (MAE) logs if `NSA_DEBUG_COMPARE=1` is enabled or assert thresholds from tests.
- Bench results:
  - Decode bench stdout + generated CSV (`decode_gpu_final.csv`).
  - Triton selection bench stdout and CSV (if the bench writes one).
  - FA‑2 bench stdout (timings per configuration).

One-shot (alternative)
- `bash scripts/runner_oneshot.sh` collects env, routing, sanity, Triton FWD/BWD, decode bench + summary, FA‑2 probe (+optional), ranges sample, and training showcase into `artifacts/runner/<commit>/`.

Reporting Template
```
GPU Host: <hostname> / <GPU model> / Driver <ver>
Commit: <short-hash>
Torch/Triton/CUDA: torch <ver>, triton <ver>, cuda <ver>

Tests:
- Triton FWD parity: PASS/FAIL (<runtime>)
- Triton BWD parity: PASS/FAIL (<runtime>)
- FA‑2 varlen: PASS/FAIL (<runtime>)
- Grad/Train smokes: PASS/FAIL (<runtime>)

Benches:
- Decode bench: <summary numbers, lines>
- Triton selection bench: <summary>
- FA‑2 bench: <summary>

Artifacts:
- Attach logs (txt) and CSVs.
```

Operational Notes
- Keep `NSA_TRITON_SEL_FORCE=1` limited to tests on Ada (4090); default remains fallback to packed SDPA.
- If FA‑2 wheels are unavailable, FA‑2 tests will fallback/skip; note this in the report.
- Use `NSA_FA2_MIN_LEN_*` to avoid tiny‑length slowdowns when probing performance.
