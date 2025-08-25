# Env & Build Guardrails

Goals: reproducible env on A100, deterministic behavior, early failure on mispairings.

What to implement
- Python 3.11/3.12; torch 2.4 + cu121 pins (see `requirements-gpu-cu121-torch24.txt`).
- `scripts/_env_guard.py` to validate: GPU arch, Torch+CUDA versions, TF32/bf16 policy, tokenizer availability, dataset/cache paths.
- Default policies: enable TF32 on matmul; autocast bf16 on A100; disable torch.compile by default.
- Bootstrap snippet: print device, dtype, TF32 flags, deterministic settings at startup.

Checks
- Device: sm80+ detected; error if consumer Ada (SM 8.9) unless in bench mode.
- Data: `HF_HOME`, `HF_DATASETS_CACHE`, `NSA_DATA_CACHE` present/writable if streaming.
- Repro: capture `pip freeze` to `artifacts/env/constraints.txt`.

Deliverables
- `scripts/_env_guard.py` (executable as module) — DONE
- Constraints file template if needed — PENDING

