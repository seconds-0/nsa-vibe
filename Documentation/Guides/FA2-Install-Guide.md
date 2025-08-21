Flash-Attention (FA‑2) Install Guide — RTX 4090 and Similar

Summary
- On RTX 4090 with PyTorch 2.3.1, FA‑2 often compiles from source, which can hang if Ninja or parallel jobs are misconfigured.
- Easiest path: upgrade to a Torch version with available wheels (2.4+). Otherwise, constrain Ninja job count.

Option A — Fix Ninja and Limit Parallelism (stay on Torch 2.3.1)
```bash
# Ensure a working Ninja
pip uninstall -y ninja
pip install ninja
ninja --version  # should print a version and exit 0

# Constrain build parallelism to avoid RAM exhaustion
MAX_JOBS=2 pip install flash-attn --no-build-isolation
```

Notes
- Use MAX_JOBS=2..4 depending on system RAM (e.g., <96GB RAM → prefer 2–4).
- --no-build-isolation ensures Ninja from your environment is used.

Option B — Prefer Wheels (upgrade Torch 2.4+)
```bash
# Example for CUDA 12.1 wheels
pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0+cu121
pip install flash-attn
```

Why this helps
- Torch 2.4+ has prebuilt FA‑2 wheels for Ada (SM 8.9), avoiding long source builds.

Operational Tips
- Run on a CUDA Linux host; avoid building on CPU-only CI.
- If compile still fails or runs >45 minutes, verify Ninja and reduce MAX_JOBS.
- If wheels are unavailable, temporarily disable FA‑2 (our routing falls back to SDPA) and proceed with tests/benches.

