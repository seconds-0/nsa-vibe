Title: FA‑2 vs SDPA on A100 (SM80) — Notes and Repro

Scope
- Hardware: NVIDIA A100 (SM80). Variants: 40GB SXM4, 80GB (PCIe/SXM4).
- Software: PyTorch 2.4–2.5, CUDA 12.x, flash-attn 2.5.x.
- Workloads: NSA M7 prefill/decode shapes (seq 1–2K, heads 8–64, d_k 64–192), batch 1–2.

Summary
- For current NSA shapes on A100, PyTorch SDPA paths (flash/mem_efficient) match or beat FA‑2.
- Net result: we disable FA‑2 by default on A100 to avoid unnecessary kernel overhead and variance.

Reproduction
- Prefill: `PYTHONPATH=. uv run python bench/bench_prefill.py --config configs/base.yaml`
- Decode:  `PYTHONPATH=. uv run python bench/bench_decode.py  --config configs/base.yaml`
- Force FA‑2 for probe (do not use for final numbers):
  `NSA_USE_FA2=1 NSA_FA2_MIN_LEN_WIN=1 NSA_FA2_MIN_LEN_CMP=1 ...`

Artifacts
- Use CI perf guard (PR42) to capture decode totals per commit; store A100 baselines under `baselines/a100_decode_guard.json`.
- Attach canary results to the PR when available; avoid committing hard numbers due to hardware variance.

Policy
- A100 defaults: `runtime.fa2_enabled: false`, `fa2_min_len_*: -1`.
- Precedence: env > runtime config > code default. See production runbook for details.

