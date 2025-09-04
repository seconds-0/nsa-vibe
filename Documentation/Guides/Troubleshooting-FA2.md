Troubleshooting FA‑2 (NSA)

Symptom → probable cause → fix

- Illegal memory access
  - Cause: malformed `cu_seqlens_*` (dtype/monotonic/size) or non‑contiguous packed buffers.
  - Fix: enforce int32 `(B+1)`, `cu[0]=0`, `cu[-1]=nnz`, monotonic; ensure `.contiguous()`; repro via `scripts/repro_fa2_crash.py`.

- Immediate FPE on H100
  - Cause: dtype drift to fp32 or non‑contiguous views.
  - Fix: cast to fp16/bf16 and `.contiguous()` before FA‑2 calls.

- NaNs after FA‑2
  - Cause: tiny windows + extreme logits; numerical instabilities.
  - Fix: route to SDPA for very small lengths (e.g., T<16) and record inputs.

- “not implemented for head_dim …”
  - Cause: head_dim not multiple of 8 or exceeds SM limits.
  - Fix: gate by `%8==0`; SM8x default ≤128 (optionally ≤256 behind `NSA_FA2_ALLOW_D>128`); SM9x ≤256.

How to capture diagnostics
- Export: `NSA_DEBUG_TIMING=1 NSA_SDPA_AUDIT=1`.
- Run repro: `PYTHONPATH=. uv run -q python scripts/repro_fa2_crash.py`.
- Artifacts: see `artifacts/2025-09-04/fa2_harden/` (JSON env + results, CSV from benches).

