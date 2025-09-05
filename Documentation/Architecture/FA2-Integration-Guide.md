FA‑2 Integration Guide (NSA)

What’s routed to FA‑2
- Optional only (OFF by default). When explicitly enabled and contracts pass:
  - Sliding window (win): dense FA‑2 (`flash_attn_func`), else SDPA masked.
  - Compressed (cmp): dense FA‑2 per‑row batches, else SDPA masked.
  - Selection: remains SDPA (packed/masked); future varlen FA‑2 optional.

Preconditions (runtime enforced)
- Device: CUDA on SM80/89/90 GPUs.
- Dtype: fp16 or bf16; fp32 falls back.
- Head dim: multiple of 8; ≤128 on SM8x by default (guarded), ≤256 on SM9x.
- Contiguity: q/k/v or qkv must be contiguous along packed dims.
- Varlen: `cu_seqlens_*` must be int32, monotonic, `(B+1)`, `cu[0]=0`, `cu[-1]=nnz`.

Fallback rules
- Contracts fail → SDPA path with detailed log when `NSA_DEBUG_TIMING=1`.
- Kernel import fails or raises → SDPA path (dense or masked) with log.

Flags
- `NSA_USE_FA2` (master switch), `NSA_USE_FA2_WIN`, `NSA_USE_FA2_CMP` (per‑branch). Default OFF.
- `NSA_FA2_MIN_LEN_WIN`, `NSA_FA2_MIN_LEN_CMP` (min lengths to choose FA‑2). Use large sentinels (e.g., 8192) until sweeps justify lower.
- `NSA_FA2_PREF_FP16` (prefer fp16 casts), `NSA_FA2_ALLOW_FP32` (rarely true).
- `NSA_FA2_ALLOW_D_GT_128` (permit D>128 on SM8x with caution; default 0).
- `NSA_FA2_AUDIT` (optional startup probe), `NSA_FA2_HARD_FAIL` (raise if eligible but blocked).

Thresholds (starting points; tune via benches)
- A100 (SM80):
  - D≤128: win min Tk ≈192; cmp min Lc ≈64
  - D=192: win ≈256; cmp ≈96
  - D=256: win ≈320; cmp ≈128
- H100 (SM90):
  - D≤128: win ≈128; cmp ≈48
  - D=192: win ≈160; cmp ≈64
  - D=256: win ≈192; cmp ≈96

Startup audit (optional)
- Log device, SM, torch/flash‑attn versions; probe a tiny forward and record pass/fail and fallback reason.

First responder checklist
- Illegal memory access: validate `cu_seqlens_*` and contiguity; run `scripts/repro_fa2_crash.py` to reproduce.
- Immediate FPE/NaNs: check dtype (fp32 drift) and strides; enforce `.to(fp16/bf16).contiguous()`.
- “not implemented for head_dim”: verify `%8==0` and SM‑specific upper limits.
