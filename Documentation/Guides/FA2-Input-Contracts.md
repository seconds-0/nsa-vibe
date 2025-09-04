FA‑2 Input Contracts (NSA)

Scope: FA‑2 fast paths used by NSA for compressed and sliding window decode (dense + varlen/kvpacked). Selection remains SDPA for now.

Device / SW
- GPUs: Ampere (SM80), Ada (SM89), Hopper (SM90).
- Torch ≥ 2.4 (CUDA ≥ 12.1). flash-attn ≥ 2.5.5 recommended.
- Kernel entry points: `flash_attn_func`, `flash_attn_qkvpacked_func`, and varlen siblings.

Dtypes
- Allowed: fp16, bf16. Disallowed: fp32 (force SDPA fallback).
- Numerics: prefer fp16 for throughput unless model requires bf16.

Head dimension
- `head_dim % 8 == 0` (required).
- SM80/SM89: allow up to 128 by default; permit 192/256 only with `NSA_FA2_ALLOW_D_GT_128=1`.
- SM90: allow up to 256 by default.

Contiguity / strides
- Dense/QKV‑packed: inputs must be contiguous in the last 2 dims `(nheads, head_dim)` and the QKV packing dimension.
- Varlen packed views must be contiguous slices; no strided/narrowed non‑unit strides along the packed dim.

Varlen (`flash_attn_varlen_*`)
- `qkv_unpadded`: shape `(nnz, 3, nheads, head_dim)` or `(nnz, nheads, head_dim)` depending on function.
- `cu_seqlens`: `int32`, shape `(B+1,)`, monotonic non‑decreasing, `cu[0]=0`, `cu[-1]=nnz`.
- For `kvpacked`, `cu_seqlens_k` may differ from `cu_seqlens_q` (prefill vs decode).

Causal semantics & windows
- NSA decode often slices K/V for strict causality; passing `causal=False` is valid iff K/V are sliced to `≤ t`.
- For dense full‑length inputs (no slicing), set `causal=True` to avoid off‑by‑one on the last row.
- Sliding window (`window_size=(L,R)`): pass inclusive `[i-L, i+R]`. With NSA’s win branch masking to `≤ t`, keep `causal=True` unless you pre‑slice K.

Cheap runtime asserts to add
- dtype check: {fp16,bf16}; else hard fallback.
- head_dim: `%8==0` and `≤ 128 (SM8x)` or `≤ 256 (SM9x)` by default.
- contiguity: `.is_contiguous()` on q/k/v or qkv along the packed dims.
- varlen: `cu.dtype==torch.int32`; `cu.numel()==B+1`; `cu[0]==0`; `torch.all(cu[1:]>=cu[:-1])`.
- decode safety: when passing `causal=False`, assert that the K slice ends at t.
- batch size: if effective batch `B > 1024`, require `NSA_FA2_ALLOW_LARGE_BATCH=1` or fallback.
