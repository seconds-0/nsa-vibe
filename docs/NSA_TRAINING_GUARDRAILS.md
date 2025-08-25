# NSA Training Guardrails (Interim)

Purpose: Reduce OOM risk until architectural fixes land.

Defaults (recommended)
- Sequence length cap: `S ≤ 512` for dim≈768.
- Selection backend: `NSA_USE_SEL_PACK=1`, `NSA_USE_SEL_MASK=0`.
- Sliding backend: prefer FA‑2 varlen; else parity/gather (avoid masked).
- Compressed backend: avoid masked; use parity or FA‑2 compressed if available.
- Allocator: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256`.

Rationale
- Avoid building O(S²) masks and large score tensors; reduce peak grad memory.
- Packed/varlen SDPA keeps memory bounded per row.

Validation Checklist
- Confirm `env.json` contains intended NSA_* flags.
- Peak `gpu_mem_reserved` < 35–40 GiB/GPU at S=512.
- No hangs at S=512; pass smoke steps including backward.

Escalation
- If S>512 is required ahead of fixes, consider head/group tensor parallel sharding inside NSA (engineering effort).

