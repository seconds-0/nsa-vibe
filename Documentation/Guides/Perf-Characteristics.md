Performance Characteristics — NSA Execution Paths

Summary
- NSA routes attention paths based on device capability and flags with conservative defaults on RTX 4090. This guide summarizes expected characteristics and when each path is used.

Paths
- SDPA (packed/masked):
  - Pros: Robust, available everywhere, strong numerical stability
  - Cons: Not the fastest at long sequence lengths
  - Default on 4090 for selection and for cmp/win when FA‑2 not available
- FlashAttention‑2 (FA‑2), varlen/dense:
  - Pros: Significant speedups for long windows/many compressed tokens
  - Cons: Requires compatible wheels (Torch 2.4+ on 4090); numeric checks applied; falls back on non‑finite outputs
  - Tuned via thresholds (fa2_min_len_win/fa2_min_len_cmp)
- Triton selection (dense/varlen):
  - Pros: Group‑centric design can minimize KV reads; used for selection branch
  - Cons: Disabled by default on 4090 per ADR; parity/emergency fallback to SDPA
  - Enabled on compatible GPUs via flags/profiles; requires aligned D/Dv and fp16/bf16

Routing Summary
- 4090 (SM 8.9): Triton off by default; FA‑2 optional on Torch 2.4+ with thresholds; SDPA fallback always available.
- A100/H100: Triton allowed under guardrails; FA‑2 enabled via thresholds.

Observability
- Set NSA_DEBUG_TIMING=1 for timing logs on packing/buckets/kernels.
- Logs include bucket histograms and fallback reasons when triggered.

Bench Guidance
- Use bench/bench_decode.py with --branch_force_mode env to measure single-branch paths.
- Summarize with bench/summarize_decode_csv.py to track cmp/sel/win percentages and read counts.

