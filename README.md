# NSA-Vibe

This repository implements Native Sparse Attention (NSA) with Triton-accelerated selection kernels and FA-2 integration.

## Kernel Status (M4)
- Triton selection forward is implemented (dense + varlen) with guards and fallbacks.
- Broadcasting issues on Triton 3.1.0 were fixed by explicit 2D pointer loads.
- Backward pass is currently routed through packed SDPA via a custom autograd wrapper; Triton backward will land in M5.
- Until M4 benches land, sel_triton_min_L should remain conservative; SDPA fallback is preserved.

Known limitations:
- Supported dtypes: fp16/bf16. Alignment requirement by default (D and Dv multiples of 8).
- Training uses packed SDPA backward unless NSA_SEL_TRITON_ALLOW_GRAD=1 (which still calls packed backward).

## Diagnostics
- NSA_DEBUG_TIMING=1 per-bucket timing
- NSA_DEBUG_SHAPES=1 shapes/strides logging
- NSA_DEBUG_COMPARE=1 parity MAE logging

See Documentation/M4-Triton-Selection-Test-Plan.md for GPU validation/bench steps.
