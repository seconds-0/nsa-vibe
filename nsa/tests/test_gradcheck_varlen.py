import os

import pytest
import torch

from nsa.core.attention_kernels import (
    batched_causal_attention_compressed_masked,
    sliding_window_attention_masked,
)

RUN_GC = (
    os.getenv("NSA_TEST_TRAIN", "0").lower() in ("1", "true", "yes") and torch.cuda.is_available()
)


@pytest.mark.skipif(
    not RUN_GC, reason="Gradcheck (GPU opt-in); set NSA_TEST_TRAIN=1 and ensure CUDA"
)
def test_gradcheck_sliding_tiny():
    # Tiny shapes for gradcheck
    B, S, G, h, Dk, Dv, w = 1, 4, 1, 1, 8, 8, 3
    device = torch.device("cuda")
    Q = torch.randn(B, S, G, h, Dk, dtype=torch.double, device=device, requires_grad=True)
    K = torch.randn(B, G, S, Dk, dtype=torch.double, device=device, requires_grad=True)
    V = torch.randn(B, G, S, Dv, dtype=torch.double, device=device, requires_grad=True)

    def func(*args):
        return sliding_window_attention_masked(*args, w).to(torch.double)

    assert torch.autograd.gradcheck(func, (Q, K, V), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(
    not RUN_GC, reason="Gradcheck (GPU opt-in); set NSA_TEST_TRAIN=1 and ensure CUDA"
)
def test_gradcheck_compressed_tiny():
    B, S, G, h, Dk, Dv, l, d = 1, 6, 1, 1, 8, 8, 4, 2
    device = torch.device("cuda")
    Q = torch.randn(B, S, G, h, Dk, dtype=torch.double, device=device, requires_grad=True)
    S_cmp = (S - l) // d + 1
    Kc = torch.randn(B, G, S_cmp, Dk, dtype=torch.double, device=device, requires_grad=True)
    Vc = torch.randn(B, G, S_cmp, Dv, dtype=torch.double, device=device, requires_grad=True)

    def func(*args):
        return batched_causal_attention_compressed_masked(*args, l, d).to(torch.double)

    assert torch.autograd.gradcheck(func, (Q, Kc, Vc), eps=1e-6, atol=1e-4, rtol=1e-3)
