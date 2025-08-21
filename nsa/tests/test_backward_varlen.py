import os

import pytest
import torch

from nsa.core.attention_kernels import (
    batched_causal_attention_compressed_masked,
    compressed_attention_fa2,
    sliding_window_attention_fa2,
    sliding_window_attention_masked,
)

RUN_GPU = (
    os.getenv("NSA_TEST_TRAIN", "0").lower() in ("1", "true", "yes") and torch.cuda.is_available()
)


@pytest.mark.skipif(
    not RUN_GPU, reason="GPU opt-in tests; set NSA_TEST_TRAIN=1 and ensure CUDA available"
)
def test_backward_parity_sliding_gpu():
    B, S, G, h, Dk, Dv, w = 1, 16, 1, 2, 64, 64, 8
    device = torch.device("cuda")
    Q = torch.randn(B, S, G, h, Dk, device=device, requires_grad=True)
    K = torch.randn(B, G, S, Dk, device=device, requires_grad=True)
    V = torch.randn(B, G, S, Dv, device=device, requires_grad=True)
    # Forward
    O_ref = sliding_window_attention_masked(Q, K, V, w)
    loss_ref = O_ref.pow(2).mean()
    loss_ref.backward()
    grads_ref = (Q.grad.clone(), K.grad.clone(), V.grad.clone())
    # Reset grads
    Q.grad = K.grad = V.grad = None
    O_fa2 = sliding_window_attention_fa2(Q, K, V, w)
    loss_fa2 = O_fa2.pow(2).mean()
    loss_fa2.backward()
    grads_fa2 = (Q.grad, K.grad, V.grad)
    mae = sum((a - b).abs().mean().item() for a, b in zip(grads_ref, grads_fa2)) / 3.0
    assert mae <= 2e-4


@pytest.mark.skipif(
    not RUN_GPU, reason="GPU opt-in tests; set NSA_TEST_TRAIN=1 and ensure CUDA available"
)
def test_backward_parity_compressed_gpu():
    B, S, G, h, Dk, Dv, l, d = 1, 16, 1, 2, 64, 64, 8, 4
    device = torch.device("cuda")
    Q = torch.randn(B, S, G, h, Dk, device=device, requires_grad=True)
    K_raw = torch.randn(B, G, S, Dk, device=device)
    V_raw = torch.randn(B, G, S, Dv, device=device)
    S_cmp = (S - l) // d + 1
    K_cmp = (
        torch.stack([K_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
        .detach()
        .requires_grad_(True)
    )
    V_cmp = (
        torch.stack([V_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
        .detach()
        .requires_grad_(True)
    )
    O_ref = batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
    loss_ref = O_ref.pow(2).mean()
    loss_ref.backward()
    grads_ref = (Q.grad.clone(), K_cmp.grad.clone(), V_cmp.grad.clone())
    Q.grad = K_cmp.grad = V_cmp.grad = None
    O_fa2 = compressed_attention_fa2(Q, K_cmp, V_cmp, l, d)
    loss_fa2 = O_fa2.pow(2).mean()
    loss_fa2.backward()
    mae = (
        sum(
            (a - b).abs().mean().item() for a, b in zip(grads_ref, (Q.grad, K_cmp.grad, V_cmp.grad))
        )
        / 3.0
    )
    assert mae <= 2e-4
