import torch

from nsa.core.attention_kernels import (
    batched_causal_attention_compressed,
    grouped_selection_attention,
    sliding_window_attention,
)


def test_gradcheck_sliding_cpu_tiny():
    B, S, G, h, Dk, Dv, w = 1, 4, 1, 1, 8, 8, 3
    Q = torch.randn(B, S, G, h, Dk, dtype=torch.double, requires_grad=True)
    K = torch.randn(B, G, S, Dk, dtype=torch.double, requires_grad=True)
    V = torch.randn(B, G, S, Dv, dtype=torch.double, requires_grad=True)
    func = lambda *args: sliding_window_attention(*args, w).to(torch.double)
    assert torch.autograd.gradcheck(func, (Q, K, V), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_compressed_cpu_tiny():
    B, S, G, h, Dk, Dv, l, d = 1, 6, 1, 1, 8, 8, 4, 2
    Q = torch.randn(B, S, G, h, Dk, dtype=torch.double, requires_grad=True)
    S_cmp = (S - l) // d + 1
    Kc = torch.randn(B, G, S_cmp, Dk, dtype=torch.double, requires_grad=True)
    Vc = torch.randn(B, G, S_cmp, Dv, dtype=torch.double, requires_grad=True)
    func = lambda *args: batched_causal_attention_compressed(*args, l, d).to(torch.double)
    assert torch.autograd.gradcheck(func, (Q, Kc, Vc), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_selection_cpu_tiny():
    B, S, G, h, Dk, Dv = 1, 2, 1, 1, 8, 8
    Q = torch.randn(B, S, G, h, Dk, dtype=torch.double, requires_grad=True)
    K = torch.randn(B, G, 16, Dk, dtype=torch.double, requires_grad=True)
    V = torch.randn(B, G, 16, Dv, dtype=torch.double, requires_grad=True)
    ranges = torch.tensor(
        [
            [[[0, 4], [8, 12]]],
            [[[4, 8], [12, 16]]],
        ],
        dtype=torch.int64,
    ).unsqueeze(0)
    func = lambda *args: grouped_selection_attention(*args).to(torch.double)
    assert torch.autograd.gradcheck(func, (Q, K, V, ranges), eps=1e-6, atol=1e-4, rtol=1e-3)
