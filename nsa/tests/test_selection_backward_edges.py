import torch


def _run_ref_backward(Q, K, V, ranges, dO):
    from nsa.kernels.triton_sel_kernel import selection_attention_backward_reference
    return selection_attention_backward_reference(Q, K, V, ranges, dO)


def _run_autograd(Q, K, V, ranges, dO):
    from nsa.core.attention_kernels import grouped_selection_attention_packed
    Qr = Q.detach().requires_grad_(True)
    Kr = K.detach().requires_grad_(True)
    Vr = V.detach().requires_grad_(True)
    Or = grouped_selection_attention_packed(Qr, Kr, Vr, ranges)
    gQ, gK, gV = torch.autograd.grad(Or, (Qr, Kr, Vr), dO, allow_unused=True)
    if gQ is None:
        gQ = torch.zeros_like(Q)
    if gK is None:
        gK = torch.zeros_like(K)
    if gV is None:
        gV = torch.zeros_like(V)
    return gQ, gK, gV


def test_selection_backward_empty_ranges_cpu():
    torch.manual_seed(0)
    B, S, G, h, D, Dv = 1, 1, 1, 2, 8, 8
    S_kv = 16
    Q = torch.randn(B, S, G, h, D)
    K = torch.randn(B, G, S_kv, D)
    V = torch.randn(B, G, S_kv, Dv)
    Q.requires_grad_(True); K.requires_grad_(True); V.requires_grad_(True)
    # All ranges empty
    ranges = torch.tensor([[[[[0, 0], [4, 4]]]]], dtype=torch.int64)
    dO = torch.randn(B, S, G, h, Dv)
    dQ_ref, dK_ref, dV_ref = _run_ref_backward(Q, K, V, ranges, dO)
    # With empty ranges, forward is zero and disconnected; autograd has no path.
    # Validate that reference grads are zeros of correct shape.
    assert torch.allclose(dQ_ref, torch.zeros_like(Q))
    assert torch.allclose(dK_ref, torch.zeros_like(K))
    assert torch.allclose(dV_ref, torch.zeros_like(V))


def test_selection_backward_adjacent_and_duplicates_cpu():
    torch.manual_seed(1)
    B, S, G, h, D, Dv = 1, 2, 1, 2, 8, 8
    S_kv = 24
    Q = torch.randn(B, S, G, h, D)
    K = torch.randn(B, G, S_kv, D)
    V = torch.randn(B, G, S_kv, Dv)
    Q.requires_grad_(True); K.requires_grad_(True); V.requires_grad_(True)
    # Adjacent + duplicate spans
    ranges = torch.tensor(
        [
            [[[0, 4], [4, 8]]],    # adjacent
            [[[8, 12], [12, 16]]], # adjacent
        ],
        dtype=torch.int64,
    ).unsqueeze(0)  # [B,S,G,n,2]
    dO = torch.randn(B, S, G, h, Dv)
    dQ_ref, dK_ref, dV_ref = _run_ref_backward(Q, K, V, ranges, dO)
    dQ_auto, dK_auto, dV_auto = _run_autograd(Q, K, V, ranges, dO)
    assert torch.allclose(dQ_ref, dQ_auto, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dK_ref, dK_auto, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dV_ref, dV_auto, atol=1e-5, rtol=1e-5)


def test_selection_backward_long_L_bucket_cpu():
    torch.manual_seed(2)
    B, S, G, h, D, Dv = 1, 1, 1, 2, 8, 8
    S_kv = 64
    Q = torch.randn(B, S, G, h, D)
    K = torch.randn(B, G, S_kv, D)
    V = torch.randn(B, G, S_kv, Dv)
    Q.requires_grad_(True); K.requires_grad_(True); V.requires_grad_(True)
    # One long span (L=32)
    ranges = torch.tensor([[[[[10, 42], [0, 0]]]]], dtype=torch.int64)
    dO = torch.randn(B, S, G, h, Dv)
    dQ_ref, dK_ref, dV_ref = _run_ref_backward(Q, K, V, ranges, dO)
    dQ_auto, dK_auto, dV_auto = _run_autograd(Q, K, V, ranges, dO)
    assert torch.allclose(dQ_ref, dQ_auto, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dK_ref, dK_auto, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dV_ref, dV_auto, atol=1e-5, rtol=1e-5)
