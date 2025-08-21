import torch


def test_selection_backward_reference_cpu_parity():
    # Small deterministic shape
    torch.manual_seed(0)
    B, S, G, h, D, Dv = 1, 2, 1, 2, 8, 8
    S_kv = 24
    Q = torch.randn(B, S, G, h, D, requires_grad=True)
    K = torch.randn(B, G, S_kv, D, requires_grad=True)
    V = torch.randn(B, G, S_kv, Dv, requires_grad=True)
    # Two spans per row with overlap-free segments
    ranges = torch.tensor(
        [
            [[[0, 6], [12, 18]]],
            [[[4, 10], [18, 22]]],
        ],
        dtype=torch.int64,
    ).unsqueeze(0)  # [B,S,G,n,2]

    from nsa.core.attention_kernels import grouped_selection_attention
    from nsa.kernels.triton_sel_kernel import selection_attention_backward_reference

    # Forward via packed reference
    O = grouped_selection_attention(Q, K, V, ranges).contiguous()
    # Make an explicit upstream gradient and use a scalar loss to avoid view/version issues
    dO = torch.randn_like(O)
    (O * dO).sum().backward()
    gQ_ref, gK_ref, gV_ref = Q.grad.clone(), K.grad.clone(), V.grad.clone()

    # Backward via reference implementation
    dQ_ref, dK_ref, dV_ref = selection_attention_backward_reference(Q, K, V, ranges, dO)

    # Compare
    assert torch.allclose(dQ_ref, gQ_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dK_ref, gK_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dV_ref, gV_ref, atol=1e-5, rtol=1e-5)
