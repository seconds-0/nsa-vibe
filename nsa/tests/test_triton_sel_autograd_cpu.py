import os

import torch


def test_autograd_wrapper_fallback_grads_cpu():
    os.environ["NSA_USE_TRITON_SEL"] = "1"
    os.environ["NSA_SEL_TRITON_ALLOW_GRAD"] = "1"
    B, S, G, H, D, Dv = 2, 1, 1, 2, 8, 8
    S_kv = 32
    Q = torch.randn(B, S, G, H, D, requires_grad=True)
    K = torch.randn(B, G, S_kv, D, requires_grad=True)
    V = torch.randn(B, G, S_kv, Dv, requires_grad=True)
    ranges = torch.tensor(
        [
            [[[[0, 8], [16, 24]]]],
            [[[[4, 12], [20, 28]]]],
        ],
        dtype=torch.int64,
    )
    from nsa.core.attention_kernels import grouped_selection_attention_packed
    from nsa.kernels.triton_sel_kernel import selection_attention_triton

    out_tri = selection_attention_triton(Q, K, V, ranges)
    loss_tri = out_tri.square().sum()
    loss_tri.backward()
    gQ_tri, gK_tri, gV_tri = Q.grad.clone(), K.grad.clone(), V.grad.clone()

    # Reset grads
    Q.grad = None
    K.grad = None
    V.grad = None
    out_ref = grouped_selection_attention_packed(Q, K, V, ranges)
    loss_ref = out_ref.square().sum()
    loss_ref.backward()
    gQ_ref, gK_ref, gV_ref = Q.grad, K.grad, V.grad

    assert torch.allclose(gQ_tri, gQ_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(gK_tri, gK_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(gV_tri, gV_ref, atol=1e-5, rtol=1e-5)
