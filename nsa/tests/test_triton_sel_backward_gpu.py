import os
import pytest
import torch


CUDA_OK = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_OK, reason="CUDA required for Triton backward parity test")
def test_triton_selection_backward_parity_gpu():
    # Force-enable Triton wrapper + backward path
    os.environ["NSA_USE_TRITON_SEL"] = "1"
    os.environ["NSA_SEL_TRITON_ALLOW_GRAD"] = "1"
    os.environ["NSA_TRITON_SEL_FORCE"] = "1"  # bypass SM 8.9 ADR in tests

    device = torch.device("cuda")
    B, S, G, h, D, Dv = 1, 2, 1, 2, 16, 16
    S_kv = 48
    torch.manual_seed(0)
    Q = torch.randn(B, S, G, h, D, device=device, requires_grad=True, dtype=torch.float16)
    K = torch.randn(B, G, S_kv, D, device=device, requires_grad=True, dtype=torch.float16)
    V = torch.randn(B, G, S_kv, Dv, device=device, requires_grad=True, dtype=torch.float16)
    # Two rows with different ranges (varlen + dense)
    ranges = torch.tensor(
        [
            [[[0, 12], [20, 28], [28, 36]]],
            [[[4, 20], [20, 20], [36, 44]]],
        ],
        dtype=torch.int64,
        device=device,
    ).unsqueeze(2)  # [S,1,n,2] -> [S,G,n,2]
    ranges = ranges.unsqueeze(0)  # [B,S,G,n,2]

    from nsa.kernels.triton_sel_kernel import selection_attention_triton
    from nsa.core.attention_kernels import grouped_selection_attention_packed

    # Triton forward + custom backward
    out_tri = selection_attention_triton(Q, K, V, ranges)
    loss_tri = (out_tri.float() ** 2).sum()
    loss_tri.backward()
    gQ_tri, gK_tri, gV_tri = Q.grad.clone(), K.grad.clone(), V.grad.clone()

    # Reset grads
    Q.grad = None
    K.grad = None
    V.grad = None

    # Reference path
    out_ref = grouped_selection_attention_packed(Q, K, V, ranges)
    loss_ref = (out_ref.float() ** 2).sum()
    loss_ref.backward()
    gQ_ref, gK_ref, gV_ref = Q.grad, K.grad, V.grad

    assert torch.allclose(gQ_tri, gQ_ref, atol=5e-3, rtol=5e-3)
    assert torch.allclose(gK_tri, gK_ref, atol=5e-3, rtol=5e-3)
    assert torch.allclose(gV_tri, gV_ref, atol=5e-3, rtol=5e-3)

