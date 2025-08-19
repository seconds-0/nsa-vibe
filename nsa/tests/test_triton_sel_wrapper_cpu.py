import os
import torch


def test_wrapper_falls_back_on_cpu():
    os.environ["NSA_USE_TRITON_SEL"] = "1"
    B, S, G, H, D, Dv = 2, 1, 1, 2, 8, 8
    S_kv = 32
    Q = torch.randn(B, S, G, H, D)
    K = torch.randn(B, G, S_kv, D)
    V = torch.randn(B, G, S_kv, Dv)
    # two ranges per row
    ranges = torch.tensor(
        [
            [[[[0, 8], [16, 24]]]],
            [[[[4, 12], [20, 28]]]],
        ],
        dtype=torch.int64,
    )
    from nsa.kernels.triton_sel_kernel import selection_attention_triton
    from nsa.core.attention_kernels import grouped_selection_attention_packed

    out_ref = grouped_selection_attention_packed(Q, K, V, ranges)
    out = selection_attention_triton(Q, K, V, ranges)
    assert torch.allclose(out, out_ref, atol=1e-4, rtol=1e-4)


