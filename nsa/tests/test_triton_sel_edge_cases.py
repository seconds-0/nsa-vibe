import os

import torch


def test_empty_ranges_cpu_fallback():
    os.environ["NSA_USE_TRITON_SEL"] = "1"
    B, S, G, H, D, Dv = 1, 1, 1, 2, 8, 8
    S_kv = 16
    Q = torch.randn(B, S, G, H, D)
    K = torch.randn(B, G, S_kv, D)
    V = torch.randn(B, G, S_kv, Dv)
    # All-empty ranges
    ranges = torch.zeros((B, S, G, 4, 2), dtype=torch.int64)
    from nsa.core.attention_kernels import grouped_selection_attention_packed
    from nsa.kernels.triton_sel_kernel import selection_attention_triton

    out_ref = grouped_selection_attention_packed(Q, K, V, ranges)
    out = selection_attention_triton(Q, K, V, ranges)
    assert torch.allclose(out, out_ref, atol=1e-4, rtol=1e-4)


def test_invalid_ranges_clamped_cpu():
    os.environ["NSA_USE_TRITON_SEL"] = "1"
    B, S, G, H, D, Dv = 1, 1, 1, 2, 8, 8
    S_kv = 16
    Q = torch.randn(B, S, G, H, D)
    K = torch.randn(B, G, S_kv, D)
    V = torch.randn(B, G, S_kv, Dv)
    # Invalid ranges (negative and > S_kv) should be clamped
    ranges = torch.tensor([[[[[-5, -1], [10, 100]]]]], dtype=torch.int64)
    from nsa.core.attention_kernels import grouped_selection_attention_packed
    from nsa.kernels.triton_sel_kernel import selection_attention_triton

    # Build clamped reference
    ranges_ref = ranges.clamp(0, S_kv)
    out_ref = grouped_selection_attention_packed(Q, K, V, ranges_ref)
    out = selection_attention_triton(Q, K, V, ranges)
    assert torch.allclose(out, out_ref, atol=1e-4, rtol=1e-4)
