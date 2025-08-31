import torch

from nsa.core.attention_kernels import grouped_selection_attention_masked


def test_masked_selection_handles_empty_rows_without_nan():
    # Construct a tiny problem where selection ranges are all empty
    torch.manual_seed(0)
    B, S, G, h, Dk, Dv = 1, 2, 1, 2, 8, 8
    S_kv = 4
    Q = torch.randn(B, S, G, h, Dk)
    K = torch.randn(B, G, S_kv, Dk)
    V = torch.randn(B, G, S_kv, Dv)
    # ranges with start==end produce no allowed keys
    ranges = torch.zeros((B, S, G, 1, 2), dtype=torch.int64)

    out = grouped_selection_attention_masked(Q, K, V, ranges)
    # Output must be finite and rows with empty ranges should be zeros
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out), atol=0.0)

