import os

import torch


def test_group_consistency_identical_ranges_cpu():
    os.environ["NSA_USE_TRITON_SEL"] = "1"
    B, S, G, H, D, Dv = 1, 1, 1, 4, 16, 16
    S_kv = 64
    Q = torch.randn(B, S, G, H, D)
    K = torch.randn(B, G, S_kv, D)
    V = torch.randn(B, G, S_kv, Dv)
    # identical ranges for all heads in the group
    ranges = torch.tensor([[[[[8, 24], [32, 48]]]]], dtype=torch.int64)
    from nsa.kernels.triton_sel_kernel import selection_attention_triton

    selection_attention_triton(Q, K, V, ranges)
    # outputs across heads should differ only due to Q; ensure that if Q heads equal, outputs equal
    Q[:, :, :, 1:] = Q[:, :, :, :1]
    O2 = selection_attention_triton(Q, K, V, ranges)
    # Now all heads in group have identical Q and identical ranges; outputs must be identical
    diff = (O2 - O2[:, :, :, :1]).abs().max().item()
    assert diff < 1e-6
