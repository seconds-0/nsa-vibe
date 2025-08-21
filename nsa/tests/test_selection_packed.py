import os

import pytest
import torch

from nsa.core.attention_kernels import (
    grouped_selection_attention,
    grouped_selection_attention_masked,
    grouped_selection_attention_packed,
)

RUN_PACKED = os.getenv("NSA_TEST_SEL_PACK", "0").lower() in ("1", "true", "yes")
RUN_MASK = os.getenv("NSA_TEST_SEL_MASK", "0").lower() in ("1", "true", "yes")


def _random_ranges(B, S, G, n, S_kv):
    ranges = torch.zeros(B, S, G, n, 2, dtype=torch.int32)
    for b in range(B):
        for t in range(S):
            for g in range(G):
                start = max(0, t - 2)
                end = min(S_kv, t + 1)
                ranges[b, t, g, 0, 0] = start
                ranges[b, t, g, 0, 1] = end
    return ranges


@pytest.mark.skipif(
    not RUN_PACKED, reason="Selection packing parity scaffold; enable with NSA_TEST_SEL_PACK=1"
)
def test_selection_packed_parity_scaffold():
    torch.manual_seed(0)
    B, S, G, h, Dk, Dv = 1, 8, 2, 2, 8, 8
    Q = torch.randn(B, S, G, h, Dk)
    S_kv = 16
    K = torch.randn(B, G, S_kv, Dk)
    V = torch.randn(B, G, S_kv, Dv)
    n = 1
    ranges = _random_ranges(B, S, G, n, S_kv)

    out_ref = grouped_selection_attention(Q, K, V, ranges)
    out_packed = grouped_selection_attention_packed(Q, K, V, ranges)

    mae = (out_ref - out_packed).abs().mean().item()
    assert mae < 1e-8


@pytest.mark.skipif(
    not RUN_MASK, reason="Masked selection parity is opt-in; enable with NSA_TEST_SEL_MASK=1"
)
def test_selection_masked_parity():
    torch.manual_seed(1)
    B, S, G, h, Dk, Dv = 1, 8, 2, 2, 8, 8
    Q = torch.randn(B, S, G, h, Dk)
    S_kv = 16
    K = torch.randn(B, G, S_kv, Dk)
    V = torch.randn(B, G, S_kv, Dv)
    n = 1
    ranges = _random_ranges(B, S, G, n, S_kv)

    out_ref = grouped_selection_attention(Q, K, V, ranges)
    out_masked = grouped_selection_attention_masked(Q, K, V, ranges)

    mae = (out_ref - out_masked).abs().mean().item()
    assert mae < 1e-8
