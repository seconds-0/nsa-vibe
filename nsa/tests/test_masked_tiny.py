import os
import pytest
import torch

from nsa.core.attention_kernels import (
    sliding_window_attention_masked,
    batched_causal_attention_compressed_masked,
)
from nsa.kernels.flash_wrappers import attention_bgh


RUN_MASKED = os.getenv("NSA_TEST_MASKED", "0").lower() in ("1", "true", "yes")


def _ref_sliding(Q, K, V, w):
    B, S, G, h, Dk = Q.shape
    Dv = V.shape[-1]
    out = torch.zeros(B, S, G, h, Dv, dtype=V.dtype)
    for t in range(S):
        end = t + 1
        start = max(0, end - w)
        out[:, t] = attention_bgh(Q[:, t], K[:, :, start:end], V[:, :, start:end], causal=True)
    return out


def _ref_compressed(Q, K_cmp, V_cmp, l, d):
    B, S, G, h, Dk = Q.shape
    Dv = V_cmp.shape[-1]
    out = torch.zeros(B, S, G, h, Dv, dtype=V_cmp.dtype)
    S_cmp = K_cmp.shape[2]
    for t in range(S):
        L = 0 if (t + 1) < l else min(((t + 1 - l) // d) + 1, S_cmp)
        if L > 0:
            out[:, t] = attention_bgh(Q[:, t], K_cmp[:, :, :L], V_cmp[:, :, :L], causal=True)
    return out


@pytest.mark.skipif(not RUN_MASKED, reason="Tiny masked parity tests are opt-in; set NSA_TEST_MASKED=1")
@pytest.mark.parametrize("S", [1, 2, 3, 4])
def test_masked_sliding_tiny_parity(S):
    torch.manual_seed(0)
    B, G, h, Dk, Dv = 1, 1, 2, 8, 8
    Q = torch.randn(B, S, G, h, Dk)
    K = torch.randn(B, G, S, Dk)
    V = torch.randn(B, G, S, Dv)
    for w in range(1, S + 1):
        out_ref = _ref_sliding(Q, K, V, w)
        out = sliding_window_attention_masked(Q, K, V, w)
        max_err = (out - out_ref).abs().max().item()
        assert max_err < 1e-6


@pytest.mark.skipif(not RUN_MASKED, reason="Tiny masked parity tests are opt-in; set NSA_TEST_MASKED=1")
@pytest.mark.parametrize("S", [1, 2, 3, 4])
@pytest.mark.parametrize("l,d", [(1, 1), (2, 1)])
def test_masked_compressed_tiny_parity(S, l, d):
    torch.manual_seed(1)
    B, G, h, Dk, Dv = 1, 1, 2, 8, 8
    Q = torch.randn(B, S, G, h, Dk)
    # Build simple deterministic K_cmp/V_cmp by mean pooling over stride windows
    S_cmp = 0 if S < l else (S - l) // d + 1
    if S_cmp == 0:
        K_cmp = torch.zeros(B, G, 0, Dk)
        V_cmp = torch.zeros(B, G, 0, Dv)
    else:
        K_raw = torch.randn(B, G, S, Dk)
        V_raw = torch.randn(B, G, S, Dv)
        K_cmp = torch.stack([K_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
        V_cmp = torch.stack([V_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    out_ref = _ref_compressed(Q, K_cmp, V_cmp, l, d)
    out = batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
    max_err = (out - out_ref).abs().max().item()
    assert max_err < 1e-6


@pytest.mark.skipif(not RUN_MASKED, reason="Determinism checks are opt-in; set NSA_TEST_MASKED=1")
def test_masked_sliding_determinism():
    torch.manual_seed(2)
    B, S, G, h, Dk, Dv, w = 1, 8, 1, 2, 8, 8, 4
    Q = torch.randn(B, S, G, h, Dk)
    K = torch.randn(B, G, S, Dk)
    V = torch.randn(B, G, S, Dv)
    out1 = sliding_window_attention_masked(Q, K, V, w)
    out2 = sliding_window_attention_masked(Q, K, V, w)
    assert torch.allclose(out1, out2, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not RUN_MASKED, reason="Determinism checks are opt-in; set NSA_TEST_MASKED=1")
def test_masked_compressed_determinism():
    torch.manual_seed(3)
    B, S, G, h, Dk, Dv, l, d = 1, 8, 1, 2, 8, 8, 4, 2
    Q = torch.randn(B, S, G, h, Dk)
    S_cmp = (S - l) // d + 1
    K_raw = torch.randn(B, G, S, Dk)
    V_raw = torch.randn(B, G, S, Dv)
    K_cmp = torch.stack([K_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    V_cmp = torch.stack([V_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    out1 = batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
    out2 = batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
    assert torch.allclose(out1, out2, atol=0.0, rtol=0.0)


