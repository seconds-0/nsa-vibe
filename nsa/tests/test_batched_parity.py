import os
import torch
import pytest

from nsa.core.attention_kernels import sliding_window_attention_masked, batched_causal_attention_compressed_masked
from nsa.kernels.flash_wrappers import attention_bgh


RUN_MASKED = os.getenv("NSA_TEST_MASKED", "0").lower() in ("1", "true", "yes")


def _cmp_parity_sliding(B=1, S=8, G=1, h=2, Dk=8, Dv=8, w=4):
    torch.manual_seed(0)
    Q = torch.randn(B, S, G, h, Dk)
    K = torch.randn(B, G, S, Dk)
    V = torch.randn(B, G, S, Dv)
    # Reference per-t
    out_ref = torch.zeros(B, S, G, h, Dv)
    for t in range(S):
        end = t + 1
        start = max(0, end - w)
        out_ref[:, t] = attention_bgh(Q[:, t], K[:, :, start:end], V[:, :, start:end], causal=True)
    # Candidate masked SDPA path
    out = sliding_window_attention_masked(Q, K, V, w)
    return (out - out_ref).abs().max().item()


def _cmp_parity_compressed(B=1, S=8, G=1, h=2, Dk=8, Dv=8, l=4, d=2):
    torch.manual_seed(1)
    Q = torch.randn(B, S, G, h, Dk)
    # Build compressed K/V by slicing raw; for parity test we don’t rely on ϕ
    S_cmp = max(0, (S - l) // d + 1) if S >= l else 0
    if S_cmp == 0:
        K_cmp = torch.zeros(B, G, 0, Dk)
        V_cmp = torch.zeros(B, G, 0, Dv)
    else:
        K_raw = torch.randn(B, G, S, Dk)
        V_raw = torch.randn(B, G, S, Dv)
        # naive pool to create deterministic K_cmp/V_cmp windows (d-stride, length l)
        K_cmp = torch.stack([K_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
        V_cmp = torch.stack([V_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    # Reference per-t
    out_ref = torch.zeros(B, S, G, h, Dv)
    for t in range(S):
        L = 0 if (t + 1) < l else min(((t + 1 - l) // d) + 1, S_cmp)
        if L > 0:
            out_ref[:, t] = attention_bgh(Q[:, t], K_cmp[:, :, :L], V_cmp[:, :, :L], causal=True)
    out = batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
    return (out - out_ref).abs().max().item()


@pytest.mark.skipif(not RUN_MASKED, reason="Masked parity tests are opt-in; set NSA_TEST_MASKED=1 to run")
@pytest.mark.parametrize("w", [1, 4, 8])
@pytest.mark.parametrize("S", [4, 8])
def test_parity_sliding_masked(S, w):
    max_err = _cmp_parity_sliding(S=S, w=w)
    assert max_err < 1e-6


@pytest.mark.skipif(not RUN_MASKED, reason="Masked parity tests are opt-in; set NSA_TEST_MASKED=1 to run")
@pytest.mark.parametrize("l,d", [(4, 2), (2, 1)])
@pytest.mark.parametrize("S", [4, 8])
def test_parity_compressed_masked(S, l, d):
    max_err = _cmp_parity_compressed(S=S, l=l, d=d)
    assert max_err < 1e-6
