import os
import pytest
import torch

from nsa.core.attention_kernels import (
    sliding_window_attention_masked,
    batched_causal_attention_compressed_masked,
    sliding_window_attention_fa2,
    compressed_attention_fa2,
)
from nsa.kernels.flash_wrappers import is_flash_available
from nsa.kernels.flash_wrappers import attention_bgh


RUN_FA2 = os.getenv("NSA_TEST_FA2", "0").lower() in ("1", "true", "yes")


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
@pytest.mark.skipif(not is_flash_available(), reason="flash-attn not available")
def test_fa2_env_ready():
    assert is_flash_available()


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
def test_sliding_parity_placeholder():
    # Placeholder: currently masked SDPA path; FA-2 to be integrated in M1
    torch.manual_seed(0)
    B, S, G, h, Dk, Dv, w = 1, 8, 1, 2, 8, 8, 4
    Q = torch.randn(B, S, G, h, Dk)
    K = torch.randn(B, G, S, Dk)
    V = torch.randn(B, G, S, Dv)
    # reference per-token
    out_ref = torch.zeros(B, S, G, h, Dv)
    for t in range(S):
        end = t + 1
        start = max(0, end - w)
        out_ref[:, t] = attention_bgh(Q[:, t], K[:, :, start:end], V[:, :, start:end], causal=True)
    out = sliding_window_attention_fa2(Q, K, V, w)
    assert (out - out_ref).abs().max().item() < 1e-6


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
def test_compressed_parity_placeholder():
    torch.manual_seed(1)
    B, S, G, h, Dk, Dv, l, d = 1, 8, 1, 2, 8, 8, 4, 2
    Q = torch.randn(B, S, G, h, Dk)
    S_cmp = (S - l) // d + 1
    K_raw = torch.randn(B, G, S, Dk)
    V_raw = torch.randn(B, G, S, Dv)
    K_cmp = torch.stack([K_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    V_cmp = torch.stack([V_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    out_ref = torch.zeros(B, S, G, h, Dv)
    for t in range(S):
        L = 0 if (t + 1) < l else min(((t + 1 - l) // d) + 1, S_cmp)
        if L > 0:
            out_ref[:, t] = attention_bgh(Q[:, t], K_cmp[:, :, :L], V_cmp[:, :, :L], causal=True)
    out = compressed_attention_fa2(Q, K_cmp, V_cmp, l, d)
    assert (out - out_ref).abs().max().item() < 1e-6


