import os
import pytest
import torch

from nsa.core.attention_kernels import (
    sliding_window_attention_masked,
    batched_causal_attention_compressed_masked,
    sliding_window_attention_fa2,
    compressed_attention_fa2,
    sliding_window_attention_fa2_decode,
    compressed_attention_fa2_decode,
)
from nsa.kernels.flash_wrappers import is_flash_available, fa2_supported
from nsa.kernels.flash_wrappers import attention_bgh


RUN_FA2 = os.getenv("NSA_TEST_FA2", "0").lower() in ("1", "true", "yes")


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
@pytest.mark.skipif(not is_flash_available(), reason="flash-attn not available")
def test_fa2_env_ready():
    assert is_flash_available()


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
def test_sliding_parity_dense_bucket():
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
def test_sliding_packing_equivalence():
    # Row vs bucket equivalence under same kernel path
    torch.manual_seed(3)
    B, S, G, h, Dk, Dv, w = 1, 9, 1, 2, 8, 8, 4
    Q = torch.randn(B, S, G, h, Dk)
    K = torch.randn(B, G, S, Dk)
    V = torch.randn(B, G, S, Dv)
    # Row path: per-token reference
    out_row = torch.zeros(B, S, G, h, Dv)
    for t in range(S):
        end = t + 1
        start = max(0, end - w)
        out_row[:, t] = attention_bgh(Q[:, t], K[:, :, start:end], V[:, :, start:end], causal=True)
    # Bucket path using FA‑2 wrapper (falls back on CPU)
    out_bucket = sliding_window_attention_fa2(Q, K, V, w)
    assert (out_bucket - out_row).abs().max().item() < 1e-6


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
def test_compressed_parity_dense_bucket():
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


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
def test_decode_paths_dense_bucket():
    torch.manual_seed(2)
    B, S, G, h, Dk, Dv, w = 1, 6, 1, 2, 8, 8, 4
    Q = torch.randn(B, S, G, h, Dk)
    K = torch.randn(B, G, S, Dk)
    V = torch.randn(B, G, S, Dv)
    # sliding decode
    for t in range(1, S + 1):
        out_ref = torch.zeros(B, G, h, Dv)
        start = max(0, t - w)
        out_ref = attention_bgh(Q[:, t - 1], K[:, :, start:t], V[:, :, start:t], causal=True)
        out = sliding_window_attention_fa2_decode(Q[:, t - 1], K[:, :, :t], V[:, :, :t], w)
        assert (out - out_ref).abs().max().item() < 1e-6
    # compressed decode
    l, d = 4, 2
    S_cmp = (S - l) // d + 1
    K_raw = torch.randn(B, G, S, Dk)
    V_raw = torch.randn(B, G, S, Dv)
    K_cmp = torch.stack([K_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    V_cmp = torch.stack([V_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    for t in range(1, S + 1):
        L = 0 if t < l else min(((t - l) // d) + 1, S_cmp)
        out_ref = attention_bgh(Q[:, t - 1], K_cmp[:, :, :L], V_cmp[:, :, :L], causal=True) if L > 0 else torch.zeros(B, G, h, Dv)
        out = compressed_attention_fa2_decode(Q[:, t - 1], K_cmp, V_cmp, L)
        assert (out - out_ref).abs().max().item() < 1e-6


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
def test_head_dim_constraint_xfail():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        pytest.skip("CPU: no FA-2 support")
    assert not fa2_supported(device, torch.float16, head_dim=7)


