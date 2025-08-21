import os

import pytest
import torch

from nsa.core.attention_kernels import (
    compressed_attention_fa2,
    sliding_window_attention_fa2,
)
from nsa.kernels.flash_wrappers import attention_bgh, fa2_supported, is_flash_available

RUN_FA2 = os.getenv("NSA_TEST_FA2", "0").lower() in ("1", "true", "yes")


def _gpu_ready(head_dim: int) -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    return is_flash_available() and fa2_supported(device, torch.float16, head_dim)


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
def test_sliding_varlen_vs_sdpa_gpu():
    if not _gpu_ready(64):
        pytest.skip("GPU+FA2 not available or head_dim unsupported")
    torch.manual_seed(0)
    device = torch.device("cuda")
    B, S, G, h, Dk, Dv, w = 2, 64, 2, 2, 64, 64, 32
    Q = torch.randn(B, S, G, h, Dk, device=device, dtype=torch.float16)
    K = torch.randn(B, G, S, Dk, device=device, dtype=torch.float16)
    V = torch.randn(B, G, S, Dv, device=device, dtype=torch.float16)
    # reference per-token SDPA (FP16 inputs allowed; SDPA accum FP32)
    out_ref = torch.zeros(B, S, G, h, Dv, device=device, dtype=torch.float16)
    for t in range(S):
        end = t + 1
        start = max(0, end - w)
        out_ref[:, t] = attention_bgh(
            Q[:, t].to(torch.float32),
            K[:, :, start:end].to(torch.float32),
            V[:, :, start:end].to(torch.float32),
            causal=True,
        ).to(torch.float16)
    # force varlen path
    os.environ["NSA_FA2_FORCE_VARLEN"] = "1"
    out = sliding_window_attention_fa2(Q, K, V, w)
    mae = (out.to(torch.float32) - out_ref.to(torch.float32)).abs().mean().item()
    assert mae <= 2e-4


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
def test_compressed_varlen_vs_sdpa_gpu():
    if not _gpu_ready(64):
        pytest.skip("GPU+FA2 not available or head_dim unsupported")
    torch.manual_seed(1)
    device = torch.device("cuda")
    B, S, G, h, Dk, Dv, l, d = 2, 64, 2, 2, 64, 64, 16, 8
    Q = torch.randn(B, S, G, h, Dk, device=device, dtype=torch.float16)
    K_raw = torch.randn(B, G, S, Dk, device=device, dtype=torch.float16)
    V_raw = torch.randn(B, G, S, Dv, device=device, dtype=torch.float16)
    S_cmp = (S - l) // d + 1
    K_cmp = torch.stack([K_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    V_cmp = torch.stack([V_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    out_ref = torch.zeros(B, S, G, h, Dv, device=device, dtype=torch.float16)
    for t in range(S):
        L = 0 if (t + 1) < l else min(((t + 1 - l) // d) + 1, S_cmp)
        if L > 0:
            out_ref[:, t] = attention_bgh(
                Q[:, t].to(torch.float32),
                K_cmp[:, :, :L].to(torch.float32),
                V_cmp[:, :, :L].to(torch.float32),
                causal=True,
            ).to(torch.float16)
    os.environ["NSA_FA2_FORCE_VARLEN"] = "1"
    out = compressed_attention_fa2(Q, K_cmp, V_cmp, l, d)
    mae = (out.to(torch.float32) - out_ref.to(torch.float32)).abs().mean().item()
    assert mae <= 2e-4


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
def test_compressed_row_vs_varlen_equivalence_gpu():
    if not _gpu_ready(64):
        pytest.skip("GPU+FA2 not available or head_dim unsupported")
    torch.manual_seed(3)
    device = torch.device("cuda")
    B, S, G, h, Dk, Dv, l, d = 1, 48, 1, 2, 64, 64, 16, 8
    Q = torch.randn(B, S, G, h, Dk, device=device, dtype=torch.float16)
    K_raw = torch.randn(B, G, S, Dk, device=device, dtype=torch.float16)
    V_raw = torch.randn(B, G, S, Dv, device=device, dtype=torch.float16)
    S_cmp = (S - l) // d + 1
    K_cmp = torch.stack([K_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    V_cmp = torch.stack([V_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    os.environ["NSA_FA2_FORCE_DENSE"] = "1"
    out_dense = compressed_attention_fa2(Q, K_cmp, V_cmp, l, d)
    os.environ["NSA_FA2_FORCE_DENSE"] = "0"
    os.environ["NSA_FA2_FORCE_VARLEN"] = "1"
    out_varlen = compressed_attention_fa2(Q, K_cmp, V_cmp, l, d)
    mae = (out_dense.to(torch.float32) - out_varlen.to(torch.float32)).abs().mean().item()
    assert mae <= 5e-5


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
def test_gpu_determinism_note():
    if not _gpu_ready(64):
        pytest.skip("GPU+FA2 not available or head_dim unsupported")
    torch.manual_seed(2)
    device = torch.device("cuda")
    B, S, G, h, Dk, Dv, w = 1, 64, 1, 2, 64, 64, 32
    Q = torch.randn(B, S, G, h, Dk, device=device, dtype=torch.float16)
    K = torch.randn(B, G, S, Dk, device=device, dtype=torch.float16)
    V = torch.randn(B, G, S, Dv, device=device, dtype=torch.float16)
    os.environ["NSA_FA2_FORCE_VARLEN"] = "1"
    out1 = sliding_window_attention_fa2(Q, K, V, w)
    out2 = sliding_window_attention_fa2(Q, K, V, w)
    # Non-bitwise determinism acceptable; check small tolerance
    mae = (out1.to(torch.float32) - out2.to(torch.float32)).abs().mean().item()
    assert mae <= 1e-5
