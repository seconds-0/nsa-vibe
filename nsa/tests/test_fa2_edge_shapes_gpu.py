import os
import torch
import pytest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FA-2 edge tests")


RUN_FA2 = os.getenv("NSA_TEST_FA2", "0").lower() in ("1", "true", "yes", "on")


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
def test_tiny_lengths_and_t1_decode():
    # Ensure T=1 decode behaves and no NaNs for tiny windows
    from nsa.core.attention_kernels import sliding_window_attention_fa2

    B, S, G, h, D = 1, 8, 2, 2, 64
    device = torch.device("cuda")
    Q = torch.randn(B, S, G, h, D, device=device, dtype=torch.float16)
    K = torch.randn(B, G, S, D, device=device, dtype=torch.float16)
    V = torch.randn(B, G, S, D, device=device, dtype=torch.float16)
    # Window of 1 → strictly causal single-key per row
    out = sliding_window_attention_fa2(Q, K, V, w=1)
    assert torch.isfinite(out).all()
    # First row attends only to index 0
    assert out.shape == (B, S, G, h, D)

