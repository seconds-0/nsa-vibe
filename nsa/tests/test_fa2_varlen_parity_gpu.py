import os
import torch
import pytest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FA-2 varlen tests")


RUN_FA2 = os.getenv("NSA_TEST_FA2", "0").lower() in ("1", "true", "yes", "on")


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
def test_selection_varlen_v2_matches_masked_small():
    # Small case with simple ranges; ensures v2 varlen path matches masked reference
    from nsa.core.attention_kernels import (
        selection_attention_varlen_all_v2,
        grouped_selection_attention_masked,
    )

    B, S, G, h, D = 1, 4, 2, 2, 64
    device = torch.device("cuda")
    Q = torch.randn(B, S, G, h, D, device=device, dtype=torch.float16)
    K = torch.randn(B, G, S, D, device=device, dtype=torch.float16)
    V = torch.randn(B, G, S, D, device=device, dtype=torch.float16)
    # Ranges: allow [0..t] per row (causal)
    n = 2
    ranges = torch.zeros(B, S, G, n, 2, device=device, dtype=torch.int32)
    for t in range(S):
        ranges[:, t, :, 0, 0] = 0
        ranges[:, t, :, 0, 1] = t + 1
        ranges[:, t, :, 1, 0] = t + 1
        ranges[:, t, :, 1, 1] = t + 1  # empty second segment beyond t

    out_masked = grouped_selection_attention_masked(Q, K, V, ranges)
    out_v2 = selection_attention_varlen_all_v2(Q, K, V, ranges)

    torch.testing.assert_close(out_v2, out_masked, atol=5e-3, rtol=5e-3)

