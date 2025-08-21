import os

import pytest
import torch

from nsa.core.attention_kernels import grouped_selection_attention_packed
from nsa.kernels.cuda_sel_kernel import selection_attention_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sel_cuda_gpu_small_shapes():
    os.environ["NSA_SEL_CUDA_BUILD"] = "1"
    os.environ["NSA_SEL_CUDA"] = "1"
    torch.manual_seed(0)
    device = torch.device("cuda")
    B, S, G, h, Dk, Dv = 2, 1, 2, 2, 32, 32
    L = 128
    Q = torch.randn(B, S, G, h, Dk, device=device, dtype=torch.float16)
    K = torch.randn(B, G, L, Dk, device=device, dtype=torch.float16)
    V = torch.randn(B, G, L, Dv, device=device, dtype=torch.float16)
    ranges = torch.zeros(B, S, G, 2, 2, dtype=torch.int32, device=device)
    ranges[..., 0, 0] = 0
    ranges[..., 0, 1] = L // 2
    ranges[..., 1, 0] = L // 2
    ranges[..., 1, 1] = L
    O_cuda = selection_attention_cuda(Q, K, V, ranges)
    O_ref = grouped_selection_attention_packed(Q, K, V, ranges)
    mae = (O_cuda - O_ref).abs().mean().item()
    assert mae < 1e-3
