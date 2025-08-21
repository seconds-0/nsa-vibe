import torch

from nsa.core.attention_kernels import grouped_selection_attention_packed
from nsa.kernels.triton_sel_kernel import selection_attention_triton


def test_triton_wrapper_falls_back_on_float32():
    # Triton path only supports fp16/bf16; float32 must fallback to packed SDPA
    B, S, G, h, Dk, Dv, S_kv = 1, 1, 1, 2, 8, 8, 4
    Q = torch.randn(B, S, G, h, Dk, dtype=torch.float32)
    K = torch.randn(B, G, S_kv, Dk, dtype=torch.float32)
    V = torch.randn(B, G, S_kv, Dv, dtype=torch.float32)
    ranges = torch.tensor([[[[[0, 2], [1, 4]]]]], dtype=torch.int32)
    out_wrap = selection_attention_triton(Q, K, V, ranges)
    out_ref = grouped_selection_attention_packed(Q, K, V, ranges)
    assert torch.allclose(out_wrap, out_ref, atol=1e-5)
