import torch

from nsa.core.attention_kernels import grouped_selection_attention_packed
from nsa.kernels.triton_sel_kernel import selection_attention_triton


def test_triton_wrapper_falls_back_on_cpu(monkeypatch):
    # Force Triton usage via env, but on CPU it should fallback to packed path
    monkeypatch.setenv("NSA_USE_TRITON_SEL", "1")
    B, S, G, h, Dk, Dv, S_kv, n = 1, 1, 1, 2, 8, 8, 4, 2
    Q = torch.randn(B, S, G, h, Dk)
    K = torch.randn(B, G, S_kv, Dk)
    V = torch.randn(B, G, S_kv, Dv)
    ranges = torch.tensor([[[[[0, 2], [1, 4]]]]], dtype=torch.int32)
    out_wrap = selection_attention_triton(Q, K, V, ranges)
    out_ref = grouped_selection_attention_packed(Q, K, V, ranges)
    assert torch.allclose(out_wrap, out_ref, atol=1e-5)
