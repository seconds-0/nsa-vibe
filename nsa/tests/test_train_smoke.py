import torch

from nsa.model.llama_block_nsa import LlamaBlockNSA


def test_llama_block_forward_shapes():
    B, S, dim = 2, 16, 64
    model = LlamaBlockNSA(
        dim=dim, n_heads=4, n_kv_groups=2, d_k=16, d_v=16, l=8, d=4, l_sel=16, n_sel=4, w=16
    )
    x = torch.randn(B, S, dim)
    y = model(x)
    assert y.shape == (B, S, dim)
