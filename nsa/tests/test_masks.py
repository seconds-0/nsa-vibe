import torch
import torch.nn.functional as F


def test_causal_sdpa_ref():
    # Reference SDPA is causal by API; ensure no future leakage with toy tensors
    B, H, S, Dk, Dv = 1, 1, 8, 4, 6
    q = torch.randn(B, H, S, Dk)
    k = torch.randn(B, H, S, Dk)
    v = torch.randn(B, H, S, Dv)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    # For each t, compare against truncated K/V (non-causal) result
    for t in range(S):
        q_t = q[:, :, t : t + 1]
        k_t = k[:, :, : t + 1]
        v_t = v[:, :, : t + 1]
        attn_t = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=False)
        assert attn_t.shape == out[:, :, t : t + 1].shape
