import torch
import torch.nn.functional as F

from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta
from nsa.core.nsa_attention import NSAAttention
from nsa.core.rope import apply_rope


def full_attention_reference_from_nsa_weights(x: torch.Tensor, nsa: NSAAttention) -> torch.Tensor:
    # Use NSA's W_Q, W_K_win, W_V_win, and output to create a fair reference
    B, S, _ = x.shape
    n_heads = nsa.n_heads
    d_k = nsa.d_k
    d_v = nsa.d_v
    W_Q = nsa.W_Q
    W_K = nsa.W_K_win
    W_V = nsa.W_V_win
    W_O = nsa.out
    Q = W_Q(x).view(B, S, n_heads, d_k).permute(0, 2, 1, 3)  # [B,H,S,Dk]
    K = (
        W_K(x)
        .view(B, S, nsa.n_kv_groups, d_k)
        .repeat_interleave(nsa.h_per_group, dim=2)
        .permute(0, 2, 1, 3)
    )  # [B,H,S,Dk]
    V = (
        W_V(x)
        .view(B, S, nsa.n_kv_groups, d_v)
        .repeat_interleave(nsa.h_per_group, dim=2)
        .permute(0, 2, 1, 3)
    )  # [B,H,S,Dv]
    # Apply RoPE to align with model invariants
    pos = torch.arange(S, device=x.device)
    scale = getattr(nsa, "rope_scale", 1.0)
    Q = apply_rope(Q, pos, scale=scale)
    K = apply_rope(K, pos, scale=scale)
    out = []
    for t in range(S):
        q = Q[:, :, t : t + 1]
        k = K[:, :, : t + 1]
        v = V[:, :, : t + 1]
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B,H,1,Dv]
        out.append(attn)
    O = torch.cat(out, dim=2)  # [B,H,S,Dv]
    O = O.permute(0, 2, 1, 3).reshape(B, S, n_heads * d_v)
    return W_O(O)


def test_smallS_equivalence():
    torch.manual_seed(0)
    B, S, dim = 1, 8, 64
    n_heads, G, d_k, d_v = 4, 1, 16, 16
    x = torch.randn(B, S, dim)
    # NSA configured to cover all tokens: w ≥ S and n*l' ≥ S
    l, d, l_sel, n, w = 4, 2, 4, 4, 16
    nsa = NSAAttention(
        dim=dim,
        n_heads=n_heads,
        n_kv_groups=G,
        d_k=d_k,
        d_v=d_v,
        l=l,
        d=d,
        l_sel=l_sel,
        n_sel=n,
        w=w,
    )
    meta = build_block_meta(S, l, d, l_sel, n_sel=n, w=w)
    device = x.device
    kv = NSA_KV(
        K_sel=torch.zeros((B, G, 0, d_k), device=device),
        V_sel=torch.zeros((B, G, 0, d_v), device=device),
        K_win=torch.zeros((B, G, 0, d_k), device=device),
        V_win=torch.zeros((B, G, 0, d_v), device=device),
        K_cmp_raw_seq=torch.zeros((B, G, 0, d_k), device=device),
        V_cmp_raw_seq=torch.zeros((B, G, 0, d_v), device=device),
        K_cmp=torch.zeros((B, G, 0, d_k), device=device),
        V_cmp=torch.zeros((B, G, 0, d_v), device=device),
        win_ptr=torch.zeros((B, G), dtype=torch.int32, device=device),
        cmp_emit_next=torch.zeros((B, G), dtype=torch.int32, device=device),
        reads_pred=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_total=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_sel=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_cmp=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_win=torch.zeros((0,), dtype=torch.int64, device=device),
        meta=meta,
    )
    # Force gates to sliding branch only
    nsa.gate.fc2.bias.data = torch.tensor([-1000.0, -1000.0, 1000.0])
    y_nsa, _ = nsa(x, kv, prefill=True)
    y_ref = full_attention_reference_from_nsa_weights(x, nsa)
    mae = (y_nsa - y_ref).abs().mean().item()
    assert mae < 1e-5
