import torch
import torch.nn as nn
import torch.nn.functional as F

from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta
from nsa.core.nsa_attention import NSAAttention


def full_attn_ref_from_branch_weights(
    x: torch.Tensor,
    W_Q: nn.Linear,
    W_K: nn.Linear,
    W_V: nn.Linear,
    W_O: nn.Linear,
    n_heads: int,
    n_kv_groups: int,
    d_k: int,
    d_v: int,
) -> torch.Tensor:
    B, S, _ = x.shape
    Q = W_Q(x).view(B, S, n_heads, d_k)
    K = W_K(x).view(B, S, n_kv_groups, d_k).repeat_interleave(n_heads // n_kv_groups, dim=2)
    V = W_V(x).view(B, S, n_kv_groups, d_v).repeat_interleave(n_heads // n_kv_groups, dim=2)
    Q = Q.permute(0, 2, 1, 3)
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)
    outs = []
    for t in range(S):
        q = Q[:, :, t : t + 1]
        k = K[:, :, : t + 1]
        v = V[:, :, : t + 1]
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B,H,1,Dv]
        outs.append(attn)
    O = torch.cat(outs, dim=2).permute(0, 2, 1, 3).reshape(B, S, n_heads * d_v)
    return W_O(O)


def test_selection_full_coverage_equiv():
    torch.manual_seed(0)
    B, S, dim = 1, 8, 32
    # Full coverage by selection: l=d=l_sel=1, n_sel=S, w=0, G=1, H=1
    n_heads, G, d_k, d_v = 1, 1, 16, 16
    nsa = NSAAttention(
        dim=dim, n_heads=n_heads, n_kv_groups=G, d_k=d_k, d_v=d_v, l=1, d=1, l_sel=1, n_sel=S, w=0
    )
    x = torch.randn(B, S, dim)
    meta = build_block_meta(S, 1, 1, 1, n_sel=S, w=0)
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
    # Force selection-only gating
    nsa.gate.fc2.bias.data = torch.tensor([-1000.0, 1000.0, -1000.0])
    y_nsa, _ = nsa(x, kv, prefill=True)
    y_ref = full_attn_ref_from_branch_weights(
        x, nsa.W_Q, nsa.W_K_sel, nsa.W_V_sel, nsa.out, n_heads, G, d_k, d_v
    )
    mae = (y_nsa - y_ref).abs().mean().item()
    assert mae < 1e-5


def test_compressed_full_coverage_equiv():
    torch.manual_seed(0)
    B, S, dim = 1, 8, 32
    # Full coverage by compressed: l=d=1 makes compressed == raw; w=0, selection doesn't matter
    n_heads, G, d_k, d_v = 1, 1, 16, 16
    nsa = NSAAttention(
        dim=dim, n_heads=n_heads, n_kv_groups=G, d_k=d_k, d_v=d_v, l=1, d=1, l_sel=1, n_sel=1, w=0
    )
    x = torch.randn(B, S, dim)
    meta = build_block_meta(S, 1, 1, 1, n_sel=1, w=0)
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
    # Force compressed-only gating
    nsa.gate.fc2.bias.data = torch.tensor([1000.0, -1000.0, -1000.0])
    y_nsa, _ = nsa(x, kv, prefill=True)
    y_ref = full_attn_ref_from_branch_weights(
        x, nsa.W_Q, nsa.W_K_cmp, nsa.W_V_cmp, nsa.out, n_heads, G, d_k, d_v
    )
    mae = (y_nsa - y_ref).abs().mean().item()
    assert mae < 1e-5
