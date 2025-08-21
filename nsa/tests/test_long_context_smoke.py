import torch

from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta
from nsa.core.nsa_attention import NSAAttention


def _kv(B, G, d_k, d_v, S_ctx, w, device):
    return NSA_KV(
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
        meta=build_block_meta(S_ctx, l=32, d=16, l_sel=64, n_sel=8, w=w),
    )


def test_long_context_smoke_scaled():
    # Scaled to keep CPU runtime reasonable; verifies no crashes and monotonic counters
    device = torch.device("cpu")
    B, dim, heads, groups, d_k, d_v, w = 1, 64, 4, 2, 16, 16, 128
    S_ctx = 2048
    nsa = NSAAttention(
        dim=dim,
        n_heads=heads,
        n_kv_groups=groups,
        d_k=d_k,
        d_v=d_v,
        l=32,
        d=16,
        l_sel=64,
        n_sel=8,
        w=w,
    ).to(device)
    kv = _kv(B, groups, d_k, d_v, S_ctx + w, w, device)
    x_ctx = torch.randn(B, S_ctx, dim, device=device)
    with torch.no_grad():
        _, kv = nsa(x_ctx, kv, prefill=True)
    # Decode a few steps; ensure counters increase
    prev = int(kv.reads_act_total[-1].item()) if kv.reads_act_total.numel() else 0
    for _ in range(4):
        x_tok = torch.randn(B, 1, dim, device=device)
        with torch.no_grad():
            _, kv = nsa(x_tok, kv, prefill=False)
        now = int(kv.reads_act_total[-1].item())
        assert now >= prev
        prev = now
