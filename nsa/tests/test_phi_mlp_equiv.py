import torch

from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta
from nsa.core.nsa_attention import NSAAttention


def _empty_kv(
    B: int,
    G: int,
    d_k: int,
    d_v: int,
    S: int,
    l: int,
    d: int,
    l_sel: int,
    n_sel: int,
    w: int,
    device,
):
    meta = build_block_meta(S, l, d, l_sel, n_sel=n_sel, w=w)
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
        meta=meta,
    )


def test_phi_mlp_matches_avg_prefill():
    torch.manual_seed(0)
    device = torch.device("cpu")
    B, S = 2, 16
    dim, n_heads, G = 256, 8, 4
    n_heads // G
    d_k, d_v = 64, 64
    l, d, l_sel, n_sel, w = 4, 2, 8, 4, 16

    nsa_avg = NSAAttention(dim, n_heads, G, d_k, d_v, l, d, l_sel, n_sel, w, phi="avg").to(device)
    nsa_mlp = NSAAttention(dim, n_heads, G, d_k, d_v, l, d, l_sel, n_sel, w, phi="mlp").to(device)
    # Copy shared weights for identical projections/gates
    sd = nsa_avg.state_dict()
    for k, v in sd.items():
        if "phi_" in k:
            continue
        nsa_mlp.state_dict()[k].copy_(v)

    x = torch.randn(B, S, dim, device=device)
    kv_avg = _empty_kv(B, G, d_k, d_v, S, l, d, l_sel, n_sel, w, device)
    kv_mlp = _empty_kv(B, G, d_k, d_v, S, l, d, l_sel, n_sel, w, device)
    y_avg, _ = nsa_avg(x, kv_avg, prefill=True)
    y_mlp, _ = nsa_mlp(x, kv_mlp, prefill=True)
    mae = (y_avg - y_mlp).abs().mean().item()
    assert mae < 1e-5


def test_phi_mlp_matches_avg_decode():
    torch.manual_seed(1)
    device = torch.device("cpu")
    B, S = 1, 8
    dim, n_heads, G = 128, 4, 2
    n_heads // G
    d_k, d_v = 32, 32
    l, d, l_sel, n_sel, w = 4, 2, 8, 4, 8
    nsa_avg = NSAAttention(dim, n_heads, G, d_k, d_v, l, d, l_sel, n_sel, w, phi="avg").to(device)
    nsa_mlp = NSAAttention(dim, n_heads, G, d_k, d_v, l, d, l_sel, n_sel, w, phi="mlp").to(device)
    sd = nsa_avg.state_dict()
    for k, v in sd.items():
        if "phi_" in k:
            continue
        nsa_mlp.state_dict()[k].copy_(v)

    kv_avg = _empty_kv(B, G, d_k, d_v, 0, l, d, l_sel, n_sel, w, device)
    kv_mlp = _empty_kv(B, G, d_k, d_v, 0, l, d, l_sel, n_sel, w, device)
    for t in range(S):
        x_t = torch.randn(B, 1, dim, device=device)
        y_avg, kv_avg = nsa_avg(x_t, kv_avg, prefill=False)
        y_mlp, kv_mlp = nsa_mlp(x_t, kv_mlp, prefill=False)
        mae = (y_avg - y_mlp).abs().mean().item()
        assert mae < 1e-5
