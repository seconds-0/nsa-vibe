import torch

from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta
from nsa.core.nsa_attention import NSAAttention


def run_ref_branch(x: torch.Tensor, nsa: NSAAttention, branch: str) -> torch.Tensor:
    # Force gate one-hot to branch
    if branch == "cmp":
        bias = torch.tensor([1000.0, -1000.0, -1000.0])
    elif branch == "sel":
        bias = torch.tensor([-1000.0, 1000.0, -1000.0])
    else:
        bias = torch.tensor([-1000.0, -1000.0, 1000.0])
    nsa.gate.fc2.bias.data = bias
    B, S, dim = x.shape
    G, d_k, d_v = nsa.n_kv_groups, nsa.d_k, nsa.d_v
    meta = build_block_meta(S, nsa.l, nsa.d, nsa.l_sel, nsa.n_sel, nsa.w)
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
    y, _ = nsa(x, kv, prefill=True)
    return y


def test_ablation_sliding_equiv():
    torch.manual_seed(0)
    B, S, dim = 1, 8, 64
    nsa = NSAAttention(
        dim=dim, n_heads=4, n_kv_groups=1, d_k=16, d_v=16, l=4, d=2, l_sel=4, n_sel=4, w=16
    )
    x = torch.randn(B, S, dim)
    y = run_ref_branch(x, nsa, "win")
    assert y.shape == (B, S, dim)


def test_ablation_compressed_equiv():
    torch.manual_seed(1)
    B, S, dim = 1, 8, 64
    nsa = NSAAttention(
        dim=dim, n_heads=4, n_kv_groups=1, d_k=16, d_v=16, l=4, d=2, l_sel=4, n_sel=4, w=16
    )
    x = torch.randn(B, S, dim)
    y = run_ref_branch(x, nsa, "cmp")
    assert y.shape == (B, S, dim)
