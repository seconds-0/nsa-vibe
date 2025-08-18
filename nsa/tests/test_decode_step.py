import torch

from nsa.core.nsa_attention import NSAAttention
from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta


def num_cmp(S: int, l: int, d: int) -> int:
    return 0 if S < l else (S - l) // d + 1


def reads(S: int, l: int, d: int, n: int, l_sel: int, w: int) -> int:
    return num_cmp(S, l, d) + n * l_sel + min(w, S)


def test_decode_step_reads_small():
    torch.manual_seed(0)
    B, dim = 1, 64
    nsa = NSAAttention(dim=dim, n_heads=4, n_kv_groups=1, d_k=16, d_v=16, l=4, d=2, l_sel=4, n_sel=4, w=8)
    G, d_k, d_v = nsa.n_kv_groups, nsa.d_k, nsa.d_v
    device = torch.device("cpu")
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
        meta=build_block_meta(64, nsa.l, nsa.d, nsa.l_sel, nsa.n_sel, nsa.w),
    )
    # simulate 5 decode steps with random tokens
    total_S = 5
    for S in range(1, total_S + 1):
        x = torch.randn(B, 1, dim)
        y, kv = nsa(x, kv, prefill=False)
        assert y.shape == (B, 1, dim)
        expected = reads(S, nsa.l, nsa.d, nsa.n_sel, nsa.l_sel, nsa.w)
        assert int(kv.reads_pred[-1].item()) == expected
        # Actual total should match predicted in M0 reference path
        assert int(kv.reads_act_total[-1].item()) == expected
        # Branch breakdown invariant
        assert int(kv.reads_act_sel[-1].item()) == nsa.n_sel * nsa.l_sel
        assert int(kv.reads_act_cmp[-1].item()) == (0 if S < nsa.l else (S - nsa.l) // nsa.d + 1)
        assert int(kv.reads_act_win[-1].item()) == min(nsa.w, S)


