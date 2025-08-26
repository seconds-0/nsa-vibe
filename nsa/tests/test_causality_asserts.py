import os
import torch
import pytest

from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta
from nsa.core.nsa_attention import NSAAttention


def _empty_kv(B: int, G: int, d_k: int, d_v: int, device: torch.device):
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
        meta=build_block_meta(64, 4, 2, 4, 2, 8),
    )


def test_decode_selection_causality_strict_asserts(monkeypatch):
    # Save and set environment variable
    old_val = os.environ.get("NSA_STRICT_ASSERTS")
    os.environ["NSA_STRICT_ASSERTS"] = "1"
    torch.manual_seed(0)
    B, dim = 1, 32
    nsa = NSAAttention(dim=dim, n_heads=2, n_kv_groups=1, d_k=8, d_v=8, l=4, d=2, l_sel=4, n_sel=2, w=4)
    device = torch.device("cpu")
    kv = _empty_kv(B, nsa.n_kv_groups, nsa.d_k, nsa.d_v, device)

    # Monkeypatch selection to return a future-reaching range at decode step 1 (current_pos=0)
    def fake_select_topn_ranges(p_grp, meta, n_sel, current_pos, merge_adjacent, local_count):
        # Shape [B,G,n,2], with end=2 (> current_pos+1 == 1)
        B = p_grp.shape[0]
        G = p_grp.shape[1]
        out = torch.zeros((B, G, 1, 2), dtype=torch.int32, device=p_grp.device)
        out[..., 0] = 0  # start
        out[..., 1] = 2  # illegal end
        return out

    import nsa.core.nsa_attention as mod

    monkeypatch.setattr(mod, "select_topn_ranges", fake_select_topn_ranges)

    x = torch.randn(B, 1, dim, device=device)
    try:
        with pytest.raises(AssertionError):
            _ = nsa(x, kv, prefill=False)
    finally:
        # Restore original value
        if old_val is None:
            os.environ.pop("NSA_STRICT_ASSERTS", None)
        else:
            os.environ["NSA_STRICT_ASSERTS"] = old_val


def test_prefill_batched_selection_causality_strict_asserts(monkeypatch):
    # Save and set environment variables
    old_strict = os.environ.get("NSA_STRICT_ASSERTS")
    old_batched = os.environ.get("NSA_PREFILL_BATCHED")
    os.environ["NSA_STRICT_ASSERTS"] = "1"
    os.environ["NSA_PREFILL_BATCHED"] = "1"
    torch.manual_seed(0)
    B, dim, S = 1, 32, 3
    nsa = NSAAttention(dim=dim, n_heads=2, n_kv_groups=1, d_k=8, d_v=8, l=4, d=2, l_sel=4, n_sel=2, w=4)
    device = torch.device("cpu")
    kv = _empty_kv(B, nsa.n_kv_groups, nsa.d_k, nsa.d_v, device)

    def fake_select_topn_ranges_batched(p_grp_all, meta, n_sel, S_q, merge_adjacent, local_count):
        # Shape [B,S,G,n,2]; make first timestep include end=2 (> t+1 = 1)
        B = p_grp_all.shape[0]
        S = p_grp_all.shape[1]
        G = p_grp_all.shape[2]
        out = torch.zeros((B, S, G, 1, 2), dtype=torch.int32, device=p_grp_all.device)
        out[..., 0, 0] = 0  # all starts at 0
        out[:, 0, :, 0, 1] = 2  # illegal end for t=0
        out[:, 1:, :, 0, 1] = 1  # legal for t>=1
        return out

    import nsa.core.nsa_attention as mod

    monkeypatch.setattr(mod, "select_topn_ranges_batched", fake_select_topn_ranges_batched)

    x = torch.randn(B, S, dim, device=device)
    try:
        with pytest.raises(AssertionError):
            _ = nsa(x, kv, prefill=True)
    finally:
        # Restore original values
        if old_strict is None:
            os.environ.pop("NSA_STRICT_ASSERTS", None)
        else:
            os.environ["NSA_STRICT_ASSERTS"] = old_strict
        if old_batched is None:
            os.environ.pop("NSA_PREFILL_BATCHED", None)
        else:
            os.environ["NSA_PREFILL_BATCHED"] = old_batched


def test_prefill_batched_compressed_bounds_ok_strict():
    # Save and set environment variables
    old_strict = os.environ.get("NSA_STRICT_ASSERTS")
    old_batched = os.environ.get("NSA_PREFILL_BATCHED")
    os.environ["NSA_STRICT_ASSERTS"] = "1"
    os.environ.pop("NSA_PREFILL_BATCHED", None)  # Use default sequential prefill to exercise logic safely
    torch.manual_seed(0)
    B, dim, S = 1, 32, 8
    nsa = NSAAttention(dim=dim, n_heads=2, n_kv_groups=1, d_k=8, d_v=8, l=4, d=2, l_sel=4, n_sel=2, w=4)
    device = torch.device("cpu")
    kv = _empty_kv(B, nsa.n_kv_groups, nsa.d_k, nsa.d_v, device)
    x = torch.randn(B, S, dim, device=device)
    try:
        # Should not raise under strict asserts for normal inputs
        y, kv2 = nsa(x, kv, prefill=True)
        assert y.shape == (B, S, dim)
    finally:
        # Restore original values
        if old_strict is None:
            os.environ.pop("NSA_STRICT_ASSERTS", None)
        else:
            os.environ["NSA_STRICT_ASSERTS"] = old_strict
        if old_batched is None:
            os.environ.pop("NSA_PREFILL_BATCHED", None)
        else:
            os.environ["NSA_PREFILL_BATCHED"] = old_batched
