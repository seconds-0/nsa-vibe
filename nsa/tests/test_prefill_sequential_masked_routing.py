import os

import torch

from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta
from nsa.core.nsa_attention import NSAAttention


def test_prefill_sequential_honors_force_masked(monkeypatch, capsys):
    # Ensure sequential prefill path and force masked selection
    monkeypatch.setenv("NSA_PREFILL_BATCHED", "0")
    monkeypatch.setenv("NSA_FORCE_PARITY", "0")
    monkeypatch.setenv("NSA_FORCE_SEL_MASK", "1")
    monkeypatch.setenv("NSA_USE_SEL_MASK", "1")
    monkeypatch.setenv("NSA_USE_SEL_PACK", "0")
    monkeypatch.setenv("NSA_USE_TRITON_SEL", "0")
    monkeypatch.setenv("NSA_SEL_CUDA", "0")
    monkeypatch.setenv("NSA_DEBUG_LOG", "1")
    # Don't cap logs for this test
    monkeypatch.delenv("NSA_LOG_LIMIT", raising=False)

    # Build a tiny model and KV on CPU
    torch.manual_seed(0)
    B, S, dim = 1, 4, 32
    nsa = NSAAttention(
        dim=dim,
        n_heads=2,
        n_kv_groups=1,
        d_k=8,
        d_v=8,
        l=4,
        d=2,
        l_sel=4,
        n_sel=2,
        w=4,
    )
    device = torch.device("cpu")
    G, d_k, d_v = nsa.n_kv_groups, nsa.d_k, nsa.d_v
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

    # Run a short sequential prefill and capture logs
    x = torch.randn(B, S, dim, device=device)
    _ = nsa(x, kv, prefill=True)

    out = capsys.readouterr().out
    assert "NSA-LOG prefill.sel.path path=masked_forced" in out

