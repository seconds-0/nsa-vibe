import os

import pytest
import torch

from nsa.cache.kv_cache import NSA_KV
from nsa.core.attention_kernels import sliding_window_attention
from nsa.core.nsa_attention import NSAAttention
from nsa.core.block_index import build_block_meta


@pytest.mark.parametrize("S,w", [(8, 1), (8, 4), (8, 8)])
def test_sliding_window_attention_no_nan_cpu(S, w):
    torch.manual_seed(0)
    B, G, h, Dk, Dv = 2, 1, 2, 16, 16
    Q = torch.randn(B, S, G, h, Dk, dtype=torch.float32)
    K = torch.randn(B, G, S, Dk, dtype=torch.float32)
    V = torch.randn(B, G, S, Dv, dtype=torch.float32)
    out = sliding_window_attention(Q, K, V, w)
    assert out.shape == (B, S, G, h, Dv)
    assert torch.isfinite(out).all(), "Output contains non-finite values"


def test_prefill_batched_sliding_no_nan_cpu(monkeypatch):
    # Ensure batched prefill path is used
    old_env = dict(os.environ)
    monkeypatch.setenv("NSA_PREFILL_BATCHED", "1")
    # Do not allow sliding FA2 semantics so it falls back to SDPA sliding path
    monkeypatch.delenv("NSA_ALLOW_SLIDING_FA2", raising=False)
    monkeypatch.setenv("NSA_USE_FA2", "1")  # enter FA2 branch, then fall back inside
    monkeypatch.delenv("NSA_FA2_FORCE_VARLEN", raising=False)
    monkeypatch.delenv("NSA_FA2_FORCE_DENSE", raising=False)

    torch.manual_seed(0)
    B, S, dim = 1, 8, 64
    n_heads, G, d_k, d_v = 4, 1, 16, 16
    x = torch.randn(B, S, dim)
    # NSA configured to cover all tokens: w ≥ S and n*l' ≥ S
    l, d, l_sel, n, w = 4, 2, 4, 4, 8
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
        use_flash=True,
    )
    # Force gates to sliding branch only
    with torch.no_grad():
        nsa.gate.fc2.bias.copy_(torch.tensor([-1000.0, -1000.0, 1000.0]))

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
    y, _ = nsa(x, kv, prefill=True)
    assert torch.isfinite(y).all(), "Prefill batched output contains non-finite values"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("S,w", [(8, 1), (8, 4), (8, 8)])
def test_sliding_window_attention_no_nan_cuda(S, w, dtype):
    torch.manual_seed(0)
    B, G, h, Dk, Dv = 2, 1, 2, 16, 16
    device = torch.device("cuda")
    Q = torch.randn(B, S, G, h, Dk, dtype=dtype, device=device)
    K = torch.randn(B, G, S, Dk, dtype=dtype, device=device)
    V = torch.randn(B, G, S, Dv, dtype=dtype, device=device)
    out = sliding_window_attention(Q, K, V, w)
    assert out.shape == (B, S, G, h, Dv)
    assert torch.isfinite(out).all(), "CUDA output contains non-finite values"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
def test_prefill_batched_sliding_no_nan_cuda(monkeypatch):
    # Ensure batched prefill path is used; disallow sliding FA2 semantics
    monkeypatch.setenv("NSA_PREFILL_BATCHED", "1")
    monkeypatch.delenv("NSA_ALLOW_SLIDING_FA2", raising=False)
    monkeypatch.setenv("NSA_USE_FA2", "1")
    monkeypatch.delenv("NSA_FA2_FORCE_VARLEN", raising=False)
    monkeypatch.delenv("NSA_FA2_FORCE_DENSE", raising=False)

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, S, dim = 1, 8, 64
    n_heads, G, d_k, d_v = 4, 1, 16, 16
    x = torch.randn(B, S, dim, device=device, dtype=torch.float16)
    l, d, l_sel, n, w = 4, 2, 4, 4, 8
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
        use_flash=True,
    ).to(device).half()
    with torch.no_grad():
        nsa.gate.fc2.bias.copy_(torch.tensor([-1000.0, -1000.0, 1000.0], device=device, dtype=torch.float16))

    meta = build_block_meta(S, l, d, l_sel, n_sel=n, w=w)
    kv = NSA_KV(
        K_sel=torch.zeros((B, G, 0, d_k), device=device, dtype=torch.float16),
        V_sel=torch.zeros((B, G, 0, d_v), device=device, dtype=torch.float16),
        K_win=torch.zeros((B, G, 0, d_k), device=device, dtype=torch.float16),
        V_win=torch.zeros((B, G, 0, d_v), device=device, dtype=torch.float16),
        K_cmp_raw_seq=torch.zeros((B, G, 0, d_k), device=device, dtype=torch.float16),
        V_cmp_raw_seq=torch.zeros((B, G, 0, d_v), device=device, dtype=torch.float16),
        K_cmp=torch.zeros((B, G, 0, d_k), device=device, dtype=torch.float16),
        V_cmp=torch.zeros((B, G, 0, d_v), device=device, dtype=torch.float16),
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
    assert torch.isfinite(y).all(), "CUDA prefill batched output contains non-finite values"
