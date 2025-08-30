import os
import pytest
import torch

from nsa.core.attention_kernels import (
    grouped_selection_attention_packed,
    selection_attention_varlen_all,
)


def _make_simple_ranges(B: int, S: int, G: int, n: int, S_kv: int, device: str):
    # Build ascending, clamped ranges per (b,t,g): [max(0,t-3), t+1)
    rng = torch.zeros((B, S, G, n, 2), dtype=torch.int32, device=device)
    for b in range(B):
        for t in range(S):
            for g in range(G):
                s0 = max(0, t - 3)
                e0 = min(S_kv, t + 1)
                if e0 > s0:
                    rng[b, t, g, 0, 0] = s0
                    rng[b, t, g, 0, 1] = e0
    return rng


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
        ),
    ],
)
def test_selection_varlen_matches_packed(device: str, monkeypatch):
    # Force varlen to use parity semantics (causal=True) to match packed reference
    monkeypatch.setenv("NSA_SEL_VARLEN_FORCE_PARITY", "1")
    torch.manual_seed(0)
    B, S, G, h, Dk, Dv, S_kv = 2, 6, 1, 2, 32, 32, 16
    dev = torch.device(device)
    Q = torch.randn(B, S, G, h, Dk, device=dev)
    K = torch.randn(B, G, S_kv, Dk, device=dev)
    V = torch.randn(B, G, S_kv, Dv, device=dev)
    ranges = _make_simple_ranges(B, S, G, n=2, S_kv=S_kv, device=device)

    O_varlen = selection_attention_varlen_all(Q, K, V, ranges)
    O_packed = grouped_selection_attention_packed(Q, K, V, ranges)
    mae = (O_varlen - O_packed).abs().mean().item()
    assert mae < 1e-5
