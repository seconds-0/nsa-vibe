import os
import pytest
import torch


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Ensure FA-2 env flags are clean per test
    for k in [
        "NSA_USE_FA2",
        "NSA_USE_FA2_WIN",
        "NSA_USE_FA2_CMP",
        "NSA_FA2_MIN_LEN_WIN",
        "NSA_FA2_MIN_LEN_CMP",
        "NSA_ALLOW_SLIDING_FA2",
        "NSA_PREFILL_BATCHED",
    ]:
        monkeypatch.delenv(k, raising=False)
    yield


def _make_nsa_module():
    from nsa.core.nsa_attention import NSAAttention

    # Small, CPU-friendly module
    return NSAAttention(
        dim=64,
        n_heads=4,
        n_kv_groups=1,
        d_k=16,
        d_v=16,
        l=4,
        d=2,
        l_sel=4,
        n_sel=2,
        w=4,
        use_flash=True,
    )


def _make_kv(S: int, d_k: int, d_v: int):
    from nsa.core.block_index import build_block_meta
    from nsa.cache.kv_cache import NSA_KV

    B, G = 1, 1
    device = torch.device("cpu")
    meta = build_block_meta(S, l=4, d=2, l_sel=4, n_sel=2, w=4)
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


def test_fa2_disabled_via_env_avoids_call(monkeypatch):
    # Force batched prefill path
    monkeypatch.setenv("NSA_PREFILL_BATCHED", "1")

    # Monkeypatch FA-2 compressed entry to raise if called
    import nsa.core.attention_kernels as K

    called = {"cnt": 0}

    def _boom(*args, **kwargs):
        called["cnt"] += 1
        raise RuntimeError("compressed_attention_fa2 should not be called when disabled")

    monkeypatch.setattr(K, "compressed_attention_fa2", _boom)

    # Explicitly disable FA-2 via env
    monkeypatch.setenv("NSA_USE_FA2", "0")
    monkeypatch.setenv("NSA_USE_FA2_CMP", "0")

    nsa = _make_nsa_module()
    x = torch.randn(1, 8, 64)
    kv = _make_kv(S=8, d_k=16, d_v=16)

    # Should not raise even though the FA-2 function would explode if called
    y, _ = nsa(x, kv, prefill=True)
    assert torch.isfinite(y).all()
    assert called["cnt"] == 0, "FA-2 path was invoked despite being disabled"


def test_fa2_enabled_attempts_call(monkeypatch):
    # Force batched prefill path
    monkeypatch.setenv("NSA_PREFILL_BATCHED", "1")
    # Enable FA-2 via env
    monkeypatch.setenv("NSA_USE_FA2", "1")
    monkeypatch.setenv("NSA_USE_FA2_CMP", "1")

    # Monkeypatch FA-2 compressed entry to raise to prove it was attempted
    import nsa.core.attention_kernels as K

    def _boom(*args, **kwargs):
        raise RuntimeError("FA-2 compressed path invoked (expected under enable)")

    monkeypatch.setattr(K, "compressed_attention_fa2", _boom)

    nsa = _make_nsa_module()
    x = torch.randn(1, 8, 64)
    kv = _make_kv(S=8, d_k=16, d_v=16)

    with pytest.raises(RuntimeError, match="FA-2 compressed path invoked"):
        nsa(x, kv, prefill=True)

