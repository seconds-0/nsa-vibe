import torch
import torch.nn.functional as F

from nsa.core.nsa_attention import NSAAttention
from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta
from nsa.core.compress_pool import avg_pool_phi_rope_kv


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


def test_decode_group_consistency_and_causality_decode():
    torch.manual_seed(1)
    B, dim = 1, 64
    # Multiple heads per group to exercise group-consistency
    n_heads, G, d_k, d_v = 4, 2, 16, 16
    nsa = NSAAttention(dim=dim, n_heads=n_heads, n_kv_groups=G, d_k=d_k, d_v=d_v, l=4, d=2, l_sel=4, n_sel=4, w=6)
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
    # simulate a few decode steps and ensure outputs shape; causality implicitly enforced by attention kernels
    for _ in range(6):
        x = torch.randn(B, 1, dim)
        y, kv = nsa(x, kv, prefill=False)
        assert y.shape == (B, 1, dim)


def test_decode_smallS_equivalence_sliding_only():
    torch.manual_seed(2)
    B, dim = 1, 64
    n_heads, G, d_k, d_v = 4, 1, 16, 16
    # Configure coverage to include all tokens via sliding
    l, d, l_sel, n, w = 4, 2, 4, 4, 64
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
    )
    device = torch.device("cpu")
    # Force gates to sliding branch only
    nsa.gate.fc2.bias.data = torch.tensor([-1000.0, -1000.0, 1000.0])
    # Build empty KV and prefill a short context
    S_init = 6
    x_ctx = torch.randn(B, S_init, dim)
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
        meta=build_block_meta(S_init, l, d, l_sel, n_sel=n, w=w),
    )
    with torch.no_grad():
        _, kv = nsa(x_ctx, kv, prefill=True)

    steps = 4
    for step in range(steps):
        x_tok = torch.randn(B, 1, dim)
        y_nsa, kv = nsa(x_tok, kv, prefill=False)
        # Build full-attn reference using K_win/V_win caches (already RoPE'd), per-head
        H = n_heads
        h_per_group = nsa.h_per_group
        # Compute Q with absolute position as in decode path
        W_Q = nsa.W_Q
        Q_lin = W_Q(x_tok).view(B, 1, H, d_k)  # [B,1,H,Dk]
        # Reshape to [B,H,1,Dk] and apply RoPE with absolute position t
        Qh = Q_lin.permute(0, 2, 1, 3)  # [B,H,1,Dk]
        t_abs = kv.K_win.shape[2] - 1  # current last index
        pos = torch.arange(t_abs, t_abs + 1)
        from nsa.core.rope import apply_rope as rope_apply
        Qh = rope_apply(Qh, pos)
        # Expand cached K/V per head from group caches
        Kh = kv.K_win.repeat_interleave(h_per_group, dim=1)  # [B,H,S,Dk]
        Vh = kv.V_win.repeat_interleave(h_per_group, dim=1)  # [B,H,S,Dv]
        attn = F.scaled_dot_product_attention(Qh, Kh, Vh, is_causal=True)  # [B,H,1,Dv]
        O = attn.permute(0, 2, 1, 3).reshape(B, 1, H * d_v)
        y_ref = nsa.out(O)
        mae = (y_nsa - y_ref).abs().mean().item()
        assert mae < 1e-5


def test_decode_selection_only_equivalence_full_sel():
    torch.manual_seed(4)
    B, dim = 1, 48
    n_heads, G, d_k, d_v = 4, 1, 12, 12
    # Configure selection to cover all tokens: l=d=l_sel=1; n_sel large; w small
    l, d, l_sel, n, w = 1, 1, 1, 64, 0
    device = torch.device("cpu")
    nsa = NSAAttention(dim=dim, n_heads=n_heads, n_kv_groups=G, d_k=d_k, d_v=d_v, l=l, d=d, l_sel=l_sel, n_sel=n, w=w)
    # Force gates to selection branch only
    nsa.gate.fc2.bias.data = torch.tensor([-1000.0, 1000.0, -1000.0])
    # Prefill a short context
    S_init = 6
    x_ctx = torch.randn(B, S_init, dim, device=device)
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
        meta=build_block_meta(S_init, l, d, l_sel, n_sel=n, w=w),
    )
    with torch.no_grad():
        _, kv = nsa(x_ctx, kv, prefill=True)
    steps = 4
    for _ in range(steps):
        x_tok = torch.randn(B, 1, dim, device=device)
        y_nsa, kv = nsa(x_tok, kv, prefill=False)
        # Build full-attn reference using selection projections (covers all 0..t)
        H = n_heads
        W_Q = nsa.W_Q
        W_K = nsa.W_K_sel
        W_V = nsa.W_V_sel
        Qh = W_Q(x_tok).view(B, 1, H, d_k).permute(0, 2, 1, 3)
        Kh = kv.K_sel.repeat_interleave(nsa.h_per_group, dim=1)  # [B,H,S,Dk]
        Vh = kv.V_sel.repeat_interleave(nsa.h_per_group, dim=1)  # [B,H,S,Dv]
        attn = F.scaled_dot_product_attention(Qh, Kh, Vh, is_causal=True)
        y_ref = nsa.out(attn.permute(0, 2, 1, 3).reshape(B, 1, H * d_v))
        mae = (y_nsa - y_ref).abs().mean().item()
        assert mae < 1e-5


def test_cmp_decode_emission_parity_prefill_vs_decode():
    torch.manual_seed(3)
    B, S, dim = 1, 12, 32
    n_heads, G, d_k, d_v = 2, 1, 16, 16
    l, d, l_sel, n, w = 4, 2, 4, 4, 8
    device = torch.device("cpu")
    x = torch.randn(B, S, dim, device=device)
    nsa = NSAAttention(dim=dim, n_heads=n_heads, n_kv_groups=G, d_k=d_k, d_v=d_v, l=l, d=d, l_sel=l_sel, n_sel=n, w=w)
    # Build raw cmp K/V by projecting and shaping
    with torch.no_grad():
        K_raw = nsa._shape_kv(nsa.W_K_cmp(x), B, S)
        V_raw = nsa._shape_kv(nsa.W_V_cmp(x), B, S)
    # Prefill compressed with absolute positions 0..S-1
    K_cmp_full, V_cmp_full = avg_pool_phi_rope_kv(K_raw, V_raw, l, d, pos=torch.arange(S))
    # Decode incremental emission
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
        meta=build_block_meta(S, l, d, l_sel, n_sel=n, w=w),
    )
    # run decode steps feeding one token at a time
    for t in range(S):
        x_tok = x[:, t : t + 1]
        _, kv = nsa(x_tok, kv, prefill=False)
    # Compare compressed streams
    assert kv.K_cmp.shape == K_cmp_full.shape
    assert kv.V_cmp.shape == V_cmp_full.shape
    assert torch.allclose(kv.K_cmp, K_cmp_full, atol=1e-6)
    assert torch.allclose(kv.V_cmp, V_cmp_full, atol=1e-6)


