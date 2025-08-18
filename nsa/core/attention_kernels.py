import os
import torch
import torch.nn.functional as F
from nsa.core.debug import log
from nsa.kernels.flash_wrappers import attention_bgh, fa2_supported, is_flash_varlen_available
from nsa.core.packing import (
    compute_sliding_lengths,
    compute_compressed_lengths,
    build_length_buckets,
    build_cu_seqlens_for_buckets,
)


def batched_causal_attention_compressed(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K_cmp: torch.Tensor,  # [B,G,S_cmp,Dk]
    V_cmp: torch.Tensor,  # [B,G,S_cmp,Dv]
    l: int,
    d: int,
) -> torch.Tensor:  # [B,S,G,h,Dv]
    """
    Compressed branch attention with per-row causal mask derived from emission schedule.
    We cannot rely on is_causal=True due to S_q != S_kv and variable allowed lengths per t.
    """
    B, S, G, h, Dk = Q.shape
    S_cmp = K_cmp.shape[2]
    device = Q.device

    # num_cmp(t) = 0 if t+1 < l else floor((t+1 - l) / d) + 1, clamped to S_cmp
    tpos = torch.arange(S, device=device)
    num_cmp = torch.where(tpos + 1 < l, 0, ((tpos + 1 - l) // d) + 1).clamp(max=S_cmp)
    col = torch.arange(S_cmp, device=device).view(1, S_cmp)
    # disallowed mask: True means masked
    disallowed = col >= num_cmp.view(S, 1)  # [S,S_cmp]
    # Enforce token-level causality as well: no compressed tokens emitted from future blocks beyond t
    # When l=d=1, S_cmp == S and this reduces to standard causal

    # Parity-first: exact per-t using attention_bgh
    out = torch.zeros((B, S, G, h, V_cmp.shape[-1]), dtype=V_cmp.dtype, device=V_cmp.device)
    log("cmp.begin", B=B, S=S, S_cmp=int(S_cmp), l=l, d=d)
    for t in range(S):
        L = int(num_cmp[t].item())
        if L <= 0:
            out[:, t] = 0.0
            continue
        q_t = Q[:, t]
        k_t = K_cmp[:, :, :L, :]
        v_t = V_cmp[:, :, :L, :]
        out[:, t] = attention_bgh(q_t, k_t, v_t, causal=True)
        log("cmp.step", t=int(t), L=L)
    return out


def sliding_window_attention(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K: torch.Tensor,  # [B,G,S,Dk]
    V: torch.Tensor,  # [B,G,S,Dv]
    w: int,
) -> torch.Tensor:  # [B,S,G,h,Dv]
    B, S, G, h, Dk = Q.shape
    if w <= 0 or K.shape[2] == 0:
        return torch.zeros((B, S, G, h, V.shape[-1]), dtype=V.dtype, device=V.device)
    # Parity-first: per-t attention via attention_bgh
    out = torch.zeros((B, S, G, h, V.shape[-1]), dtype=V.dtype, device=V.device)
    for t in range(S):
        end = t + 1
        start = max(0, end - w)
        q_t = Q[:, t]
        k_t = K[:, :, start:end, :]
        v_t = V[:, :, start:end, :]
        out[:, t] = attention_bgh(q_t, k_t, v_t, causal=True)
        log("win.step", t=int(t), start=int(start), end=int(end))
    return out


def grouped_selection_attention(
    Q: torch.Tensor,      # [B,S,G,h,Dk]
    K: torch.Tensor,      # [B,G,S_kv,Dk]
    V: torch.Tensor,      # [B,G,S_kv,Dv]
    ranges: torch.Tensor, # [B,S,G,n,2]
) -> torch.Tensor:       # [B,S,G,h,Dv]
    B, S, G, h, Dk = Q.shape
    S_kv = K.shape[2]
    device = Q.device

    # Path 1: exact sequential-equivalence gather per (b,t,g)
    out = torch.zeros((B, S, G, h, V.shape[-1]), dtype=V.dtype, device=V.device)
    for b in range(B):
        for t in range(S):
            for g in range(G):
                # build exact gather index list
                idxs = []
                for i in range(ranges.shape[3]):
                    s0 = int(ranges[b, t, g, i, 0].item())
                    e0 = int(ranges[b, t, g, i, 1].item())
                    if e0 > s0:
                        idxs.append(torch.arange(s0, e0, device=V.device))
                if idxs:
                    idx = torch.cat(idxs)
                    k = K[b, g, idx]  # [L,Dk]
                    v = V[b, g, idx]  # [L,Dv]
                    q = Q[b, t, g]    # [h,Dk]
                    # Expand per-head kv and add query-length dim for SDPA
                    q_btgh = q.unsqueeze(0).unsqueeze(2)                 # [1,h,1,Dk]
                    k_btgh = k.unsqueeze(0).unsqueeze(0).expand(1, q.shape[0], k.shape[0], k.shape[1])  # [1,h,L,Dk]
                    v_btgh = v.unsqueeze(0).unsqueeze(0).expand(1, q.shape[0], v.shape[0], v.shape[1])  # [1,h,L,Dv]
                    attn = F.scaled_dot_product_attention(q_btgh, k_btgh, v_btgh, is_causal=True)  # [1,h,1,Dv]
                    out[b, t, g] = attn.squeeze(0).squeeze(1)  # [h,Dv]
                    log("sel.step", b=int(b), t=int(t), g=int(g), L=int(k.shape[0]))
                else:
                    out[b, t, g] = 0.0
                    log("sel.step", b=int(b), t=int(t), g=int(g), L=0)
    return out


def sliding_window_attention_masked(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K: torch.Tensor,  # [B,G,S,Dk]
    V: torch.Tensor,  # [B,G,S,Dv]
    w: int,
) -> torch.Tensor:  # [B,S,G,h,Dv]
    # Row-packed masked SDPA that mirrors current per-token + is_causal semantics:
    # within the [start..t] window, only the first element (start) is attended.
    B, S, G, h, Dk = Q.shape
    # Small-length auto-switch can be applied at the caller; leave here pure masked-SDPA
    if w <= 0 or K.shape[2] == 0:
        return torch.zeros((B, S, G, h, V.shape[-1]), dtype=V.dtype, device=V.device)
    device = Q.device
    BGH = B * G * h
    # Flatten groups/heads and build per-row packed Q
    Qfg = Q.permute(0, 2, 3, 1, 4).reshape(BGH, S, Dk)  # [BGH,S,Dk]
    Kfg = K.unsqueeze(2).expand(-1, -1, h, -1, -1).reshape(BGH, S, Dk)  # [BGH,S,Dk]
    Vfg = V.unsqueeze(2).expand(-1, -1, h, -1, -1).reshape(BGH, S, V.shape[-1])  # [BGH,S,Dv]
    Qrf = Qfg.reshape(BGH * S, 1, Dk)  # [BGH*S,1,Dk]
    Krf = Kfg.repeat_interleave(S, dim=0)  # [BGH*S,S,Dk]
    Vrf = Vfg.repeat_interleave(S, dim=0)  # [BGH*S,S,Dv]
    # Build allowed mask per row: only start = max(0, t-w+1) is allowed
    tpos = torch.arange(S, device=device)
    start = (tpos - (w - 1)).clamp_min(0)  # [S]
    # A[t,j] = (j == start[t])
    A = torch.zeros((S, S), dtype=torch.bool, device=device)
    A[torch.arange(S, device=device), start] = True
    zeros = torch.zeros((), dtype=Qrf.dtype, device=device)
    neg_inf = torch.full((), float("-inf"), dtype=Qrf.dtype, device=device)
    Mf = torch.where(A.unsqueeze(0).expand(BGH, S, S).reshape(BGH * S, S), zeros, neg_inf).unsqueeze(1)
    Of = F.scaled_dot_product_attention(Qrf, Krf, Vrf, attn_mask=Mf)
    Of = Of.reshape(BGH, S, V.shape[-1]).reshape(B, G, h, S, V.shape[-1]).permute(0, 3, 1, 2, 4)
    Of = torch.nan_to_num(Of, nan=0.0)
    return Of


def batched_causal_attention_compressed_masked(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K_cmp: torch.Tensor,  # [B,G,S_cmp,Dk]
    V_cmp: torch.Tensor,  # [B,G,S_cmp,Dv]
    l: int,
    d: int,
) -> torch.Tensor:  # [B,S,G,h,Dv]
    # Row-packed masked SDPA mirroring per-token + is_causal semantics:
    # when num_cmp(t)>0, only compressed index 0 is attended; else zero.
    B, S, G, h, Dk = Q.shape
    S_cmp = K_cmp.shape[2]
    device = Q.device
    if S_cmp == 0:
        return torch.zeros((B, S, G, h, V_cmp.shape[-1]), dtype=V_cmp.dtype, device=V_cmp.device)
    BGH = B * G * h
    Qfg = Q.permute(0, 2, 3, 1, 4).reshape(BGH, S, Dk)  # [BGH,S,Dk]
    Kfg = K_cmp.unsqueeze(2).expand(-1, -1, h, -1, -1).reshape(BGH, S_cmp, Dk)  # [BGH,S_cmp,Dk]
    Vfg = V_cmp.unsqueeze(2).expand(-1, -1, h, -1, -1).reshape(BGH, S_cmp, V_cmp.shape[-1])  # [BGH,S_cmp,Dv]
    Qrf = Qfg.reshape(BGH * S, 1, Dk)  # [BGH*S,1,Dk]
    Krf = Kfg.repeat_interleave(S, dim=0)  # [BGH*S,S_cmp,Dk]
    Vrf = Vfg.repeat_interleave(S, dim=0)  # [BGH*S,S_cmp,Dv]
    # Build per-row mask: allow only j==0 when num_cmp(t)>0
    tpos = torch.arange(S, device=device)
    num_cmp = torch.where(tpos + 1 < l, 0, ((tpos + 1 - l) // d) + 1).clamp(min=0, max=S_cmp)  # [S]
    A = torch.zeros((S, S_cmp), dtype=torch.bool, device=device)
    A[num_cmp > 0, 0] = True
    zeros = torch.zeros((), dtype=Qrf.dtype, device=device)
    neg_inf = torch.full((), float("-inf"), dtype=Qrf.dtype, device=device)
    Mf = torch.where(A.unsqueeze(0).expand(BGH, S, S_cmp).reshape(BGH * S, S_cmp), zeros, neg_inf).unsqueeze(1)
    Of = F.scaled_dot_product_attention(Qrf, Krf, Vrf, attn_mask=Mf)
    Of = Of.reshape(BGH, S, V_cmp.shape[-1]).reshape(B, G, h, S, V_cmp.shape[-1]).permute(0, 3, 1, 2, 4)
    Of = torch.nan_to_num(Of, nan=0.0)
    return Of

def grouped_selection_attention_packed(
    Q: torch.Tensor,      # [B,S,G,h,Dk]
    K: torch.Tensor,      # [B,G,S_kv,Dk]
    V: torch.Tensor,      # [B,G,S_kv,Dv]
    ranges: torch.Tensor, # [B,S,G,n,2]
) -> torch.Tensor:       # [B,S,G,h,Dv]
    """
    Bucketed varlen packing by row length L with parity to gather path.
    For each (b,t,g), build its flat index list from ranges, bucket rows
    by identical L, and run one SDPA per bucket.
    """
    B, S, G, h, Dk = Q.shape
    S_kv = K.shape[2]
    device = Q.device
    # Initialize output
    out = torch.zeros((B, S, G, h, V.shape[-1]), dtype=V.dtype, device=device)
    # Flatten to row indices
    rows = []  # list of (b,t,g, idx_tensor[L])
    lengths = []
    for b in range(B):
        for t in range(S):
            for g in range(G):
                idxs = []
                for i in range(ranges.shape[3]):
                    s0 = int(ranges[b, t, g, i, 0].item())
                    e0 = int(ranges[b, t, g, i, 1].item())
                    if e0 > s0:
                        idxs.append(torch.arange(s0, e0, device=device))
                if idxs:
                    idx = torch.cat(idxs)
                else:
                    idx = torch.empty((0,), dtype=torch.long, device=device)
                rows.append((b, t, g, idx))
                lengths.append(idx.numel())
    if not rows:
        return out
    lengths_t = torch.tensor(lengths, device=device)
    unique_L = torch.unique(lengths_t)
    for Lval in unique_L.tolist():
        L = int(Lval)
        # collect row indices for this bucket
        bucket_idx = [i for i, Lx in enumerate(lengths) if Lx == L]
        if L == 0 or len(bucket_idx) == 0:
            # rows with L=0 remain zeros
            continue
        N = len(bucket_idx)
        # Build Q, K, V batches
        Qb = torch.zeros((N, h, Dk), dtype=Q.dtype, device=device)
        Kb = torch.zeros((N, L, Dk), dtype=K.dtype, device=device)
        Vb = torch.zeros((N, L, V.shape[-1]), dtype=V.dtype, device=device)
        map_rows = []
        for j, ridx in enumerate(bucket_idx):
            b, t, g, idx = rows[ridx]
            Qb[j] = Q[b, t, g]                     # [h,Dk]
            Kb[j] = K[b, g, idx]                   # [L,Dk]
            Vb[j] = V[b, g, idx]                   # [L,Dv]
            map_rows.append((b, t, g))
        # SDPA per bucket: expand per-head
        q_btgh = Qb.unsqueeze(1)                   # [N,1,h,Dk]
        q_btgh = q_btgh.permute(0, 2, 1, 3)       # [N,h,1,Dk]
        k_btgh = Kb.unsqueeze(1).expand(N, h, L, Dk)
        v_btgh = Vb.unsqueeze(1).expand(N, h, L, V.shape[-1])
        attn = F.scaled_dot_product_attention(q_btgh, k_btgh, v_btgh, is_causal=True)  # [N,h,1,Dv]
        Ob = attn.squeeze(2)  # [N,h,Dv]
        # Scatter back
        for j, (b, t, g) in enumerate(map_rows):
            out[b, t, g] = Ob[j]
    return out


def grouped_selection_attention_masked(
    Q: torch.Tensor,      # [B,S,G,h,Dk]
    K: torch.Tensor,      # [B,G,S_kv,Dk]
    V: torch.Tensor,      # [B,G,S_kv,Dv]
    ranges: torch.Tensor, # [B,S,G,n,2]
) -> torch.Tensor:       # [B,S,G,h,Dv]
    """
    Fully batched selection attention using an additive -inf mask.
    Constructs an allowed mask from ranges for each (B,S,G) and runs a single
    SDPA per (B,G*h).
    """
    B, S, G, h, Dk = Q.shape
    S_kv = K.shape[2]
    device = Q.device

    # Build allowed mask [B,S,G,S_kv] from ranges
    allowed = torch.zeros((B, S, G, S_kv), dtype=torch.bool, device=device)
    n = ranges.shape[3]
    for b in range(B):
        for t in range(S):
            for g in range(G):
                for i in range(n):
                    s0 = int(ranges[b, t, g, i, 0].item())
                    e0 = int(ranges[b, t, g, i, 1].item())
                    if e0 > s0:
                        allowed[b, t, g, s0:e0] = True

    # Prepare SDPA tensors: [B,G*h,S, D*] and mask [B,G*h,S,S_kv]
    Qf = Q.reshape(B, S, G * h, Dk).transpose(1, 2)  # [B,G*h,S,Dk]
    Kf = K.unsqueeze(2).expand(-1, -1, h, -1, -1).reshape(B, G * h, S_kv, Dk)
    Vf = V.unsqueeze(2).expand(-1, -1, h, -1, -1).reshape(B, G * h, S_kv, V.shape[-1])
    zeros = torch.zeros((B, G * h, S, S_kv), dtype=Qf.dtype, device=device)
    neg_inf = torch.full((B, G * h, S, S_kv), float("-inf"), dtype=Qf.dtype, device=device)
    Mf = torch.where(allowed.reshape(B, S, G, S_kv).transpose(1, 2).reshape(B, G, S, S_kv)
                     .unsqueeze(2).expand(-1, -1, h, -1, -1)
                     .reshape(B, G * h, S, S_kv), zeros, neg_inf)

    Of = F.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=Mf)  # [B,G*h,S,Dv]
    return Of.transpose(1, 2).reshape(B, S, G, h, V.shape[-1])


# ===== FA-2 integration scaffolding (M1) =====

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "1" if default else "0").lower()
    return v in ("1", "true", "yes", "on")


def sliding_window_attention_fa2(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K: torch.Tensor,  # [B,G,S,Dk]
    V: torch.Tensor,  # [B,G,S,Dv]
    w: int,
    min_len_for_fa2: int = 16,
) -> torch.Tensor:
    """
    Planned FA-2 path for sliding with safe fallbacks.
    Currently falls back to masked SDPA to preserve numerics until FA-2 is wired.
    """
    B, S, G, h, Dk = Q.shape
    device = Q.device
    # Compute effective per-row window lengths and buckets
    lengths = compute_sliding_lengths(S, w, device)
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    # Allow override via env
    try:
        min_len_for_fa2 = int(os.getenv("NSA_FA2_MIN_LEN_WIN", str(min_len_for_fa2)))
    except Exception:
        pass
    buckets = build_length_buckets(lengths)
    if buckets:
        log("fa2.win.buckets", n=len(buckets), max_len=max_len)
        # Build cu_seqlens per bucket (for future FA-2 varlen call)
        for idx in buckets:
            blens = lengths[idx]
            _ = build_cu_seqlens_for_buckets(blens)
    # Small-length auto-switch to masked SDPA
    if max_len < min_len_for_fa2:
        return sliding_window_attention_masked(Q, K, V, w)
    # Capability check
    if not fa2_supported(device, Q.dtype, Dk) or not is_flash_varlen_available():
        return sliding_window_attention_masked(Q, K, V, w)
    # Attempt dense FA-2 per-bucket with query len=1 rows (fallback on any error)
    try:
        from flash_attn import flash_attn_func  # type: ignore
        B, S, G, h, Dk = Q.shape
        Dv = V.shape[-1]
        out = torch.zeros((B, S, G, h, Dv), dtype=V.dtype, device=V.device)
        for idx in buckets:
            if idx.numel() == 0:
                continue
            L = int(lengths[idx[0]].item())
            rows_q = []
            rows_k = []
            rows_v = []
            tgt = []
            for t in idx.tolist():
                start = max(0, (t + 1) - w)
                end = t + 1
                for b in range(B):
                    for g in range(G):
                        q_bgh = Q[b, t, g]  # [h,Dk]
                        k_seg = K[b, g, start:end]  # [L,Dk]
                        v_seg = V[b, g, start:end]  # [L,Dv]
                        # Expand K/V across heads
                        k_b = k_seg.unsqueeze(1).expand(L, h, Dk)  # [L,h,Dk]
                        v_b = v_seg.unsqueeze(1).expand(L, h, Dv)  # [L,h,Dv]
                        rows_q.append(q_bgh.unsqueeze(0))           # [1,h,Dk]
                        rows_k.append(k_b.unsqueeze(0))             # [1,L,h,Dk]
                        rows_v.append(v_b.unsqueeze(0))             # [1,L,h,Dv]
                        tgt.append((b, t, g))
            if not rows_q:
                continue
            q_batch = torch.cat(rows_q, dim=0).unsqueeze(1)       # [N,1,h,Dk]
            k_batch = torch.cat(rows_k, dim=0)                    # [N,L,h,Dk]
            v_batch = torch.cat(rows_v, dim=0)                    # [N,L,h,Dv]
            o_batch = flash_attn_func(q_batch, k_batch, v_batch, dropout_p=0.0, softmax_scale=None, causal=False)
            # o_batch: [N,1,h,Dv]
            for i, (b, t, g) in enumerate(tgt):
                out[b, t, g] = o_batch[i, 0]
        return out
    except Exception:
        return sliding_window_attention_masked(Q, K, V, w)


def compressed_attention_fa2(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K_cmp: torch.Tensor,  # [B,G,S_cmp,Dk]
    V_cmp: torch.Tensor,  # [B,G,S_cmp,Dv]
    l: int,
    d: int,
    min_len_for_fa2: int = 16,
) -> torch.Tensor:
    """
    Planned FA-2 path for compressed with safe fallbacks.
    Currently falls back to masked SDPA to preserve numerics until FA-2 is wired.
    """
    B, S, G, h, Dk = Q.shape
    device = Q.device
    S_cmp = K_cmp.shape[2]
    if S_cmp == 0:
        return torch.zeros((B, S, G, h, V_cmp.shape[-1]), dtype=V_cmp.dtype, device=V_cmp.device)
    num_cmp = compute_compressed_lengths(S, l, d, S_cmp, device)
    max_len = int(num_cmp.max().item()) if num_cmp.numel() > 0 else 0
    try:
        min_len_for_fa2 = int(os.getenv("NSA_FA2_MIN_LEN_CMP", str(min_len_for_fa2)))
    except Exception:
        pass
    buckets = build_length_buckets(num_cmp)
    if buckets:
        log("fa2.cmp.buckets", n=len(buckets), max_len=max_len)
        for idx in buckets:
            blens = num_cmp[idx]
            _ = build_cu_seqlens_for_buckets(blens)
    if max_len < min_len_for_fa2:
        return batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
    if not fa2_supported(device, Q.dtype, Dk) or not is_flash_varlen_available():
        return batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
    try:
        from flash_attn import flash_attn_func  # type: ignore
        Dv = V_cmp.shape[-1]
        out = torch.zeros((B, S, G, h, Dv), dtype=V_cmp.dtype, device=V_cmp.device)
        for idx in buckets:
            if idx.numel() == 0:
                continue
            L = int(num_cmp[idx[0]].item())
            rows_q = []
            rows_k = []
            rows_v = []
            tgt = []
            for t in idx.tolist():
                if L <= 0:
                    continue
                for b in range(B):
                    for g in range(G):
                        q_bgh = Q[b, t, g]                  # [h,Dk]
                        k_seg = K_cmp[b, g, :L]             # [L,Dk]
                        v_seg = V_cmp[b, g, :L]             # [L,Dv]
                        k_b = k_seg.unsqueeze(1).expand(L, h, Dk)
                        v_b = v_seg.unsqueeze(1).expand(L, h, Dv)
                        rows_q.append(q_bgh.unsqueeze(0))    # [1,h,Dk]
                        rows_k.append(k_b.unsqueeze(0))      # [1,L,h,Dk]
                        rows_v.append(v_b.unsqueeze(0))      # [1,L,h,Dv]
                        tgt.append((b, t, g))
            if not rows_q:
                continue
            q_batch = torch.cat(rows_q, dim=0).unsqueeze(1)       # [N,1,h,Dk]
            k_batch = torch.cat(rows_k, dim=0)                    # [N,L,h,Dk]
            v_batch = torch.cat(rows_v, dim=0)                    # [N,L,h,Dv]
            o_batch = flash_attn_func(q_batch, k_batch, v_batch, dropout_p=0.0, softmax_scale=None, causal=False)
            for i, (b, t, g) in enumerate(tgt):
                out[b, t, g] = o_batch[i, 0]
        return out
    except Exception:
        return batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
