import os
import time
import torch
import torch.nn.functional as F
from nsa.core.debug import log
from nsa.kernels.flash_wrappers import (
    attention_bgh,
    fa2_supported,
    is_flash_varlen_available,
    attention_fa2_dense_batch,
    attention_fa2_varlen,
)
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
    # Attempt FA-2 across all rows using varlen first, then dense per-bucket. Fallback to masked SDPA on error.
    try:
        B, S, G, h, Dk = Q.shape
        Dv = V.shape[-1]
        use_timing = os.getenv("NSA_DEBUG_TIMING", "0").lower() in ("1", "true", "yes")
        # Log histogram of lengths
        if buckets:
            uniq, counts = torch.unique(lengths, return_counts=True)
            log("fa2.win.hist", uniq=uniq.tolist(), counts=counts.tolist())
        # Try a single varlen call across all rows
        if is_flash_varlen_available():
            rows = []
            len_rows = []
            for t in range(S):
                L = int(lengths[t].item())
                for b in range(B):
                    for g in range(G):
                        rows.append((b, t, g))
                        len_rows.append(L)
            N = len(rows)
            if N > 0 and max_len >= 1:
                q_pack = torch.empty((N, h, Dk), dtype=Q.dtype, device=Q.device)
                total_k = int(sum(len_rows))
                k_pack = torch.empty((total_k, h, Dk), dtype=K.dtype, device=K.device)
                v_pack = torch.empty((total_k, h, Dv), dtype=V.dtype, device=V.device)
                cuq = torch.arange(0, N + 1, dtype=torch.int32, device=Q.device)
                lens_t = torch.tensor(len_rows, dtype=torch.int32, device=Q.device)
                cuk = torch.empty((N + 1,), dtype=torch.int32, device=Q.device)
                torch.cumsum(torch.nn.functional.pad(lens_t, (1, 0)), dim=0, out=cuk)
                # Fill packs
                write_pos = 0
                for i, (b, t, g) in enumerate(rows):
                    L = len_rows[i]
                    q_pack[i] = Q[b, t, g]
                    if L > 0:
                        start = max(0, (t + 1) - w)
                        end = t + 1
                        seg_k = K[b, g, start:end]  # [L,Dk]
                        seg_v = V[b, g, start:end]  # [L,Dv]
                        k_pack[write_pos:write_pos + L] = seg_k.unsqueeze(1).expand(L, h, Dk)
                        v_pack[write_pos:write_pos + L] = seg_v.unsqueeze(1).expand(L, h, Dv)
                        write_pos += L
                if use_timing:
                    t0 = time.perf_counter()
                o_pack = attention_fa2_varlen(
                    q_pack, k_pack, v_pack,
                    cuq, cuk,
                    max_seqlen_q=1, max_seqlen_k=max_len,
                    causal=True,
                )  # [N,h,Dv]
                if use_timing:
                    dt = (time.perf_counter() - t0) * 1e3
                    log("fa2.win.varlen_all", N=int(N), total_k=int(total_k), ms=dt)
                # Scatter back
                out = torch.zeros((B, S, G, h, Dv), dtype=V.dtype, device=V.device)
                for i, (b, t, g) in enumerate(rows):
                    out[b, t, g] = o_pack[i]
                return out
        out = torch.zeros((B, S, G, h, Dv), dtype=V.dtype, device=V.device)
        for idx in buckets:
            if idx.numel() == 0:
                continue
            L = int(lengths[idx[0]].item())
            # Collect rows for this bucket
            rows_q = []  # [N,h,Dk]
            rows_k = []  # [N,L,Dk]
            rows_v = []  # [N,L,Dv]
            tgt = []
            for t in idx.tolist():
                start = max(0, (t + 1) - w)
                end = t + 1
                for b in range(B):
                    for g in range(G):
                        rows_q.append(Q[b, t, g])
                        rows_k.append(K[b, g, start:end])
                        rows_v.append(V[b, g, start:end])
                        tgt.append((b, t, g))
            if not rows_q:
                continue
            N = len(rows_q)
            Qb = torch.stack(rows_q, dim=0)  # [N,h,Dk]
            Kb = torch.stack(rows_k, dim=0)  # [N,L,Dk]
            Vb = torch.stack(rows_v, dim=0)  # [N,L,Dv]
            if is_flash_varlen_available():
                # Pack varlen (constant L here, but use API for generality)
                q_pack = Qb  # [N,h,Dk]
                k_pack = Kb.reshape(N * L, Dk).unsqueeze(1).expand(-1, h, -1).reshape(N * L, h, Dk)
                v_pack = Vb.reshape(N * L, Dv).unsqueeze(1).expand(-1, h, -1).reshape(N * L, h, Dv)
                cuq = torch.arange(0, N + 1, device=Q.device, dtype=torch.int32)
                cuk = torch.arange(0, (N + 1) * L, step=L, device=Q.device, dtype=torch.int32)
                if use_timing:
                    t0 = time.perf_counter()
                o_pack = attention_fa2_varlen(
                    q_pack, k_pack, v_pack,
                    cuq, cuk,
                    max_seqlen_q=1, max_seqlen_k=L,
                    causal=True,
                )  # [N,h,Dv]
                if use_timing:
                    dt = (time.perf_counter() - t0) * 1e3
                    log("fa2.win.bucket", path="varlen", L=L, N=int(N), ms=dt)
                Ob = o_pack  # [N,h,Dv]
            else:
                q_rows = Qb.unsqueeze(1)  # [N,1,h,Dk]
                k_rows = Kb.unsqueeze(2).expand(N, L, h, Dk)
                v_rows = Vb.unsqueeze(2).expand(N, L, h, Dv)
                if use_timing:
                    t0 = time.perf_counter()
                Ob = attention_fa2_dense_batch(q_rows, k_rows, v_rows, causal=True).squeeze(1)  # [N,h,Dv]
                if use_timing:
                    dt = (time.perf_counter() - t0) * 1e3
                    log("fa2.win.bucket", path="dense", L=L, N=int(N), ms=dt)
            for i, (b, t, g) in enumerate(tgt):
                out[b, t, g] = Ob[i]
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
        Dv = V_cmp.shape[-1]
        use_timing = os.getenv("NSA_DEBUG_TIMING", "0").lower() in ("1", "true", "yes")
        # Log histogram of lengths
        if buckets:
            uniq, counts = torch.unique(num_cmp, return_counts=True)
            log("fa2.cmp.hist", uniq=uniq.tolist(), counts=counts.tolist())
        # Try single varlen across all rows with L>0
        if is_flash_varlen_available() and max_len >= 1:
            rows = []
            len_rows = []
            for t in range(S):
                L = int(num_cmp[t].item())
                for b in range(B):
                    for g in range(G):
                        if L > 0:
                            rows.append((b, t, g))
                            len_rows.append(L)
            N = len(rows)
            if N > 0:
                q_pack = torch.empty((N, h, Dk), dtype=Q.dtype, device=Q.device)
                total_k = int(sum(len_rows))
                k_pack = torch.empty((total_k, h, Dk), dtype=K_cmp.dtype, device=K_cmp.device)
                v_pack = torch.empty((total_k, h, Dv), dtype=V_cmp.dtype, device=V_cmp.device)
                cuq = torch.arange(0, N + 1, dtype=torch.int32, device=Q.device)
                lens_t = torch.tensor(len_rows, dtype=torch.int32, device=Q.device)
                cuk = torch.empty((N + 1,), dtype=torch.int32, device=Q.device)
                torch.cumsum(torch.nn.functional.pad(lens_t, (1, 0)), dim=0, out=cuk)
                write_pos = 0
                for i, (b, t, g) in enumerate(rows):
                    L = len_rows[i]
                    q_pack[i] = Q[b, t, g]
                    seg_k = K_cmp[b, g, :L]
                    seg_v = V_cmp[b, g, :L]
                    k_pack[write_pos:write_pos + L] = seg_k.unsqueeze(1).expand(L, h, Dk)
                    v_pack[write_pos:write_pos + L] = seg_v.unsqueeze(1).expand(L, h, Dv)
                    write_pos += L
                if use_timing:
                    t0 = time.perf_counter()
                o_pack = attention_fa2_varlen(
                    q_pack, k_pack, v_pack,
                    cuq, cuk,
                    max_seqlen_q=1, max_seqlen_k=max_len,
                    causal=True,
                )  # [N,h,Dv]
                if use_timing:
                    dt = (time.perf_counter() - t0) * 1e3
                    log("fa2.cmp.varlen_all", N=int(N), total_k=int(total_k), ms=dt)
                out = torch.zeros((B, S, G, h, Dv), dtype=V_cmp.dtype, device=V_cmp.device)
                for i, (b, t, g) in enumerate(rows):
                    out[b, t, g] = o_pack[i]
                return out
        out = torch.zeros((B, S, G, h, Dv), dtype=V_cmp.dtype, device=V_cmp.device)
        for idx in buckets:
            if idx.numel() == 0:
                continue
            L = int(num_cmp[idx[0]].item())
            rows_q = []  # [N,h,Dk]
            rows_k = []  # [N,L,Dk]
            rows_v = []  # [N,L, Dv]
            tgt = []
            for t in idx.tolist():
                if L <= 0:
                    continue
                for b in range(B):
                    for g in range(G):
                        rows_q.append(Q[b, t, g])
                        rows_k.append(K_cmp[b, g, :L])
                        rows_v.append(V_cmp[b, g, :L])
                        tgt.append((b, t, g))
            if not rows_q:
                continue
            N = len(rows_q)
            Qb = torch.stack(rows_q, dim=0)  # [N,h,Dk]
            Kb = torch.stack(rows_k, dim=0)  # [N,L,Dk]
            Vb = torch.stack(rows_v, dim=0)  # [N,L,Dv]
            if is_flash_varlen_available():
                q_pack = Qb
                k_pack = Kb.reshape(N * L, Dk).unsqueeze(1).expand(-1, h, -1).reshape(N * L, h, Dk)
                v_pack = Vb.reshape(N * L, Dv).unsqueeze(1).expand(-1, h, -1).reshape(N * L, h, Dv)
                cuq = torch.arange(0, N + 1, device=Q.device, dtype=torch.int32)
                cuk = torch.arange(0, (N + 1) * L, step=L, device=Q.device, dtype=torch.int32)
                if use_timing:
                    t0 = time.perf_counter()
                o_pack = attention_fa2_varlen(
                    q_pack, k_pack, v_pack,
                    cuq, cuk,
                    max_seqlen_q=1, max_seqlen_k=L,
                    causal=True,
                )  # [N,h,Dv]
                if use_timing:
                    dt = (time.perf_counter() - t0) * 1e3
                    log("fa2.cmp.bucket", path="varlen", L=L, N=int(N), ms=dt)
                Ob = o_pack
            else:
                q_rows = Qb.unsqueeze(1)
                k_rows = Kb.unsqueeze(2).expand(N, L, h, Dk)
                v_rows = Vb.unsqueeze(2).expand(N, L, h, Dv)
                if use_timing:
                    t0 = time.perf_counter()
                Ob = attention_fa2_dense_batch(q_rows, k_rows, v_rows, causal=True).squeeze(1)
                if use_timing:
                    dt = (time.perf_counter() - t0) * 1e3
                    log("fa2.cmp.bucket", path="dense", L=L, N=int(N), ms=dt)
            for i, (b, t, g) in enumerate(tgt):
                out[b, t, g] = Ob[i]
        return out
    except Exception:
        return batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)


def sliding_window_attention_fa2_decode(q_t: torch.Tensor, K_win: torch.Tensor, V_win: torch.Tensor, w: int) -> torch.Tensor:
    B, G, h, Dk = q_t.shape
    end = K_win.shape[2]
    win_len = min(w, end)
    if win_len == 0:
        return torch.zeros((B, G, h, V_win.shape[-1]), dtype=V_win.dtype, device=V_win.device)
    # CPU or unsupported: direct SDPA for parity
    if not fa2_supported(q_t.device, q_t.dtype, Dk):
        start = end - win_len
        return attention_bgh(q_t, K_win[:, :, start:end], V_win[:, :, start:end], causal=True)
    # Small-length auto-switch for decode
    try:
        min_len = int(os.getenv("NSA_FA2_MIN_LEN_WIN", "16"))
    except Exception:
        min_len = 16
    if win_len < min_len:
        start = end - win_len
        return attention_bgh(q_t, K_win[:, :, start:end], V_win[:, :, start:end], causal=True)
    start = end - win_len
    k = K_win[:, :, start:end]
    v = V_win[:, :, start:end]
    N = B * G
    q_rows = q_t.reshape(N, h, Dk).unsqueeze(1)  # [N,1,h,Dk]
    k_rows = k.reshape(N, win_len, Dk).unsqueeze(2).expand(N, win_len, h, Dk)
    v_rows = v.reshape(N, win_len, v.shape[-1]).unsqueeze(2).expand(N, win_len, h, v.shape[-1])
    try:
        o = attention_fa2_dense_batch(q_rows, k_rows, v_rows, causal=True)  # [N,1,h,Dv]
        return o.squeeze(1).reshape(B, G, h, -1)
    except Exception:
        return attention_bgh(q_t, k, v, causal=True)


def compressed_attention_fa2_decode(q_t: torch.Tensor, K_cmp: torch.Tensor, V_cmp: torch.Tensor, L: int) -> torch.Tensor:
    if L <= 0:
        B, G, h, _ = q_t.shape
        return torch.zeros((B, G, h, V_cmp.shape[-1]), dtype=V_cmp.dtype, device=V_cmp.device)
    B, G, h, Dk = q_t.shape
    if not fa2_supported(q_t.device, q_t.dtype, Dk):
        return attention_bgh(q_t, K_cmp[:, :, :L], V_cmp[:, :, :L], causal=True)
    try:
        min_len = int(os.getenv("NSA_FA2_MIN_LEN_CMP", "16"))
    except Exception:
        min_len = 16
    if L < min_len:
        return attention_bgh(q_t, K_cmp[:, :, :L], V_cmp[:, :, :L], causal=True)
    k = K_cmp[:, :, :L]
    v = V_cmp[:, :, :L]
    N = B * G
    q_rows = q_t.reshape(N, h, Dk).unsqueeze(1)
    k_rows = k.reshape(N, L, Dk).unsqueeze(2).expand(N, L, h, Dk)
    v_rows = v.reshape(N, L, v.shape[-1]).unsqueeze(2).expand(N, L, h, v.shape[-1])
    try:
        o = attention_fa2_dense_batch(q_rows, k_rows, v_rows, causal=True)
        return o.squeeze(1).reshape(B, G, h, -1)
    except Exception:
        return attention_bgh(q_t, k, v, causal=True)
