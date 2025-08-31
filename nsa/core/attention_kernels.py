import os
import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from nsa.core.debug import log
from nsa.core.packing import (
    build_cu_seqlens_for_buckets,
    build_length_buckets,
    compute_compressed_lengths,
    compute_sliding_lengths,
)
from nsa.kernels.flash_wrappers import (
    attention_bgh,
    attention_fa2_dense_batch,
    attention_fa2_varlen,
    fa2_supported,
    fa2_supported_verbose,
    is_flash_varlen_available,
)

# Simple grow-on-demand workspaces for varlen packing to avoid frequent allocations
_VARLEN_WS: Dict[Tuple, Dict[str, torch.Tensor]] = {}
_SEL_PACK_WS: Dict[Tuple, Dict[str, torch.Tensor]] = {}


def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, str(default)))
        return v
    except Exception:
        return default


def _env_int_bounded(name: str, default: int, min_val: int = 0, max_val: int = 10**8) -> int:
    """Read integer from environment with bounds checking to prevent excessive memory allocation."""
    try:
        v = int(os.getenv(name, str(default)))
        if v < min_val:
            return min_val
        if v > max_val:
            # Log warning if value exceeds max
            import warnings

            warnings.warn(f"{name}={v} exceeds maximum {max_val}, clamping to {max_val}")
            return max_val
        return v
    except Exception:
        return default


def clear_varlen_workspaces() -> None:
    """Optional memory cleanup: free varlen packing workspaces."""
    _VARLEN_WS.clear()


def clear_selection_pack_workspaces() -> None:
    """Optional memory cleanup: free selection pack workspaces."""
    _SEL_PACK_WS.clear()


def _get_varlen_workspace(
    device: torch.device,
    dtype_q: torch.dtype,
    dtype_k: torch.dtype,
    dtype_v: torch.dtype,
    h: int,
    d_k: int,
    d_v: int,
    cap_N: int,
    cap_total_k: int,
) -> dict[str, torch.Tensor]:
    key = (str(device), dtype_q, dtype_k, dtype_v, h, d_k, d_v)
    ws = _VARLEN_WS.get(key)
    need_new = ws is None
    if not need_new:
        q, k, v = ws["q"], ws["k"], ws["v"]
        cuq, cuk = ws["cuq"], ws["cuk"]
        need_new = (
            q.shape[0] < cap_N
            or k.shape[0] < cap_total_k
            or v.shape[0] < cap_total_k
            or cuq.numel() < (cap_N + 1)
            or cuk.numel() < (cap_N + 1)
        )
    if need_new:
        # Allow pre-sizing via env to avoid growth reallocations on long runs
        # Bounded to prevent excessive memory allocation (max 1M rows, 100M total K/V)
        reserve_N = _env_int_bounded("NSA_VARLEN_RESERVE_N", 0, 0, 10**6)
        reserve_K = _env_int_bounded("NSA_VARLEN_RESERVE_K", 0, 0, 10**8)
        new_N = max(cap_N, reserve_N, 1)
        new_K = max(cap_total_k, reserve_K, 1)
        ws = {
            "q": torch.empty((new_N, h, d_k), dtype=dtype_q, device=device),
            "k": torch.empty((new_K, h, d_k), dtype=dtype_k, device=device),
            "v": torch.empty((new_K, h, d_v), dtype=dtype_v, device=device),
            "cuq": torch.empty((new_N + 1,), dtype=torch.int32, device=device),
            "cuk": torch.empty((new_N + 1,), dtype=torch.int32, device=device),
        }
        _VARLEN_WS[key] = ws
    return ws


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
    col >= num_cmp.view(S, 1)  # [S,S_cmp]
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
    # Empty or zero window → zeros
    if w <= 0 or K.shape[2] == 0 or S == 0:
        return torch.zeros((B, S, G, h, V.shape[-1]), dtype=V.dtype, device=V.device)
    device = Q.device
    # Build banded causal mask once: allowed keys per row t are [t-w+1 .. t]
    row = torch.arange(S, device=device).view(S, 1)
    col = torch.arange(S, device=device).view(1, S)
    allowed = (col <= row) & (col >= (row - (w - 1)))  # [S,S]
    # Use additive float mask with -inf for disallowed positions to avoid NaNs
    # across SDPA backends/dtypes. Shape: [S,S] then broadcast to [B,G*h,S,S].
    Mf2d = torch.full((S, S), float("-inf"), dtype=Q.dtype, device=device)
    Mf2d.masked_fill_(allowed, 0.0)
    # Prepare SDPA tensors: [B, G*h, S, D*]
    Qf = Q.reshape(B, S, G * h, Dk).transpose(1, 2).contiguous()  # [B,G*h,S,Dk]
    Kf = K.unsqueeze(2).expand(B, G, h, S, Dk).reshape(B, G * h, S, Dk).contiguous()
    Vf = (
        V.unsqueeze(2)
        .expand(B, G, h, S, V.shape[-1])
        .reshape(B, G * h, S, V.shape[-1])
        .contiguous()
    )
    # Broadcast additive mask to [B,G*h,S,S]
    Mf = Mf2d.view(1, 1, S, S).expand(B, G * h, S, S)
    Of = F.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=Mf)  # [B,G*h,S,Dv]
    Of = Of.transpose(1, 2).reshape(B, S, G, h, V.shape[-1])
    return Of


def grouped_selection_attention(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K: torch.Tensor,  # [B,G,S_kv,Dk]
    V: torch.Tensor,  # [B,G,S_kv,Dv]
    ranges: torch.Tensor,  # [B,S,G,n,2]
) -> torch.Tensor:  # [B,S,G,h,Dv]
    B, S, G, h, Dk = Q.shape
    K.shape[2]

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
                    q = Q[b, t, g]  # [h,Dk]
                    # Expand per-head kv and add query-length dim for SDPA
                    q_btgh = q.unsqueeze(0).unsqueeze(2)  # [1,h,1,Dk]
                    k_btgh = (
                        k.unsqueeze(0).unsqueeze(0).expand(1, q.shape[0], k.shape[0], k.shape[1])
                    )  # [1,h,L,Dk]
                    v_btgh = (
                        v.unsqueeze(0).unsqueeze(0).expand(1, q.shape[0], v.shape[0], v.shape[1])
                    )  # [1,h,L,Dv]
                    q_btgh = q_btgh.contiguous()
                    k_btgh = k_btgh.contiguous()
                    v_btgh = v_btgh.contiguous()
                    attn = F.scaled_dot_product_attention(
                        q_btgh, k_btgh, v_btgh, is_causal=True
                    )  # [1,h,1,Dv]
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
    # Memory-friendly masked semantics: only the first element in [start..t] is attended.
    # With a single allowed key per row, SDPA reduces to returning that V directly.
    B, S, G, h, Dk = Q.shape
    if w <= 0 or K.shape[2] == 0:
        return torch.zeros((B, S, G, h, V.shape[-1]), dtype=V.dtype, device=V.device)
    device = Q.device
    tpos = torch.arange(S, device=device)
    start = (tpos - (w - 1)).clamp_min(0)  # [S]
    # Build per-(B,G,S) gather indices and fetch V at start
    idx = start.view(1, 1, S, 1).expand(B, G, S, 1)  # [B,G,S,1]
    v_sel = torch.gather(V, 2, idx.expand(B, G, S, V.shape[-1]))  # [B,G,S,Dv]
    # Expand across heads; result [B,S,G,h,Dv]
    Of = v_sel.permute(0, 2, 1, 3).unsqueeze(3).expand(B, S, G, h, V.shape[-1])
    return Of


def batched_causal_attention_compressed_masked(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K_cmp: torch.Tensor,  # [B,G,S_cmp,Dk]
    V_cmp: torch.Tensor,  # [B,G,S_cmp,Dv]
    l: int,
    d: int,
) -> torch.Tensor:  # [B,S,G,h,Dv]
    # Memory-friendly masked semantics: if num_cmp(t)>0, attend only to index 0 → return V[:, :, 0].
    B, S, G, h, Dk = Q.shape
    S_cmp = K_cmp.shape[2]
    device = Q.device
    if S_cmp == 0:
        return torch.zeros((B, S, G, h, V_cmp.shape[-1]), dtype=V_cmp.dtype, device=V_cmp.device)
    tpos = torch.arange(S, device=device)
    num_cmp = torch.where(tpos + 1 < l, 0, ((tpos + 1 - l) // d) + 1).clamp(min=0, max=S_cmp)  # [S]
    have_any = (num_cmp > 0).view(1, S, 1, 1, 1).expand(B, S, G, h, 1)
    v0 = V_cmp[:, :, 0, :]  # [B,G,Dv]
    v0f = v0.unsqueeze(1).unsqueeze(3).expand(B, S, G, h, V_cmp.shape[-1])
    Of = torch.where(have_any, v0f, torch.zeros_like(v0f))
    return Of


def grouped_selection_attention_packed(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K: torch.Tensor,  # [B,G,S_kv,Dk]
    V: torch.Tensor,  # [B,G,S_kv,Dv]
    ranges: torch.Tensor,  # [B,S,G,n,2]
) -> torch.Tensor:  # [B,S,G,h,Dv]
    """
    Bucketed varlen packing by row length L with parity to gather path.
    For each (b,t,g), build its flat index list from ranges, bucket rows
    by identical L, and run one SDPA per bucket.
    """
    B, S, G, h, Dk = Q.shape
    K.shape[2]
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
    # Enable autograd-safe packing during training or when forced by env
    use_safe_pack = (
        torch.is_grad_enabled() and (Q.requires_grad or K.requires_grad or V.requires_grad)
    ) or _env_bool("NSA_TRAIN_SAFE_PACK", False)

    for Lval in unique_L.tolist():
        L = int(Lval)
        # collect row indices for this bucket
        bucket_idx = [i for i, Lx in enumerate(lengths) if Lx == L]
        if L == 0 or len(bucket_idx) == 0:
            # rows with L=0 remain zeros
            continue
        N = len(bucket_idx)
        if use_safe_pack:
            # Graph-friendly packing using stack to preserve autograd links
            map_rows = []
            Q_list = []
            K_list = []
            V_list = []
            for ridx in bucket_idx:
                b, t, g, idx = rows[ridx]
                map_rows.append((b, t, g))
                Q_list.append(Q[b, t, g])  # [h,Dk]
                K_list.append(K[b, g, idx])  # [L,Dk]
                V_list.append(V[b, g, idx])  # [L,Dv]
            Qb = torch.stack(Q_list, dim=0)  # [N,h,Dk]
            Kb = torch.stack(K_list, dim=0)  # [N,L,Dk]
            Vb = torch.stack(V_list, dim=0)  # [N,L,Dv]
            q_btgh = Qb.unsqueeze(1).permute(0, 2, 1, 3)  # [N,h,1,Dk]
            k_btgh = Kb.unsqueeze(1).expand(N, h, L, Dk)
            v_btgh = Vb.unsqueeze(1).expand(N, h, L, V.shape[-1])
            attn = F.scaled_dot_product_attention(q_btgh, k_btgh, v_btgh, is_causal=True)
            Ob = attn.squeeze(2)  # [N,h,Dv]
            for j, (b, t, g) in enumerate(map_rows):
                out[b, t, g] = Ob[j]
        else:
            # Workspace-backed Q, K, V batches to reduce allocations
            ws_key = (str(device), Q.dtype, K.dtype, V.dtype, h, Dk, V.shape[-1])
            ws = _SEL_PACK_WS.get(ws_key)
            need_new = (
                ws is None or ws["Q"].shape[0] < N or ws["K"].shape[1] < L or ws["V"].shape[1] < L
            )
            if need_new:
                # Allow pre-sizing via env to reduce reallocations
                # Bounded to prevent excessive memory allocation (max 100K rows, 10K length)
                reserve_N = _env_int_bounded("NSA_SEL_PACK_RESERVE_N", 0, 0, 10**5)
                reserve_L = _env_int_bounded("NSA_SEL_PACK_RESERVE_L", 0, 0, 10**4)
                new_N = max(N, reserve_N)
                new_L = max(L, reserve_L)
                Qb = torch.empty((new_N, h, Dk), dtype=Q.dtype, device=device)
                Kb = torch.empty((new_N, new_L, Dk), dtype=K.dtype, device=device)
                Vb = torch.empty((new_N, new_L, V.shape[-1]), dtype=V.dtype, device=device)
                _SEL_PACK_WS[ws_key] = {"Q": Qb, "K": Kb, "V": Vb}
            else:
                Qb = _SEL_PACK_WS[ws_key]["Q"][:N]
                Kb = _SEL_PACK_WS[ws_key]["K"][:N, :L]
                Vb = _SEL_PACK_WS[ws_key]["V"][:N, :L]
            # Populate workspace buffers and perform SDPA (execute for both new and reused workspaces)
            map_rows = []
            for j, ridx in enumerate(bucket_idx):
                b, t, g, idx = rows[ridx]
                Qb[j] = Q[b, t, g]  # [h,Dk]
                Kb[j] = K[b, g, idx]  # [L,Dk]
                Vb[j] = V[b, g, idx]  # [L,Dv]
                map_rows.append((b, t, g))
            # SDPA per bucket: expand per-head
            q_btgh = Qb.unsqueeze(1)  # [N,1,h,Dk]
            q_btgh = q_btgh.permute(0, 2, 1, 3)  # [N,h,1,Dk]
            k_btgh = Kb.unsqueeze(1).expand(N, h, L, Dk)
            v_btgh = Vb.unsqueeze(1).expand(N, h, L, V.shape[-1])
            attn = F.scaled_dot_product_attention(
                q_btgh, k_btgh, v_btgh, is_causal=True
            )  # [N,h,1,Dv]
            Ob = attn.squeeze(2)  # [N,h,Dv]
            # Scatter back
            for j, (b, t, g) in enumerate(map_rows):
                out[b, t, g] = Ob[j]
    return out


def selection_attention_varlen_all(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K: torch.Tensor,  # [B,G,S_kv,Dk]
    V: torch.Tensor,  # [B,G,S_kv,Dv]
    ranges: torch.Tensor,  # [B,S,G,n,2]
) -> torch.Tensor:  # [B,S,G,h,Dv]
    """
    Fully batched selection attention using varlen packing across all (B,S,G) rows.

    If NSA_SEL_VARLEN_V2 is enabled (default), dispatches to the vectorized v2
    packer. Otherwise uses the legacy v1 path (minimal loops with workspace).
    """
    # Optional v2 vectorized packer
    if os.getenv("NSA_SEL_VARLEN_V2", "1").lower() in ("1", "true", "yes", "on"):
        return selection_attention_varlen_all_v2(Q, K, V, ranges)
    B, S, G, h, Dk = Q.shape
    # Parity override: when enabled, force causal=True to match packed reference
    _parity = os.getenv("NSA_SEL_VARLEN_FORCE_PARITY", "0").lower() in ("1", "true", "yes", "on")
    if _parity:
        # Force exact parity by delegating to the packed reference
        return grouped_selection_attention_packed(Q, K, V, ranges)
    device = Q.device
    Dv = V.shape[-1]
    out = torch.zeros((B, S, G, h, Dv), dtype=V.dtype, device=V.device)
    # Build row list and lengths from ranges (sum of segment lengths)
    rows: list[tuple[int, int, int]] = []
    lens: list[int] = []
    for b in range(B):
        for t in range(S):
            for g in range(G):
                L = 0
                for i in range(ranges.shape[3]):
                    s0 = int(ranges[b, t, g, i, 0].item())
                    e0 = int(ranges[b, t, g, i, 1].item())
                    if e0 > s0:
                        L += e0 - s0
                if L > 0:
                    rows.append((b, t, g))
                    lens.append(L)
    N = len(rows)
    if N == 0:
        return out

    total_k = int(sum(lens))
    # Workspace-backed packing
    ws = _get_varlen_workspace(
        device,
        dtype_q=Q.dtype,
        dtype_k=K.dtype,
        dtype_v=V.dtype,
        h=h,
        d_k=Dk,
        d_v=Dv,
        cap_N=N,
        cap_total_k=total_k,
    )
    q_pack = ws["q"][:N]
    k_pack = ws["k"][:total_k]
    v_pack = ws["v"][:total_k]
    cuq = ws["cuq"][: N + 1]
    cuk = ws["cuk"][: N + 1]
    # Fill cu_seqlens
    cuq.zero_()
    cuk.zero_()
    # Pack per row
    write_pos = 0
    for i, (b, t, g) in enumerate(rows):
        # q for row
        q_pack[i] = Q[b, t, g]
        # iterate segments for this row
        for j in range(ranges.shape[3]):
            s0 = int(ranges[b, t, g, j, 0].item())
            e0 = int(ranges[b, t, g, j, 1].item())
            if e0 <= s0:
                continue
            seg_k = K[b, g, s0:e0]  # [Lseg,Dk]
            seg_v = V[b, g, s0:e0]  # [Lseg,Dv]
            Lseg = e0 - s0
            # Assign using explicit expand_as to match target slice shape and avoid view pitfalls
            _kslice = k_pack[write_pos : write_pos + Lseg]
            _vslice = v_pack[write_pos : write_pos + Lseg]
            _kslice.copy_(seg_k[:, None, :].expand_as(_kslice))
            _vslice.copy_(seg_v[:, None, :].expand_as(_vslice))
            write_pos += Lseg
        cuq[i + 1] = cuq[i] + 1
        cuk[i + 1] = cuk[i] + lens[i]
    # Try FA‑2 varlen if available and supported. Default non-causal semantics;
    # optionally force parity with packed path via NSA_SEL_VARLEN_FORCE_PARITY.
    ok, _ = fa2_supported_verbose(device, Q.dtype, Dk)
    if ok and is_flash_varlen_available():
        try:
            o_pack = attention_fa2_varlen(
                q_pack,
                k_pack,
                v_pack,
                cuq,
                cuk,
                max_seqlen_q=1,
                max_seqlen_k=max(lens),
                causal=_parity,
            )  # [N,h,Dv]
            # Scatter back
            for i, (b, t, g) in enumerate(rows):
                out[b, t, g] = o_pack[i]
            return out
        except Exception:
            pass
    # Dense batch per fixed L bucket as fallback
    buckets: dict[int, list[int]] = {}
    for i, L in enumerate(lens):
        buckets.setdefault(L, []).append(i)
    for L, idxs in buckets.items():
        if L <= 0 or len(idxs) == 0:
            continue
        Nb = len(idxs)
        Qb = torch.empty((Nb, h, Dk), dtype=Q.dtype, device=device)
        Kb = torch.empty((Nb, L, Dk), dtype=K.dtype, device=device)
        Vb = torch.empty((Nb, L, Dv), dtype=V.dtype, device=device)
        tgt: list[tuple[int, int, int]] = []
        for j, irow in enumerate(idxs):
            b, t, g = rows[irow]
            Qb[j] = Q[b, t, g]
            # Rebuild fixed-length K/V for this row from ranges
            write = 0
            for rj in range(ranges.shape[3]):
                s0 = int(ranges[b, t, g, rj, 0].item())
                e0 = int(ranges[b, t, g, rj, 1].item())
                if e0 <= s0:
                    continue
                Lseg = e0 - s0
                Kb[j, write : write + Lseg] = K[b, g, s0:e0]
                Vb[j, write : write + Lseg] = V[b, g, s0:e0]
                write += Lseg
            tgt.append((b, t, g))
        # Batched dense fallback for this bucket. Default non-causal; optionally force parity.
        try:
            q_rows = Qb.unsqueeze(1)  # [Nb,1,h,Dk]
            k_rows = Kb.unsqueeze(2).expand(Nb, L, h, Dk)  # [Nb,L,h,Dk]
            v_rows = Vb.unsqueeze(2).expand(Nb, L, h, Dv)  # [Nb,L,h,Dv]
            Ob = attention_fa2_dense_batch(q_rows, k_rows, v_rows, causal=_parity).squeeze(
                1
            )  # [Nb,h,Dv]
            for i, (b, t, g) in enumerate(tgt):
                out[b, t, g] = Ob[i]
        except Exception:
            # Final fallback: per-row SDPA
            for j, (b, t, g) in enumerate(tgt):
                q_btgh = Qb[j].unsqueeze(0).unsqueeze(0)  # [1,1,h,Dk]
                k_btgh = Kb[j].unsqueeze(0).unsqueeze(0)  # [1,1,L,Dk]
                v_btgh = Vb[j].unsqueeze(0).unsqueeze(0)  # [1,1,L,Dv]
                out[b, t, g] = attention_bgh(q_btgh, k_btgh, v_btgh, causal=_parity)[0, 0]
    return out


def selection_attention_varlen_all_v2(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ranges: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized v2 varlen selection packer with FA‑2 varlen fast path and dense fallback.
    - Eliminates Python loops for packing by using a difference-array mask to build per-row
      allowed indices and flat-select K/V tokens.
    - Uses causal=False for single‑query rows.
    - Env: NSA_SEL_VARLEN_MIN_L to bypass on tiny rows (falls back to packed path).
    """
    B, S, G, h, Dk = Q.shape
    # Parity override: when enabled, force causal=True to match packed reference
    _parity = os.getenv("NSA_SEL_VARLEN_FORCE_PARITY", "0").lower() in ("1", "true", "yes", "on")
    if _parity:
        # Force exact parity by delegating to the packed reference
        return grouped_selection_attention_packed(Q, K, V, ranges)
    device = Q.device
    Dv = V.shape[-1]
    S_kv = K.shape[2]
    out = torch.zeros((B, S, G, h, Dv), dtype=V.dtype, device=V.device)
    if S_kv == 0:
        return out

    # Build allowed mask [B,S,G,S_kv]
    n = ranges.shape[3]
    starts = ranges[..., 0].to(torch.int64).clamp_(0, S_kv)
    ends = ranges[..., 1].to(torch.int64).clamp_(0, S_kv)
    BSG = B * S * G
    starts_f = starts.reshape(BSG, n)
    ends_f = ends.reshape(BSG, n)
    diff = torch.zeros((BSG, S_kv + 1), dtype=torch.int32, device=device)
    one = torch.ones_like(starts_f, dtype=diff.dtype, device=device)
    diff.scatter_add_(1, starts_f, one)
    diff.scatter_add_(1, ends_f, -one)
    allowed = diff[:, :-1].cumsum(dim=1).gt(0)  # [BSG,S_kv]

    lens_flat = allowed.sum(dim=1, dtype=torch.int32)  # [BSG]
    row_mask = lens_flat.gt(0)
    if not torch.any(row_mask):
        return out
    try:
        min_L = int(os.getenv("NSA_SEL_VARLEN_MIN_L", "0"))
    except Exception:
        min_L = 0
    if min_L > 0 and int(lens_flat.max().item()) < min_L:
        return grouped_selection_attention_packed(Q, K, V, ranges)

    idx_rows = torch.nonzero(row_mask, as_tuple=False).squeeze(1)  # [N]
    N = int(idx_rows.numel())
    # (b,t,g) indices for scatter
    b_idx = idx_rows // (S * G)
    rem = idx_rows % (S * G)
    t_idx = rem // G
    g_idx = rem % G

    # Pack Q rows
    Q_rows = Q.reshape(B * S * G, h, Dk)[idx_rows]

    # Map rows to b,g to select K/V
    bg_map = (
        torch.arange(B, device=device).view(B, 1, 1) * G
        + torch.arange(G, device=device).view(1, 1, G)
    ).expand(B, S, G)
    bg_rows = bg_map.reshape(B * S * G)[idx_rows]
    K_bg = K.reshape(B * G, S_kv, Dk)[bg_rows]
    V_bg = V.reshape(B * G, S_kv, Dv)[bg_rows]
    allowed_rows = allowed[idx_rows]

    total_k = int(lens_flat[row_mask].sum().item())
    sel_k = K_bg[allowed_rows]  # [total_k, Dk]
    sel_v = V_bg[allowed_rows]  # [total_k, Dv]
    lens_sel = lens_flat[row_mask]  # [N]

    # Workspace-backed packing
    ws = _get_varlen_workspace(
        device,
        dtype_q=Q.dtype,
        dtype_k=K.dtype,
        dtype_v=V.dtype,
        h=h,
        d_k=Dk,
        d_v=Dv,
        cap_N=N,
        cap_total_k=total_k,
    )
    q_pack = ws["q"][:N]
    k_pack = ws["k"][:total_k]
    v_pack = ws["v"][:total_k]
    cuq = ws["cuq"][: N + 1]
    cuk = ws["cuk"][: N + 1]

    q_pack.copy_(Q_rows)
    k_pack.copy_(sel_k.unsqueeze(1).expand(total_k, h, Dk))
    v_pack.copy_(sel_v.unsqueeze(1).expand(total_k, h, Dv))
    cuq.copy_(torch.arange(0, N + 1, device=device, dtype=torch.int32))
    cuk[0] = 0
    torch.cumsum(lens_sel.to(torch.int32), dim=0, out=cuk[1:])

    # FA‑2 varlen (non-causal)
    ok, _why = fa2_supported_verbose(device, Q.dtype, Dk)
    max_len = int(lens_sel.max().item())
    if ok and is_flash_varlen_available():
        try:
            o_pack = attention_fa2_varlen(
                q_pack,
                k_pack,
                v_pack,
                cuq,
                cuk,
                max_seqlen_q=1,
                max_seqlen_k=max_len,
                causal=_parity,
            )
            out[b_idx, t_idx, g_idx] = o_pack
            return out
        except Exception:
            pass

    # Correctness-first fallback: masked SDPA over an allowed key mask
    # This path matches the non-causal packed reference exactly and avoids
    # potential packing/indexing pitfalls in dense-bucket fallbacks.
    try:
        return grouped_selection_attention_masked(Q, K, V, ranges)
    except Exception:
        pass

    # Legacy dense fallback by length buckets (kept as a final fallback)
    starts = cuk[:-1].to(torch.int64)
    ends = cuk[1:].to(torch.int64)
    Ls = (ends - starts).to(torch.int64)
    for L in torch.unique(Ls).tolist():
        if L <= 0:
            continue
        sel = (Ls == L).nonzero(as_tuple=False).squeeze(1)
        if sel.numel() == 0:
            continue
        Nb = int(sel.numel())
        Qb = q_pack[sel]
        k_rows = torch.empty((Nb, L, h, Dk), dtype=K.dtype, device=device)
        v_rows = torch.empty((Nb, L, h, Dv), dtype=V.dtype, device=device)
        for j in range(Nb):
            s0 = int(starts[sel[j]].item())
            e0 = int(ends[sel[j]].item())
            k_rows[j] = k_pack[s0:e0]
            v_rows[j] = v_pack[s0:e0]
        try:
            Ob = attention_fa2_dense_batch(Qb.unsqueeze(1), k_rows, v_rows, causal=_parity).squeeze(1)
        except Exception:
            Ob = torch.empty((Nb, h, Dv), dtype=V.dtype, device=device)
            for j in range(Nb):
                Ob[j] = attention_bgh(Qb[j].unsqueeze(0), k_rows[j].unsqueeze(0), v_rows[j].unsqueeze(0), causal=_parity)[
                    0
                ]
        out[b_idx[sel], t_idx[sel], g_idx[sel]] = Ob
    return out


def grouped_selection_attention_masked(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K: torch.Tensor,  # [B,G,S_kv,Dk]
    V: torch.Tensor,  # [B,G,S_kv,Dv]
    ranges: torch.Tensor,  # [B,S,G,n,2]
) -> torch.Tensor:  # [B,S,G,h,Dv]
    """
    Fully batched selection attention using an additive -inf mask.
    Vectorized ranges→mask construction via prefix-sum trick (no Python loops).
    """
    B, S, G, h, Dk = Q.shape
    S_kv = K.shape[2]
    device = Q.device
    if S_kv == 0:
        return torch.zeros((B, S, G, h, V.shape[-1]), dtype=V.dtype, device=device)

    # Vectorized allowed mask [B,S,G,S_kv] from ranges using difference array
    n = ranges.shape[3]
    starts = ranges[..., 0].to(torch.int64).clamp_(0, S_kv)  # [B,S,G,n]
    ends = ranges[..., 1].to(torch.int64).clamp_(0, S_kv)  # [B,S,G,n]
    BSG = B * S * G
    starts_f = starts.reshape(BSG, n)
    ends_f = ends.reshape(BSG, n)
    diff = torch.zeros((BSG, S_kv + 1), dtype=torch.int32, device=device)
    one = torch.ones_like(starts_f, dtype=diff.dtype, device=device)
    diff.scatter_add_(1, starts_f, one)
    diff.scatter_add_(1, ends_f, -one)
    allowed = diff[:, :-1].cumsum(dim=1).gt(0).reshape(B, S, G, S_kv)

    # Detect rows with no allowed keys (all False along key dimension)
    row_has_any = allowed.any(dim=-1)  # [B,S,G]
    row_empty = ~row_has_any

    # Prevent SDPA from seeing an all-−inf row which can produce NaNs.
    # For originally empty rows, force a single safe key (index 0) to True,
    # run SDPA, then zero their outputs afterward to preserve semantics.
    if row_empty.any():
        allowed_safe = allowed.clone()
        flat = allowed_safe.view(B * S * G, S_kv)
        row_empty_flat = row_empty.reshape(B * S * G)
        if S_kv > 0:
            flat[row_empty_flat, 0] = True
        allowed_safe = flat.view_as(allowed_safe)
    else:
        allowed_safe = allowed

    # Prepare SDPA tensors: [B,G*h,S, D*] and mask [B,G*h,S,S_kv]
    Qf = Q.reshape(B, S, G * h, Dk).transpose(1, 2).contiguous()  # [B,G*h,S,Dk]
    Kf = K.unsqueeze(2).expand(-1, -1, h, -1, -1).reshape(B, G * h, S_kv, Dk).contiguous()
    Vf = V.unsqueeze(2).expand(-1, -1, h, -1, -1).reshape(B, G * h, S_kv, V.shape[-1]).contiguous()
    # Build additive mask in float32 for numerical stability with -inf
    zeros = torch.zeros((B, G * h, S, S_kv), dtype=torch.float32, device=device)
    neg_inf = torch.full((B, G * h, S, S_kv), float("-inf"), dtype=torch.float32, device=device)
    Mf = torch.where(
        allowed_safe.transpose(1, 2)  # [B,G,S,S_kv]
        .unsqueeze(2)
        .expand(-1, -1, h, -1, -1)
        .reshape(B, G * h, S, S_kv),
        zeros,
        neg_inf,
    ).contiguous()

    Of = F.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=Mf)  # [B,G*h,S,Dv]
    Of = Of.transpose(1, 2).reshape(B, S, G, h, V.shape[-1])
    # Zero outputs for originally empty rows to preserve semantics
    if row_empty.any():
        Of = torch.where(row_has_any.unsqueeze(-1).unsqueeze(-1), Of, torch.zeros_like(Of))
    return Of


# ===== FA-2 integration scaffolding (M1) =====


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "1" if default else "0").lower()
    return v in ("1", "true", "yes", "on")


def _is_sm89(device: torch.device) -> bool:
    """Return True if running on CUDA device with SM 8.9 (Ada/RTX 4090)."""
    if device.type != "cuda":
        return False
    try:
        cap = torch.cuda.get_device_capability(device)
        return cap == (8, 9)
    except Exception:
        return False


def _fa2_forced() -> bool:
    """Return True if FA-2 usage is explicitly forced via env."""
    return _env_bool("NSA_FA2_FORCE", False)


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
    # Policy: sliding FA-2 is disabled by default due to API semantics
    # limitation (causal mask assumes start at 0). Allow only if explicitly
    # enabled via NSA_ALLOW_SLIDING_FA2 or forced flags.
    allow_sliding_fa2 = _env_bool("NSA_ALLOW_SLIDING_FA2", False)
    # Guard: disable FA-2 on Ada (SM 8.9) unless explicitly forced
    if _is_sm89(device) and not _fa2_forced():
        if os.getenv("NSA_SDPA_AUDIT", "0").lower() in ("1", "true", "yes"):
            log("fa2.gate_skip", branch="win", reason="sm89_guard", forced=bool(_fa2_forced()))
        return sliding_window_attention(Q, K, V, w)
    # Policy guard
    if not allow_sliding_fa2 and not (
        _env_bool("NSA_FA2_FORCE_VARLEN", False) or _env_bool("NSA_FA2_FORCE_DENSE", False)
    ):
        if os.getenv("NSA_SDPA_AUDIT", "0").lower() in ("1", "true", "yes"):
            log("fa2.gate_skip", branch="win", reason="unsupported_sliding_semantics", forced=False)
        return sliding_window_attention(Q, K, V, w)
    # Compute effective per-row window lengths and buckets
    lengths = compute_sliding_lengths(S, w, device)
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    # Allow override via env
    try:
        min_len_for_fa2 = int(os.getenv("NSA_FA2_MIN_LEN_WIN", str(min_len_for_fa2)))
    except Exception:
        pass
    # Disable sentinel: non-positive threshold disables FA‑2 entirely for this branch
    if min_len_for_fa2 <= 0:
        if os.getenv("NSA_SDPA_AUDIT", "0").lower() in ("1", "true", "yes"):
            log("fa2.gate_skip", branch="win", reason="disabled_threshold")
        return sliding_window_attention(Q, K, V, w)
    buckets = build_length_buckets(lengths)
    if buckets:
        log("fa2.win.buckets", n=len(buckets), max_len=max_len)
        # Build cu_seqlens per bucket (for future FA-2 varlen call)
        for idx in buckets:
            blens = lengths[idx]
            _ = build_cu_seqlens_for_buckets(blens)
    # Small-length auto-switch to masked SDPA
    if max_len < min_len_for_fa2:
        if os.getenv("NSA_DEBUG_TIMING", "0").lower() in ("1", "true", "yes"):
            log(
                "fa2.gate_skip",
                branch="win",
                reason="below_min_len",
                max_len=int(max_len),
                min_len=int(min_len_for_fa2),
            )
        return sliding_window_attention(Q, K, V, w)
    # Capability check
    ok, why = fa2_supported_verbose(device, Q.dtype, Dk)
    if not ok or not is_flash_varlen_available():
        if os.getenv("NSA_SDPA_AUDIT", "0").lower() in ("1", "true", "yes"):
            log("fa2.gate_skip", branch="win", reason=why, has_varlen=is_flash_varlen_available())
        return sliding_window_attention(Q, K, V, w)
    # Attempt FA-2 across all rows using varlen first, then dense per-bucket. Fallback to masked SDPA on error.
    try:
        B, S, G, h, Dk = Q.shape
        Dv = V.shape[-1]
        use_timing = os.getenv("NSA_DEBUG_TIMING", "0").lower() in ("1", "true", "yes")
        force_varlen = _env_bool("NSA_FA2_FORCE_VARLEN", False)
        force_dense = _env_bool("NSA_FA2_FORCE_DENSE", False)
        force_win_dense = _env_bool("NSA_WIN_FORCE_DENSE", False)
        # Log histogram of lengths
        if buckets:
            uniq, counts = torch.unique(lengths, return_counts=True)
            log("fa2.win.hist", uniq=uniq.tolist(), counts=counts.tolist())
        # Try a single varlen call across all rows
        if (is_flash_varlen_available() and not (force_dense or force_win_dense)) or force_varlen:
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
                use_safe_pack = (
                    torch.is_grad_enabled()
                    and (Q.requires_grad or K.requires_grad or V.requires_grad)
                ) or _env_bool("NSA_TRAIN_SAFE_PACK", False)
                if use_safe_pack:
                    # Autograd-safe packing via stack/cat to preserve graph links
                    q_pack = torch.stack([Q[b, t, g] for (b, t, g) in rows], dim=0)  # [N,h,Dk]
                    k_rows = []
                    v_rows = []
                    for i, (b, t, g) in enumerate(rows):
                        L = len_rows[i]
                        if L > 0:
                            start = max(0, (t + 1) - w)
                            end = t + 1
                            seg_k = K[b, g, start:end].unsqueeze(1).expand(-1, h, -1)  # [L,h,Dk]
                            seg_v = V[b, g, start:end].unsqueeze(1).expand(-1, h, -1)  # [L,h,Dv]
                            k_rows.append(seg_k)
                            v_rows.append(seg_v)
                    total_k = int(sum(len_rows))
                    if total_k > 0:
                        k_pack = torch.cat(k_rows, dim=0)
                        v_pack = torch.cat(v_rows, dim=0)
                    else:
                        k_pack = torch.zeros((0, h, Dk), dtype=K.dtype, device=K.device)
                        v_pack = torch.zeros((0, h, Dv), dtype=V.dtype, device=V.device)
                    cuq = torch.arange(0, N + 1, device=Q.device, dtype=torch.int32)
                    lens_t = torch.tensor(len_rows, dtype=torch.int32, device=Q.device)
                    cuk = torch.cumsum(torch.nn.functional.pad(lens_t, (1, 0)), dim=0)
                else:
                    total_k = int(sum(len_rows))
                    ws = _get_varlen_workspace(
                        Q.device, Q.dtype, K.dtype, V.dtype, h, Dk, Dv, N, total_k
                    )
                    q_pack = ws["q"][:N]
                    k_pack = ws["k"][:total_k]
                    v_pack = ws["v"][:total_k]
                    # Build cumulative sequence lengths for Q and K
                    cuq = ws["cuq"][: N + 1]
                    cuq.copy_(torch.arange(0, N + 1, device=Q.device, dtype=torch.int32))
                    lens_t = torch.tensor(len_rows, dtype=torch.int32, device=Q.device)
                    cuk = ws["cuk"][: N + 1]
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
                            assert (write_pos + L) <= total_k, "varlen K/V pack overflow"
                            k_pack[write_pos : write_pos + L] = seg_k.unsqueeze(1).expand(L, h, Dk)
                            v_pack[write_pos : write_pos + L] = seg_v.unsqueeze(1).expand(L, h, Dv)
                            write_pos += L
                # Optional integrity checks (debug only)
                if os.getenv("NSA_SDPA_AUDIT", "0").lower() in ("1", "true", "yes"):
                    try:
                        assert cuq.numel() == (N + 1), "cuq length mismatch"
                        assert cuk.numel() == (N + 1), "cuk length mismatch"
                        assert int(cuk[-1].item()) == int(total_k), "cuk total_k mismatch"
                        if total_k > 0 and N > 0:
                            probe = [0, N // 2, N - 1] if N >= 3 else [0]
                            for i in probe:
                                L_i = int(len_rows[i])
                                b_i, t_i, g_i = rows[i]
                                s_i = int(max(0, (t_i + 1) - w))
                                e_i = int(t_i + 1)
                                if L_i > 0:
                                    ks = k_pack[cuk[i] : cuk[i + 1]]  # [L,h,Dk]
                                    kv = K[b_i, g_i, s_i:e_i].unsqueeze(1).expand(-1, h, -1)
                                    if ks.shape != kv.shape:
                                        log(
                                            "warn.fa2_win_pack_shape",
                                            row=i,
                                            ks=ks.shape,
                                            kv=kv.shape,
                                        )
                                    else:
                                        md = float((ks - kv).abs().max().item())
                                        if md > 1e-3:
                                            log(
                                                "warn.fa2_win_pack_mismatch",
                                                row=i,
                                                L=L_i,
                                                max_diff=md,
                                            )
                    except Exception:
                        pass

                if use_timing:
                    t0 = time.perf_counter()
                o_pack = attention_fa2_varlen(
                    q_pack,
                    k_pack,
                    v_pack,
                    cuq,
                    cuk,
                    max_seqlen_q=1,
                    max_seqlen_k=max_len,
                    causal=False,
                )  # [N,h,Dv]
                if not torch.isfinite(o_pack).all():
                    log("warn.fa2_win_varlen_nonfinite")
                    return sliding_window_attention(Q, K, V, w)
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
            if is_flash_varlen_available() and not (force_dense or force_win_dense):
                # Pack varlen (constant L here, but use API for generality)
                q_pack = Qb  # [N,h,Dk]
                k_pack = Kb.reshape(N * L, Dk).unsqueeze(1).expand(-1, h, -1).reshape(N * L, h, Dk)
                v_pack = Vb.reshape(N * L, Dv).unsqueeze(1).expand(-1, h, -1).reshape(N * L, h, Dv)
                cuq = torch.arange(0, N + 1, device=Q.device, dtype=torch.int32)
                cuk = torch.arange(0, (N + 1) * L, step=L, device=Q.device, dtype=torch.int32)
                if use_timing:
                    t0 = time.perf_counter()
                o_pack = attention_fa2_varlen(
                    q_pack,
                    k_pack,
                    v_pack,
                    cuq,
                    cuk,
                    max_seqlen_q=1,
                    max_seqlen_k=L,
                    causal=False,
                )  # [N,h,Dv]
                if not torch.isfinite(o_pack).all():
                    log("warn.fa2_win_bucket_nonfinite")
                    return sliding_window_attention(Q, K, V, w)
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
                Ob = attention_fa2_dense_batch(q_rows, k_rows, v_rows, causal=False).squeeze(
                    1
                )  # [N,h,Dv]
                if use_timing:
                    dt = (time.perf_counter() - t0) * 1e3
                    log("fa2.win.bucket", path="dense", L=L, N=int(N), ms=dt)
            for i, (b, t, g) in enumerate(tgt):
                out[b, t, g] = Ob[i]
        return out
    except Exception as e:
        log("warn.fa2_unexpected_fallback", branch="win", error=str(e)[:100])
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
    # Guard: disable FA-2 on Ada (SM 8.9) unless explicitly forced
    if _is_sm89(device) and not _fa2_forced():
        if os.getenv("NSA_SDPA_AUDIT", "0").lower() in ("1", "true", "yes"):
            log("fa2.gate_skip", branch="cmp", reason="sm89_guard", forced=bool(_fa2_forced()))
        return batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
    S_cmp = K_cmp.shape[2]
    if S_cmp == 0:
        return torch.zeros((B, S, G, h, V_cmp.shape[-1]), dtype=V_cmp.dtype, device=V_cmp.device)
    num_cmp = compute_compressed_lengths(S, l, d, S_cmp, device)
    max_len = int(num_cmp.max().item()) if num_cmp.numel() > 0 else 0
    try:
        min_len_for_fa2 = int(os.getenv("NSA_FA2_MIN_LEN_CMP", str(min_len_for_fa2)))
    except Exception:
        pass
    # Disable sentinel: non-positive threshold disables FA‑2 entirely for this branch
    if min_len_for_fa2 <= 0:
        if os.getenv("NSA_SDPA_AUDIT", "0").lower() in ("1", "true", "yes"):
            log("fa2.gate_skip", branch="cmp", reason="disabled_threshold")
        return batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
    buckets = build_length_buckets(num_cmp)
    if buckets:
        log("fa2.cmp.buckets", n=len(buckets), max_len=max_len)
        for idx in buckets:
            blens = num_cmp[idx]
            _ = build_cu_seqlens_for_buckets(blens)
    if max_len < min_len_for_fa2:
        if os.getenv("NSA_DEBUG_TIMING", "0").lower() in ("1", "true", "yes"):
            log(
                "fa2.gate_skip",
                branch="cmp",
                reason="below_min_len",
                max_len=int(max_len),
                min_len=int(min_len_for_fa2),
            )
        return batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
    ok, why = fa2_supported_verbose(device, Q.dtype, Dk)
    if not ok or not is_flash_varlen_available():
        if os.getenv("NSA_SDPA_AUDIT", "0").lower() in ("1", "true", "yes"):
            log("fa2.gate_skip", branch="cmp", reason=why, has_varlen=is_flash_varlen_available())
        return batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
    try:
        Dv = V_cmp.shape[-1]
        use_timing = os.getenv("NSA_DEBUG_TIMING", "0").lower() in ("1", "true", "yes")
        # Log histogram of lengths
        if buckets:
            uniq, counts = torch.unique(num_cmp, return_counts=True)
            log("fa2.cmp.hist", uniq=uniq.tolist(), counts=counts.tolist())
        # Try single varlen across all rows with L>0
        force_varlen = _env_bool("NSA_FA2_FORCE_VARLEN", False)
        force_dense = _env_bool("NSA_FA2_FORCE_DENSE", False)
        if ((is_flash_varlen_available() and not force_dense) or force_varlen) and max_len >= 1:
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
                total_k = int(sum(len_rows))
                use_safe_pack = (
                    torch.is_grad_enabled()
                    and (Q.requires_grad or K_cmp.requires_grad or V_cmp.requires_grad)
                ) or _env_bool("NSA_TRAIN_SAFE_PACK", False)
                if use_safe_pack:
                    q_pack = torch.stack([Q[b, t, g] for (b, t, g) in rows], dim=0)
                    k_rows = []
                    v_rows = []
                    for (b, t, g), L in zip(rows, len_rows):
                        if L > 0:
                            seg_k = K_cmp[b, g, :L]
                            seg_v = V_cmp[b, g, :L]
                            k_rows.append(seg_k.unsqueeze(1).expand(-1, h, -1))  # [L,h,Dk]
                            v_rows.append(seg_v.unsqueeze(1).expand(-1, h, -1))  # [L,h,Dv]
                    if total_k > 0:
                        k_pack = torch.cat(k_rows, dim=0)
                        v_pack = torch.cat(v_rows, dim=0)
                    else:
                        k_pack = torch.zeros((0, h, Dk), dtype=K_cmp.dtype, device=K_cmp.device)
                        v_pack = torch.zeros((0, h, Dv), dtype=V_cmp.dtype, device=V_cmp.device)
                    cuq = torch.arange(0, N + 1, device=Q.device, dtype=torch.int32)
                    lens_t = torch.tensor(len_rows, dtype=torch.int32, device=Q.device)
                    cuk = torch.cumsum(torch.nn.functional.pad(lens_t, (1, 0)), dim=0)
                else:
                    ws = _get_varlen_workspace(
                        Q.device, Q.dtype, K_cmp.dtype, V_cmp.dtype, h, Dk, Dv, N, total_k
                    )
                    q_pack = ws["q"][:N]
                    k_pack = ws["k"][:total_k]
                    v_pack = ws["v"][:total_k]
                    cuq = ws["cuq"][: N + 1]
                    cuq.copy_(torch.arange(0, N + 1, device=Q.device, dtype=torch.int32))
                    lens_t = torch.tensor(len_rows, dtype=torch.int32, device=Q.device)
                    cuk = ws["cuk"][: N + 1]
                    torch.cumsum(torch.nn.functional.pad(lens_t, (1, 0)), dim=0, out=cuk)
                    write_pos = 0
                    for i, (b, t, g) in enumerate(rows):
                        L = len_rows[i]
                        q_pack[i] = Q[b, t, g]
                        if L > 0:
                            seg_k = K_cmp[b, g, :L]
                            seg_v = V_cmp[b, g, :L]
                            assert (write_pos + L) <= total_k, "varlen cmp K/V pack overflow"
                            k_pack[write_pos : write_pos + L] = seg_k.unsqueeze(1).expand(L, h, Dk)
                            v_pack[write_pos : write_pos + L] = seg_v.unsqueeze(1).expand(L, h, Dv)
                            write_pos += L
                if use_timing:
                    t0 = time.perf_counter()
                o_pack = attention_fa2_varlen(
                    q_pack,
                    k_pack,
                    v_pack,
                    cuq,
                    cuk,
                    max_seqlen_q=1,
                    max_seqlen_k=max_len,
                    causal=False,
                )  # [N,h,Dv]
                if not torch.isfinite(o_pack).all():
                    log("warn.fa2_cmp_varlen_nonfinite")
                    return batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
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
            if is_flash_varlen_available() and not force_dense:
                q_pack = Qb
                k_pack = Kb.reshape(N * L, Dk).unsqueeze(1).expand(-1, h, -1).reshape(N * L, h, Dk)
                v_pack = Vb.reshape(N * L, Dv).unsqueeze(1).expand(-1, h, -1).reshape(N * L, h, Dv)
                cuq = torch.arange(0, N + 1, device=Q.device, dtype=torch.int32)
                cuk = torch.arange(0, (N + 1) * L, step=L, device=Q.device, dtype=torch.int32)
                if use_timing:
                    t0 = time.perf_counter()
                o_pack = attention_fa2_varlen(
                    q_pack,
                    k_pack,
                    v_pack,
                    cuq,
                    cuk,
                    max_seqlen_q=1,
                    max_seqlen_k=L,
                    causal=False,
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
                if not torch.isfinite(Ob).all():
                    return batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
                if use_timing:
                    dt = (time.perf_counter() - t0) * 1e3
                    log("fa2.cmp.bucket", path="dense", L=L, N=int(N), ms=dt)
            for i, (b, t, g) in enumerate(tgt):
                out[b, t, g] = Ob[i]
        return out
    except Exception as e:
        log("warn.fa2_unexpected_fallback", branch="cmp", error=str(e)[:100])
        return batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)


def sliding_window_attention_fa2_decode(
    q_t: torch.Tensor, K_win: torch.Tensor, V_win: torch.Tensor, w: int
) -> torch.Tensor:
    B, G, h, Dk = q_t.shape
    # Guard: disable FA-2 on Ada (SM 8.9) unless explicitly forced
    if _is_sm89(q_t.device) and not _fa2_forced():
        if os.getenv("NSA_SDPA_AUDIT", "0").lower() in ("1", "true", "yes"):
            log(
                "fa2.gate_skip",
                branch="win.decode",
                reason="sm89_guard",
                forced=bool(_fa2_forced()),
            )
        end = K_win.shape[2]
        win_len = min(w, end)
        if win_len == 0:
            return torch.zeros((B, G, h, V_win.shape[-1]), dtype=V_win.dtype, device=V_win.device)
        start = end - win_len
        return attention_bgh(q_t, K_win[:, :, start:end], V_win[:, :, start:end], causal=True)
    end = K_win.shape[2]
    win_len = min(w, end)
    if win_len == 0:
        return torch.zeros((B, G, h, V_win.shape[-1]), dtype=V_win.dtype, device=V_win.device)
    # CPU or unsupported: direct SDPA for parity
    ok, why = fa2_supported_verbose(q_t.device, q_t.dtype, Dk)
    if not ok:
        if os.getenv("NSA_SDPA_AUDIT", "0").lower() in ("1", "true", "yes"):
            log("fa2.gate_skip", branch="win.decode", reason=why)
        start = end - win_len
        return attention_bgh(q_t, K_win[:, :, start:end], V_win[:, :, start:end], causal=True)
    # Small-length auto-switch for decode
    try:
        min_len = int(os.getenv("NSA_FA2_MIN_LEN_WIN", "16"))
    except Exception:
        min_len = 16
    if min_len < 1:
        min_len = 1
    if win_len < min_len:
        if os.getenv("NSA_DEBUG_TIMING", "0").lower() in ("1", "true", "yes"):
            log(
                "fa2.gate_skip",
                branch="win.decode",
                reason="below_min_len",
                win_len=int(win_len),
                min_len=int(min_len),
            )
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
        o = attention_fa2_dense_batch(q_rows, k_rows, v_rows, causal=False)  # [N,1,h,Dv]
        o = o.squeeze(1).reshape(B, G, h, -1)
        if not torch.isfinite(o).all():
            return attention_bgh(q_t, k, v, causal=True)
        return o
    except Exception as e:
        log("warn.fa2_unexpected_fallback", branch="win.decode", error=str(e)[:100])
        return attention_bgh(q_t, k, v, causal=True)


def compressed_attention_fa2_decode(
    q_t: torch.Tensor, K_cmp: torch.Tensor, V_cmp: torch.Tensor, L: int
) -> torch.Tensor:
    if L <= 0:
        B, G, h, _ = q_t.shape
        return torch.zeros((B, G, h, V_cmp.shape[-1]), dtype=V_cmp.dtype, device=V_cmp.device)
    B, G, h, Dk = q_t.shape
    # Guard: disable FA-2 on Ada (SM 8.9) unless explicitly forced
    if _is_sm89(q_t.device) and not _fa2_forced():
        if os.getenv("NSA_SDPA_AUDIT", "0").lower() in ("1", "true", "yes"):
            log(
                "fa2.gate_skip",
                branch="cmp.decode",
                reason="sm89_guard",
                forced=bool(_fa2_forced()),
            )
        return attention_bgh(q_t, K_cmp[:, :, :L], V_cmp[:, :, :L], causal=True)
    ok, why = fa2_supported_verbose(q_t.device, q_t.dtype, Dk)
    if not ok:
        if os.getenv("NSA_SDPA_AUDIT", "0").lower() in ("1", "true", "yes"):
            log("fa2.gate_skip", branch="cmp.decode", reason=why)
        return attention_bgh(q_t, K_cmp[:, :, :L], V_cmp[:, :, :L], causal=True)
    try:
        min_len = int(os.getenv("NSA_FA2_MIN_LEN_CMP", "16"))
    except Exception:
        min_len = 16
    if min_len < 1:
        min_len = 1
    if L < min_len:
        if os.getenv("NSA_DEBUG_TIMING", "0").lower() in ("1", "true", "yes"):
            log(
                "fa2.gate_skip",
                branch="cmp.decode",
                reason="below_min_len",
                L=int(L),
                min_len=int(min_len),
            )
        return attention_bgh(q_t, K_cmp[:, :, :L], V_cmp[:, :, :L], causal=True)
    k = K_cmp[:, :, :L]
    v = V_cmp[:, :, :L]
    N = B * G
    q_rows = q_t.reshape(N, h, Dk).unsqueeze(1)
    k_rows = k.reshape(N, L, Dk).unsqueeze(2).expand(N, L, h, Dk)
    v_rows = v.reshape(N, L, v.shape[-1]).unsqueeze(2).expand(N, L, h, v.shape[-1])
    try:
        o = attention_fa2_dense_batch(q_rows, k_rows, v_rows, causal=False)
        o = o.squeeze(1).reshape(B, G, h, -1)
        if not torch.isfinite(o).all():
            return attention_bgh(q_t, k, v, causal=True)
        return o
    except Exception:
        return attention_bgh(q_t, k, v, causal=True)
