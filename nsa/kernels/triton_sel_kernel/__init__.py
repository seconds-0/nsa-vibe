import os
from typing import Optional, Tuple

import torch


def _env_true(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "1" if default else "0").lower()
    return v in ("1", "true", "yes", "on")


def triton_sel_available() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


def selection_attention_triton(
    Q: torch.Tensor,      # [B,S,G,h,Dk]
    K: torch.Tensor,      # [B,G,S_kv,Dk]
    V: torch.Tensor,      # [B,G,S_kv,Dv]
    ranges: torch.Tensor, # [B,S,G,n,2]
    *,
    use_packed_fallback: bool = True,
) -> torch.Tensor:
    """
    Selection attention via Triton kernel (M4). Until kernels are implemented,
    this wrapper falls back to the packed or gather SDPA reference.
    """
    # If Triton disabled or unavailable, fall back to reference implementations
    if not (_env_true("NSA_USE_TRITON_SEL", False) and triton_sel_available()):
        from nsa.core.attention_kernels import (
            grouped_selection_attention_packed,
            grouped_selection_attention,
        )
        if use_packed_fallback:
            return grouped_selection_attention_packed(Q, K, V, ranges)
        return grouped_selection_attention(Q, K, V, ranges)

    # TODO(M4): Implement Triton forward path; for now, defer to packed fallback when rows have small L
    # Simple heuristic: use Triton only if total selected length per row ≥ sel_triton_min_L
    try:
        min_L = int(os.getenv("NSA_SEL_TRITON_MIN_L", "64"))
    except Exception:
        min_L = 64

    B, S, G, n, _ = ranges.shape
    # Compute total selected length per (b,t,g)
    lengths = (ranges[..., 1] - ranges[..., 0]).clamp_min(0)  # [B,S,G,n]
    total_L = lengths.sum(dim=-1)  # [B,S,G]
    if (total_L < min_L).all():
        from nsa.core.attention_kernels import grouped_selection_attention_packed
        return grouped_selection_attention_packed(Q, K, V, ranges)

    # Pack rows by identical total_L to minimize padding; call Triton per bucket
    B, S, G, h, D = Q.shape
    Dv = V.shape[-1]
    # Build index list and L per (b,t,g)
    idx_map = []  # list of (b,t,g,L, spans)
    for b in range(B):
        for t in range(S):
            for g in range(G):
                r = ranges[b, t, g]
                spans = []
                for i in range(r.shape[0]):
                    s0 = int(r[i, 0].item())
                    e0 = int(r[i, 1].item())
                    if e0 > s0:
                        spans.append((s0, e0))
                L_i = sum(e - s for (s, e) in spans)
                idx_map.append((b, t, g, L_i, spans))
    # Bucket by L
    buckets: dict[int, list[Tuple[int, int, int, list[Tuple[int, int]]]]] = {}
    for b, t, g, L_i, spans in idx_map:
        buckets.setdefault(L_i, []).append((b, t, g, spans))
    O = torch.zeros((B, S, G, h, Dv), device=Q.device, dtype=V.dtype)
    try:
        from .sel_fwd import sel_attn_fwd_dense, sel_attn_fwd_varlen
        for L_i, items in buckets.items():
            if L_i == 0:
                continue
            N = len(items)
            Q_pack = torch.empty((N, h, D), device=Q.device, dtype=Q.dtype)
            # Build varlen packed K/V and cu_seqlens
            total_L = N * L_i
            K_pack = torch.empty((total_L, D), device=Q.device, dtype=Q.dtype)
            V_pack = torch.empty((total_L, Dv), device=Q.device, dtype=V.dtype)
            cu = torch.empty((N + 1,), device=Q.device, dtype=torch.int32)
            cu[0] = 0
            write = 0
            for j, (b, t, g, spans) in enumerate(items):
                Q_pack[j] = Q[b, t, g]
                idx = torch.cat([torch.arange(s, e, device=Q.device) for (s, e) in spans], dim=0)
                Lw = idx.shape[0]
                K_pack[write : write + Lw] = K[b, g, idx]
                V_pack[write : write + Lw] = V[b, g, idx]
                write += Lw
                cu[j + 1] = write
            O_pack = sel_attn_fwd_varlen(Q_pack, K_pack, V_pack, cu)
            # Scatter back
            for j, (b, t, g, _) in enumerate(items):
                O[b, t, g] = O_pack[j]
        # Observability: estimate tokens/bytes read
        from nsa.core.debug import log
        total_tokens = int(sum(L_i * len(items) for L_i, items in buckets.items())) if buckets else 0
        bytes_k = int(total_tokens * D * Q.element_size() if buckets else 0)
        bytes_v = int(total_tokens * Dv * V.element_size() if buckets else 0)
        log("sel.triton.reads", total_tokens=total_tokens, bytes_k=bytes_k, bytes_v=bytes_v, buckets=len(buckets))
    except Exception:
        from nsa.core.attention_kernels import grouped_selection_attention_packed
        return grouped_selection_attention_packed(Q, K, V, ranges)
    return O


