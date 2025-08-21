import os

import torch

from nsa.core.flags import env_true as _env_true


def triton_sel_available() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


def _normalize_ranges_tensor(ranges: torch.Tensor, S_kv: int) -> torch.Tensor:
    """Normalize selection ranges to shape [B,S,G,n,2] and clamp to [0, S_kv].
    Accepts extra singleton dimensions (over-nesting) and squeezes them safely.
    If 4D is provided (missing batch), unsqueezes batch at dim 0.
    """
    t = ranges
    # Squeeze any extra singleton dimensions until <= 5D
    while t.dim() > 5:
        squeezed = False
        for d in range(t.dim()):
            if t.size(d) == 1:
                t = t.squeeze(d)
                squeezed = True
                break
        if not squeezed:
            break
    if t.dim() == 4:
        t = t.unsqueeze(0)
    if t.dim() != 5:
        raise ValueError(f"ranges must be 5D [B,S,G,n,2] after normalization, got {t.shape}")
    if t.shape[-1] != 2:
        raise ValueError(f"ranges last dim must be 2 (start,end), got {t.shape}")
    return t.clamp(min=0, max=S_kv)


_PACK_CACHE: dict[int, dict[str, torch.Tensor]] = {}
_DEVICE_LOGGED: bool = False
_PACK_CACHE_MAX_ENTRIES: int = int(os.getenv("NSA_SEL_TRITON_PACK_CACHE_MAX_ENTRIES", "4"))
_PACK_CACHE_MAX_MB: int = int(os.getenv("NSA_SEL_TRITON_PACK_CACHE_MAX_MB", "512"))  # soft cap


def _pack_cache_total_bytes() -> int:
    total = 0
    for entry in _PACK_CACHE.values():
        for k in ("K", "V"):
            t = entry.get(k)
            if t is not None:
                total += t.numel() * t.element_size()
        cu = entry.get("cu")
        if cu is not None:
            total += cu.numel() * cu.element_size()
    return total


def _get_pack_buffers(
    device: torch.device,
    total_L: int,
    D: int,
    Dv: int,
    N: int,
    dtype_k: torch.dtype,
    dtype_v: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = device.index if device.type == "cuda" else -1
    buf = _PACK_CACHE.get(key)
    need_new = True
    if buf is not None:
        Kb, Vb, cu = buf["K"], buf["V"], buf["cu"]
        if Kb.numel() >= total_L * D and Vb.numel() >= total_L * Dv and cu.numel() >= (N + 1):
            need_new = False
    if need_new:
        Kb = torch.empty((total_L, D), device=device, dtype=dtype_k)
        Vb = torch.empty((total_L, Dv), device=device, dtype=dtype_v)
        cu = torch.empty((N + 1,), device=device, dtype=torch.int32)
        # simple size-limited cache to prevent unbounded growth
        if len(_PACK_CACHE) >= _PACK_CACHE_MAX_ENTRIES:
            _PACK_CACHE.clear()
        _PACK_CACHE[key] = {"K": Kb, "V": Vb, "cu": cu}
        # Soft memory pressure guard: evict all if exceeding cap
        try:
            max_bytes = _PACK_CACHE_MAX_MB * 1024 * 1024
            if _pack_cache_total_bytes() > max_bytes:
                from nsa.core.debug import log

                log("sel.triton.pack_cache_evict", reason="over_cap", cap_mb=_PACK_CACHE_MAX_MB)
                _PACK_CACHE.clear()
                _PACK_CACHE[key] = {"K": Kb, "V": Vb, "cu": cu}
        except Exception:
            pass
    else:
        Kb = _PACK_CACHE[key]["K"][:total_L]
        Vb = _PACK_CACHE[key]["V"][:total_L]
        cu = _PACK_CACHE[key]["cu"][: N + 1]
    return Kb, Vb, cu


def _log_device_once() -> None:
    global _DEVICE_LOGGED
    if _DEVICE_LOGGED:
        return
    try:
        import triton  # noqa: F401

        triton_ver = triton.__version__
    except Exception:
        triton_ver = "<no-triton>"
    try:
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        cuda_ver = torch.version.cuda
    except Exception:
        dev, cap, cuda_ver = "<unknown>", (0, 0), "<unknown>"
    from nsa.core.debug import log

    log("sel.triton.device", device=dev, sm=f"{cap[0]}.{cap[1]}", cuda=cuda_ver, triton=triton_ver)
    _DEVICE_LOGGED = True


class _SelAttnTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, ranges: torch.Tensor
    ) -> torch.Tensor:
        # Normalize ranges to 5D and clamp to valid [0, S_kv]
        S_kv = K.shape[2]
        ranges = _normalize_ranges_tensor(ranges, S_kv)
        ctx.save_for_backward(Q, K, V, ranges)
        # Call underlying implementation but mark we’re inside wrapper to avoid re-wrapping
        return selection_attention_triton(Q, K, V, ranges, _inside_wrapper=True)

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        Q, K, V, ranges = ctx.saved_tensors
        # Use the reference analytical backward to compute gradients directly
        dQ, dK, dV = _selection_attention_backward(Q, K, V, ranges, dO)
        return dQ, dK, dV, None


def _build_row_indices_from_ranges(
    ranges_row: torch.Tensor, S_kv: int, device: torch.device
) -> torch.Tensor:
    # ranges_row: [n,2] (int), clamp to [0, S_kv]
    n = ranges_row.shape[0]
    idx_parts = []
    for i in range(n):
        s0 = int(ranges_row[i, 0].item())
        e0 = int(ranges_row[i, 1].item())
        s0 = max(0, min(s0, S_kv))
        e0 = max(s0, min(e0, S_kv))
        if e0 > s0:
            idx_parts.append(torch.arange(s0, e0, device=device, dtype=torch.long))
    if not idx_parts:
        return torch.empty((0,), device=device, dtype=torch.long)
    return torch.cat(idx_parts, dim=0)


def _selection_attention_backward(
    Q: torch.Tensor,  # [B,S,G,h,D]
    K: torch.Tensor,  # [B,G,S_kv,D]
    V: torch.Tensor,  # [B,G,S_kv,Dv]
    ranges: torch.Tensor,  # [B,S,G,n,2]
    dO: torch.Tensor,  # [B,S,G,h,Dv]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Analytic grads matching packed SDPA semantics (q_len=1 causal => only first key allowed)
    assert Q.ndim == 5 and K.ndim == 4 and V.ndim == 4 and dO.ndim == 5
    B, S, G, h, D = Q.shape
    Dv = V.shape[-1]
    S_kv = K.shape[2]
    device = Q.device
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    rows = []
    lengths = []
    for b in range(B):
        for t in range(S):
            for g_ in range(G):
                idx = _build_row_indices_from_ranges(ranges[b, t, g_], S_kv, device)
                rows.append((b, t, g_, idx))
                lengths.append(int(idx.numel()))
    if not rows:
        return dQ, dK, dV
    lengths_t = torch.tensor(lengths, device=device)
    unique_L = torch.unique(lengths_t)
    scale = 1.0 / (D**0.5)
    for Lval in unique_L.tolist():
        L = int(Lval)
        bucket_idx = [i for i, Lx in enumerate(lengths) if Lx == L]
        if len(bucket_idx) == 0:
            continue
        if L == 0:
            continue
        N = len(bucket_idx)
        Qb = torch.empty((N, h, D), device=device, dtype=Q.dtype)
        Kb = torch.empty((N, L, D), device=device, dtype=K.dtype)
        Vb = torch.empty((N, L, Dv), device=device, dtype=V.dtype)
        dOb = torch.empty((N, h, Dv), device=device, dtype=dO.dtype)
        tgt = []
        for j, ridx in enumerate(bucket_idx):
            b, t, g_, idx = rows[ridx]
            Qb[j] = Q[b, t, g_]
            Kb[j] = K[b, g_, idx]
            Vb[j] = V[b, g_, idx]
            dOb[j] = dO[b, t, g_]
            tgt.append((b, t, g_, idx))
        Qf = Qb.to(torch.float32)
        Kf = Kb.to(torch.float32)
        Vf = Vb.to(torch.float32)
        dOf = dOb.to(torch.float32)
        logits = torch.matmul(Qf, Kf.transpose(1, 2)) * scale
        # Mirror packed path (q_len=1 causal): only first key contributes
        if logits.shape[-1] > 1:
            logits[..., 1:] = float("-inf")
        P = torch.softmax(logits, dim=-1)
        dVb = torch.einsum("nhl,nhv->nlv", P, dOf)
        dP = torch.matmul(dOf, Vf.transpose(1, 2))
        dp_dot_p = (dP * P).sum(dim=-1, keepdim=True)
        dS = (dP - dp_dot_p) * P
        dQb = torch.matmul(dS, Kf) * scale
        dKb = torch.einsum("nhl,nhd->nld", dS, Qf) * scale
        for j, (b, t, g_, idx) in enumerate(tgt):
            dQ[b, t, g_] = dQ[b, t, g_] + dQb[j].to(dQ.dtype)
            dV[b, g_, idx] = dV[b, g_, idx] + dVb[j].to(dV.dtype)
            dK[b, g_, idx] = dK[b, g_, idx] + dKb[j].to(dK.dtype)
    return dQ, dK, dV


def selection_attention_backward_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ranges: torch.Tensor,
    dO: torch.Tensor,
):
    return _selection_attention_backward(Q, K, V, ranges, dO)


def selection_attention_triton(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K: torch.Tensor,  # [B,G,S_kv,Dk]
    V: torch.Tensor,  # [B,G,S_kv,Dv]
    ranges: torch.Tensor,  # [B,S,G,n,2]
    *,
    use_packed_fallback: bool = True,
    _inside_wrapper: bool = False,
) -> torch.Tensor:
    # If gradients are enabled and allowed, route through autograd wrapper (avoid recursion)
    if (
        (not _inside_wrapper)
        and torch.is_grad_enabled()
        and (Q.requires_grad or K.requires_grad or V.requires_grad)
    ):
        if _env_true("NSA_SEL_TRITON_ALLOW_GRAD", False):
            return _SelAttnTritonFn.apply(Q, K, V, ranges)
        else:
            from nsa.core.attention_kernels import grouped_selection_attention_packed

            return grouped_selection_attention_packed(Q, K, V, ranges)
    S_kv = K.shape[2]
    ranges = _normalize_ranges_tensor(ranges, S_kv)
    if not (_env_true("NSA_USE_TRITON_SEL", False) and triton_sel_available()):
        from nsa.core.attention_kernels import (
            grouped_selection_attention,
            grouped_selection_attention_packed,
        )

        return (
            grouped_selection_attention_packed(Q, K, V, ranges)
            if use_packed_fallback
            else grouped_selection_attention(Q, K, V, ranges)
        )

    # Assertions / logs for contiguity & strides (opt-in)
    if _env_true("NSA_DEBUG_SHAPES", False):
        from nsa.core.debug import log

        log("sel.triton.input", Q=Q, K=K, V=V, ranges=ranges)
        assert Q.is_contiguous(), "Q must be contiguous"
        assert K.is_contiguous(), "K must be contiguous"
        assert V.is_contiguous(), "V must be contiguous"

    # Past this point, either no grad or we're already inside wrapper
    allowed_dtypes = {torch.float16, torch.bfloat16}
    if (
        Q.dtype not in allowed_dtypes
        or K.dtype not in allowed_dtypes
        or V.dtype not in allowed_dtypes
    ):
        from nsa.core.attention_kernels import grouped_selection_attention_packed

        return grouped_selection_attention_packed(Q, K, V, ranges)
    if _env_true("NSA_SEL_TRITON_REQUIRE_ALIGN", True):
        D = Q.shape[-1]
        Dv = V.shape[-1]
        if (D % 8 != 0) or (Dv % 8 != 0):
            from nsa.core.attention_kernels import grouped_selection_attention_packed

            return grouped_selection_attention_packed(Q, K, V, ranges)

    # Simple heuristic: use Triton only if total selected length per row ≥ sel_triton_min_L
    try:
        # Conservative default: keep Triton effectively off unless explicitly enabled.
        min_L = int(os.getenv("NSA_SEL_TRITON_MIN_L", "4096"))
    except Exception:
        min_L = 4096

    B, S, G, n, _ = ranges.shape
    lengths = (ranges[..., 1] - ranges[..., 0]).clamp_min(0)
    total_L = lengths.sum(dim=-1)
    if (total_L < min_L).all():
        from nsa.core.attention_kernels import grouped_selection_attention_packed

        return grouped_selection_attention_packed(Q, K, V, ranges)

    B, S, G, h, D = Q.shape
    Dv = V.shape[-1]
    _log_device_once()

    # Hard disable on RTX 4090 (Ada, SM 8.9) unless explicitly forced
    try:
        dev = Q.device
        if dev.type == "cuda":
            cap = torch.cuda.get_device_capability(dev)  # accepts device
        else:
            cap = (0, 0)
    except Exception:
        cap = (0, 0)
    if cap == (8, 9) and not _env_true("NSA_TRITON_SEL_FORCE", False):
        from nsa.core.attention_kernels import grouped_selection_attention_packed
        from nsa.core.debug import log

        log("sel.triton.disabled_adr", reason="ADR-2025-08-M4-02", sm=f"{cap[0]}.{cap[1]}")
        return grouped_selection_attention_packed(Q, K, V, ranges)

    # Build index list and L per (b,t,g)
    idx_map = []
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

    buckets: dict[int, list[tuple[int, int, int, list[tuple[int, int]]]]] = {}
    for b, t, g, L_i, spans in idx_map:
        buckets.setdefault(L_i, []).append((b, t, g, spans))

    O = torch.zeros((B, S, G, h, Dv), device=Q.device, dtype=V.dtype)
    try:
        from .sel_fwd import sel_attn_fwd_dense, sel_attn_fwd_varlen

        force = os.getenv("NSA_SEL_TRITON_FORCE_PATH", "auto").lower()
        for L_i, items in buckets.items():
            if L_i == 0:
                continue
            N = len(items)
            Q_pack = torch.empty((N, h, D), device=Q.device, dtype=Q.dtype)
            total_L = N * L_i
            K_pack, V_pack, cu = _get_pack_buffers(Q.device, total_L, D, Dv, N, Q.dtype, V.dtype)
            cu[0] = 0
            write = 0
            # Optional CUDA memory/timing diagnostics
            use_timing = torch.cuda.is_available() and _env_true("NSA_DEBUG_TIMING", False)
            start_evt = end_evt = None
            if use_timing:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
            pack_t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            pack_t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if pack_t0 is not None:
                pack_t0.record()
            for j, (b, t, g, spans) in enumerate(items):
                Q_pack[j] = Q[b, t, g]
                idx = torch.cat([torch.arange(s, e, device=Q.device) for (s, e) in spans], dim=0)
                Lw = idx.shape[0]
                K_pack[write : write + Lw] = K[b, g, idx]
                V_pack[write : write + Lw] = V[b, g, idx]
                write += Lw
                cu[j + 1] = write
            if pack_t1 is not None:
                pack_t1.record()
                pack_t1.synchronize()
                from nsa.core.debug import log

                log(
                    "sel.triton.pack_time_ms",
                    L=int(L_i),
                    N=int(N),
                    ms=float(pack_t0.elapsed_time(pack_t1)),
                )
            if force == "dense":
                use_dense = True
            elif force == "varlen":
                use_dense = False
            else:
                use_dense = True
                for _, _, _, spans in items:
                    if len(spans) != 1:
                        use_dense = False
                        break
            if use_dense:
                K_dense = torch.empty((N, L_i, D), device=Q.device, dtype=K.dtype)
                V_dense = torch.empty((N, L_i, Dv), device=Q.device, dtype=V.dtype)
                for j, (b, t, g, spans) in enumerate(items):
                    s0, e0 = spans[0]
                    K_dense[j] = K[b, g, s0:e0]
                    V_dense[j] = V[b, g, s0:e0]
                if _env_true("NSA_DEBUG_SHAPES", False):
                    from nsa.core.debug import log

                    log("sel.triton.shapes", path="dense", Q=Q_pack, K=K_dense, V=V_dense)
                try:
                    # Optional group-centric path for better KV reuse across heads
                    if _env_true("NSA_SEL_TRITON_GROUP", False):
                        from .sel_fwd import sel_attn_fwd_dense_group

                        O_pack = sel_attn_fwd_dense_group(Q_pack, K_dense, V_dense)
                        if _env_true("NSA_DEBUG_TIMING", False):
                            from nsa.core.debug import log

                            log("sel.triton.path", kind="dense_group", L=int(L_i), N=int(N))
                    else:
                        O_pack = sel_attn_fwd_dense(Q_pack, K_dense, V_dense)
                        if _env_true("NSA_DEBUG_TIMING", False):
                            from nsa.core.debug import log

                            log("sel.triton.path", kind="dense_per_head", L=int(L_i), N=int(N))
                except Exception:
                    # Dense Triton failed; fall back to full packed SDPA for safety.
                    from nsa.core.attention_kernels import grouped_selection_attention_packed
                    from nsa.core.debug import log

                    log("sel.triton.fallback", path="dense", reason="exception")
                    return grouped_selection_attention_packed(Q, K, V, ranges)
            else:
                if _env_true("NSA_DEBUG_SHAPES", False):
                    from nsa.core.debug import log

                    log("sel.triton.shapes", path="varlen", Q=Q_pack, K=K_pack, V=V_pack)
                try:
                    if _env_true("NSA_SEL_TRITON_GROUP", False):
                        from .sel_fwd import sel_attn_fwd_varlen_group

                        O_pack = sel_attn_fwd_varlen_group(Q_pack, K_pack, V_pack, cu)
                        if _env_true("NSA_DEBUG_TIMING", False):
                            from nsa.core.debug import log

                            log("sel.triton.path", kind="varlen_group", L=int(L_i), N=int(N))
                    else:
                        O_pack = sel_attn_fwd_varlen(Q_pack, K_pack, V_pack, cu)
                        if _env_true("NSA_DEBUG_TIMING", False):
                            from nsa.core.debug import log

                            log("sel.triton.path", kind="varlen_per_head", L=int(L_i), N=int(N))
                except Exception:
                    # Varlen Triton failed; fall back to full packed SDPA for safety.
                    from nsa.core.attention_kernels import grouped_selection_attention_packed
                    from nsa.core.debug import log

                    log("sel.triton.fallback", path="varlen", reason="exception")
                    return grouped_selection_attention_packed(Q, K, V, ranges)
            for j, (b, t, g, _) in enumerate(items):
                O[b, t, g] = O_pack[j]
            if use_timing and start_evt is not None:
                end_evt.record()
                torch.cuda.synchronize()
                ms = start_evt.elapsed_time(end_evt)
                from nsa.core.debug import log

                bytes_rw = float(L_i * (D + Dv) * Q.element_size() * N)
                gbps = bytes_rw / (ms / 1e3) / 1e9 if ms > 0 else 0.0
                log(
                    "sel.triton.bucket_timing",
                    L=L_i,
                    N=N,
                    path=("dense" if use_dense else "varlen"),
                    time_ms=float(ms),
                    gbps=gbps,
                )
        # Observability: estimate tokens/bytes read
        from nsa.core.debug import log

        total_tokens = (
            int(sum(L_i * len(items) for L_i, items in buckets.items())) if buckets else 0
        )
        bytes_k = int(total_tokens * D * Q.element_size() if buckets else 0)
        bytes_v = int(total_tokens * Dv * V.element_size() if buckets else 0)
        hist = {int(L_i): len(items) for L_i, items in buckets.items()}
        log(
            "sel.triton.reads",
            total_tokens=total_tokens,
            bytes_k=bytes_k,
            bytes_v=bytes_v,
            buckets=len(buckets),
            hist=str(hist),
        )
        if _env_true("NSA_DEBUG_COMPARE", False):
            from nsa.core.attention_kernels import grouped_selection_attention_packed

            O_ref = grouped_selection_attention_packed(Q, K, V, ranges)
            mae = (O - O_ref).abs().mean().item()
            log("sel.triton.parity", mae=mae)
    except Exception:
        from nsa.core.attention_kernels import grouped_selection_attention_packed

        # Log shapes/strides/dtypes to help diagnose CUDA issues
        from nsa.core.debug import log

        try:
            log("sel.triton.error_ctx", Q=Q, K=K, V=V, ranges=ranges)
        except Exception:
            pass
        return grouped_selection_attention_packed(Q, K, V, ranges)
    return O
