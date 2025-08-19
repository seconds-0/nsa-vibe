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


_PACK_CACHE: dict[int, dict[str, torch.Tensor]] = {}
_DEVICE_LOGGED: bool = False
_PACK_CACHE_MAX_ENTRIES: int = 4


def _get_pack_buffers(device: torch.device, total_L: int, D: int, Dv: int, N: int, dtype_k: torch.dtype, dtype_v: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
        # Save tensors for backward; ranges is int so safe
        ctx.save_for_backward(Q, K, V, ranges)
        # Use triton wrapper in forward (may still fall back per guards)
        return selection_attention_triton(Q, K, V, ranges)

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        Q, K, V, ranges = ctx.saved_tensors
        # Compute grads via packed SDPA reference path for correctness
        Q_r = Q.detach().requires_grad_(True)
        K_r = K.detach().requires_grad_(True)
        V_r = V.detach().requires_grad_(True)
        from nsa.core.attention_kernels import grouped_selection_attention_packed
        O_ref = grouped_selection_attention_packed(Q_r, K_r, V_r, ranges)
        O_ref.backward(dO)
        dQ = Q_r.grad
        dK = K_r.grad
        dV = V_r.grad
        return dQ, dK, dV, None


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
    # Clamp ranges to valid [0, S_kv] first for consistent semantics
    S_kv = K.shape[2]
    ranges = ranges.clamp(min=0, max=S_kv)
    # If Triton disabled or unavailable, fall back to reference implementations
    if not (_env_true("NSA_USE_TRITON_SEL", False) and triton_sel_available()):
        from nsa.core.attention_kernels import (
            grouped_selection_attention_packed,
            grouped_selection_attention,
        )
        if use_packed_fallback:
            return grouped_selection_attention_packed(Q, K, V, ranges)
        return grouped_selection_attention(Q, K, V, ranges)

    # Defensive input checks and gatekeeping
    # - If training/grad enabled and not explicitly allowed, fall back to packed
    if torch.is_grad_enabled() and (Q.requires_grad or K.requires_grad or V.requires_grad):
        if _env_true("NSA_SEL_TRITON_ALLOW_GRAD", False):
            return _SelAttnTritonFn.apply(Q, K, V, ranges)
        else:
            from nsa.core.attention_kernels import grouped_selection_attention_packed
            return grouped_selection_attention_packed(Q, K, V, ranges)
    # - Dtype support: only half/bfloat16
    allowed_dtypes = {torch.float16, torch.bfloat16}
    if Q.dtype not in allowed_dtypes or K.dtype not in allowed_dtypes or V.dtype not in allowed_dtypes:
        from nsa.core.attention_kernels import grouped_selection_attention_packed
        return grouped_selection_attention_packed(Q, K, V, ranges)
    # - Alignment requirement (optional)
    if _env_true("NSA_SEL_TRITON_REQUIRE_ALIGN", True):
        D = Q.shape[-1]
        Dv = V.shape[-1]
        if (D % 8 != 0) or (Dv % 8 != 0):
            from nsa.core.attention_kernels import grouped_selection_attention_packed
            return grouped_selection_attention_packed(Q, K, V, ranges)

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
    _log_device_once()
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
            for j, (b, t, g, spans) in enumerate(items):
                Q_pack[j] = Q[b, t, g]
                idx = torch.cat([torch.arange(s, e, device=Q.device) for (s, e) in spans], dim=0)
                Lw = idx.shape[0]
                K_pack[write : write + Lw] = K[b, g, idx]
                V_pack[write : write + Lw] = V[b, g, idx]
                write += Lw
                cu[j + 1] = write
            # Dense fast-path when each row is a single contiguous span
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
                O_pack = sel_attn_fwd_dense(Q_pack, K_dense, V_dense)
            else:
                if _env_true("NSA_DEBUG_SHAPES", False):
                    from nsa.core.debug import log
                    log("sel.triton.shapes", path="varlen", Q=Q_pack, K=K_pack, V=V_pack)
                O_pack = sel_attn_fwd_varlen(Q_pack, K_pack, V_pack, cu)
            # Scatter back
            for j, (b, t, g, _) in enumerate(items):
                O[b, t, g] = O_pack[j]
            if use_timing and start_evt is not None:
                end_evt.record(); torch.cuda.synchronize()
                ms = start_evt.elapsed_time(end_evt)
                from nsa.core.debug import log
                bytes_rw = float(L_i * (D + Dv) * Q.element_size() * N)
                gbps = bytes_rw / (ms / 1e3) / 1e9 if ms > 0 else 0.0
                log("sel.triton.bucket_timing", L=L_i, N=N, path=("dense" if use_dense else "varlen"), time_ms=float(ms), gbps=gbps)
        # Observability: estimate tokens/bytes read
        from nsa.core.debug import log
        total_tokens = int(sum(L_i * len(items) for L_i, items in buckets.items())) if buckets else 0
        bytes_k = int(total_tokens * D * Q.element_size() if buckets else 0)
        bytes_v = int(total_tokens * Dv * V.element_size() if buckets else 0)
        # Histogram of lengths per bucket
        hist = {int(L_i): len(items) for L_i, items in buckets.items()}
        log("sel.triton.reads", total_tokens=total_tokens, bytes_k=bytes_k, bytes_v=bytes_v, buckets=len(buckets), hist=str(hist))
        # Optional parity compare for observability
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


