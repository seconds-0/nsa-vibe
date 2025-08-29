from __future__ import annotations

import torch
import torch.nn.functional as F
from nsa.core.debug import log


def _env_bool(name: str, default: bool = False) -> bool:
    v = str(name and __import__("os").getenv(name, "1" if default else "0")).lower()
    return v in ("1", "true", "yes", "on")


def flash_attn_version() -> str | None:
    """Return flash-attn version string if importable, else None."""
    try:
        import flash_attn as _fa  # type: ignore

        return getattr(_fa, "__version__", None)
    except Exception:
        return None


def is_flash_available() -> bool:
    """Return True if flash-attn dense API is importable."""
    try:
        from flash_attn import flash_attn_func  # type: ignore

        _ = flash_attn_func  # silence linter
        return True
    except Exception:
        return False


def is_flash_varlen_available() -> bool:
    """Return True if a varlen API is importable (either QKV or KV-packed)."""
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore

        _ = flash_attn_varlen_func
        return True
    except Exception:
        try:
            from flash_attn import flash_attn_varlen_kvpacked_func  # type: ignore

            _ = flash_attn_varlen_kvpacked_func
            return True
        except Exception:
            return False


def fa2_supported_verbose(
    device: torch.device, dtype: torch.dtype, head_dim: int
) -> tuple[bool, str]:
    """
    Conservative capability probe with a reason string for logging.
    We do not hard-fail on dtype, relying on try/except at call sites.
    """
    if device.type != "cuda":
        return False, "device_not_cuda"
    if head_dim % 8 != 0:
        return False, "head_dim_not_multiple_of_8"
    if not (is_flash_varlen_available() or is_flash_available()):
        return False, "flash_attn_not_importable"
    # Optional version floor (best-effort)
    ver = flash_attn_version()
    if ver is None:
        # Unknown version; still allow
        return True, "ok"
    # Allow all known versions; attach for logs
    return True, f"ok_v{ver}"


def fa2_supported(device: torch.device, dtype: torch.dtype, head_dim: int) -> bool:
    ok, _ = fa2_supported_verbose(device, dtype, head_dim)
    return ok


def attention_bgh(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """
    Q: [B,G,h,Dk], K/V: [B,G,S,D*] -> out [B,G,h,Dv]
    Prefer flash-attn if available; fallback to SDPA.
    """
    B, G, h, Dk = Q.shape
    S = K.shape[2]
    # Try FA-2 dense path first
    if is_flash_available():
        try:
            from flash_attn import flash_attn_func  # type: ignore

            # Reshape without materializing copies
            q = Q.transpose(1, 2).reshape(B, G * h, 1, Dk)  # [B,G*h,1,Dk]
            k = (
                K.unsqueeze(2)
                .expand(B, G, h, S, Dk)
                .reshape(B, G * h, S, Dk)
            )  # [B,G*h,S,Dk]
            v = (
                V.unsqueeze(2)
                .expand(B, G, h, S, V.shape[-1])
                .reshape(B, G * h, S, V.shape[-1])
            )  # [B,G*h,S,Dv]
            if _env_bool("NSA_DEBUG_TIMING"):
                log("fa2.bgh.path", path="fa2.dense", B=B, G=G, h=h, S=S, Dk=Dk)
            o = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=causal)
            o = o.reshape(B, G, h, -1)
            if not torch.isfinite(o).all():
                log("warn.flash_bgh_nonfinite", path="fa2.dense")
            return torch.nan_to_num(o, nan=0.0)
        except Exception:
            pass
    # SDPA fallback
    if _env_bool("NSA_DEBUG_TIMING"):
        log("fa2.bgh.path", path="sdpa", B=B, G=G, h=h, S=S, Dk=Dk)
    # Expand heads via view/expand to avoid materializing copies
    q2 = Q.reshape(B * G * h, 1, Dk).contiguous()
    k2 = (
        K.unsqueeze(2)
        .expand(B, G, h, S, Dk)
        .reshape(B * G * h, S, Dk)
        .contiguous()
    )
    v2 = (
        V.unsqueeze(2)
        .expand(B, G, h, S, V.shape[-1])
        .reshape(B * G * h, S, V.shape[-1])
        .contiguous()
    )
    attn = F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
    o = attn.squeeze(1).reshape(B, G, h, -1)
    return torch.nan_to_num(o, nan=0.0)


def attention_fa2_varlen_stub(*args, **kwargs):
    """
    Stub for FA-2 varlen attention; will be implemented in M1.
    Intentionally raises to direct callers to fallback.
    """
    raise NotImplementedError("FA-2 varlen attention not yet implemented")


def attention_fa2_dense_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """
    Best-effort dense FA-2 call for a batch of independent rows.
    Shapes:
    - q: [N, Tq, h, D]
    - k: [N, Tk, h, D]
    - v: [N, Tk, h, Dv]
    Returns: o [N, Tq, h, Dv]
    Falls back to SDPA if flash-attn unavailable.
    """
    # Ensure contiguous tensors for FA-2
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    try:
        from flash_attn import flash_attn_func  # type: ignore

        if _env_bool("NSA_DEBUG_TIMING"):
            log("fa2.batch.path", path="fa2.dense", N=int(q.shape[0]), Tq=int(q.shape[1]), Tk=int(k.shape[1]))
        return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=causal)
    except Exception:
        # SDPA fallback per row
        N, Tq, h, D = q.shape
        Tk = k.shape[1]
        Dv = v.shape[-1]
        if _env_bool("NSA_DEBUG_TIMING"):
            log("fa2.batch.path", path="sdpa", N=int(N), Tq=int(Tq), Tk=int(Tk))
        q2 = q.reshape(N * h, Tq, D)
        k2 = k.reshape(N * h, Tk, D)
        v2 = v.reshape(N * h, Tk, Dv)
        out = F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
        return out.reshape(N, h, Tq, Dv).permute(0, 2, 1, 3).contiguous()


def attention_fa2_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    causal: bool,
) -> torch.Tensor:
    """
    Best-effort varlen FA-2 call with separate Q/K/V packing.
    Shapes:
    - q: [total_q, h, D], k: [total_k, h, D], v: [total_k, h, Dv]
    - cu_seqlens_*: int32 [N+1]
    Returns: [total_q, h, Dv] packed output.
    Falls back to dense batching by padding per bucket if varlen API unavailable.
    """
    # Ensure contiguous tensors for FA-2
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore

        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=None,
            causal=causal,
        )
    except Exception:
        # Try KV-packed API variant
        try:
            from flash_attn import flash_attn_varlen_kvpacked_func  # type: ignore

            # Build KV packed as [total_k, 2, h, D]
            kv_packed = torch.stack([k, v], dim=1).contiguous()
            return flash_attn_varlen_kvpacked_func(
                q,
                kv_packed,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=causal,
            )
        except Exception:
            raise NotImplementedError("FA-2 varlen API not available; caller should fallback")
