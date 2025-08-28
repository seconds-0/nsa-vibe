from __future__ import annotations

import torch
import torch.nn.functional as F


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


def fa2_supported(device: torch.device, dtype: torch.dtype, head_dim: int) -> bool:
    """
    Conservative support check for FA-2 paths.
    - Require CUDA device
    - Require head_dim multiple of 8 (typical constraint)
    - Guard on availability import probe
    """
    if device.type != "cuda":
        return False
    if head_dim % 8 != 0:
        return False
    # Prefer varlen availability probe for FA-2 usage
    return is_flash_varlen_available() or is_flash_available()


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
            o = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=causal)
            o = o.reshape(B, G, h, -1)
            return torch.nan_to_num(o, nan=0.0)
        except Exception:
            pass
    # SDPA fallback
    q2 = Q.reshape(B * G * h, 1, Dk)
    k2 = K.repeat_interleave(h, dim=1).reshape(B * G * h, S, Dk)
    v2 = V.repeat_interleave(h, dim=1).reshape(B * G * h, S, V.shape[-1])
    q2 = q2.contiguous(); k2 = k2.contiguous(); v2 = v2.contiguous()
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

        return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=causal)
    except Exception:
        # SDPA fallback per row
        N, Tq, h, D = q.shape
        Tk = k.shape[1]
        Dv = v.shape[-1]
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
