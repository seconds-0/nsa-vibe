import os
from typing import Optional

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

    # TODO(M4): Implement Triton forward path; for now, defer to packed fallback
    from nsa.core.attention_kernels import grouped_selection_attention_packed

    return grouped_selection_attention_packed(Q, K, V, ranges)


