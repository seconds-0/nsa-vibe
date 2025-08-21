import os
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load as _load_ext

from nsa.core.debug import log as _log
from nsa.core.flags import env_true as _env_true

_EXT: Optional[object] = None


def _load_extension() -> Optional[object]:
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        return None
    # Only build on explicit request
    if not _env_true("NSA_SEL_CUDA_BUILD", False):
        return None
    this_dir = os.path.dirname(__file__)
    sources = [
        os.path.join(this_dir, "sel_cuda.cpp"),
        os.path.join(this_dir, "sel_cuda_kernel.cu"),
    ]
    try:
        _EXT = _load_ext(
            name="sel_cuda",
            sources=sources,
            verbose=_env_true("NSA_DEBUG_BUILD", False),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
        )
        return _EXT
    except (RuntimeError, OSError) as e:
        # Narrow exception handling; preserve KeyboardInterrupt/SystemExit
        _log("sel.cuda.build_failed", err=str(e))
        _EXT = None
        return None


def cuda_sel_available() -> bool:
    return _load_extension() is not None


def selection_attention_cuda(
    Q: torch.Tensor,  # [B,S,G,h,Dk]
    K: torch.Tensor,  # [B,G,S_kv,Dk]
    V: torch.Tensor,  # [B,G,S_kv,Dv]
    ranges: torch.Tensor,  # [B,S,G,n,2]
) -> torch.Tensor:  # [B,S,G,h,Dv]
    """
    Experimental CUDA selection wrapper. For now, this is a strict fallback to the
    packed SDPA reference path. When the CUDA kernel is implemented, this function
    will dispatch to it when `NSA_SEL_CUDA=1` and constraints are met.
    """
    use_cuda = _env_true("NSA_SEL_CUDA", False)
    ext = _load_extension() if use_cuda else None
    if use_cuda and ext is not None:
        try:
            return ext.sel_forward(Q, K, V, ranges)
        except (RuntimeError, ValueError) as e:
            _log("sel.cuda.forward_failed", err=str(e))
            # fall through to fallback
    from nsa.core.attention_kernels import grouped_selection_attention_packed

    return grouped_selection_attention_packed(Q, K, V, ranges)
