from __future__ import annotations
from typing import Optional

import torch


def build_inv_freq(
    dim: int, base: float = 10000.0, device: Optional[torch.device] = None
) -> torch.Tensor:
    assert dim % 2 == 0, "RoPE requires even dimension"
    half = dim // 2
    idx = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = base ** (-2 * idx / dim)
    return inv_freq  # [half]


def apply_rope(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply rotary position embeddings along the last dimension.

    x: [..., S, D] tensor with even D
    pos: [S] or [..., S] integer positions
    returns: same shape as x
    """
    D = x.shape[-1]
    assert D % 2 == 0, "RoPE requires even dimension"
    device = x.device
    inv_freq = build_inv_freq(D, base=base, device=device)  # [D/2]
    # pos shape broadcasting to [..., S, D/2]
    while pos.dim() < x.dim() - 1:
        pos = pos.unsqueeze(0)
    angles = pos.to(x.dtype).unsqueeze(-1) * inv_freq  # [..., S, D/2]
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    x_2 = x.view(*x.shape[:-1], D // 2, 2)
    x0, x1 = x_2[..., 0], x_2[..., 1]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    y = torch.stack((y0, y1), dim=-1).view_as(x)
    return y
