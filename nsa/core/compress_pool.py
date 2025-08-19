from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from .rope import apply_rope


def avg_pool_phi_rope_kv(
    K_raw: torch.Tensor,
    V_raw: torch.Tensor,
    l: int,
    d: int,
    pos: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Apply RoPE to K before ϕ; use absolute positions if provided
    S = K_raw.shape[2]
    if pos is None:
        pos = torch.arange(S, device=K_raw.device)
    K_rope = apply_rope(K_raw, pos)
    V_rope = V_raw
    # Expect shapes [B,G,S,D*]
    B, G, S, Dk = K_rope.shape
    # If sequence shorter than kernel, no compressed tokens yet
    if S < l:
        return (
            torch.zeros((B, G, 0, Dk), device=K_rope.device, dtype=K_rope.dtype),
            torch.zeros((B, G, 0, V_rope.shape[-1]), device=V_rope.device, dtype=V_rope.dtype),
        )
    # Unfold over time with stride d and kernel l (causal pooling over past)
    Kf = K_rope.reshape(B * G, S, Dk).transpose(1, 2).unsqueeze(3)  # [B*G, Dk, S, 1]
    Vf = V_rope.reshape(B * G, S, -1).transpose(1, 2).unsqueeze(3)
    Kp = F.avg_pool2d(Kf[:, :, : S, :], kernel_size=(l, 1), stride=(d, 1))  # [B*G, Dk, S_cmp, 1]
    Vp = F.avg_pool2d(Vf[:, :, : S, :], kernel_size=(l, 1), stride=(d, 1))
    S_cmp = Kp.shape[2]
    K_cmp = Kp.squeeze(3).transpose(1, 2).reshape(B, G, S_cmp, Dk)
    V_cmp = Vp.squeeze(3).transpose(1, 2).reshape(B, G, S_cmp, V_rope.shape[-1])
    return K_cmp, V_cmp


