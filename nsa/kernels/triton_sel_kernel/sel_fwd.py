import math
import os
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU CI
    triton = None
    tl = None


if triton is not None:
    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=4, num_stages=2),
            triton.Config({}, num_warps=8, num_stages=2),
            triton.Config({}, num_warps=4, num_stages=3),
        ],
        key=["D", "Dv"],
    )
    @triton.jit
    def _sel_attn_fwd_kernel(
        Q_ptr,  # [N, H, D]
        K_ptr,  # [N, L, D]
        V_ptr,  # [N, L, Dv]
        O_ptr,  # [N, H, Dv]
        N, H, L, D, Dv,
        stride_qn, stride_qh, stride_qd,
        stride_kn, stride_kL, stride_kd,
        stride_vn, stride_vL, stride_vd,
        stride_on, stride_oh, stride_odv,
        inv_sqrt_d,
        BLOCK_D: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_DV: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        n = pid // H
        # Base pointers for this (n,h)
        q_base = Q_ptr + n * stride_qn + h * stride_qh
        o_base = O_ptr + n * stride_on + h * stride_oh

        # Pass 1: compute global m and lse across L (no V)
        m = tl.full((1,), float('-inf'), dtype=tl.float32)
        lse = tl.zeros((1,), dtype=tl.float32)
        offs_L = tl.arange(0, BLOCK_L)
        for l0 in range(0, L, BLOCK_L):
            Lmask = l0 + offs_L < L
            logits_tile = tl.zeros((BLOCK_L,), dtype=tl.float32)
            offs_D = tl.arange(0, BLOCK_D)
            for d0 in range(0, D, BLOCK_D):
                Dmask = d0 + offs_D < D
                q_vec = tl.load(q_base + (d0 + offs_D) * stride_qd, mask=Dmask, other=0.0)
                rows = (l0 + offs_L).to(tl.int32)[:, None]
                cols = (d0 + offs_D).to(tl.int32)[None, :]
                k_ptrs = K_ptr + n * stride_kn + rows * stride_kL + cols * stride_kd
                k_tile = tl.load(
                    k_ptrs,
                    mask=tl.broadcast_to(Lmask[:, None], k_ptrs.shape) & tl.broadcast_to(Dmask[None, :], k_ptrs.shape),
                    other=0.0,
                )
                logits_tile += tl.sum(k_tile * q_vec[None, :], axis=1)
            logits_tile *= inv_sqrt_d
            tile_max = tl.max(tl.where(Lmask, logits_tile, float('-inf')))
            new_m = tl.maximum(m, tile_max)
            lse = lse * tl.exp(m - new_m) + tl.sum(tl.where(Lmask, tl.exp(logits_tile - new_m), 0.0))
            m = new_m

        # Pass 2: accumulate outputs over V using normalized probabilities
        offs_Dv = tl.arange(0, BLOCK_DV)
        for dv0 in range(0, Dv, BLOCK_DV):
            DVmask = dv0 + offs_Dv < Dv
            acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)
            for l0 in range(0, L, BLOCK_L):
                Lmask = l0 + offs_L < L
                # recompute logits tile for p
                logits_tile = tl.zeros((BLOCK_L,), dtype=tl.float32)
                offs_D = tl.arange(0, BLOCK_D)
                for d0 in range(0, D, BLOCK_D):
                    Dmask = d0 + offs_D < D
                    q_vec = tl.load(q_base + (d0 + offs_D) * stride_qd, mask=Dmask, other=0.0)
                    rows = (l0 + offs_L).to(tl.int32)[:, None]
                    cols = (d0 + offs_D).to(tl.int32)[None, :]
                    k_ptrs = K_ptr + n * stride_kn + rows * stride_kL + cols * stride_kd
                    k_tile = tl.load(
                        k_ptrs,
                        mask=tl.broadcast_to(Lmask[:, None], k_ptrs.shape) & tl.broadcast_to(Dmask[None, :], k_ptrs.shape),
                        other=0.0,
                    )
                    logits_tile += tl.sum(k_tile * q_vec[None, :], axis=1)
                logits_tile *= inv_sqrt_d
                p = tl.where(Lmask, tl.exp(logits_tile - m) / lse, 0.0)
                rows = (l0 + offs_L).to(tl.int32)[:, None]
                cols_v = (dv0 + offs_Dv).to(tl.int32)[None, :]
                v_ptrs = V_ptr + n * stride_vn + rows * stride_vL + cols_v * stride_vd
                v_tile = tl.load(
                    v_ptrs,
                    mask=tl.broadcast_to(Lmask[:, None], v_ptrs.shape) & tl.broadcast_to(DVmask[None, :], v_ptrs.shape),
                    other=0.0,
                )
                acc += tl.sum(v_tile * p[:, None], axis=0)
            tl.store(o_base + (dv0 + offs_Dv) * stride_odv, acc, mask=DVmask)

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=4, num_stages=2),
            triton.Config({}, num_warps=8, num_stages=2),
            triton.Config({}, num_warps=4, num_stages=3),
        ],
        key=["D", "Dv"],
    )
    @triton.jit
    def _sel_attn_fwd_varlen_kernel(
        Q_ptr,   # [N, H, D]
        K_ptr,   # [TotalL, D]
        V_ptr,   # [TotalL, Dv]
        CU_ptr,  # [N+1] int32 cu_seqlens for rows
        O_ptr,   # [N, H, Dv]
        N, H, D, Dv,
        stride_qn, stride_qh, stride_qd,
        stride_kL, stride_kd,
        stride_vL, stride_vd,
        stride_on, stride_oh, stride_odv,
        inv_sqrt_d,
        BLOCK_D: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_DV: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        n = pid // H
        q_base = Q_ptr + n * stride_qn + h * stride_qh
        o_base = O_ptr + n * stride_on + h * stride_oh
        # Load row start/end
        # Load cu_seqlens with defensive clamping to [0, cu[N]]
        cuN = tl.load(CU_ptr + N)
        rs = tl.load(CU_ptr + n)
        re = tl.load(CU_ptr + n + 1)
        zero_i32 = tl.full((1,), 0, dtype=tl.int32)
        row_start = tl.maximum(zero_i32, tl.minimum(rs, cuN))
        row_end = tl.maximum(row_start, tl.minimum(re, cuN))
        L = row_end - row_start
        # Pass 1: compute m, lse across row L
        m = tl.full((1,), float('-inf'), dtype=tl.float32)
        lse = tl.zeros((1,), dtype=tl.float32)
        offs_L = tl.arange(0, BLOCK_L)
        for l0 in range(0, L, BLOCK_L):
            Lmask = l0 + offs_L < L
            logits_tile = tl.zeros((BLOCK_L,), dtype=tl.float32)
            offs_D = tl.arange(0, BLOCK_D)
            for d0 in range(0, D, BLOCK_D):
                Dmask = d0 + offs_D < D
                q_vec = tl.load(q_base + (d0 + offs_D) * stride_qd, mask=Dmask, other=0.0)
                # row index = row_start + l0 + offs_L
                rows = (row_start + l0 + offs_L).to(tl.int32)[:, None]
                cols = (d0 + offs_D).to(tl.int32)[None, :]
                k_ptrs = K_ptr + rows * stride_kL + cols * stride_kd
                k_tile = tl.load(
                    k_ptrs,
                    mask=tl.broadcast_to(Lmask[:, None], k_ptrs.shape) & tl.broadcast_to(Dmask[None, :], k_ptrs.shape),
                    other=0.0,
                )
                logits_tile += tl.sum(k_tile * q_vec[None, :], axis=1)
            logits_tile *= inv_sqrt_d
            tile_max = tl.max(tl.where(Lmask, logits_tile, float('-inf')))
            new_m = tl.maximum(m, tile_max)
            lse = lse * tl.exp(m - new_m) + tl.sum(tl.where(Lmask, tl.exp(logits_tile - new_m), 0.0))
            m = new_m
        # Pass 2: accumulate outputs
        offs_Dv = tl.arange(0, BLOCK_DV)
        for dv0 in range(0, Dv, BLOCK_DV):
            DVmask = dv0 + offs_Dv < Dv
            acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)
            for l0 in range(0, L, BLOCK_L):
                Lmask = l0 + offs_L < L
                logits_tile = tl.zeros((BLOCK_L,), dtype=tl.float32)
                offs_D = tl.arange(0, BLOCK_D)
                for d0 in range(0, D, BLOCK_D):
                    Dmask = d0 + offs_D < D
                    q_vec = tl.load(q_base + (d0 + offs_D) * stride_qd, mask=Dmask, other=0.0)
                    rows = (row_start + l0 + offs_L).to(tl.int32)[:, None]
                    cols = (d0 + offs_D).to(tl.int32)[None, :]
                    k_ptrs = K_ptr + rows * stride_kL + cols * stride_kd
                    k_tile = tl.load(
                        k_ptrs,
                        mask=tl.broadcast_to(Lmask[:, None], k_ptrs.shape) & tl.broadcast_to(Dmask[None, :], k_ptrs.shape),
                        other=0.0,
                    )
                    logits_tile += tl.sum(k_tile * q_vec[None, :], axis=1)
                logits_tile *= inv_sqrt_d
                p = tl.where(Lmask, tl.exp(logits_tile - m) / lse, 0.0)
                rows = (row_start + l0 + offs_L).to(tl.int32)[:, None]
                cols_v = (dv0 + offs_Dv).to(tl.int32)[None, :]
                v_ptrs = V_ptr + rows * stride_vL + cols_v * stride_vd
                v_tile = tl.load(
                    v_ptrs,
                    mask=tl.broadcast_to(Lmask[:, None], v_ptrs.shape) & tl.broadcast_to(DVmask[None, :], v_ptrs.shape),
                    other=0.0,
                )
                acc += tl.sum(v_tile * p[:, None], axis=0)
            tl.store(o_base + (dv0 + offs_Dv) * stride_odv, acc, mask=DVmask)


def _get_block_sizes(D: int, L: int, Dv: int) -> tuple[int, int, int]:
    """Read optional env overrides for block sizes with safe defaults."""
    def _env_int(name: str, default: int) -> int:
        try:
            v = int(os.getenv(name, str(default)))
            return max(16, v)
        except Exception:
            return default
    BLOCK_D = _env_int("NSA_SEL_TRITON_BLOCK_D", 64 if D >= 64 else 32)
    BLOCK_L = _env_int("NSA_SEL_TRITON_BLOCK_L", 128 if L >= 128 else 64)
    BLOCK_DV = _env_int("NSA_SEL_TRITON_BLOCK_DV", 64 if Dv >= 64 else 32)
    return BLOCK_D, BLOCK_L, BLOCK_DV


def sel_attn_fwd_dense(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Triton forward for selection attention on a dense packed batch:
    Q: [N, H, D], K: [N, L, D], V: [N, L, Dv] -> O: [N, H, Dv]
    """
    assert triton is not None and torch.cuda.is_available(), "Triton/GPU required"
    N, H, D = Q.shape
    L = K.shape[1]
    Dv = V.shape[2]
    # Ensure contiguity for correct strides
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous()
    O = torch.empty((N, H, Dv), device=Q.device, dtype=V.dtype)
    # Strides
    stride_qn, stride_qh, stride_qd = Q.stride(0), Q.stride(1), Q.stride(2)
    stride_kn, stride_kL, stride_kd = K.stride(0), K.stride(1), K.stride(2)
    stride_vn, stride_vL, stride_vd = V.stride(0), V.stride(1), V.stride(2)
    stride_on, stride_oh, stride_odv = O.stride(0), O.stride(1), O.stride(2)
    # Launch grid over N*H programs
    grid = (N * H,)
    BLOCK_D, BLOCK_L, BLOCK_DV = _get_block_sizes(D, L, Dv)
    inv_sqrt_d = 1.0 / math.sqrt(D)
    _sel_attn_fwd_kernel[grid](
        Q, K, V, O,
        N, H, L, D, Dv,
        stride_qn, stride_qh, stride_qd,
        stride_kn, stride_kL, stride_kd,
        stride_vn, stride_vL, stride_vd,
        stride_on, stride_oh, stride_odv,
        inv_sqrt_d,
        BLOCK_D=BLOCK_D, BLOCK_L=BLOCK_L, BLOCK_DV=BLOCK_DV,
        num_warps=4, num_stages=2,
    )
    return O


def sel_attn_fwd_varlen(Q: torch.Tensor, K_all: torch.Tensor, V_all: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """
    Triton forward for varlen packed selection:
    Q: [N, H, D]; K_all: [TotalL, D]; V_all: [TotalL, Dv]; cu_seqlens: [N+1] int32
    """
    assert triton is not None and torch.cuda.is_available(), "Triton/GPU required"
    N, H, D = Q.shape
    Dv = V_all.shape[1]
    # Ensure contiguity for correct strides
    Q = Q.contiguous(); K_all = K_all.contiguous(); V_all = V_all.contiguous()
    O = torch.empty((N, H, Dv), device=Q.device, dtype=V_all.dtype)
    stride_qn, stride_qh, stride_qd = Q.stride(0), Q.stride(1), Q.stride(2)
    stride_kL, stride_kd = K_all.stride(0), K_all.stride(1)
    stride_vL, stride_vd = V_all.stride(0), V_all.stride(1)
    stride_on, stride_oh, stride_odv = O.stride(0), O.stride(1), O.stride(2)
    grid = (N * H,)
    # Estimate a typical L for tuning from average row length
    total_L = int(cu_seqlens[-1].item())
    avg_L = max(1, total_L // max(1, N))
    BLOCK_D, BLOCK_L, BLOCK_DV = _get_block_sizes(D, avg_L, Dv)
    inv_sqrt_d = 1.0 / math.sqrt(D)
    _sel_attn_fwd_varlen_kernel[grid](
        Q, K_all, V_all, cu_seqlens,
        O,
        N, H, D, Dv,
        stride_qn, stride_qh, stride_qd,
        stride_kL, stride_kd,
        stride_vL, stride_vd,
        stride_on, stride_oh, stride_odv,
        inv_sqrt_d,
        BLOCK_D=BLOCK_D, BLOCK_L=BLOCK_L, BLOCK_DV=BLOCK_DV,
        num_warps=4, num_stages=2,
    )
    return O


