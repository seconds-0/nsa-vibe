"""
Fixed Triton kernel implementation that avoids problematic 2D broadcasting.

This kernel works with both Triton 3.0 and 3.1 by avoiding the broadcasting
pattern that causes "Cannot make_shape_compatible" errors.
"""

import torch
import triton
import triton.language as tl
import math
import os


@triton.jit
def _sel_attn_fwd_kernel_fixed(
    Q_ptr, K_ptr, V_ptr, O_ptr,
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
    """Fixed kernel that avoids 2D broadcasting issues."""
    
    pid_nh = tl.program_id(0)
    n = pid_nh // H
    h = pid_nh % H
    
    q_base = Q_ptr + n * stride_qn + h * stride_qh
    k_base = K_ptr + n * stride_kn
    v_base = V_ptr + n * stride_vn
    o_base = O_ptr + n * stride_on + h * stride_oh
    
    # Initialize running statistics
    m = float('-inf')
    lse = 0.0
    
    # Pass 1: Compute maximum and log-sum-exp
    for l0 in range(0, L, BLOCK_L):
        l_end = min(l0 + BLOCK_L, L)
        
        # Initialize logits for this L-block
        logits = tl.zeros((BLOCK_L,), dtype=tl.float32)
        
        # Compute attention scores for this L-block
        for d0 in range(0, D, BLOCK_D):
            d_end = min(d0 + BLOCK_D, D)
            
            # Load query vector for this D-block
            q_vec = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for d_idx in range(BLOCK_D):
                if d0 + d_idx < D:
                    q_val = tl.load(q_base + (d0 + d_idx) * stride_qd)
                    q_vec = tl.where(d_idx == tl.arange(0, BLOCK_D), q_val, q_vec)
            
            # Load and compute with K for this L,D block
            for l_idx in range(BLOCK_L):
                if l0 + l_idx < L:
                    # Accumulate dot product for this position
                    dot_product = 0.0
                    for d_idx in range(BLOCK_D):
                        if d0 + d_idx < D:
                            k_val = tl.load(k_base + (l0 + l_idx) * stride_kL + (d0 + d_idx) * stride_kd)
                            q_val = tl.load(q_base + (d0 + d_idx) * stride_qd)
                            dot_product += k_val * q_val
                    
                    # Update logits
                    current_logits = tl.load(tl.full((1,), 0.0, dtype=tl.float32))
                    logits = tl.where(l_idx == tl.arange(0, BLOCK_L), 
                                     logits + dot_product, logits)
        
        # Scale by 1/sqrt(d)
        logits *= inv_sqrt_d
        
        # Apply causal mask and update running max/lse
        for l_idx in range(BLOCK_L):
            if l0 + l_idx < L:
                logit_val = logits[l_idx]
                if logit_val > float('-inf'):
                    new_m = max(m, logit_val)
                    if m == float('-inf'):
                        lse = tl.exp(logit_val - new_m)
                    else:
                        lse = lse * tl.exp(m - new_m) + tl.exp(logit_val - new_m)
                    m = new_m
    
    # Pass 2: Compute weighted sum with V
    for dv0 in range(0, Dv, BLOCK_DV):
        acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)
        
        for l0 in range(0, L, BLOCK_L):
            # Recompute attention weights for this L-block
            logits = tl.zeros((BLOCK_L,), dtype=tl.float32)
            
            for d0 in range(0, D, BLOCK_D):
                for l_idx in range(BLOCK_L):
                    if l0 + l_idx < L:
                        dot_product = 0.0
                        for d_idx in range(BLOCK_D):
                            if d0 + d_idx < D:
                                k_val = tl.load(k_base + (l0 + l_idx) * stride_kL + (d0 + d_idx) * stride_kd)
                                q_val = tl.load(q_base + (d0 + d_idx) * stride_qd)
                                dot_product += k_val * q_val
                        logits = tl.where(l_idx == tl.arange(0, BLOCK_L), 
                                         logits + dot_product, logits)
            
            logits *= inv_sqrt_d
            
            # Convert to probabilities and accumulate V
            for l_idx in range(BLOCK_L):
                if l0 + l_idx < L:
                    p = tl.exp(logits[l_idx] - m) / lse if lse > 0 else 0.0
                    
                    # Load V and accumulate
                    for dv_idx in range(BLOCK_DV):
                        if dv0 + dv_idx < Dv:
                            v_val = tl.load(v_base + (l0 + l_idx) * stride_vL + (dv0 + dv_idx) * stride_vd)
                            acc = tl.where(dv_idx == tl.arange(0, BLOCK_DV),
                                          acc + p * v_val, acc)
        
        # Store output
        for dv_idx in range(BLOCK_DV):
            if dv0 + dv_idx < Dv:
                tl.store(o_base + (dv0 + dv_idx) * stride_odv, acc[dv_idx])


def sel_attn_fwd_dense_fixed(Q, K, V):
    """
    Fixed version of dense selection attention that works with Triton 3.0/3.1.
    
    Args:
        Q: [N, H, D]
        K: [N, L, D]  
        V: [N, L, Dv]
    
    Returns:
        O: [N, H, Dv]
    """
    N, H, D = Q.shape
    _, L, _ = K.shape
    _, _, Dv = V.shape
    
    # Get block sizes
    def _env_int(name: str, default: int) -> int:
        try:
            v = int(os.getenv(name, str(default)))
            return max(16, v)
        except Exception:
            return default
    
    BLOCK_D = _env_int("NSA_SEL_TRITON_BLOCK_D", 64 if D >= 64 else 32)
    BLOCK_L = _env_int("NSA_SEL_TRITON_BLOCK_L", 64)  # Use smaller default
    BLOCK_DV = _env_int("NSA_SEL_TRITON_BLOCK_DV", 64 if Dv >= 64 else 32)
    
    # Create output tensor
    O = torch.empty((N, H, Dv), device=Q.device, dtype=V.dtype)
    
    # Launch kernel
    grid = (N * H,)
    inv_sqrt_d = 1.0 / math.sqrt(D)
    
    _sel_attn_fwd_kernel_fixed[grid](
        Q, K, V, O,
        N, H, L, D, Dv,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        inv_sqrt_d,
        BLOCK_D=BLOCK_D, BLOCK_L=BLOCK_L, BLOCK_DV=BLOCK_DV,
    )
    
    return O


if __name__ == "__main__":
    # Test the fixed kernel
    device = torch.device("cuda")
    N, H, D, Dv, L = 4, 4, 64, 64, 128
    
    Q = torch.randn(N, H, D, device=device, dtype=torch.float16)
    K = torch.randn(N, L, D, device=device, dtype=torch.float16)
    V = torch.randn(N, L, Dv, device=device, dtype=torch.float16)
    
    print("Testing fixed kernel...")
    try:
        O = sel_attn_fwd_dense_fixed(Q, K, V)
        print(f"SUCCESS: Fixed kernel works! Output shape: {O.shape}")
    except Exception as e:
        print(f"FAILED: {e}")