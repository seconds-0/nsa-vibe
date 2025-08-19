import os
import pytest
import torch

RUN_TRITON_SEL = os.getenv("NSA_TEST_TRITON_SEL", "0").lower() in ("1", "true", "yes")


@pytest.mark.skipif(not RUN_TRITON_SEL, reason="Triton selection tests are opt-in; set NSA_TEST_TRITON_SEL=1")
def test_triton_sel_dense_parity_small():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("Triton not available")

    torch.manual_seed(0)
    device = torch.device("cuda")
    N, H, D, Dv, L = 8, 4, 64, 64, 96
    Q = torch.randn(N, H, D, device=device, dtype=torch.float16)
    K = torch.randn(N, L, D, device=device, dtype=torch.float16)
    V = torch.randn(N, L, Dv, device=device, dtype=torch.float16)

    from nsa.kernels.triton_sel_kernel.sel_fwd import sel_attn_fwd_dense
    O_tri = sel_attn_fwd_dense(Q, K, V).float()

    # Reference: SDPA over packed rows -> treat each (n,h) as query row with causal over full L
    # Build SDPA inputs [B,H,S,D] with S=1 and K/V [B,H,L,D*]
    Q_sdpa = Q  # [N,H,D]
    K_sdpa = K.unsqueeze(0).expand(N, L, D).unsqueeze(1).expand(N, H, L, D)  # [N,H,L,D]
    V_sdpa = V.unsqueeze(0).expand(N, L, Dv).unsqueeze(1).expand(N, H, L, Dv)
    O_ref = torch.nn.functional.scaled_dot_product_attention(
        Q_sdpa, K_sdpa, V_sdpa, is_causal=True
    )  # [N,H,1,Dv]
    O_ref = O_ref.squeeze(2).float()

    mae = (O_tri - O_ref).abs().mean().item()
    assert mae < 1e-3


