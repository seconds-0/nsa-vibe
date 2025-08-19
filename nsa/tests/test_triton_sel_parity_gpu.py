import os
import pytest
import torch


RUN_TRITON_SEL = os.getenv("NSA_TEST_TRITON_SEL", "0").lower() in ("1", "true", "yes")


@pytest.mark.skipif(not RUN_TRITON_SEL, reason="Triton selection tests are opt-in; set NSA_TEST_TRITON_SEL=1")
def test_triton_wrapper_parity_multirange_small_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("Triton not available")

    os.environ["NSA_USE_TRITON_SEL"] = "1"
    torch.manual_seed(0)
    device = torch.device("cuda")
    B, S, G, H, D, Dv = 4, 1, 1, 2, 64, 64
    S_kv = 192
    Q = torch.randn(B, S, G, H, D, device=device, dtype=torch.float16)
    K = torch.randn(B, G, S_kv, D, device=device, dtype=torch.float16)
    V = torch.randn(B, G, S_kv, Dv, device=device, dtype=torch.float16)
    # Build multi-range per row
    n = 3
    ranges = torch.zeros((B, S, G, n, 2), device=device, dtype=torch.int64)
    for b in range(B):
        ranges[b, 0, 0, 0, 0] = 16
        ranges[b, 0, 0, 0, 1] = 16 + 24
        ranges[b, 0, 0, 1, 0] = 64
        ranges[b, 0, 0, 1, 1] = 64 + 32
        ranges[b, 0, 0, 2, 0] = 120
        ranges[b, 0, 0, 2, 1] = 120 + 40

    from nsa.kernels.triton_sel_kernel import selection_attention_triton
    from nsa.core.attention_kernels import grouped_selection_attention_packed
    O_ref = grouped_selection_attention_packed(Q, K, V, ranges).float()
    O_tri = selection_attention_triton(Q, K, V, ranges).float()
    mae = (O_tri - O_ref).abs().mean().item()
    assert mae < 1e-3


