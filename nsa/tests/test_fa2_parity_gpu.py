import os
import torch
import pytest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FA-2 parity tests")


RUN_FA2 = os.getenv("NSA_TEST_FA2", "0").lower() in ("1", "true", "yes", "on")


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
def test_fa2_env_ready():
    try:
        import flash_attn  # noqa: F401
    except Exception as e:
        pytest.skip(f"flash-attn not importable: {e}")


def _rand(B, T, H, D, dt):
    return torch.randn(B, T, H, D, device="cuda", dtype=dt)


@pytest.mark.skipif(not RUN_FA2, reason="FA-2 parity tests are opt-in; set NSA_TEST_FA2=1")
@pytest.mark.parametrize("D", [64, 96, 128, 192, 256])
@pytest.mark.parametrize("T", [1, 8, 16, 32, 96, 128])
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_dense_vs_sdpa(D, T, dt):
    from nsa.attn.fa2_contracts import fa2_supported_verbose
    from flash_attn import flash_attn_func

    q = _rand(1, T, 8, D, dt)
    k = _rand(1, T, 8, D, dt)
    v = _rand(1, T, 8, D, dt)
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, dropout_p=0.0
    )
    ok, _ = fa2_supported_verbose(q=q, k=k, v=v, head_dim=D, is_varlen=False)
    if not ok:
        pytest.skip("FA-2 not supported for this config by contracts")
    out_fa = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    atol = 5e-3 if dt == torch.float16 else 7e-3
    rtol = 5e-3 if dt == torch.float16 else 7e-3
    torch.testing.assert_close(out_fa, out_sdpa, atol=atol, rtol=rtol)

