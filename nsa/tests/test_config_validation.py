import pytest

from nsa.core.nsa_attention import NSAAttention


def test_invalid_divisibility_raises():
    with pytest.raises(ValueError):
        _ = NSAAttention(
            dim=64, n_heads=4, n_kv_groups=1, d_k=16, d_v=16, l=30, d=16, l_sel=64, n_sel=4, w=16
        )
    with pytest.raises(ValueError):
        _ = NSAAttention(
            dim=64, n_heads=4, n_kv_groups=1, d_k=16, d_v=16, l=32, d=12, l_sel=60, n_sel=4, w=16
        )
