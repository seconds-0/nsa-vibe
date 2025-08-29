import torch
from nsa.core.rope import apply_rope


def test_apply_rope_preserves_input_dtype():
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(2, 3, 8, dtype=dtype)
        pos = torch.arange(3, dtype=torch.int64)
        y = apply_rope(x, pos)
        assert y.dtype == dtype
