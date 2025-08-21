import torch

from nsa.kernels.triton_sel_kernel import _normalize_ranges_tensor


def test_normalize_ranges_accepts_overnested_and_4d():
    S_kv = 50
    # 6D with extra singleton dims
    t6 = torch.zeros((1, 1, 2, 1, 1, 2), dtype=torch.int32)
    # Extra singleton dims should be squeezed to 5D
    out6 = _normalize_ranges_tensor(t6, S_kv)
    assert out6.dim() == 5
    assert out6.dtype == torch.int32

    # 4D missing batch dimension -> becomes [1,S,G,n,2]
    t4 = torch.tensor([[[[0, 10], [40, 60]]]], dtype=torch.int32)  # [S=1,G=1,n=2,2]
    out4 = _normalize_ranges_tensor(t4, S_kv)
    assert out4.shape == (1, 1, 1, 2, 2)
    # Clamped to [0,S_kv]
    assert int(out4[0, 0, 0, 1, 1].item()) == S_kv
