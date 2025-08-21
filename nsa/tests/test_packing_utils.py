import torch

from nsa.core.packing import (
    build_cu_seqlens_for_buckets,
    pack_batch_by_lengths,
    unpack_packed_to_padded,
)


def test_cu_seqlens_basic():
    lens = torch.tensor([3, 0, 5, 2], dtype=torch.int32)
    cu = build_cu_seqlens_for_buckets(lens)
    assert cu.tolist() == [0, 3, 3, 8, 10]


def test_pack_unpack_roundtrip():
    B, S_max, D = 3, 6, 4
    x = torch.zeros((B, S_max, D))
    lens = torch.tensor([3, 0, 5], dtype=torch.int32)
    # Fill with identifiable rows
    x[0, :3] = 1.0
    x[2, :5] = 2.0
    packed, cu = pack_batch_by_lengths(x, lens)
    padded, mask = unpack_packed_to_padded(packed, cu, S_max)
    # Check shapes
    assert packed.shape == (8, D)
    assert padded.shape == (B, S_max, D)
    # Check content where mask True
    assert torch.allclose(padded[0, :3], torch.ones((3, D)))
    assert torch.allclose(padded[2, :5], torch.full((5, D), 2.0))
    # Mask assertions
    assert mask.dtype == torch.bool
    assert mask[0, :3].all() and not mask[0, 3:].any()
    assert not mask[1].any()
    assert mask[2, :5].all() and not mask[2, 5:].any()
