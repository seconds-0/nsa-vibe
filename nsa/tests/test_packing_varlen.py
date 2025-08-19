import torch

from nsa.core.packing import pack_batch_by_lengths, unpack_packed_to_padded, build_cu_seqlens_for_buckets


def test_pack_unpack_roundtrip():
	B, S_max, D = 3, 7, 5
	x = torch.arange(B*S_max*D, dtype=torch.float32).reshape(B, S_max, D)
	lengths = torch.tensor([3, 0, 5])
	packed, cu = pack_batch_by_lengths(x, lengths)
	padded, mask = unpack_packed_to_padded(packed, cu, S_max)
	assert mask[0, :3].all() and not mask[0, 3:].any()
	assert mask[1].sum().item() == 0
	assert mask[2, :5].all() and not mask[2, 5:].any()
	# First row payload preserved
	assert torch.allclose(padded[0, :3], x[0, :3])
	# Third row payload preserved
	assert torch.allclose(padded[2, :5], x[2, :5])


