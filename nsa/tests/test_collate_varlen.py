import torch

from nsa.core.collate import collate_token_batch


def test_collate_token_batch_masks_and_labels():
	seqs = [list(range(5)), [7, 8], []]
	input_ids, labels, attn_mask, loss_mask, lengths, cu = collate_token_batch(seqs, pad_id=0)
	assert input_ids.shape == (3, 5)
	assert labels.shape == (3, 5)
	assert attn_mask[0].sum().item() == 5
	assert attn_mask[1].sum().item() == 2
	assert attn_mask[2].sum().item() == 0
	# label shift: first row labels[0,0..3] == input_ids[0,1..4], labels[0,4] untouched
	assert torch.all(labels[0, :4] == input_ids[0, 1:5])
	assert loss_mask[0, :4].all() and not loss_mask[0, 4]
	# cu_seqlens monotonic
	assert cu.tolist() == [0, 5, 7, 7]


