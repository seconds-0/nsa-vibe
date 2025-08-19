from typing import List, Tuple

import torch


def collate_token_batch(
	sequences: List[List[int]],
	*,
	pad_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Collate token id sequences (var-length) into padded tensors and masks with label shift.

	Args:
		sequences: list of token id lists
		pad_id: id used for padding
	Returns:
		input_ids: [B,S_max]
		labels:    [B,S_max]  (next-token labels; last position masked out)
		attn_mask: [B,S_max]  (True for valid tokens)
		loss_mask: [B,S_max]  (True for positions to include in loss)
		lengths:   [B]
		cu_seqlens:[B+1]  cumulative lengths
	"""
	B = len(sequences)
	lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.int32)
	S_max = int(lengths.max().item()) if B > 0 else 0
	input_ids = torch.full((B, S_max), pad_id, dtype=torch.long)
	labels = torch.full((B, S_max), pad_id, dtype=torch.long)
	attn_mask = torch.zeros((B, S_max), dtype=torch.bool)
	loss_mask = torch.zeros((B, S_max), dtype=torch.bool)
	for b, seq in enumerate(sequences):
		L = len(seq)
		if L == 0:
			continue
		input_ids[b, :L] = torch.tensor(seq, dtype=torch.long)
		attn_mask[b, :L] = True
		# next-token labels (shifted left by 1), last token has no next label
		labels[b, : L - 1] = input_ids[b, 1:L]
		loss_mask[b, : L - 1] = True
	# cu_seqlens for varlen APIs
	cu = torch.zeros((B + 1,), dtype=torch.int32)
	cu[1:] = torch.cumsum(lengths, dim=0)
	return input_ids, labels, attn_mask, loss_mask, lengths, cu


