import torch


def compute_sliding_lengths(S: int, w: int, device: torch.device) -> torch.Tensor:
	"""
	Return per-row window lengths for sliding attention: L_t = min(w, t+1)
	Shape: [S]
	"""
	tpos = torch.arange(S, device=device)
	return (tpos + 1).clamp_max(w)


def compute_compressed_lengths(S: int, l: int, d: int, S_cmp: int, device: torch.device) -> torch.Tensor:
	"""
	Return per-row valid compressed lengths: num_cmp(t)
	Shape: [S]
	"""
	tpos = torch.arange(S, device=device)
	return torch.where(tpos + 1 < l, 0, ((tpos + 1 - l) // d) + 1).clamp(min=0, max=S_cmp)


def build_length_buckets(lengths: torch.Tensor) -> list[torch.Tensor]:
	"""
	Group row indices by identical length.
	Args:
		lengths: [S] int tensor
	Returns:
		List of index tensors, one per unique length (descending by length)
	"""
	if lengths.numel() == 0:
		return []
	unique = torch.unique(lengths, sorted=True)
	# sort descending so larger buckets processed first
	unique = torch.flip(unique, dims=[0])
	buckets: list[torch.Tensor] = []
	for L in unique.tolist():
		idx = torch.nonzero(lengths == int(L), as_tuple=False).flatten()
		buckets.append(idx)
	return buckets


def build_cu_seqlens_for_buckets(bucket_lengths: torch.Tensor) -> torch.Tensor:
	"""
	Build cumulative sequence lengths (cu_seqlens) for varlen APIs from a vector of lengths.
	Args:
		bucket_lengths: [N] lengths per row in a bucket
	Returns:
		cu_seqlens: [N+1] with cu_seqlens[0]=0 and cu_seqlens[i+1]=sum_{j<=i} len[j]
	"""
	if bucket_lengths.numel() == 0:
		return torch.zeros((1,), dtype=torch.int32, device=bucket_lengths.device)
	cs = torch.zeros((bucket_lengths.numel() + 1,), dtype=torch.int32, device=bucket_lengths.device)
	cs[1:] = torch.cumsum(bucket_lengths.to(dtype=torch.int32), dim=0)
	return cs


