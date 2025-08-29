from typing import List

import torch


def compute_sliding_lengths(S: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Return per-row window lengths for sliding attention: L_t = min(w, t+1)
    Shape: [S]
    """
    tpos = torch.arange(S, device=device)
    return (tpos + 1).clamp_max(w)


def compute_compressed_lengths(
    S: int, l: int, d: int, S_cmp: int, device: torch.device
) -> torch.Tensor:
    """
    Return per-row valid compressed lengths: num_cmp(t)
    Shape: [S]
    """
    tpos = torch.arange(S, device=device)
    return torch.where(tpos + 1 < l, 0, ((tpos + 1 - l) // d) + 1).clamp(min=0, max=S_cmp)


def build_length_buckets(lengths: torch.Tensor) -> List[torch.Tensor]:
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
    buckets: List[torch.Tensor] = []
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


def pack_batch_by_lengths(
    x: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack a batch of padded rows into a contiguous buffer with cu_seqlens.

    Args:
            x: [B,S_max,D]
            lengths: [B] valid lengths per row
    Returns:
            (packed: [sum(lengths), D], cu_seqlens: [B+1])
    """
    device = x.device
    B, S_max, D = x.shape
    assert lengths.shape[0] == B
    cu = build_cu_seqlens_for_buckets(lengths.to(torch.int32))
    N = int(cu[-1].item())
    packed = torch.empty((N, D), dtype=x.dtype, device=device)
    write = 0
    for b in range(B):
        L = int(lengths[b].item())
        if L > 0:
            packed[write : write + L] = x[b, :L]
            write += L
    return packed, cu


def unpack_packed_to_padded(
    packed: torch.Tensor, cu_seqlens: torch.Tensor, S_max: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack a packed buffer back to padded batch and mask.

    Args:
            packed: [N,D]
            cu_seqlens: [B+1]
            S_max: target padded length
    Returns:
            (padded [B,S_max,D], mask [B,S_max])
    """
    device = packed.device
    B = cu_seqlens.shape[0] - 1
    D = packed.shape[-1]
    padded = torch.zeros((B, S_max, D), dtype=packed.dtype, device=device)
    mask = torch.zeros((B, S_max), dtype=torch.bool, device=device)
    for b in range(B):
        start = int(cu_seqlens[b].item())
        end = int(cu_seqlens[b + 1].item())
        L = end - start
        if L > 0:
            padded[b, :L] = packed[start:end]
            mask[b, :L] = True
    return padded, mask
