from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class BlockMeta:
    l: int
    d: int
    l_sel: int
    n_sel: int
    w: int
    cmp_starts: torch.Tensor  # [S_cmp]
    sel_starts: torch.Tensor  # [S_sel]
    # CSR representation: (indptr, indices, values) mapping cmp_idx -> {sel_idx: weight}
    M_csl_indptr: torch.Tensor
    M_csl_indices: torch.Tensor
    M_csl_values: torch.Tensor
    # COO representation for fast batched matmul
    M_csl_coo_indices: torch.Tensor  # [2, nnz] rows, cols
    M_csl_coo_values: torch.Tensor   # [nnz]


def build_block_starts(seq_len: int, l: int, d: int, l_sel: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if d <= 0 or l <= 0 or l_sel <= 0:
        raise ValueError("Block parameters must be positive")
    # compression blocks (overlapped)
    max_cmp = 0 if seq_len < l else (seq_len - l) // d + 1
    cmp_starts = torch.arange(max_cmp, dtype=torch.int32) * d
    # selection blocks (non-overlapped)
    max_sel = 0 if seq_len <= 0 else (seq_len + l_sel - 1) // l_sel
    sel_starts = torch.arange(max_sel, dtype=torch.int32) * l_sel
    return cmp_starts, sel_starts


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def build_M_csl_csr(seq_len: int, l: int, d: int, l_sel: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Build CSR with fractional-overlap weights from cmp blocks to sel blocks
    cmp_starts, sel_starts = build_block_starts(seq_len, l, d, l_sel)
    indptr = [0]
    indices: list[int] = []
    values: list[float] = []
    for cmp_i, s in enumerate(cmp_starts.tolist()):
        a0, a1 = s, s + l
        total = 0
        row_pairs: list[Tuple[int, int]] = []
        for sel_j, t in enumerate(sel_starts.tolist()):
            b0, b1 = t, t + l_sel
            ov = _overlap_len(a0, a1, b0, b1)
            if ov > 0:
                row_pairs.append((sel_j, ov))
                total += ov
        # normalize by total overlap to get fractional weights
        if total > 0:
            for sel_j, ov in row_pairs:
                indices.append(sel_j)
                values.append(ov / total)
        indptr.append(len(indices))
    return (
        torch.tensor(indptr, dtype=torch.int32),
        torch.tensor(indices, dtype=torch.int32),
        torch.tensor(values, dtype=torch.float32),
    )


def build_block_meta(seq_len: int, l: int, d: int, l_sel: int, n_sel: int, w: int) -> BlockMeta:
    if l % d != 0 or l_sel % d != 0:
        # Enforce divisibility by default (per PRD); general overlaps allowed later if needed
        raise ValueError("Require d|l and d|l_sel in M0")
    cmp_starts, sel_starts = build_block_starts(seq_len, l, d, l_sel)
    indptr, indices, values = build_M_csl_csr(seq_len, l, d, l_sel)
    # Build COO from CSR
    rows: list[int] = []
    for r in range(len(cmp_starts)):
        start, end = int(indptr[r].item()), int(indptr[r + 1].item())
        rows.extend([r] * (end - start))
    coo_indices = torch.stack([torch.tensor(rows, dtype=torch.int32), indices.clone()], dim=0)
    return BlockMeta(
        l=l,
        d=d,
        l_sel=l_sel,
        n_sel=n_sel,
        w=w,
        cmp_starts=cmp_starts,
        sel_starts=sel_starts,
        M_csl_indptr=indptr,
        M_csl_indices=indices,
        M_csl_values=values,
        M_csl_coo_indices=coo_indices,
        M_csl_coo_values=values.clone(),
    )


