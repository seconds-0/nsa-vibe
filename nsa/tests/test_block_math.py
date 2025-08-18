import math

import pytest
import torch

from nsa.core.block_index import build_block_meta


@pytest.mark.parametrize("l,d,l_sel", [(32, 16, 64), (16, 8, 32)])
def test_eq9_mapping_fractional_overlap(l, d, l_sel):
    S = 1024
    meta = build_block_meta(S, l, d, l_sel, n_sel=16, w=512)
    # random p_cmp per (S_cmp)
    S_cmp = meta.cmp_starts.numel()
    p_cmp = torch.rand(S_cmp)
    p_cmp = p_cmp / p_cmp.sum()
    # reference p_slc via fractional overlaps
    S_sel = meta.sel_starts.numel()
    p_ref = torch.zeros(S_sel)
    for i, s in enumerate(meta.cmp_starts.tolist()):
        a0, a1 = s, s + l
        total = 0
        ov = []
        for j, t in enumerate(meta.sel_starts.tolist()):
            b0, b1 = t, t + l_sel
            x = max(0, min(a1, b1) - max(a0, b0))
            ov.append(x)
            total += x
        if total > 0:
            for j, x in enumerate(ov):
                if x > 0:
                    p_ref[j] += p_cmp[i] * (x / total)
    # Compute via CSR
    indptr, indices, values = meta.M_csl_indptr, meta.M_csl_indices, meta.M_csl_values
    p_mdc = torch.zeros_like(p_ref)
    for r in range(S_cmp):
        start, end = int(indptr[r].item()), int(indptr[r + 1].item())
        cols = indices[start:end]
        w = values[start:end]
        if cols.numel() > 0:
            p_mdc.index_add_(0, cols, p_cmp[r] * w)
    assert torch.allclose(p_mdc, p_ref, atol=1e-6)


def test_divisibility_guards():
    with pytest.raises(ValueError):
        _ = build_block_meta(1024, l=30, d=16, l_sel=64, n_sel=16, w=512)
    with pytest.raises(ValueError):
        _ = build_block_meta(1024, l=32, d=12, l_sel=60, n_sel=16, w=512)


