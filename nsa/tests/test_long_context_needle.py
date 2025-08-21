import os
import math
import pytest
import torch

from nsa.core.block_index import build_block_meta
from nsa.core.selection_scorer import (
    map_pcmp_to_pslc_batched,
    group_reduce_pslc,
    select_topn_ranges,
)


def _select_block_index_containing(meta, pos: int) -> int:
    starts = meta.sel_starts
    l_sel = int(meta.l_sel)
    # Find the selection block index whose range [start, start+l_sel) contains pos
    s = int(torch.nonzero((starts <= pos) & (pos < starts + l_sel), as_tuple=False)[-1].item())
    return s


def _best_compressed_row_for_selection_block(meta, sel_block_idx: int) -> int:
    # From COO mapping, find compressed row(s) that contribute to this selection column
    rows, cols = meta.M_csl_coo_indices
    vals = meta.M_csl_coo_values
    mask = cols == sel_block_idx
    assert mask.any(), "No compressed rows map to the target selection block — meta invalid?"
    rows_sel = rows[mask]
    vals_sel = vals[mask]
    # Pick the compressed row with the highest mapping weight to strengthen selection
    j = int(torch.argmax(vals_sel).item())
    return int(rows_sel[j].item())


def _assert_range_covers_pos(ranges: torch.Tensor, pos: int) -> None:
    # ranges: [B,G,n_top,2]
    s = ranges[..., 0]
    e = ranges[..., 1]
    covered = ((s <= pos) & (pos < e)).any()
    assert bool(covered), f"No selected range covers position {pos}"


def _run_needle_check(S_ctx: int, device: torch.device) -> None:
    # Default PRD-like params (reduced n_sel for speed)
    l, d, l_sel, n_sel, w = 32, 16, 64, 8, 512
    meta = build_block_meta(S_ctx, l=l, d=d, l_sel=l_sel, n_sel=n_sel, w=w)

    # Place needle somewhere away from edges to avoid force-local bias dominating
    pos = int(S_ctx // 2)
    sel_idx = _select_block_index_containing(meta, pos)
    cmp_row = _best_compressed_row_for_selection_block(meta, sel_idx)

    # Build a p_cmp distribution that puts all mass on the chosen compressed row
    B, Sq, G, h = 1, 1, 2, 1
    S_cmp = int(meta.cmp_starts.numel())
    p_cmp_all = torch.zeros((B, Sq, G, h, S_cmp), device=device, dtype=torch.float32)
    p_cmp_all[..., cmp_row] = 1.0

    # Map to selection scores and reduce over heads per group
    p_slc_all = map_pcmp_to_pslc_batched(p_cmp_all, meta)
    p_grp = group_reduce_pslc(p_slc_all.squeeze(1))  # [B,G,S_sel]

    # Select ranges at token t = S_ctx - 1
    t_token = S_ctx - 1
    ranges = select_topn_ranges(p_grp, meta, n_top=n_sel, t_token=t_token, force_init=True, force_local=2)

    # Assert the needle position is covered by at least one selected range in all groups
    for g in range(G):
        _assert_range_covers_pos(ranges[:, g], pos)


def test_selection_mapping_includes_needle_cpu_small():
    # CPU-friendly size; validates Eq.9 mapping + top‑n range formation covers the needle
    torch.manual_seed(0)
    _run_needle_check(S_ctx=4096, device=torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for 64k variant")
def test_selection_mapping_includes_needle_64k_cuda():
    # GPU-only long-context stress; same semantic check at 64k
    torch.manual_seed(0)
    _run_needle_check(S_ctx=65536, device=torch.device("cuda"))

