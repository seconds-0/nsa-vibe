import pytest
import torch

from nsa.core.block_index import build_block_meta
from nsa.core.selection_scorer import (
    group_reduce_pslc,
    map_pcmp_to_pslc_batched,
    select_topn_ranges,
)

# Test configuration constants (PRD-like defaults, optimized for speed)
DEFAULT_L = 32  # Compressed block size
DEFAULT_D = 16  # Compressed block stride
DEFAULT_L_SEL = 64  # Selection block size
DEFAULT_N_SEL = 8  # Number of selected blocks (reduced from 16 for speed)
DEFAULT_W = 512  # Sliding window size
DEFAULT_B = 1  # Batch size
DEFAULT_SQ = 1  # Sequence query dimension
DEFAULT_G = 2  # Number of groups
DEFAULT_H = 1  # Heads per group


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
    # Use configurable test parameters
    meta = build_block_meta(
        S_ctx, l=DEFAULT_L, d=DEFAULT_D, l_sel=DEFAULT_L_SEL, n_sel=DEFAULT_N_SEL, w=DEFAULT_W
    )

    # Place needle somewhere away from edges to avoid force-local bias dominating
    pos = int(S_ctx // 2)
    sel_idx = _select_block_index_containing(meta, pos)
    cmp_row = _best_compressed_row_for_selection_block(meta, sel_idx)

    # Build a p_cmp distribution that puts all mass on the chosen compressed row
    S_cmp = int(meta.cmp_starts.numel())
    p_cmp_all = torch.zeros(
        (DEFAULT_B, DEFAULT_SQ, DEFAULT_G, DEFAULT_H, S_cmp), device=device, dtype=torch.float32
    )
    p_cmp_all[..., cmp_row] = 1.0

    # Map to selection scores and reduce over heads per group
    p_slc_all = map_pcmp_to_pslc_batched(p_cmp_all, meta)
    p_grp = group_reduce_pslc(p_slc_all.squeeze(1))  # [B,G,S_sel]

    # Select ranges at token t = S_ctx - 1
    t_token = S_ctx - 1
    ranges = select_topn_ranges(
        p_grp, meta, n_top=DEFAULT_N_SEL, t_token=t_token, force_init=True, force_local=2
    )

    # Assert the needle position is covered by at least one selected range in all groups
    for g in range(DEFAULT_G):
        _assert_range_covers_pos(ranges[:, g], pos)


def test_selection_mapping_includes_needle_cpu_small():
    # CPU-friendly size; validates Eq.9 mapping + top‑n range formation covers the needle
    torch.manual_seed(0)
    _run_needle_check(S_ctx=4096, device=torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for 64k variant")
def test_selection_mapping_includes_needle_64k_cuda():
    # Check GPU memory availability (require at least 8GB for 64k test)
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        required_memory = 8 * 1024**3  # 8GB
        if total_memory < required_memory:
            pytest.skip(
                f"Insufficient GPU memory for 64k test: {total_memory / 1024**3:.1f}GB < {required_memory / 1024**3:.1f}GB"
            )

    # GPU-only long-context stress; same semantic check at 64k
    torch.manual_seed(0)
    _run_needle_check(S_ctx=65536, device=torch.device("cuda"))
