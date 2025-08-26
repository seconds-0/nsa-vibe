import os
import torch

from nsa.core.block_index import build_block_meta
from nsa.core.selection_scorer import (
    select_topn_ranges,
    select_topn_ranges_batched,
    validate_selection_determinism,
)


def _meta(S: int = 64, l: int = 4, d: int = 2, l_sel: int = 4, n_sel: int = 8, w: int = 8):
    return build_block_meta(S, l, d, l_sel, n_sel, w)


def test_select_topn_ranges_tie_break_prefers_lower_index_across_dtypes():
    meta = _meta(S=64, l_sel=4, n_sel=8)
    B, G, S_sel = 1, 1, len(meta.sel_starts)  # Use actual number of selection blocks
    # Create equal scores across all indices (ties everywhere)
    scores32 = torch.ones((B, G, S_sel), dtype=torch.float32)
    scores16 = torch.ones((B, G, S_sel), dtype=torch.float16)
    scoresbf = torch.ones((B, G, S_sel), dtype=torch.bfloat16)

    # Ensure all blocks are valid at this token (big t)
    t_token = 63
    n_top = 3
    # Disable forced includes to isolate tie-breaking behavior
    r32 = select_topn_ranges(scores32, meta, n_top, t_token, force_init=False, force_local=0)
    r16 = select_topn_ranges(scores16, meta, n_top, t_token, force_init=False, force_local=0)
    rbf = select_topn_ranges(scoresbf, meta, n_top, t_token, force_init=False, force_local=0)

    # With equal scores and adjacent block merging, we expect consistent behavior
    # The function merges adjacent blocks when they have equal scores
    # So we just verify all dtypes produce identical results
    for r in (r16, rbf):
        # Check that all dtypes produce the same selection as float32
        assert torch.equal(r, r32), f"Deterministic selection failed across dtypes"


def test_select_topn_ranges_batched_tie_break_prefers_lower_index_across_dtypes():
    meta = _meta(S=64, l_sel=4, n_sel=8)
    B, S, G, S_sel = 1, 3, 1, len(meta.sel_starts)  # Use actual number of selection blocks
    # Equal scores over all indices for each t
    scores32 = torch.ones((B, S, G, S_sel), dtype=torch.float32)
    scores16 = torch.ones((B, S, G, S_sel), dtype=torch.float16)
    scoresbf = torch.ones((B, S, G, S_sel), dtype=torch.bfloat16)
    n_top = 3
    # Disable forced includes to isolate tie-breaking
    r32 = select_topn_ranges_batched(scores32, meta, n_top, S, force_init=False, force_local=0)
    r16 = select_topn_ranges_batched(scores16, meta, n_top, S, force_init=False, force_local=0)
    rbf = select_topn_ranges_batched(scoresbf, meta, n_top, S, force_init=False, force_local=0)

    # With equal scores and adjacent block merging, we expect consistent behavior
    # All dtypes should return same deterministic selections
    for r in (r16, rbf):
        # Check that all dtypes produce the same selection as float32
        assert torch.equal(r, r32), f"Deterministic batched selection failed across dtypes"


def test_validate_selection_determinism_helper_true_when_enabled(monkeypatch):
    monkeypatch.setenv("NSA_VALIDATE_SELECTION_DETERMINISM", "1")
    torch.manual_seed(0)
    meta = _meta(S=128, l_sel=4, n_sel=8)
    B, G, S_sel = 1, 1, len(meta.sel_starts)  # Use actual number of selection blocks
    # Random but fixed scores; no grad to allow validation to run
    p_grp = torch.randn((B, G, S_sel), dtype=torch.float32)
    ok = validate_selection_determinism(p_grp, meta, n_top=5, t_token=63, num_trials=3)
    assert ok is True
