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
    B, G, S_sel = 1, 1, 6
    meta = _meta(S=64, l_sel=4, n_sel=8)
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

    # Ranges start at sel_starts[0], sel_starts[1], sel_starts[2] ... due to lower-index preference
    starts = meta.sel_starts[:n_top].tolist()
    for r in (r32, r16, rbf):
        got_starts = [int(r[0, 0, i, 0].item()) for i in range(n_top)]
        assert got_starts == starts


def test_select_topn_ranges_batched_tie_break_prefers_lower_index_across_dtypes():
    B, S, G, S_sel = 1, 3, 1, 6
    meta = _meta(S=64, l_sel=4, n_sel=8)
    # Equal scores over all indices for each t
    scores32 = torch.ones((B, S, G, S_sel), dtype=torch.float32)
    scores16 = torch.ones((B, S, G, S_sel), dtype=torch.float16)
    scoresbf = torch.ones((B, S, G, S_sel), dtype=torch.bfloat16)
    n_top = 3
    # Disable forced includes to isolate tie-breaking
    r32 = select_topn_ranges_batched(scores32, meta, n_top, S, force_init=False, force_local=0)
    r16 = select_topn_ranges_batched(scores16, meta, n_top, S, force_init=False, force_local=0)
    rbf = select_topn_ranges_batched(scoresbf, meta, n_top, S, force_init=False, force_local=0)

    # For each t, first range should start at the earliest block (index 0)
    expect0 = int(meta.sel_starts[0].item())
    for r in (r32, r16, rbf):
        for t in range(S):
            if r.shape[3] > 0:
                first_start = int(r[0, t, 0, 0, 0].item())
                assert first_start == expect0


def test_validate_selection_determinism_helper_true_when_enabled(monkeypatch):
    os.environ["NSA_VALIDATE_SELECTION_DETERMINISM"] = "1"
    torch.manual_seed(0)
    B, G, S_sel = 1, 1, 10
    meta = _meta(S=128, l_sel=4, n_sel=8)
    # Random but fixed scores; no grad to allow validation to run
    p_grp = torch.randn((B, G, S_sel), dtype=torch.float32)
    ok = validate_selection_determinism(p_grp, meta, n_top=5, t_token=63, num_trials=3)
    assert ok is True
