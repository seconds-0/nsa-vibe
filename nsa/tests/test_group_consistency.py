import torch

from nsa.core.block_index import build_block_meta
from nsa.core.selection_scorer import group_reduce_pslc, map_pcmp_to_pslc


def test_group_reduce_identical_across_heads():
    B, G, h, Dk = 2, 3, 4, 8
    S_cmp = 10
    # construct meta
    meta = build_block_meta(1024, l=32, d=16, l_sel=64, n_sel=16, w=512)
    # random p_cmp but identical across heads by construction
    p_base = torch.rand(B, G, 1, S_cmp)
    p_base = p_base / p_base.sum(dim=-1, keepdim=True)
    p_cmp = p_base.repeat(1, 1, h, 1)
    p_slc = map_pcmp_to_pslc(p_cmp, meta)
    p_grp = group_reduce_pslc(p_slc)
    # heads drop out; just check shape and that reduction equals h * head0 contribution
    assert p_grp.shape[:2] == (B, G)
    # sanity: sum of slc per group equals h times single head mapping (up to float error)
    single = map_pcmp_to_pslc(p_base, meta).squeeze(2)
    assert torch.allclose(p_grp, single * h, atol=1e-5)
