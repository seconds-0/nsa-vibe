import os
import torch

from nsa.core.block_index import build_block_meta
from nsa.cache.kv_cache import NSA_KV
from nsa.core.nsa_attention import NSAAttention
from nsa.kernels.cuda_sel_kernel import selection_attention_cuda
from nsa.core.attention_kernels import grouped_selection_attention_packed


def _empty_kv(B, G, d_k, d_v, device):
    return NSA_KV(
        K_sel=torch.zeros((B, G, 0, d_k), device=device),
        V_sel=torch.zeros((B, G, 0, d_v), device=device),
        K_win=torch.zeros((B, G, 0, d_k), device=device),
        V_win=torch.zeros((B, G, 0, d_v), device=device),
        K_cmp_raw_seq=torch.zeros((B, G, 0, d_k), device=device),
        V_cmp_raw_seq=torch.zeros((B, G, 0, d_v), device=device),
        K_cmp=torch.zeros((B, G, 0, d_k), device=device),
        V_cmp=torch.zeros((B, G, 0, d_v), device=device),
        win_ptr=torch.zeros((B, G), dtype=torch.int32, device=device),
        cmp_emit_next=torch.zeros((B, G), dtype=torch.int32, device=device),
        reads_pred=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_total=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_sel=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_cmp=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_win=torch.zeros((0,), dtype=torch.int64, device=device),
        meta=build_block_meta(16, 4, 2, 8, n_sel=2, w=8),
    )


def test_sel_cuda_wrapper_fallback_parity():
    os.environ["NSA_SEL_CUDA"] = "1"
    torch.manual_seed(0)
    device = torch.device("cpu")
    B, S, G, h, Dk, Dv = 2, 3, 2, 2, 16, 16
    Q = torch.randn(B, S, G, h, Dk, device=device)
    K = torch.randn(B, G, 64, Dk, device=device)
    V = torch.randn(B, G, 64, Dv, device=device)
    # Simple ranges: two small spans per (b,s,g)
    ranges = torch.zeros(B, S, G, 2, 2, dtype=torch.int32, device=device)
    ranges[..., 0, 0] = 0
    ranges[..., 0, 1] = 8
    ranges[..., 1, 0] = 16
    ranges[..., 1, 1] = 24
    O_cuda = selection_attention_cuda(Q, K, V, ranges)
    O_ref = grouped_selection_attention_packed(Q, K, V, ranges)
    mae = (O_cuda - O_ref).abs().mean().item()
    assert mae < 1e-6

