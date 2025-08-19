import os
import torch

from nsa.core.nsa_attention import NSAAttention
from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta


def main():
    torch.manual_seed(0)
    B, dim = 1, 64
    nsa = NSAAttention(dim=dim, n_heads=4, n_kv_groups=1, d_k=16, d_v=16, l=4, d=2, l_sel=4, n_sel=4, w=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nsa = nsa.to(device)
    kv = NSA_KV(
        K_sel=torch.zeros((B, nsa.n_kv_groups, 0, nsa.d_k), device=device),
        V_sel=torch.zeros((B, nsa.n_kv_groups, 0, nsa.d_v), device=device),
        K_win=torch.zeros((B, nsa.n_kv_groups, 0, nsa.d_k), device=device),
        V_win=torch.zeros((B, nsa.n_kv_groups, 0, nsa.d_v), device=device),
        K_cmp_raw_seq=torch.zeros((B, nsa.n_kv_groups, 0, nsa.d_k), device=device),
        V_cmp_raw_seq=torch.zeros((B, nsa.n_kv_groups, 0, nsa.d_v), device=device),
        K_cmp=torch.zeros((B, nsa.n_kv_groups, 0, nsa.d_k), device=device),
        V_cmp=torch.zeros((B, nsa.n_kv_groups, 0, nsa.d_v), device=device),
        win_ptr=torch.zeros((B, nsa.n_kv_groups), dtype=torch.int32, device=device),
        cmp_emit_next=torch.zeros((B, nsa.n_kv_groups), dtype=torch.int32, device=device),
        reads_pred=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_total=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_sel=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_cmp=torch.zeros((0,), dtype=torch.int64, device=device),
        reads_act_win=torch.zeros((0,), dtype=torch.int64, device=device),
        meta=build_block_meta(32, nsa.l, nsa.d, nsa.l_sel, nsa.n_sel, nsa.w),
    )
    # Prefill a short context
    x_ctx = torch.randn(B, 6, dim, device=device)
    _, kv = nsa(x_ctx, kv, prefill=True)
    # Decode a few steps with debug logs enabled
    os.environ["NSA_DEBUG_LOG"] = "1"
    for step in range(4):
        x_tok = torch.randn(B, 1, dim, device=device)
        y, kv = nsa(x_tok, kv, prefill=False)
        print(f"step={step} y_norm={(y.norm().item()):.4f} reads={int(kv.reads_act_total[-1].item())}")


if __name__ == "__main__":
    main()


