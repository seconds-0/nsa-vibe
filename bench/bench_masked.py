import time

import torch

from nsa.core.attention_kernels import (
    batched_causal_attention_compressed,
    batched_causal_attention_compressed_masked,
    sliding_window_attention,
    sliding_window_attention_masked,
)


def bench_sliding(B=1, S=256, G=1, h=4, Dk=64, Dv=64, w=64, iters=5):
    torch.manual_seed(0)
    Q = torch.randn(B, S, G, h, Dk)
    K = torch.randn(B, G, S, Dk)
    V = torch.randn(B, G, S, Dv)
    # warmup
    sliding_window_attention(Q, K, V, w)
    t0 = time.time()
    for _ in range(iters):
        sliding_window_attention(Q, K, V, w)
    t1 = time.time()
    for _ in range(iters):
        sliding_window_attention_masked(Q, K, V, w)
    t2 = time.time()
    return (t1 - t0) / iters, (t2 - t1) / iters


def bench_compressed(B=1, S=256, G=1, h=4, Dk=64, Dv=64, l=16, d=8, iters=5):
    torch.manual_seed(1)
    Q = torch.randn(B, S, G, h, Dk)
    S_cmp = 0 if S < l else (S - l) // d + 1
    if S_cmp == 0:
        K_cmp = torch.zeros(B, G, 0, Dk)
        V_cmp = torch.zeros(B, G, 0, Dv)
    else:
        K_raw = torch.randn(B, G, S, Dk)
        V_raw = torch.randn(B, G, S, Dv)
        K_cmp = torch.stack(
            [K_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2
        )
        V_cmp = torch.stack(
            [V_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2
        )
    # warmup
    batched_causal_attention_compressed(Q, K_cmp, V_cmp, l, d)
    t0 = time.time()
    for _ in range(iters):
        batched_causal_attention_compressed(Q, K_cmp, V_cmp, l, d)
    t1 = time.time()
    for _ in range(iters):
        batched_causal_attention_compressed_masked(Q, K_cmp, V_cmp, l, d)
    t2 = time.time()
    return (t1 - t0) / iters, (t2 - t1) / iters


if __name__ == "__main__":
    s_ref, s_mask = bench_sliding()
    c_ref, c_mask = bench_compressed()
    print(f"sliding  ref {s_ref * 1e3:.2f} ms  masked {s_mask * 1e3:.2f} ms")
    print(f"compressed ref {c_ref * 1e3:.2f} ms  masked {c_mask * 1e3:.2f} ms")
