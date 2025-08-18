import os
import time
import torch

from nsa.core.attention_kernels import (
    sliding_window_attention_masked,
    sliding_window_attention_fa2,
    batched_causal_attention_compressed_masked,
    compressed_attention_fa2,
)
from nsa.kernels.flash_wrappers import fa2_supported


def bench_pair(fn_ref, fn_new, *args, iters=10):
    # warmup
    fn_ref(*args)
    t0 = time.time()
    for _ in range(iters):
        fn_ref(*args)
    t1 = time.time()
    for _ in range(iters):
        fn_new(*args)
    t2 = time.time()
    return (t1 - t0) / iters, (t2 - t1) / iters


def bench_once(B, S, G, h, Dk, Dv, w, l, d, device):
    Q = torch.randn(B, S, G, h, Dk, device=device)
    K = torch.randn(B, G, S, Dk, device=device)
    V = torch.randn(B, G, S, Dv, device=device)
    # Sliding
    if device.type == "cuda" and fa2_supported(device, Q.dtype, Dk):
        s_ref, s_new = bench_pair(sliding_window_attention_masked, sliding_window_attention_fa2, Q, K, V, w)
        print(f"S={S} w={w} sliding masked {s_ref*1e3:.2f} ms  fa2 {s_new*1e3:.2f} ms  speedup x{(s_ref/max(s_new,1e-9)):.2f}")
    else:
        print("FA-2 unsupported; skipping sliding bench")
    # Compressed
    S_cmp = 0 if S < l else (S - l) // d + 1
    if S_cmp > 0:
        K_raw = torch.randn(B, G, S, Dk, device=device)
        V_raw = torch.randn(B, G, S, Dv, device=device)
        K_cmp = torch.stack([K_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
        V_cmp = torch.stack([V_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
        if device.type == "cuda" and fa2_supported(device, Q.dtype, Dk):
            c_ref, c_new = bench_pair(batched_causal_attention_compressed_masked, compressed_attention_fa2, Q, K_cmp, V_cmp, l, d)
            print(f"S={S} l={l} d={d} compressed masked {c_ref*1e3:.2f} ms  fa2 {c_new*1e3:.2f} ms  speedup x{(c_ref/max(c_new,1e-9)):.2f}")
        else:
            print("FA-2 unsupported; skipping compressed bench")


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Skipping FA-2 bench on CPU")
    else:
        for S in [256, 512, 1024]:
            for w in [64, 128, 256]:
                bench_once(B=1, S=S, G=2, h=2, Dk=64, Dv=64, w=w, l=32, d=16, device=device)


