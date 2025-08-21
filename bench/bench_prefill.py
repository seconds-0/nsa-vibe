import time

import numpy as np
import torch

from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta
from nsa.core.nsa_attention import NSAAttention


def create_empty_kv(B: int, G: int, d_k: int, d_v: int, meta) -> NSA_KV:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        meta=meta,
    )


def benchmark_prefill_scaling():
    print("=" * 60)
    print("NSA Prefill Performance Benchmark")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 256
    n_heads = 8
    n_kv_groups = 2
    d_k = 32
    d_v = 32

    configs = [
        (1, 128),
        (1, 256),
        (1, 512),
        (1, 1024),
    ]

    results = []

    for B, S in configs:
        print(f"\nTesting B={B}, S={S}...")
        nsa = NSAAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_groups=n_kv_groups,
            d_k=d_k,
            d_v=d_v,
            l=32,
            d=16,
            l_sel=64,
            n_sel=16,
            w=512,
        )
        nsa = nsa.to(device)
        x = torch.randn(B, S, dim, device=device)
        meta = build_block_meta(S, 32, 16, 64, n_sel=16, w=512)
        kv = create_empty_kv(B, n_kv_groups, d_k, d_v, meta)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _, _ = nsa(x, kv, prefill=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(10):
            start = time.perf_counter()
            with torch.no_grad():
                _, _ = nsa(x, kv, prefill=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        mean_ms = float(np.mean(times))
        std_ms = float(np.std(times))
        print(f"Mean: {mean_ms:.2f}ms ± {std_ms:.2f}ms")
        results.append({"B": B, "S": S, "mean_ms": mean_ms, "std_ms": std_ms})

    print("\n" + "=" * 60)
    print("Scaling Analysis")
    print("=" * 60)
    singles = [r for r in results if r["B"] == 1]
    if len(singles) >= 3:
        t256 = next(r for r in singles if r["S"] == 256)["mean_ms"]
        t512 = next(r for r in singles if r["S"] == 512)["mean_ms"]
        t1024 = next(r for r in singles if r["S"] == 1024)["mean_ms"]
        print(f"256->512: {t512 / t256:.2f}x; 512->1024: {t1024 / t512:.2f}x")

    return results


if __name__ == "__main__":
    benchmark_prefill_scaling()
