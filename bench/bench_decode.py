import time
import numpy as np
import torch

from nsa.core.nsa_attention import NSAAttention
from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta


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


def compute_expected_reads(S: int, l: int, d: int, n: int, l_sel: int, w: int) -> int:
    num_cmp = 0 if S < l else (S - l) // d + 1
    return num_cmp + n * l_sel + min(w, S)


def benchmark_decode_step():
    print("=" * 60)
    print("NSA Decode Performance Benchmark")
    print("=" * 60)

    B = 1
    dim = 256
    nsa = NSAAttention(dim=dim, n_heads=8, n_kv_groups=2, d_k=32, d_v=32, l=32, d=16, l_sel=64, n_sel=16, w=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nsa = nsa.to(device)

    context_sizes = [128, 256, 512, 1024]
    print(f"\n{'Context':<10} {'Time/tok':<12} {'Actual Reads':<15} {'Expected':<15} {'Match'}")
    print("-" * 70)

    for S_ctx in context_sizes:
        x_ctx = torch.randn(B, S_ctx, dim, device=device)
        meta = build_block_meta(S_ctx + 64, nsa.l, nsa.d, nsa.l_sel, n_sel=nsa.n_sel, w=nsa.w)
        kv = create_empty_kv(B, nsa.n_kv_groups, nsa.d_k, nsa.d_v, meta)
        with torch.no_grad():
            _, kv = nsa(x_ctx, kv, prefill=True)

        decode_times = []
        for step in range(32):
            x_tok = torch.randn(B, 1, dim, device=device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _, kv = nsa(x_tok, kv, prefill=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - start)

        avg_ms = float(np.mean(decode_times[8:]) * 1000)
        S_current = S_ctx + 32
        expected_reads = compute_expected_reads(S_current, nsa.l, nsa.d, nsa.n_sel, nsa.l_sel, nsa.w)
        actual_reads = kv.reads_act_total[-1].item() if len(kv.reads_act_total) > 0 else -1
        match = "✓" if actual_reads == expected_reads else "✗"
        print(f"{S_ctx:<10} {avg_ms:<12.2f} {actual_reads:<15} {expected_reads:<15} {match}")


if __name__ == "__main__":
    benchmark_decode_step()


