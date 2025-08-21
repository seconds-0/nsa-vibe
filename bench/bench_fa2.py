import time

import torch

from nsa.core.attention_kernels import (
    batched_causal_attention_compressed_masked,
    compressed_attention_fa2,
    sliding_window_attention_fa2,
    sliding_window_attention_masked,
)
from nsa.kernels.flash_wrappers import fa2_supported


def bench_pair(fn_ref, fn_new, *args, iters=10):
    # warmup
    fn_ref(*args)
    torch.cuda.synchronize()  # Ensure warmup completes
    t0 = time.time()
    for _ in range(iters):
        fn_ref(*args)
    torch.cuda.synchronize()  # Ensure all ref iterations complete
    t1 = time.time()
    for _ in range(iters):
        fn_new(*args)
    torch.cuda.synchronize()  # Ensure all new iterations complete
    t2 = time.time()
    return (t1 - t0) / iters, (t2 - t1) / iters


def bench_once(B, S, G, h, Dk, Dv, w, l, d, device):
    Q = torch.randn(B, S, G, h, Dk, device=device)
    K = torch.randn(B, G, S, Dk, device=device)
    V = torch.randn(B, G, S, Dv, device=device)
    # Sliding
    if device.type == "cuda" and fa2_supported(device, Q.dtype, Dk):
        s_ref, s_new = bench_pair(
            sliding_window_attention_masked, sliding_window_attention_fa2, Q, K, V, w
        )
        print(
            f"S={S} w={w} sliding masked {s_ref * 1e3:.2f} ms  fa2 {s_new * 1e3:.2f} ms  speedup x{(s_ref / max(s_new, 1e-9)):.2f}"
        )
        return s_ref, s_new
    else:
        print("FA-2 unsupported; skipping sliding bench")
        return None, None


def bench_cmp(B, S, G, h, Dk, Dv, l, d, device):
    Q = torch.randn(B, S, G, h, Dk, device=device)
    S_cmp = 0 if S < l else (S - l) // d + 1
    if S_cmp <= 0:
        return None, None
    K_raw = torch.randn(B, G, S, Dk, device=device)
    V_raw = torch.randn(B, G, S, Dv, device=device)
    K_cmp = torch.stack([K_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    V_cmp = torch.stack([V_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2)
    if device.type == "cuda" and fa2_supported(device, Q.dtype, Dk):
        c_ref, c_new = bench_pair(
            batched_causal_attention_compressed_masked,
            compressed_attention_fa2,
            Q,
            K_cmp,
            V_cmp,
            l,
            d,
        )
        print(
            f"S={S} l={l} d={d} compressed masked {c_ref * 1e3:.2f} ms  fa2 {c_new * 1e3:.2f} ms  speedup x{(c_ref / max(c_new, 1e-9)):.2f}"
        )
        return c_ref, c_new
    else:
        print("FA-2 unsupported; skipping compressed bench")
        return None, None


def run_sliding_benchmark(heads, dk, dv, S_list, w, iters, device):
    """Run sliding window benchmarks."""
    torch.manual_seed(0)
    print(f"# Sliding window benchmark (w={w})")
    print(f"# Device: {device}")

    for S in S_list:
        B, G, h = 1, 4, heads
        Q = torch.randn(B, S, G, h, dk, device=device)
        K = torch.randn(B, G, S, dk, device=device)
        V = torch.randn(B, G, S, dv, device=device)

        if device.type == "cuda" and fa2_supported(device, Q.dtype, dk):
            s_ref, s_new = bench_pair(
                sliding_window_attention_masked,
                sliding_window_attention_fa2,
                Q,
                K,
                V,
                w,
                iters=iters,
            )
            speedup = s_ref / max(s_new, 1e-9)
            print(
                f"S={S} w={w} sliding masked {s_ref * 1e3:.2f} ms  fa2 {s_new * 1e3:.2f} ms  speedup x{speedup:.2f}"
            )
        else:
            print(f"S={S} w={w} FA-2 unsupported; skipping")


def run_compressed_benchmark(heads, dk, dv, S_list, l, d, iters, device):
    """Run compressed attention benchmarks."""
    torch.manual_seed(0)
    print(f"# Compressed attention benchmark (l={l}, d={d})")
    print(f"# Device: {device}")

    for S in S_list:
        B, G, h = 1, 4, heads
        Q = torch.randn(B, S, G, h, dk, device=device)
        S_cmp = 0 if S < l else (S - l) // d + 1
        if S_cmp <= 0:
            print(f"S={S} l={l} d={d} compressed skipped (S_cmp={S_cmp})")
            continue

        K_raw = torch.randn(B, G, S, dk, device=device)
        V_raw = torch.randn(B, G, S, dv, device=device)
        K_cmp = torch.stack(
            [K_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2
        )
        V_cmp = torch.stack(
            [V_raw[:, :, i * d : i * d + l].mean(dim=2) for i in range(S_cmp)], dim=2
        )

        if device.type == "cuda" and fa2_supported(device, Q.dtype, dk):
            c_ref, c_new = bench_pair(
                batched_causal_attention_compressed_masked,
                compressed_attention_fa2,
                Q,
                K_cmp,
                V_cmp,
                l,
                d,
                iters=iters,
            )
            speedup = c_ref / max(c_new, 1e-9)
            print(
                f"S={S} l={l} d={d} compressed masked {c_ref * 1e3:.2f} ms  fa2 {c_new * 1e3:.2f} ms  speedup x{speedup:.2f}"
            )
        else:
            print(f"S={S} l={l} d={d} FA-2 unsupported; skipping")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FA-2 vs masked SDPA benchmarks")
    parser.add_argument(
        "--mode",
        choices=["win", "cmp", "legacy"],
        default="legacy",
        help="Benchmark mode: win=sliding window, cmp=compressed, legacy=original sweep",
    )
    parser.add_argument("--heads", type=int, default=8, help="Number of heads per group")
    parser.add_argument("--dk", type=int, default=64, help="Key dimension")
    parser.add_argument("--dv", type=int, default=64, help="Value dimension")
    parser.add_argument(
        "--S_list",
        type=str,
        default="128,256,512,1024,2048",
        help="Comma-separated sequence lengths",
    )
    parser.add_argument("--w", type=int, default=512, help="Sliding window size")
    parser.add_argument("--l", type=int, default=32, help="Compressed block size")
    parser.add_argument("--d", type=int, default=16, help="Compressed stride")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")

    args = parser.parse_args()

    # Parse S_list
    S_list = [int(s.strip()) for s in args.S_list.split(",")]

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cpu" and args.mode != "legacy":
        print("FA-2 benchmarks require CUDA; falling back to legacy mode")
        args.mode = "legacy"

    if args.mode == "win":
        run_sliding_benchmark(args.heads, args.dk, args.dv, S_list, args.w, args.iters, device)
    elif args.mode == "cmp":
        run_compressed_benchmark(
            args.heads, args.dk, args.dv, S_list, args.l, args.d, args.iters, device
        )
    else:
        # Legacy mode - original sweep functionality
        torch.manual_seed(0)
        if device.type == "cpu":
            print("Skipping FA-2 bench on CPU")
        else:
            best_win = None
            best_cmp = None
            print("Sliding sweep:")
            for S in [256, 512, 1024]:
                for w in [32, 64, 128, 256]:
                    s_ref, s_new = bench_once(
                        B=1, S=S, G=2, h=2, Dk=64, Dv=64, w=w, l=32, d=16, device=device
                    )
                    if s_ref is not None and s_new is not None:
                        if best_win is None or (s_ref - s_new) > (best_win[1] - best_win[2]):
                            best_win = (S, w, s_ref, s_new)
            print("Compressed sweep:")
            for S in [256, 512, 1024]:
                c_ref, c_new = bench_cmp(
                    B=1, S=S, G=2, h=2, Dk=64, Dv=64, l=32, d=16, device=device
                )
                if c_ref is not None and c_new is not None:
                    if best_cmp is None or (c_ref - c_new) > (best_cmp[1] - best_cmp[2]):
                        best_cmp = (S, c_ref, c_new)
            if best_win:
                print(
                    f"Recommended NSA_FA2_MIN_LEN_WIN around w where fa2>sdpa: e.g., w={best_win[1]} for S={best_win[0]}"
                )
            if best_cmp:
                print("Review compressed thresholds via S and (l,d) where FA-2 shows speedup.")
