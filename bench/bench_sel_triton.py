#!/usr/bin/env python3
"""
Triton Selection Attention Benchmark (M4) - Hybrid Version

Combines the best of both approaches:
- Simple, direct benchmarking from other agent's version
- Comprehensive analysis and reporting from my version

Usage:
    NSA_USE_TRITON_SEL=1 python bench/bench_sel_triton.py --N 256 --H 8 --D 128 --L_list 64,128,256,512 --dist many
    NSA_USE_TRITON_SEL=1 python bench/bench_sel_triton.py --csv --output results.csv
"""

import argparse
import csv
import os
import time
from typing import List, Tuple

import torch

# Precise benchmarking settings
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def build_test_data(N: int, H: int, D: int, Dv: int, L: int, dist: str, device: torch.device):
    """Build test tensors with realistic selection patterns."""
    Q = torch.randn(N, H, D, device=device, dtype=torch.bfloat16)
    
    # Create selection ranges based on distribution
    if dist == "few":
        # Single long range per row
        spans = [[(0, L)] for _ in range(N)]
    elif dist == "many":
        # 8 equal segments
        n = 8
        seg = max(1, L // n)
        spans = [[(i * seg, min((i + 1) * seg, L)) for i in range(n)] for _ in range(N)]
    else:
        # Mixed: random segments
        spans = []
        for _ in range(N):
            rem = L
            s = 0
            row = []
            while rem > 0:
                seg = min(rem, max(1, L // 8))
                row.append((s, s + seg))
                s += seg
                rem -= seg
            spans.append(row)
    
    # Build K/V from a shared pool; rows select spans from this pool
    pool_len = L * 2
    Kpool = torch.randn(pool_len, D, device=device, dtype=torch.bfloat16)
    Vpool = torch.randn(pool_len, Dv, device=device, dtype=torch.bfloat16)
    
    K_rows: List[torch.Tensor] = []
    V_rows: List[torch.Tensor] = []
    for row_spans in spans:
        idx = torch.cat([torch.arange(s, e, device=device) for (s, e) in row_spans], dim=0)
        K_rows.append(Kpool.index_select(0, idx))
        V_rows.append(Vpool.index_select(0, idx))
    
    return Q, K_rows, V_rows, spans, Kpool, Vpool


def benchmark_method(Q, K_rows, V_rows, method: str, iters: int, *, Kpool=None, Vpool=None, spans=None) -> float:
    """Benchmark single method and return average time in ms."""
    N, H, D = Q.shape
    Dv = V_rows[0].shape[1]
    
    if method == "sdpa":
        # Direct SDPA per row
        times = []
        for i in range(N):
            k = K_rows[i]
            v = V_rows[i]
            # Expand for multi-head: [1, H, L, D]
            Kf = k.unsqueeze(0).unsqueeze(0).expand(1, H, k.shape[0], D)
            Vf = v.unsqueeze(0).unsqueeze(0).expand(1, H, v.shape[0], Dv)
            Qf = Q[i:i+1].unsqueeze(2)  # [1, H, D] -> [1, H, 1, D]
            
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                _ = torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf, is_causal=True)
            torch.cuda.synchronize()
            times.append((time.time() - t0) / iters)
        
        return sum(times) / len(times) * 1000  # ms
        
    elif method == "triton":
        # Triton via selection_attention_triton wrapper
        from nsa.kernels.triton_sel_kernel import selection_attention_triton
        
        # Build NSA-compatible tensors: [B,S,G,h,D] and ranges [B,S,G,n,2]
        B, S, G = N, 1, 1
        Q_nsa = Q.view(N, 1, 1, H, D)  # [N,1,1,H,D]
        if Kpool is not None and Vpool is not None and spans is not None:
            # Use shared pool per row and build multi-range selection
            K_nsa = Kpool.unsqueeze(0).unsqueeze(0).expand(N, 1, -1, -1)  # [N,1,S_pool,D]
            V_nsa = Vpool.unsqueeze(0).unsqueeze(0).expand(N, 1, -1, -1)  # [N,1,S_pool,Dv]
            max_n = max(len(row) for row in spans)
            ranges = torch.zeros((N, 1, 1, max_n, 2), device=Q.device, dtype=torch.int64)
            for i, row_spans in enumerate(spans):
                for j, (s, e) in enumerate(row_spans):
                    ranges[i, 0, 0, j, 0] = s
                    ranges[i, 0, 0, j, 1] = e
        else:
            # Fallback: per-row contiguous selection
            K_nsa = torch.stack([k for k in K_rows], dim=0).unsqueeze(1)  # [N,1,L,D]
            V_nsa = torch.stack([v for v in V_rows], dim=0).unsqueeze(1)  # [N,1,L,Dv]
            max_n = 1
            ranges = torch.zeros((N, 1, 1, max_n, 2), device=Q.device, dtype=torch.int64)
            for i, k_row in enumerate(K_rows):
                ranges[i, 0, 0, 0, 0] = 0
                ranges[i, 0, 0, 0, 1] = k_row.shape[0]
        
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = selection_attention_triton(Q_nsa, K_nsa, V_nsa, ranges)
        torch.cuda.synchronize()
        
        return (time.time() - t0) / iters * 1000  # ms
    
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description="M4 Triton Selection Benchmark - Hybrid Version")
    parser.add_argument("--N", type=int, default=256, help="Number of rows")
    parser.add_argument("--H", type=int, default=8, help="Number of heads")
    parser.add_argument("--D", type=int, default=128, help="Head dimension")
    parser.add_argument("--Dv", type=int, default=128, help="Value dimension")
    parser.add_argument("--L_list", type=str, default="64,128,256", help="Selected lengths (comma-separated)")
    parser.add_argument("--dist", type=str, default="many", choices=["few", "many", "mixed"], help="Range distribution")
    parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--csv", action="store_true", help="Output CSV format")
    parser.add_argument("--output", type=str, help="Save results to file")
    parser.add_argument("--enable_triton", action="store_true", help="Set NSA_USE_TRITON_SEL=1 if not already set")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA required for benchmarking")
        return
    
    # Check Triton availability
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        print("Triton not available")
        return
    
    # Enable Triton for selection if requested and not already set
    if args.enable_triton and "NSA_USE_TRITON_SEL" not in os.environ:
        os.environ["NSA_USE_TRITON_SEL"] = "1"
    
    device = torch.device("cuda")
    L_list = [int(x.strip()) for x in args.L_list.split(",")]
    
    results = []
    
    if not args.csv:
        print("="*80)
        print("M4 TRITON SELECTION BENCHMARK (Hybrid)")
        print("="*80)
        print(f"Device: {torch.cuda.get_device_name()}")
        print(f"Triton: {triton_version}")
        print(f"Config: N={args.N}, H={args.H}, D={args.D}, Dv={args.Dv}, dist={args.dist}")
        print()
        print(f"{'L':>6} {'SDPA(ms)':>10} {'Triton(ms)':>12} {'Speedup':>10} {'Status':>10}")
        print("-" * 60)
    else:
        print("method,N,H,D,Dv,L,dist,time_ms,speedup")
    
    for L in L_list:
        try:
            # Build test data
            Q, K_rows, V_rows, spans, Kpool, Vpool = build_test_data(args.N, args.H, args.D, args.Dv, L, args.dist, device)
            
            # Warmup both methods
            for _ in range(args.warmup):
                _ = benchmark_method(Q, K_rows, V_rows, "sdpa", 1)
                _ = benchmark_method(Q, K_rows, V_rows, "triton", 1, Kpool=Kpool, Vpool=Vpool, spans=spans)
            
            # Benchmark
            sdpa_time = benchmark_method(Q, K_rows, V_rows, "sdpa", args.iters)
            triton_time = benchmark_method(Q, K_rows, V_rows, "triton", args.iters, Kpool=Kpool, Vpool=Vpool, spans=spans)
            speedup = sdpa_time / triton_time if triton_time > 0 else 0.0
            
            status = "FASTER" if speedup > 1.0 else "SLOWER"
            
            if args.csv:
                print(f"sdpa,{args.N},{args.H},{args.D},{args.Dv},{L},{args.dist},{sdpa_time:.3f},1.0")
                print(f"triton,{args.N},{args.H},{args.D},{args.Dv},{L},{args.dist},{triton_time:.3f},{speedup:.3f}")
            else:
                print(f"{L:>6} {sdpa_time:>10.3f} {triton_time:>12.3f} {speedup:>10.2f}x {status:>10}")
            
            results.append({
                'L': L, 'sdpa_ms': sdpa_time, 'triton_ms': triton_time, 'speedup': speedup,
                'N': args.N, 'H': args.H, 'D': args.D, 'Dv': args.Dv, 'dist': args.dist
            })
            
        except Exception as e:
            if args.csv:
                print(f"error,{args.N},{args.H},{args.D},{args.Dv},{L},{args.dist},0.0,0.0")
            else:
                print(f"{L:>6} {'ERROR':>10} {'ERROR':>12} {'0.00x':>10} {str(e)[:10]:>10}")
    
    # Analysis (only for non-CSV mode)
    if not args.csv and results:
        print()
        faster_results = [r for r in results if r['speedup'] >= 1.2]
        if faster_results:
            min_L = min(r['L'] for r in faster_results)
            print(f"✅ Triton ≥1.2x faster starting at L={min_L}")
            print(f"📊 Recommendation: sel_triton_min_L = {min_L}")
        else:
            print(f"❌ Triton not consistently faster (≥1.2x)")
            print(f"📊 Recommendation: Keep sel_triton_min_L high (≥1024)")
    
    # Save results if requested
    if args.output and results:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

