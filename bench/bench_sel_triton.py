#!/usr/bin/env python3
"""
Triton Selection Attention Benchmark (M4)

Benchmarks Triton vs Packed SDPA for selection attention across various configurations
to determine optimal thresholds and validate numerical parity.

Usage:
    NSA_USE_TRITON_SEL=1 python bench/bench_sel_triton.py --N 1024 --H 8 --D 128 --Dv 128 --L_list 64,128,256,512 --dist many --iters 50
    NSA_USE_TRITON_SEL=1 python bench/bench_sel_triton.py --decode 1 --H 4 --D 64 --Dv 64 --L_list 128,256,512 --dist few
"""

import argparse
import os
import time
from typing import List, Tuple
import numpy as np

import torch

# Ensure CUDA settings for precise benchmarking
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def generate_selection_ranges(N: int, H: int, L: int, distribution: str, n_ranges: int = None) -> torch.Tensor:
    """
    Generate selection ranges for benchmarking.
    
    Args:
        N: Number of rows
        H: Number of heads (groups)
        L: Target total selected length per row
        distribution: 'few', 'many', or 'mixed'
        n_ranges: Number of ranges (if None, inferred from distribution)
    
    Returns:
        ranges: [N, H, n_ranges, 2] tensor with start/end pairs
    """
    device = torch.device('cuda')
    
    if distribution == 'few':
        n = n_ranges or 1
        ranges = torch.zeros((N, H, n, 2), device=device, dtype=torch.int64)
        # Single long range per row
        ranges[:, :, 0, 0] = 0
        ranges[:, :, 0, 1] = L
        # Pad other ranges with 0,0 (empty)
        
    elif distribution == 'many':
        n = n_ranges or 8
        ranges = torch.zeros((N, H, n, 2), device=device, dtype=torch.int64)
        chunk_size = L // n
        remainder = L % n
        
        start = 0
        for i in range(n):
            size = chunk_size + (1 if i < remainder else 0)
            ranges[:, :, i, 0] = start
            ranges[:, :, i, 1] = start + size
            start += size + 10  # Add gaps between chunks
            
    elif distribution == 'mixed':
        n = n_ranges or np.random.choice([2, 4, 8])
        ranges = torch.zeros((N, H, n, 2), device=device, dtype=torch.int64)
        # Random splits
        for b in range(N):
            for h in range(H):
                splits = sorted(np.random.choice(range(0, L * 2, 10), n, replace=False))
                total_assigned = 0
                for i, start in enumerate(splits):
                    if total_assigned >= L:
                        break
                    remaining = L - total_assigned
                    size = min(remaining, np.random.randint(1, max(2, remaining // (n - i) + 1)))
                    ranges[b, h, i, 0] = start
                    ranges[b, h, i, 1] = start + size
                    total_assigned += size
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    return ranges


def bench_selection_attention(
    N: int, H: int, D: int, Dv: int, L: int, 
    distribution: str, iters: int = 50, warmup: int = 5
) -> Tuple[float, float, float]:
    """
    Benchmark Triton vs Packed SDPA for selection attention.
    
    Returns:
        (triton_time_ms, packed_time_ms, speedup)
    """
    device = torch.device('cuda')
    torch.manual_seed(42)
    
    # Generate test data
    Q = torch.randn(N, 1, 1, H, D, device=device, dtype=torch.bfloat16)  # [N,S=1,G=1,h,D]
    
    # Create large K/V to select from
    seq_len = L * 4  # Ensure we have enough tokens to select from
    K = torch.randn(N, 1, seq_len, D, device=device, dtype=torch.bfloat16)  # [N,G=1,S_kv,D]
    V = torch.randn(N, 1, seq_len, Dv, device=device, dtype=torch.bfloat16)  # [N,G=1,S_kv,Dv]
    
    # Generate selection ranges
    ranges = generate_selection_ranges(N, H, L, distribution)  # [N,H,n,2]
    ranges = ranges.unsqueeze(1).unsqueeze(2)  # [N,S=1,G=1,n,2]
    ranges = ranges.expand(N, 1, 1, ranges.shape[-2], 2)
    
    # Clamp ranges to valid sequence length
    ranges = torch.clamp(ranges, 0, seq_len)
    
    # Import attention functions
    from nsa.core.attention_kernels import grouped_selection_attention_packed
    from nsa.kernels.triton_sel_kernel import selection_attention_triton
    
    # Warmup
    for _ in range(warmup):
        _ = grouped_selection_attention_packed(Q, K, V, ranges)
        _ = selection_attention_triton(Q, K, V, ranges)
    
    torch.cuda.synchronize()
    
    # Benchmark Packed SDPA
    start_time = time.time()
    for _ in range(iters):
        out_packed = grouped_selection_attention_packed(Q, K, V, ranges)
    torch.cuda.synchronize()
    packed_time = (time.time() - start_time) / iters * 1000  # ms
    
    # Benchmark Triton
    start_time = time.time()
    for _ in range(iters):
        out_triton = selection_attention_triton(Q, K, V, ranges)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / iters * 1000  # ms
    
    # Verify numerical agreement
    mae = (out_packed - out_triton).abs().mean().item()
    if mae > 1e-3:
        print(f"WARNING: High MAE {mae:.6f} between Triton and Packed SDPA")
    
    speedup = packed_time / triton_time if triton_time > 0 else 0.0
    
    return triton_time, packed_time, speedup


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton Selection Attention")
    parser.add_argument("--N", type=int, default=64, help="Number of rows")
    parser.add_argument("--H", type=int, default=4, help="Number of heads")
    parser.add_argument("--D", type=int, default=64, help="Head dimension")
    parser.add_argument("--Dv", type=int, default=64, help="Value dimension")
    parser.add_argument("--L_list", type=str, default="64,128,256", help="Comma-separated list of selected lengths")
    parser.add_argument("--dist", type=str, default="mixed", choices=["few", "many", "mixed"], 
                       help="Selection range distribution")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--decode", type=int, default=0, help="Use decode-representative shapes")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA required for benchmarking")
        return
    
    # Check if Triton is available
    try:
        import triton
        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("Triton not available")
        return
    
    # Set environment for Triton usage
    os.environ["NSA_USE_TRITON_SEL"] = "1"
    
    L_values = [int(x.strip()) for x in args.L_list.split(",")]
    
    print("="*80)
    print("TRITON SELECTION ATTENTION BENCHMARK")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Rows: {args.N}, Heads: {args.H}, D: {args.D}, Dv: {args.Dv}")
    print(f"Distribution: {args.dist}, Iterations: {args.iters}")
    print(f"Selected lengths: {L_values}")
    print()
    
    # Results storage
    results = []
    
    # Benchmark header
    print(f"{'L':>6} {'Triton(ms)':>12} {'Packed(ms)':>12} {'Speedup':>10} {'GB/s':>8} {'Status':>10}")
    print("-" * 70)
    
    for L in L_values:
        try:
            triton_time, packed_time, speedup = bench_selection_attention(
                args.N, args.H, args.D, args.Dv, L, args.dist, args.iters, args.warmup
            )
            
            # Estimate memory bandwidth (rough approximation)
            bytes_per_op = args.N * L * (args.D + args.Dv) * 2  # K + V in bf16
            gb_per_s = (bytes_per_op / (triton_time / 1000)) / 1e9 if triton_time > 0 else 0
            
            status = "FASTER" if speedup > 1.0 else "SLOWER"
            
            print(f"{L:>6} {triton_time:>12.3f} {packed_time:>12.3f} {speedup:>10.2f}x {gb_per_s:>8.1f} {status:>10}")
            
            results.append({
                'L': L,
                'triton_ms': triton_time,
                'packed_ms': packed_time,
                'speedup': speedup,
                'gb_per_s': gb_per_s,
                'N': args.N,
                'H': args.H,
                'D': args.D,
                'Dv': args.Dv,
                'dist': args.dist
            })
            
        except Exception as e:
            print(f"{L:>6} {'ERROR':>12} {'ERROR':>12} {'0.00x':>10} {'0.0':>8} {str(e)[:10]:>10}")
    
    print()
    
    # Analyze results
    speedup_threshold = 1.2
    triton_wins = [r for r in results if r['speedup'] >= speedup_threshold]
    
    if triton_wins:
        min_L_triton = min(r['L'] for r in triton_wins)
        print(f"✅ Triton achieves ≥{speedup_threshold}x speedup starting at L={min_L_triton}")
        print(f"📊 Recommendation: sel_triton_min_L = {min_L_triton}")
    else:
        print(f"❌ Triton does not achieve ≥{speedup_threshold}x speedup for any tested L")
        print(f"📊 Recommendation: Keep sel_triton_min_L high (e.g., 1024)")
    
    # Save results if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()