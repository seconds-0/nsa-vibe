#!/usr/bin/env python3
"""
Quick script to check what GPUs are available on Prime Intellect right now.
"""

import os
import sys
from prime_gpu_bench import PrimeIntellectBenchmark

def main():
    try:
        benchmark = PrimeIntellectBenchmark()
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # GPU types from Prime Intellect API documentation
    gpu_types = [
        'T4',
        'L4', 
        'L40',
        'RTX_3090',
        'RTX_4090',
        'A10',
        'A30',
        'A40',
        'A100_40GB',
        'A100_80GB',
        'H100_80GB',
        'H100_NVL'
    ]
    
    print("Checking Prime Intellect GPU availability...")
    print("=" * 60)
    
    available = []
    
    for gpu in gpu_types:
        try:
            option = benchmark.find_cheapest_gpu(gpu)
            price = option['prices']['hourly']
            provider = option['provider']
            available.append((gpu, price, provider))
            print(f"✅ {gpu}: ${price:.2f}/hr ({provider})")
        except RuntimeError:
            print(f"❌ {gpu}: Not available")
    
    if available:
        print("\n" + "=" * 60)
        print("AVAILABLE GPUS (sorted by price):")
        available.sort(key=lambda x: x[1])
        for gpu, price, provider in available:
            print(f"  {gpu}: ${price:.2f}/hr ({provider})")
        
        cheapest = available[0]
        print(f"\n💡 Try: python bench/prime_gpu_bench.py --gpu-type {cheapest[0]}")
    else:
        print("\n❌ No GPUs currently available")
        print("Try again in a few minutes - availability changes frequently")

if __name__ == "__main__":
    sys.exit(main())