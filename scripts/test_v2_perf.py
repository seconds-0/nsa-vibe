#!/usr/bin/env python3
import torch
import os
import time
from nsa.core.selection_scorer import (
    convert_indices_to_ranges_batched,
    convert_indices_to_ranges_batched_v2,
    convert_indices_to_ranges_batched_dispatch
)
from nsa.core.block_index import build_block_meta

print(f"V2 enabled via env: {os.getenv('NSA_SEL_RANGES_V2', '1') == '1'}")

# Test data matching production config
B, S, G, K = 2, 256, 4, 32
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
meta = build_block_meta(seq_len=2048, l=32, d=16, l_sel=64, n_sel=16, w=512)
indices = torch.randint(-1, 100, (B, S, G, K), device=device, dtype=torch.int32)

if device == "cuda":
    # Warmup
    for _ in range(5):
        _ = convert_indices_to_ranges_batched(indices, meta, S)
        _ = convert_indices_to_ranges_batched_v2(indices, meta, S)
    
    # Time v1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = convert_indices_to_ranges_batched(indices, meta, S)
    torch.cuda.synchronize()
    v1_time = time.perf_counter() - t0
    
    # Time v2
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = convert_indices_to_ranges_batched_v2(indices, meta, S)
    torch.cuda.synchronize()
    v2_time = time.perf_counter() - t0
    
    print(f"V1 time: {v1_time*1000:.2f}ms for 100 iterations ({v1_time*10:.3f}ms per iteration)")
    print(f"V2 time: {v2_time*1000:.2f}ms for 100 iterations ({v2_time*10:.3f}ms per iteration)")
    print(f"Speedup: {v1_time/v2_time:.1f}x")
    
    # Test dispatch uses v2
    os.environ['NSA_SEL_RANGES_V2'] = '1'
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = convert_indices_to_ranges_batched_dispatch(indices, meta, S)
    torch.cuda.synchronize()
    dispatch_time = time.perf_counter() - t0
    print(f"Dispatch time: {dispatch_time*1000:.2f}ms (should match v2)")
    print(f"Dispatch uses v2: {abs(dispatch_time - v2_time) < 0.001}")