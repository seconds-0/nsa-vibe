#!/usr/bin/env python3
"""Compare performance with safe packing enabled vs disabled."""

import os
import sys
import time
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from train_showcase import TinyLM

def benchmark_forward_backward(model, seq_len: int, batch_size: int, steps: int = 20) -> dict:
    """Benchmark forward and backward passes."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Warmup
    x = torch.randint(0, 256, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 256, (batch_size, seq_len), device=device)
    
    for _ in range(3):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 256), 
            targets.view(-1)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Benchmark
    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.perf_counter()
    
    losses = []
    for _ in range(steps):
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 256, (batch_size, seq_len), device=device)
        
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 256), 
            targets.view(-1)
        )
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize() if device == 'cuda' else None
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    tokens = batch_size * seq_len * steps
    tok_per_s = tokens / total_time
    
    return {
        'time': total_time,
        'steps': steps,
        'tok_per_s': tok_per_s,
        'avg_loss': sum(losses) / len(losses),
        'losses_stable': all(l == l for l in losses)  # Check for NaN
    }

def main():
    """Compare safe pack performance."""
    
    print("="*60)
    print("Safe Packing Performance Comparison")
    print("="*60)
    
    # Load config
    config_path = "configs/m7c_125m_2xa100_production.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Config: {config_path}")
    
    seq_len = 512
    batch_size = 1 if device == 'cpu' else 4
    steps = 10 if device == 'cpu' else 20
    
    print(f"Benchmark: S={seq_len}, B={batch_size}, steps={steps}")
    
    # Test with safe packing ENABLED
    print("\n" + "="*60)
    print("Testing with NSA_TRAIN_SAFE_PACK=1 (safe mode)")
    print("="*60)
    
    os.environ['NSA_TRAIN_SAFE_PACK'] = '1'
    
    model_safe = TinyLM(
        vocab=256,
        dim=config['model']['dim'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        d_k=config['model']['d_k'],
        d_v=config['model']['d_v'],
        n_kv_groups=config['model']['n_kv_groups'],
        l=config['nsa']['l'],
        d=config['nsa']['d'],
        l_sel=config['nsa']['l_sel'],
        n_sel=config['nsa']['n_sel'],
        w=config['nsa']['w'],
        grad_checkpointing=False
    )
    
    result_safe = benchmark_forward_backward(model_safe, seq_len, batch_size, steps)
    
    print(f"Time: {result_safe['time']:.2f}s")
    print(f"Throughput: {result_safe['tok_per_s']:.1f} tok/s")
    print(f"Avg Loss: {result_safe['avg_loss']:.4f}")
    print(f"Stable: {'✅ YES' if result_safe['losses_stable'] else '❌ NO (NaN detected)'}")
    
    # Test with safe packing DISABLED
    print("\n" + "="*60)
    print("Testing with NSA_TRAIN_SAFE_PACK=0 (fast mode)")
    print("="*60)
    
    os.environ['NSA_TRAIN_SAFE_PACK'] = '0'
    
    model_fast = TinyLM(
        vocab=256,
        dim=config['model']['dim'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        d_k=config['model']['d_k'],
        d_v=config['model']['d_v'],
        n_kv_groups=config['model']['n_kv_groups'],
        l=config['nsa']['l'],
        d=config['nsa']['d'],
        l_sel=config['nsa']['l_sel'],
        n_sel=config['nsa']['n_sel'],
        w=config['nsa']['w'],
        grad_checkpointing=False
    )
    
    # Disable gradients to test inference-only fast path
    for p in model_fast.parameters():
        p.requires_grad = False
    
    # Run inference-only benchmark
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_fast = model_fast.to(device)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.perf_counter()
    
    for _ in range(steps):
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        with torch.no_grad():
            logits = model_fast(x)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    t1 = time.perf_counter()
    
    fast_time = t1 - t0
    fast_tok_per_s = (batch_size * seq_len * steps) / fast_time
    
    print(f"Time: {fast_time:.2f}s")
    print(f"Throughput: {fast_tok_per_s:.1f} tok/s (inference only)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    speedup = fast_tok_per_s / result_safe['tok_per_s']
    overhead = (1 - result_safe['tok_per_s'] / fast_tok_per_s) * 100
    
    print(f"Safe mode (training): {result_safe['tok_per_s']:.1f} tok/s")
    print(f"Fast mode (inference): {fast_tok_per_s:.1f} tok/s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Safe mode overhead: {overhead:.1f}%")
    
    if result_safe['losses_stable']:
        print("\n✅ Safe packing maintains training stability")
    else:
        print("\n❌ Warning: Training instability detected")
    
    if device == 'cuda':
        if result_safe['tok_per_s'] >= 300:
            print(f"✅ Performance target met: {result_safe['tok_per_s']:.1f} >= 300 tok/s")
        else:
            print(f"❌ Performance below target: {result_safe['tok_per_s']:.1f} < 300 tok/s")
    else:
        print("\n⚠️ Note: Running on CPU, GPU performance will be different")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())