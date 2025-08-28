#!/usr/bin/env python3
"""Test forward pass stability at different sequence lengths."""

import os
import sys
import torch
import yaml
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from train_showcase import TinyLM

def test_forward_stability(seq_len: int, config: dict) -> dict:
    """Test forward pass at a specific sequence length."""
    
    print(f"\n{'='*60}")
    print(f"Testing S={seq_len}")
    print(f"{'='*60}")
    
    # Set strict asserts for catching NaNs
    os.environ['NSA_STRICT_ASSERTS'] = '1'
    
    # Initialize model
    model = TinyLM(
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
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    print(f"Device: {device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Create test input
    batch_size = 1
    x = torch.randint(0, 256, (batch_size, seq_len), device=device)
    
    # Test forward pass
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            logits = model(x)
            
        forward_time = time.perf_counter() - t0
        
        # Check for NaNs
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        
        # Compute loss
        targets = torch.randint(0, 256, (batch_size, seq_len), device=device)
        logits_flat = logits.view(-1, 256)
        targets_flat = targets.view(-1)
        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
        
        loss_value = loss.item()
        loss_is_nan = torch.isnan(loss).item()
        loss_is_inf = torch.isinf(loss).item()
        
        print(f"Forward pass time: {forward_time:.3f}s")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
        print(f"Loss: {loss_value:.4f}")
        
        if has_nan or has_inf:
            print(f"❌ FAIL: {'NaN' if has_nan else 'Inf'} detected in logits!")
        elif loss_is_nan or loss_is_inf:
            print(f"❌ FAIL: {'NaN' if loss_is_nan else 'Inf'} loss!")
        else:
            print(f"✅ PASS: Forward pass stable, finite outputs")
            
        return {
            'seq_len': seq_len,
            'success': not (has_nan or has_inf or loss_is_nan or loss_is_inf),
            'forward_time': forward_time,
            'loss': loss_value if not (loss_is_nan or loss_is_inf) else float('nan'),
            'has_nan': has_nan,
            'has_inf': has_inf
        }
        
    except Exception as e:
        print(f"❌ FAIL: Exception during forward pass: {e}")
        return {
            'seq_len': seq_len,
            'success': False,
            'forward_time': None,
            'loss': float('nan'),
            'error': str(e)
        }

def main():
    """Test forward pass stability at multiple sequence lengths."""
    
    print("="*60)
    print("Forward Pass Stability Test")
    print("="*60)
    
    # Load config
    config_path = "configs/m7c_125m_2xa100_production.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nConfig: {config_path}")
    
    # Test sequence lengths (including problematic ones from report)
    test_lengths = [128, 512, 1024, 2048]
    
    results = []
    for seq_len in test_lengths:
        result = test_forward_stability(seq_len, config)
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Seq Len':<10} {'Status':<10} {'Loss':<10} {'Time (s)':<10}")
    print("-"*40)
    
    all_passed = True
    for r in results:
        status = '✅ PASS' if r['success'] else '❌ FAIL'
        loss_str = f"{r['loss']:.4f}" if not (r['loss'] != r['loss']) else 'NaN'
        time_str = f"{r['forward_time']:.3f}" if r.get('forward_time') else 'N/A'
        print(f"{r['seq_len']:<10} {status:<10} {loss_str:<10} {time_str:<10}")
        all_passed = all_passed and r['success']
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Forward pass stable at all sequence lengths")
    else:
        failed = [r['seq_len'] for r in results if not r['success']]
        print(f"❌ TESTS FAILED at sequence lengths: {failed}")
        print("The NaN issue may still be present at these lengths")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())