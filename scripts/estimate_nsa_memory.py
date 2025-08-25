#!/usr/bin/env python3
"""
NSA Memory Estimator
Estimates memory requirements for NSA model configurations
"""

import argparse
import math

def estimate_nsa_memory(dim, n_layers, n_heads, seq_len, batch_size=1, dtype_bytes=2):
    """
    Estimate NSA memory requirements based on quadratic scaling observation
    
    Args:
        dim: Model dimension
        n_layers: Number of layers
        n_heads: Number of attention heads
        seq_len: Sequence length
        batch_size: Batch size
        dtype_bytes: Bytes per element (2 for bf16, 4 for fp32)
    
    Returns:
        dict: Memory estimates in MB
    """
    
    # Base model parameters (approximate)
    params_per_layer = (
        4 * dim * dim +  # Q, K, V, O projections
        4 * dim * dim    # FFN
    )
    total_params = n_layers * params_per_layer + dim * 256  # embeddings
    param_memory = total_params * dtype_bytes / (1024 * 1024)
    
    # Activation memory (quadratic in sequence length based on observations)
    # From tests: dim=768, 5 layers, seq_len=128 -> 483 MB
    #            dim=768, 5 layers, seq_len=512 -> 5,165 MB
    # This suggests: memory ≈ base + k * seq_len^2
    
    # Calculate scaling factor from observations
    if dim >= 768:
        # Large model scaling (quadratic)
        base_memory = 100  # MB
        k_factor = 0.005 * dim / 768 * n_layers / 5  # Empirical scaling
        activation_memory = base_memory + k_factor * (seq_len ** 2) * batch_size
    else:
        # Small model scaling (more linear)
        base_memory = 50
        k_factor = 0.001 * dim / 128 * n_layers / 5
        activation_memory = base_memory + k_factor * seq_len * math.log(seq_len) * batch_size
    
    # Attention score memory (dominant factor)
    # Each layer stores attention scores: batch * heads * seq_len * seq_len
    attention_memory = (
        batch_size * n_layers * n_heads * seq_len * seq_len * dtype_bytes 
        / (1024 * 1024)
    )
    
    # Gradient memory (roughly 2x activations during backward)
    gradient_memory = 2 * (activation_memory + attention_memory)
    
    # Total estimate
    forward_memory = param_memory + activation_memory + attention_memory
    backward_memory = forward_memory + gradient_memory
    
    return {
        "param_memory_mb": param_memory,
        "activation_memory_mb": activation_memory,
        "attention_memory_mb": attention_memory,
        "forward_total_mb": forward_memory,
        "gradient_memory_mb": gradient_memory,
        "backward_total_mb": backward_memory,
        "forward_total_gb": forward_memory / 1024,
        "backward_total_gb": backward_memory / 1024,
    }

def format_memory_report(config, estimates):
    """Format memory estimates as readable report"""
    
    report = f"""
NSA Memory Estimate
==================
Configuration:
- Model dim: {config['dim']}
- Layers: {config['n_layers']}
- Heads: {config['n_heads']}
- Sequence length: {config['seq_len']}
- Batch size: {config['batch_size']}
- Dtype: {'bf16' if config['dtype_bytes'] == 2 else 'fp32'}

Memory Breakdown:
-----------------
Parameters:          {estimates['param_memory_mb']:8.1f} MB
Activations:         {estimates['activation_memory_mb']:8.1f} MB
Attention Scores:    {estimates['attention_memory_mb']:8.1f} MB
                     {'─' * 20}
Forward Total:       {estimates['forward_total_mb']:8.1f} MB ({estimates['forward_total_gb']:.2f} GB)

Gradients:           {estimates['gradient_memory_mb']:8.1f} MB
                     {'─' * 20}
Backward Total:      {estimates['backward_total_mb']:8.1f} MB ({estimates['backward_total_gb']:.2f} GB)

"""
    
    # Add warnings based on GPU memory
    if estimates['backward_total_gb'] > 80:
        report += "⚠️  WARNING: Exceeds A100 80GB memory! Will likely hang/OOM.\n"
    elif estimates['backward_total_gb'] > 40:
        report += "⚠️  WARNING: Exceeds A100 40GB memory. Requires 80GB GPU.\n"
    elif estimates['backward_total_gb'] > 24:
        report += "⚠️  WARNING: Exceeds RTX 4090 24GB memory. Requires larger GPU.\n"
    
    if config['seq_len'] >= 1024 and config['dim'] >= 768:
        report += "⚠️  WARNING: Large dim + long sequence detected. Expect quadratic memory growth!\n"
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Estimate NSA memory requirements')
    parser.add_argument('--dim', type=int, default=768, help='Model dimension')
    parser.add_argument('--layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--seq-len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--dtype', choices=['bf16', 'fp32'], default='bf16',
                       help='Data type')
    parser.add_argument('--compare-configs', action='store_true',
                       help='Compare common configurations')
    
    args = parser.parse_args()
    
    if args.compare_configs:
        # Compare common configurations
        configs = [
            {"name": "Small (5L, dim=128)", "dim": 128, "n_layers": 5, "n_heads": 8, "seq_len": 128},
            {"name": "Small (12L, dim=128)", "dim": 128, "n_layers": 12, "n_heads": 8, "seq_len": 128},
            {"name": "Large (5L, dim=768, S=128)", "dim": 768, "n_layers": 5, "n_heads": 12, "seq_len": 128},
            {"name": "Large (5L, dim=768, S=512)", "dim": 768, "n_layers": 5, "n_heads": 12, "seq_len": 512},
            {"name": "Large (5L, dim=768, S=1024)", "dim": 768, "n_layers": 5, "n_heads": 12, "seq_len": 1024},
            {"name": "Large (5L, dim=768, S=2048)", "dim": 768, "n_layers": 5, "n_heads": 12, "seq_len": 2048},
            {"name": "Production (12L, dim=768, S=2048)", "dim": 768, "n_layers": 12, "n_heads": 12, "seq_len": 2048},
        ]
        
        print("\nConfiguration Comparison")
        print("=" * 80)
        print(f"{'Config':<35} {'Forward':>12} {'Backward':>12} {'Status':<15}")
        print("-" * 80)
        
        for cfg in configs:
            est = estimate_nsa_memory(
                cfg['dim'], cfg['n_layers'], cfg['n_heads'], cfg['seq_len'],
                batch_size=1, dtype_bytes=2
            )
            
            status = "✅ OK"
            if est['backward_total_gb'] > 80:
                status = "❌ OOM (>80GB)"
            elif est['backward_total_gb'] > 40:
                status = "⚠️  80GB only"
            elif est['backward_total_gb'] > 24:
                status = "⚠️  40GB+"
            
            print(f"{cfg['name']:<35} {est['forward_total_gb']:>10.2f} GB {est['backward_total_gb']:>10.2f} GB  {status}")
    
    else:
        # Single configuration estimate
        config = {
            'dim': args.dim,
            'n_layers': args.layers,
            'n_heads': args.heads,
            'seq_len': args.seq_len,
            'batch_size': args.batch_size,
            'dtype_bytes': 2 if args.dtype == 'bf16' else 4
        }
        
        estimates = estimate_nsa_memory(**config)
        report = format_memory_report(config, estimates)
        print(report)

if __name__ == '__main__':
    main()
