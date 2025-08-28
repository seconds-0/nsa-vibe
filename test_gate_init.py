#!/usr/bin/env python3
"""Test script to verify gate.fc2 weights are properly initialized."""

import torch
import torch.nn as nn
import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from nsa.model.llama_block_nsa import LlamaBlockNSA

def test_gate_initialization():
    """Verify gate.fc2 weights are not zero-initialized."""
    
    print("=" * 60)
    print("Testing Gate Initialization Fix")
    print("=" * 60)
    
    # Load config
    config_path = "configs/m7c_125m_2xa100_production.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nLoading model with config: {config_path}")
    
    # Import TinyLM from train_showcase to match production setup
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    from train_showcase import TinyLM
    
    # Initialize model
    model = TinyLM(
        vocab=256,  # byte tokenizer
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
        grad_checkpointing=False  # Not testing gradient checkpointing
    )
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Check gate weights
    print("\n" + "=" * 60)
    print("Gate Weight Statistics:")
    print("=" * 60)
    
    all_gates_ok = True
    zero_gates = []
    
    for name, param in model.named_parameters():
        if 'gate.fc2' in name:
            mean = param.mean().item()
            std = param.std().item()
            min_val = param.min().item()
            max_val = param.max().item()
            
            # Bias being zero is intentional, weights should be non-zero
            is_bias = 'bias' in name
            is_zero = (std == 0.0 and mean == 0.0)
            
            if is_bias:
                status = "✅ Zero bias (intentional)" if is_zero else "⚠️ Non-zero bias"
            else:
                status = "❌ ALL ZEROS!" if is_zero else "✅ Properly initialized"
            
            print(f"\n{name}:")
            print(f"  Shape: {list(param.shape)}")
            print(f"  Mean:  {mean:.6f}")
            print(f"  Std:   {std:.6f}")
            print(f"  Min:   {min_val:.6f}")
            print(f"  Max:   {max_val:.6f}")
            print(f"  Status: {status}")
            
            # Only mark as failed if weights (not bias) are zero
            if is_zero and not is_bias:
                all_gates_ok = False
                zero_gates.append(name)
    
    # Test gate outputs
    print("\n" + "=" * 60)
    print("Testing Gate Outputs:")
    print("=" * 60)
    
    # Create dummy input
    B, S = 2, 128
    x = torch.randint(0, 256, (B, S))
    
    with torch.no_grad():
        # Get attention outputs (this will trigger gate computation)
        model.eval()
        
        # Hook to capture gate outputs
        gate_outputs = []
        def hook_fn(module, input, output):
            gate_outputs.append(output.detach())
            return output
        
        # Register hooks on gate modules
        hooks = []
        for module in model.modules():
            if hasattr(module, 'gate'):
                hooks.append(module.gate.register_forward_hook(hook_fn))
        
        # Forward pass
        _ = model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Analyze gate outputs
    print(f"\nCaptured {len(gate_outputs)} gate outputs")
    
    uniform_count = 0
    for i, gates in enumerate(gate_outputs[:3]):  # Check first 3 layers
        gates_flat = gates.reshape(-1, 3)
        
        # Check if all gates are uniform (0.333, 0.333, 0.333)
        is_uniform = torch.allclose(gates_flat, torch.ones_like(gates_flat) / 3, atol=1e-4)
        
        # Sample statistics
        mean_weights = gates_flat.mean(dim=0)
        std_weights = gates_flat.std(dim=0)
        
        print(f"\nLayer {i} gate outputs:")
        print(f"  Mean weights: [{mean_weights[0]:.4f}, {mean_weights[1]:.4f}, {mean_weights[2]:.4f}]")
        print(f"  Std weights:  [{std_weights[0]:.4f}, {std_weights[1]:.4f}, {std_weights[2]:.4f}]")
        print(f"  Uniform: {'❌ YES (all 0.333!)' if is_uniform else '✅ NO (varied)'}")
        
        if is_uniform:
            uniform_count += 1
    
    # Final verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT:")
    print("=" * 60)
    
    if all_gates_ok and uniform_count == 0:
        print("✅ PASS: Gate initialization fix is working correctly!")
        print("  - All gate.fc2 weights are properly initialized (non-zero)")
        print("  - Gate outputs show varied branch weights")
        return True
    else:
        print("❌ FAIL: Gate initialization issues detected!")
        if not all_gates_ok:
            print(f"  - Found {len(zero_gates)} zero-initialized gates:")
            for name in zero_gates:
                print(f"    - {name}")
        if uniform_count > 0:
            print(f"  - {uniform_count} layers produce uniform gate outputs (0.333 each)")
        return False

if __name__ == "__main__":
    success = test_gate_initialization()
    sys.exit(0 if success else 1)