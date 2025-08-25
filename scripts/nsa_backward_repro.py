#!/usr/bin/env python3
"""
NSA Backward Pass Reproduction Script
Isolates and profiles the backward pass hang at ≥5 layers
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_showcase import TinyLM


def get_memory_stats():
    """Get current GPU memory statistics"""
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "allocated_gb": torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
            "reserved_gb": torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
        }
    return {"allocated_mb": 0, "reserved_mb": 0, "allocated_gb": 0, "reserved_gb": 0}


def save_memory_summary(filepath):
    """Save CUDA memory summary to file"""
    if torch.cuda.is_available():
        with open(filepath, 'w') as f:
            f.write(torch.cuda.memory_summary())


def register_hooks(model, hook_log):
    """Register backward hooks on all layers"""
    def make_hook(name):
        def hook(module, grad_input, grad_output):
            hook_log.append(f"Backward: {name}")
            print(f"  [HOOK] {name} backward fired")
            return None
        return hook
    
    for i, block in enumerate(model.blocks):
        block.register_backward_hook(make_hook(f"Block_{i}"))
        if hasattr(block, 'attn'):
            block.attn.register_backward_hook(make_hook(f"Block_{i}_Attn"))
        if hasattr(block, 'mlp'):
            block.mlp.register_backward_hook(make_hook(f"Block_{i}_MLP"))


def setup_environment(args):
    """Set environment variables based on arguments"""
    env_vars = {}
    
    # Branch forcing
    if args.branch:
        env_vars['NSA_FORCE_BRANCH'] = args.branch
        os.environ['NSA_FORCE_BRANCH'] = args.branch
    
    # Selection backend
    if args.sel == 'masked':
        env_vars['NSA_USE_SEL_MASK'] = '1'
        env_vars['NSA_USE_SEL_PACK'] = '0'
        env_vars['NSA_USE_TRITON_SEL'] = '0'
        os.environ.update(env_vars)
    elif args.sel == 'packed':
        env_vars['NSA_USE_SEL_PACK'] = '1'
        env_vars['NSA_USE_SEL_MASK'] = '0'
        env_vars['NSA_USE_TRITON_SEL'] = '0'
        os.environ.update(env_vars)
    elif args.sel == 'gather':
        env_vars['NSA_USE_SEL_PACK'] = '0'
        env_vars['NSA_USE_SEL_MASK'] = '0'
        env_vars['NSA_USE_TRITON_SEL'] = '0'
        env_vars['NSA_FORCE_PARITY'] = '1'
        os.environ.update(env_vars)
    
    # Compressed backend
    if args.cmp == 'masked':
        env_vars['NSA_USE_CMP_MASK'] = '1'
        os.environ['NSA_USE_CMP_MASK'] = '1'
    elif args.cmp == 'parity':
        env_vars['NSA_USE_CMP_MASK'] = '0'
        env_vars['NSA_FORCE_PARITY'] = '1'
        os.environ.update(env_vars)
    
    # Sliding backend
    if args.win == 'masked':
        env_vars['NSA_USE_WIN_MASK'] = '1'
        os.environ['NSA_USE_WIN_MASK'] = '1'
    elif args.win == 'parity':
        env_vars['NSA_USE_WIN_MASK'] = '0'
        env_vars['NSA_FORCE_PARITY'] = '1'
        os.environ.update(env_vars)
    
    # CUDA blocking
    if args.blocking:
        env_vars['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    return env_vars


def run_test(args):
    """Run the backward pass test"""
    
    # Setup environment
    env_vars = setup_environment(args)
    
    # Create output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = args.tag or f"{args.branch or 'all'}_{args.layers}L_{args.seq_len}S"
        out_dir = Path(f"artifacts/nsa_backward/{timestamp}_{tag}")
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # Log environment
    env_log = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "env_vars": env_vars,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
    }
    
    with open(out_dir / "env.json", 'w') as f:
        json.dump(env_log, f, indent=2)
    
    print(f"Output directory: {out_dir}")
    print(f"Environment variables: {env_vars}")
    
    # Set anomaly detection
    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("Anomaly detection enabled")
    
    # Model configuration (matching train_showcase.yaml dimensions)
    if args.model_size == 'small':
        model_config = {
            'vocab': 256,
            'dim': 128,
            'n_layers': args.layers,
            'n_heads': 8,
            'n_kv_groups': 2,
            'd_k': 16,
            'd_v': 16,
            'l': 16,
            'd': 8,
            'l_sel': 32,
            'n_sel': 8,
            'w': 64,
        }
    else:  # large
        model_config = {
            'vocab': 256,
            'dim': 768,
            'n_layers': args.layers,
            'n_heads': 12,
            'n_kv_groups': 2,
            'd_k': 64,
            'd_v': 64,
            'l': 32,
            'd': 16,
            'l_sel': 64,
            'n_sel': 16,
            'w': 512,
        }
    
    print(f"\nModel config: {args.layers} layers, dim={model_config['dim']}, seq_len={args.seq_len}")
    
    # Create model
    print("Creating model...")
    model = TinyLM(**model_config, grad_checkpointing=False).cuda()
    print(f"Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Register hooks if requested
    hook_log = []
    if args.hooks:
        register_hooks(model, hook_log)
        print("Backward hooks registered")
    
    # Create input
    x = torch.randint(0, 256, (args.batch_size, args.seq_len)).cuda()
    print(f"Input shape: {x.shape}")
    
    # Memory before forward
    mem_initial = get_memory_stats()
    print(f"\nInitial memory: {mem_initial['allocated_mb']:.1f} MB allocated, {mem_initial['reserved_mb']:.1f} MB reserved")
    
    # Forward pass
    print("\nRunning forward pass...")
    t0 = time.time()
    
    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True
        ) as prof:
            with torch.profiler.record_function("forward"):
                out = model(x)
                loss = out.mean()
            
            mem_after_forward = get_memory_stats()
            print(f"After forward: {mem_after_forward['allocated_mb']:.1f} MB allocated, {mem_after_forward['reserved_mb']:.1f} MB reserved")
            save_memory_summary(out_dir / "memory_after_forward.txt")
            
            with torch.profiler.record_function("backward"):
                print("\nRunning backward pass...")
                t_backward = time.time()
                loss.backward()
                backward_time = time.time() - t_backward
        
        # Save profiler results
        prof.export_chrome_trace(str(out_dir / "trace.json"))
        with open(out_dir / "profiler_table.txt", 'w') as f:
            f.write(str(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20)))
        print(f"Profiler results saved to {out_dir}")
    else:
        out = model(x)
        loss = out.mean()
        forward_time = time.time() - t0
        print(f"Forward pass complete in {forward_time:.2f}s")
        print(f"Output shape: {out.shape}, Loss: {loss.item():.4f}")
        
        mem_after_forward = get_memory_stats()
        print(f"After forward: {mem_after_forward['allocated_mb']:.1f} MB allocated, {mem_after_forward['reserved_mb']:.1f} MB reserved")
        
        with open(out_dir / "pre_backward_mem.json", 'w') as f:
            json.dump(mem_after_forward, f, indent=2)
        
        save_memory_summary(out_dir / "memory_before_backward.txt")
        
        # Backward pass
        print("\nRunning backward pass...")
        t_backward = time.time()
        
        try:
            loss.backward()
            backward_time = time.time() - t_backward
            print(f"✓ Backward pass complete in {backward_time:.2f}s")
            
            mem_after_backward = get_memory_stats()
            print(f"After backward: {mem_after_backward['allocated_mb']:.1f} MB allocated, {mem_after_backward['reserved_mb']:.1f} MB reserved")
            
            with open(out_dir / "post_backward_mem.json", 'w') as f:
                json.dump(mem_after_backward, f, indent=2)
            
            save_memory_summary(out_dir / "memory_after_backward.txt")
            
            result = "PASS"
        except Exception as e:
            print(f"✗ Backward pass failed: {e}")
            traceback.print_exc()
            result = "FAIL"
            backward_time = time.time() - t_backward
    
    # Save hook log if used
    if args.hooks:
        with open(out_dir / "hook_log.txt", 'w') as f:
            f.write("\n".join(hook_log))
        print(f"Hook log saved ({len(hook_log)} entries)")
    
    # Save final result
    result_data = {
        "result": result if 'result' in locals() else "UNKNOWN",
        "forward_time": forward_time if 'forward_time' in locals() else None,
        "backward_time": backward_time if 'backward_time' in locals() else None,
        "mem_initial": mem_initial,
        "mem_after_forward": mem_after_forward,
        "mem_after_backward": mem_after_backward if 'mem_after_backward' in locals() else None,
        "hook_count": len(hook_log) if args.hooks else None,
    }
    
    with open(out_dir / "result.json", 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Result: {result_data['result']}")
    print(f"Output saved to: {out_dir}")
    print(f"{'='*60}")
    
    return result_data


def main():
    parser = argparse.ArgumentParser(description='NSA Backward Pass Reproduction')
    
    # Model configuration
    parser.add_argument('--layers', type=int, default=5, help='Number of layers')
    parser.add_argument('--seq-len', type=int, default=128, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--model-size', choices=['small', 'large'], default='small',
                       help='Model size (small=128d, large=768d)')
    
    # Branch/backend selection
    parser.add_argument('--branch', choices=['win', 'sel', 'cmp'], 
                       help='Force single branch (win/sel/cmp)')
    parser.add_argument('--sel', choices=['masked', 'packed', 'gather'],
                       help='Selection backend')
    parser.add_argument('--cmp', choices=['masked', 'parity'],
                       help='Compressed backend')
    parser.add_argument('--win', choices=['masked', 'parity'],
                       help='Sliding window backend')
    
    # Debugging options
    parser.add_argument('--anomaly', action='store_true',
                       help='Enable autograd anomaly detection')
    parser.add_argument('--blocking', action='store_true',
                       help='Enable CUDA_LAUNCH_BLOCKING')
    parser.add_argument('--profile', action='store_true',
                       help='Enable PyTorch profiler')
    parser.add_argument('--hooks', action='store_true',
                       help='Register backward hooks')
    
    # Output
    parser.add_argument('--out-dir', type=str, help='Output directory')
    parser.add_argument('--tag', type=str, help='Tag for output directory')
    
    args = parser.parse_args()
    
    try:
        result = run_test(args)
        sys.exit(0 if result['result'] == 'PASS' else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()