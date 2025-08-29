#!/usr/bin/env python3
"""
Validate that gradient tracing infrastructure is working correctly.
This can run on CPU to verify the setup before GPU testing.
"""

import os
import sys
import torch
import torch.nn as nn

# Set tracing flags
os.environ["NSA_TRACE_GRADS"] = "1"
os.environ["NSA_TRACE_MODULE_BWD"] = "1"

# Import after setting env vars
sys.path.append("/Users/alexanderhuth/Code/nsa-vibe")
from scripts.train_showcase import TinyLM


def test_gradient_tracing():
    """Test that gradient hooks are properly registered and fire"""

    print("=" * 60)
    print("NSA Gradient Tracing Validation")
    print("=" * 60)

    # Create small model
    model = TinyLM(
        vocab=256,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_groups=2,
        d_k=32,
        d_v=32,
        l=16,
        d=8,
        l_sel=32,
        n_sel=8,
        w=64,
        grad_checkpointing=False,
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Track gradients
    grad_seen = {}
    mod_bwd_seen = {}

    # Register gradient hooks
    def register_grad_hooks():
        for name, param in model.named_parameters():
            if param.requires_grad:

                def make_hook(n):
                    def hook(grad):
                        grad_seen[n] = (grad.shape, torch.isnan(grad).any().item())
                        return grad

                    return hook

                param.register_hook(make_hook(name))
        print(
            f"✅ Registered {len([p for p in model.parameters() if p.requires_grad])} gradient hooks"
        )

    # Register module backward hooks
    def register_module_hooks():
        def bwd_hook(mod, grad_in, grad_out):
            mod_bwd_seen[id(mod)] = mod.__class__.__name__

        for module in model.modules():
            module.register_full_backward_hook(bwd_hook)
        print(f"✅ Registered {len(list(model.modules()))} module backward hooks")

    register_grad_hooks()
    register_module_hooks()

    # Run forward and backward
    print("\n" + "-" * 40)
    print("Running forward pass...")

    batch_size = 2
    seq_len = 128
    x = torch.randint(0, 256, (batch_size, seq_len))

    out = model(x)
    loss = out.mean()

    print(f"✅ Forward complete: output shape {out.shape}, loss {loss.item():.4f}")

    print("\n" + "-" * 40)
    print("Running backward pass...")

    loss.backward()

    # Report results
    print("\n" + "=" * 60)
    print("GRADIENT TRACE RESULTS")
    print("=" * 60)

    all_params = {name for name, p in model.named_parameters() if p.requires_grad}
    missing = all_params - set(grad_seen.keys())

    print(f"\n[GRAD-TRACE] after_backward")
    print(f"  Total parameters: {len(all_params)}")
    print(f"  Gradients arrived: {len(grad_seen)}")
    print(f"  Missing gradients: {len(missing)}")

    if missing:
        print("\n  MISSING PARAMETERS:")
        for name in sorted(missing)[:10]:
            print(f"    - {name}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    else:
        print("\n  ✅ All parameter gradients arrived successfully!")

    # Check for NaN gradients
    nan_grads = [name for name, (shape, has_nan) in grad_seen.items() if has_nan]
    if nan_grads:
        print(f"\n  ⚠️ NaN gradients detected in {len(nan_grads)} parameters:")
        for name in nan_grads[:5]:
            print(f"    - {name}")

    print("\n" + "-" * 40)
    print("MODULE BACKWARD TRACE")
    print("-" * 40)

    module_types = set(mod_bwd_seen.values())
    print(f"  Module types that received backward: {len(module_types)}")
    print(f"  Types: {sorted(module_types)}")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    if len(missing) == 0 and len(grad_seen) > 0:
        print("✅ SUCCESS: Gradient tracing infrastructure is working correctly!")
        print(f"   - All {len(grad_seen)} parameter gradients tracked")
        print(f"   - {len(mod_bwd_seen)} modules received backward pass")
        return True
    else:
        print("❌ FAILURE: Issues detected with gradient tracing")
        if len(grad_seen) == 0:
            print("   - No gradients were tracked (hooks may not be firing)")
        if len(missing) > 0:
            print(f"   - {len(missing)} parameters missing gradients")
        return False


if __name__ == "__main__":
    success = test_gradient_tracing()
    sys.exit(0 if success else 1)
