#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")
from scripts.train_showcase import TinyLM
import torch
import time
import os

torch.cuda.empty_cache()

# Test at different sequence lengths
for S in [128, 512, 1024, 2048]:
    B = 1  # Single batch for testing
    model = TinyLM(256, 768, 12, 12, 2, 64, 64, 32, 16, 64, 16, 512, False).cuda()
    x = torch.randint(0, 256, (B, S), device="cuda")
    
    print(f"\n=== Testing S={S} ===")
    
    for mode in ["adaptive", "v1", "v2"]:
        os.environ.pop("NSA_SEL_RANGES_V2", None)
        os.environ["NSA_SEL_RANGES_V2_MIN_S"] = "1024"  # Adaptive threshold
        os.environ["NSA_PREFILL_BATCHED"] = "1"
        os.environ["NSA_USE_SEL_PACK"] = "1"
        os.environ["NSA_FORCE_PARITY"] = "0"
        if mode == "v1":
            os.environ["NSA_SEL_RANGES_V2"] = "0"
        elif mode == "v2":
            os.environ["NSA_SEL_RANGES_V2"] = "1"
        
        # Warmup
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        
        # Measure with timeout
        t0 = time.time()
        timeout = 30 if S <= 1024 else 60
        
        try:
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
            dt = time.time() - t0
            
            if dt > timeout:
                print(f"  {mode:8s}: TIMEOUT (>{timeout}s)")
            else:
                toks_per_sec = (B * S) / dt
                print(f"  {mode:8s}: {dt:6.2f}s, {toks_per_sec:7.1f} tok/s")
        except Exception as e:
            print(f"  {mode:8s}: ERROR - {e}")
    
    # Clean up for next iteration
    del model
    torch.cuda.empty_cache()

print("\n=== Test Complete ===")