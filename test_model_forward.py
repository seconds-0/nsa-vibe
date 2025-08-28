#!/usr/bin/env python
"""Minimal test to debug model forward hang"""
import torch
import time
from omegaconf import OmegaConf
import sys
import os

# Add path for imports
sys.path.insert(0, '.')
os.environ['NSA_PREFILL_BATCHED'] = '1'
os.environ['NSA_SEL_RANGES_V2'] = '1'

print("Loading config...")
cfg = OmegaConf.load("configs/m7c_125m_2xa100_production.yaml")

print("Importing model...")
from scripts.train_showcase import TinyLM

print("Creating model...")
model = TinyLM(
    vocab=256,
    dim=int(cfg.model.dim),
    n_layers=int(cfg.model.n_layers),
    n_heads=int(cfg.model.n_heads),
    n_kv_groups=int(cfg.model.n_kv_groups),
    d_k=int(cfg.model.d_k),
    d_v=int(cfg.model.d_v),
    l=int(cfg.nsa.l),
    d=int(cfg.nsa.d),
    l_sel=int(cfg.nsa.l_sel),
    n_sel=int(cfg.nsa.n_sel),
    w=int(cfg.nsa.w),
    grad_checkpointing=False,
).cuda().eval()

print(f"Model created, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

print("Creating input...")
x = torch.randint(0, 256, (1, 128), device='cuda')
print(f"Input shape: {x.shape}")

print("Running forward pass...")
start = time.time()
with torch.no_grad():
    try:
        out = model(x)
        print(f"✅ Forward pass successful in {time.time()-start:.2f}s")
        print(f"Output shape: {out.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()