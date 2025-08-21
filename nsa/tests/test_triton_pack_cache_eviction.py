import os
import importlib
import torch


def test_pack_cache_eviction_soft_cap(monkeypatch):
    # Import module fresh to access cache
    m = importlib.import_module('nsa.kernels.triton_sel_kernel')
    # Set tiny cap to trigger eviction
    monkeypatch.setenv('NSA_SEL_TRITON_PACK_CACHE_MAX_MB', '1')
    importlib.reload(m)
    # Allocate buffers exceeding 1MB soft cap
    device = torch.device('cpu')
    # total elements ~ 300k floats ~ 1.2MB (4 bytes each)
    total_L = 300_000
    D = 1
    Dv = 1
    N = 1
    Kb, Vb, cu = m._get_pack_buffers(device, total_L, D, Dv, N, torch.float32, torch.float32)
    # Access internal cache and check it does not exceed cap after enforcement
    bytes1 = m._pack_cache_total_bytes()
    # Re-request to test eviction logic does not grow cache unbounded
    Kb2, Vb2, cu2 = m._get_pack_buffers(device, total_L, D, Dv, N, torch.float32, torch.float32)
    bytes2 = m._pack_cache_total_bytes()
    assert len(m._PACK_CACHE) <= 1
    assert bytes2 == bytes1
