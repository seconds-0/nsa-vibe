import os
import importlib
import types
import torch


def test_selection_cuda_fallback_on_forward_error(monkeypatch):
    mod = importlib.import_module("nsa.kernels.cuda_sel_kernel")

    class BadExt:
        def sel_forward(self, Q, K, V, ranges):
            raise RuntimeError("synthetic forward failure")

    # Force the loader to return our bad extension and mark CUDA usage on
    monkeypatch.setenv("NSA_SEL_CUDA", "1")
    monkeypatch.setenv("NSA_SEL_CUDA_BUILD", "0")
    monkeypatch.setenv("NSA_DEBUG_BUILD", "0")
    monkeypatch.setenv("NSA_DEBUG_TIMING", "0")
    monkeypatch.setenv("NSA_DEBUG_SHAPES", "0")
    monkeypatch.setenv("NSA_FORCE_PARITY", "0")

    monkeypatch.setattr(mod, "_EXT", BadExt(), raising=False)
    monkeypatch.setattr(mod, "_load_extension", lambda: mod._EXT, raising=False)

    # Tiny CPU tensors
    B, S, G, h, Dk, Dv = 1, 2, 1, 2, 4, 4
    S_kv, n = 8, 2
    Q = torch.randn(B, S, G, h, Dk)
    K = torch.randn(B, G, S_kv, Dk)
    V = torch.randn(B, G, S_kv, Dv)
    ranges = torch.tensor([[[[[0, 3],[4, 6]]],[[[1, 4],[5, 7]]]]], dtype=torch.int32)

    # Fallback reference
    ref_mod = importlib.import_module("nsa.core.attention_kernels")
    O_ref = ref_mod.grouped_selection_attention_packed(Q, K, V, ranges)

    # Under failure, wrapper should fall back and match ref path shape and dtype
    O = mod.selection_attention_cuda(Q, K, V, ranges)
    assert O.shape == O_ref.shape and O.dtype == O_ref.dtype

