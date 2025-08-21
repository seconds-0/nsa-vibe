import os
import pytest
import torch
from typing import Dict, Optional


RUN_TRITON_SEL = os.getenv("NSA_TEST_TRITON_SEL", "0").lower() in ("1", "true", "yes")
SM89_BLOCK = False
if torch.cuda.is_available():
    try:
        SM89_BLOCK = torch.cuda.get_device_capability() == (8, 9)
    except (RuntimeError, AttributeError) as e:
        # RuntimeError: No CUDA devices available or device access issues
        # AttributeError: Missing CUDA functionality
        SM89_BLOCK = False
FORCED = os.getenv("NSA_TRITON_SEL_FORCE", "0").lower() in ("1", "true", "yes")


def _cleanup_env_vars(vars_to_clean: Dict[str, Optional[str]]) -> None:
    """Clean up environment variables after test, restoring original values."""
    for key, original_value in vars_to_clean.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


def _set_env_vars(vars_to_set: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Set environment variables and return original values for cleanup."""
    original_values = {}
    for key, value in vars_to_set.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    return original_values


@pytest.mark.skipif(not RUN_TRITON_SEL, reason="opt-in: set NSA_TEST_TRITON_SEL=1")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(SM89_BLOCK and not FORCED, reason="Triton disabled by ADR on SM 8.9; set NSA_TRITON_SEL_FORCE=1 for experiments")
def test_threshold_fallback_logs(capsys):
    # Set environment variables and track originals for cleanup
    env_vars = {
        "NSA_USE_TRITON_SEL": "1",
        "NSA_SEL_TRITON_MIN_L": "100000",  # force threshold fallback
        "NSA_DEBUG_LOG": "1",
        "NSA_LOG_LIMIT": "4"
    }
    original_values = _set_env_vars(env_vars)
    
    try:
        device = torch.device("cuda")
        B, S, G, H, D, Dv = 1, 1, 1, 2, 64, 64
        S_kv = 64
        Q = torch.randn(B, S, G, H, D, device=device, dtype=torch.float16)
        K = torch.randn(B, G, S_kv, D, device=device, dtype=torch.float16)
        V = torch.randn(B, G, S_kv, Dv, device=device, dtype=torch.float16)
        ranges = torch.tensor([[[[[0, 32]]]]], device=device, dtype=torch.int64)
        from nsa.kernels.triton_sel_kernel import selection_attention_triton
        _ = selection_attention_triton(Q, K, V, ranges)
        out = capsys.readouterr().out
        assert "sel.triton.fallback" in out and "reason=threshold" in out
    finally:
        _cleanup_env_vars(original_values)


@pytest.mark.skipif(not RUN_TRITON_SEL, reason="opt-in: set NSA_TEST_TRITON_SEL=1")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(SM89_BLOCK and not FORCED, reason="Triton disabled by ADR on SM 8.9; set NSA_TRITON_SEL_FORCE=1 for experiments")
def test_dtype_fallback_logs(capsys):
    # Set environment variables and track originals for cleanup
    env_vars = {
        "NSA_USE_TRITON_SEL": "1",
        "NSA_SEL_TRITON_MIN_L": "0",
        "NSA_DEBUG_LOG": "1",
        "NSA_LOG_LIMIT": "4"
    }
    original_values = _set_env_vars(env_vars)
    
    try:
        device = torch.device("cuda")
        B, S, G, H, D, Dv = 1, 1, 1, 2, 64, 64
        S_kv = 64
        # FP32 types to trigger dtype fallback
        Q = torch.randn(B, S, G, H, D, device=device, dtype=torch.float32)
        K = torch.randn(B, G, S_kv, D, device=device, dtype=torch.float32)
        V = torch.randn(B, G, S_kv, Dv, device=device, dtype=torch.float32)
        ranges = torch.tensor([[[[[0, 32]]]]], device=device, dtype=torch.int64)
        from nsa.kernels.triton_sel_kernel import selection_attention_triton
        _ = selection_attention_triton(Q, K, V, ranges)
        out = capsys.readouterr().out
        assert "sel.triton.fallback" in out and "reason=dtype" in out
    finally:
        _cleanup_env_vars(original_values)


@pytest.mark.skipif(not RUN_TRITON_SEL, reason="opt-in: set NSA_TEST_TRITON_SEL=1")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(SM89_BLOCK and not FORCED, reason="Triton disabled by ADR on SM 8.9; set NSA_TRITON_SEL_FORCE=1 for experiments")
def test_group_path_smoke():
    # Set environment variables and track originals for cleanup
    env_vars = {
        "NSA_USE_TRITON_SEL": "1",
        "NSA_SEL_TRITON_GROUP": "1",
        "NSA_SEL_TRITON_MIN_L": "1"
    }
    original_values = _set_env_vars(env_vars)
    
    try:
        device = torch.device("cuda")
        B, S, G, H, D, Dv = 2, 1, 1, 4, 64, 64
        S_kv = 128
        Q = torch.randn(B, S, G, H, D, device=device, dtype=torch.float16)
        K = torch.randn(B, G, S_kv, D, device=device, dtype=torch.float16)
        V = torch.randn(B, G, S_kv, Dv, device=device, dtype=torch.float16)
        ranges = torch.tensor([[[[[0, 40], [64, 96]]]]], device=device, dtype=torch.int64).expand(B, S, G, 2, 2).contiguous()
        from nsa.kernels.triton_sel_kernel import selection_attention_triton
        from nsa.core.attention_kernels import grouped_selection_attention_packed
        O_ref = grouped_selection_attention_packed(Q, K, V, ranges).float()
        O_tri = selection_attention_triton(Q, K, V, ranges).float()
        mae = (O_tri - O_ref).abs().mean().item()
        assert mae < 1e-3
    finally:
        _cleanup_env_vars(original_values)

