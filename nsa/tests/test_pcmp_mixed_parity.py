import os
import pytest
import torch

from nsa.core.selection_scorer import compute_pcmp_all


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
        ),
    ],
)
def test_compute_pcmp_all_mixed_parity(device: str, monkeypatch):
    torch.manual_seed(0)
    dev = torch.device(device)
    # Shapes
    B, S, G, h, Dk = 2, 8, 2, 2, 32
    S_cmp = 16
    # Use float32 baseline
    Q = torch.randn(B, S, G, h, Dk, device=dev, dtype=torch.float32)
    K = torch.randn(B, G, S_cmp, Dk, device=dev, dtype=torch.float32)
    scale = 1.0 / (Dk ** 0.5)

    # Baseline precise
    monkeypatch.setenv("NSA_P_CMP_MIXED", "0")
    p_ref = compute_pcmp_all(Q, K, scale)

    # Mixed (CUDA only path; on CPU it's same as baseline)
    monkeypatch.setenv("NSA_P_CMP_MIXED", "1")
    p_mx = compute_pcmp_all(Q, K, scale)

    mae = (p_ref - p_mx).abs().mean().item()
    # Allow small tolerance for CUDA autocast differences
    tol = 3e-4 if device == "cuda" else 1e-6
    assert mae < tol, f"MAE too high: {mae} (tol {tol})"

