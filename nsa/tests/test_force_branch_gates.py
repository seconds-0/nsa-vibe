import os

import torch

from nsa.core.nsa_attention import GateMLP


def test_force_branch_env_cmp():
    os.environ["NSA_FORCE_BRANCH"] = "cmp"
    try:
        g = GateMLP(d_k=8)
        x = torch.randn(2, 3, 8)
        p = g(x)
        assert p.shape == (2, 3, 3)
        assert torch.allclose(p.sum(dim=-1), torch.ones_like(p[..., 0]))
        idx = torch.argmax(p, dim=-1)
        assert torch.all(idx == 0)
    finally:
        os.environ.pop("NSA_FORCE_BRANCH", None)


def test_force_branch_env_sel():
    os.environ["NSA_FORCE_BRANCH"] = "sel"
    try:
        g = GateMLP(d_k=8)
        x = torch.randn(1, 4, 8)
        p = g(x)
        idx = torch.argmax(p, dim=-1)
        assert torch.all(idx == 1)
    finally:
        os.environ.pop("NSA_FORCE_BRANCH", None)


def test_force_branch_env_win():
    os.environ["NSA_FORCE_BRANCH"] = "win"
    try:
        g = GateMLP(d_k=8)
        x = torch.randn(1, 1, 8)
        p = g(x)
        idx = torch.argmax(p, dim=-1)
        assert torch.all(idx == 2)
    finally:
        os.environ.pop("NSA_FORCE_BRANCH", None)
