import os
import importlib

import torch


def test_is_sm89_cpu_false():
    mod = importlib.import_module("nsa.core.attention_kernels")
    assert mod._is_sm89(torch.device("cpu")) is False


def test_fa2_forced_env_flag(tmp_path, monkeypatch):
    mod = importlib.import_module("nsa.core.attention_kernels")
    # default off
    monkeypatch.delenv("NSA_FA2_FORCE", raising=False)
    assert mod._fa2_forced() is False
    # forced on
    monkeypatch.setenv("NSA_FA2_FORCE", "1")
    assert mod._fa2_forced() is True

