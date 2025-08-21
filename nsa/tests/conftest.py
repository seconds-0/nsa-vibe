import os
import random

import numpy as np
import pytest
import torch

from nsa.kernels.flash_wrappers import fa2_supported, is_flash_varlen_available


def _is_sm89() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability(0)
        return cap == (8, 9)
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    skip_triton_4090 = []
    skip_fa2 = []
    force_triton = os.getenv("NSA_TRITON_SEL_FORCE", "0").lower() in ("1", "true", "yes")
    want_fa2 = os.getenv("NSA_TEST_FA2", "0").lower() in ("1", "true", "yes")
    # Determine FA-2 availability with a conservative probe
    fa2_ok = False
    if torch.cuda.is_available():
        # Probe a typical head dim (64) with fp16
        fa2_ok = is_flash_varlen_available() or fa2_supported(
            torch.device("cuda"), torch.float16, 64
        )
    for item in items:
        name = item.nodeid.lower()
        # Skip Triton selection GPU tests on 4090 unless forced
        if ("triton" in name or "sel_cuda" in name) and _is_sm89() and not force_triton:
            skip_triton_4090.append(item)
        # Skip FA-2 tests unless explicitly enabled and supported
        if "fa2" in name:
            if not want_fa2 or not fa2_ok:
                skip_fa2.append(item)
    if skip_triton_4090:
        reason = "Triton selection disabled by ADR on RTX 4090; set NSA_TRITON_SEL_FORCE=1 to run"
        for it in skip_triton_4090:
            it.add_marker(pytest.mark.skip(reason=reason))
    if skip_fa2:
        reason = "FA-2 GPU tests require NSA_TEST_FA2=1 and flash-attn varlen availability"
        for it in skip_fa2:
            it.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def set_determinism():
    seed = int(os.getenv("NSA_TEST_SEED", "1337"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    yield
