import os
import random
import numpy as np
import torch
import pytest


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
