# framework/tests/conftest.py
import torch
import random
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def disable_cuda(monkeypatch):
    """Force CPU for all tests to avoid device mismatch issues."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Make all tests deterministic."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    yield
