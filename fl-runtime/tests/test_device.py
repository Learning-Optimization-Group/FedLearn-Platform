import pytest
from device import resolve_device


def test_cpu_passthrough():
    assert resolve_device("cpu") == "cpu"


def test_auto_returns_valid_device():
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}


def test_unknown_raises():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_explicit_unavailable_falls_back_to_cpu(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # mps may not exist on the runner; force the unavailable branch too
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("cuda") == "cpu"
