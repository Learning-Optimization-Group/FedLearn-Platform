"""Device resolution for the FL client.

Kept out of client.py so it is unit-testable without importing the heavy client
module (torch load, pynvml init, dataset config, etc.).
"""
from __future__ import annotations

import torch

_VALID = {"auto", "cpu", "cuda", "mps"}


def resolve_device(choice: str) -> str:
    """Resolve a --device choice to a concrete torch device string.

    ``auto`` -> ``cuda`` if available, else ``mps`` if available, else ``cpu``.
    An explicitly requested accelerator that is unavailable falls back to ``cpu``
    (with a warning) so a misdetected host still runs rather than crashing.
    """
    if choice not in _VALID:
        raise ValueError(f"--device must be one of {sorted(_VALID)}, got {choice!r}")
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if choice == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if choice == "cuda" and not cuda:
        print("[device] cuda requested but unavailable; falling back to cpu")
        return "cpu"
    if choice == "mps" and not mps:
        print("[device] mps requested but unavailable; falling back to cpu")
        return "cpu"
    return choice
