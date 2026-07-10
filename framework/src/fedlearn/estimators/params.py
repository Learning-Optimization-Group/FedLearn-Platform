# src/fedlearn/estimators/params.py
"""FR-14: the canonical trainable-parameter manifest.

Single source of truth for HOW a model's trainable parameters are enumerated, ordered, flattened,
unflattened, and counted: the ``named_parameters()`` order, filtered by ``requires_grad``. The DeComFL
client, the ZerothOrderEstimator, and the mobile ExecuTorch export all key on exactly this layout — the
flat vector that the shared-seed perturbation direction ``z`` indexes into. Any code that flattens /
unflattens / counts *trainable* params should delegate here so client and server agree on that layout;
a mismatch silently misaligns the perturbation (the model diverges with no error).

This module governs *which* parameters feed the perturbation, NOT the perturbation numerics —
``estimators.perturbation.canonical_perturbation`` is a frozen cross-language (Python<->C++) contract and
is untouched by this.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn


def param_layout(model: nn.Module) -> List[Tuple[str, torch.Size, int]]:
    """Ordered ``(name, shape, numel)`` for every TRAINABLE (``requires_grad``) parameter, in
    ``named_parameters()`` order — the manifest describing the flat vector's layout."""
    return [(name, p.shape, p.numel()) for name, p in model.named_parameters() if p.requires_grad]


def flat_params(model: nn.Module) -> torch.Tensor:
    """The trainable parameters flattened into one 1-D tensor, in :func:`param_layout` order."""
    return torch.cat([p.data.view(-1) for _, p in model.named_parameters() if p.requires_grad])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Write ``flat`` (in :func:`param_layout` order) back into the model's trainable parameters."""
    offset = 0
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].view_as(p.data))
        offset += numel


def num_trainable(model: nn.Module) -> int:
    """Number of trainable (``requires_grad``) scalar parameters — the flat vector's length."""
    return sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)


def trainable_state(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """The trainable parameters as an ``OrderedDict[name -> tensor]`` (detached snapshot) in
    :func:`param_layout` order — the CORRECT ``initial_parameters`` for a DeComFL server, so the
    server's flat layout equals the client's.

    Passing a full ``model.state_dict()`` instead includes buffers (``running_mean``/``running_var``/…)
    and frozen params (LoRA base, partial fine-tune), which the client's requires_grad flatten omits —
    so ``d_server > d_client`` and the shared-seed perturbation ``z`` silently misaligns.
    """
    return OrderedDict(
        (name, p.detach().clone()) for name, p in model.named_parameters() if p.requires_grad
    )
