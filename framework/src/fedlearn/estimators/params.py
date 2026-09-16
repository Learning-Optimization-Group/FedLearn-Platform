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


def frozen_state(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """The FROZEN, F32-only complement of :func:`trainable_state` — the bytes a DA-11 ``BASE_REF``
    backbone blob carries. Frozen (``requires_grad is False``) parameters in ``named_parameters()``
    order, then all *float* buffers in ``named_buffers()`` order, as detached clones.

    Integer buffers (e.g. ``BatchNorm.num_batches_tracked``) are excluded on purpose: they are not
    used in eval-mode forward and are not part of the platform's float32-only wire
    (``safetensors_codec``). Excluding them keeps this manifest F32-consistent and byte-deterministic
    for content addressing while remaining forward-correct for a real BatchNorm backbone (Phase 2C).
    """
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            out[name] = p.detach().clone()
    for name, b in model.named_buffers():
        if b is not None and b.is_floating_point():
            out[name] = b.detach().clone()
    return out


def federable_state(state: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
    """The subset of ``state`` that can cross the safetensors wire: the float32 tensors.

    The wire is float32-only by design — it must decode in the libtorch-free mobile C++ core, so
    other dtypes raise rather than being silently coerced. Every BatchNorm module carries an int64
    ``num_batches_tracked``, so a FULL-arm run on ANY BatchNorm model failed on the first
    GetGlobalModel. That excluded ResNets, the most common architecture in the FL literature, from
    the FULL arm entirely.

    WHAT IS DROPPED: non-float32 tensors. In practice ``num_batches_tracked``, a batch COUNTER —
    averaging it across clients is meaningless, so nothing of value is lost and each client keeps
    its own.

    WHAT IS NOT DROPPED: ``running_mean``/``running_var``. They are float32 and continue to be
    averaged. Excluding those too would be FedBN — a different algorithm with different convergence
    behaviour — rather than a fix for what the wire can carry.

    Use this on BOTH sides. Client and server must federate an identical key set, and two
    independent filters would drift; that divergence is how the frozen arm broke twice.

    Identity for a float32-only model: same keys, same order, same tensor objects, so no existing
    recipe changes behaviour.
    """
    return OrderedDict((k, v) for k, v in state.items() if v.dtype == torch.float32)


def non_federable_names(state: "OrderedDict[str, torch.Tensor]") -> "List[str]":
    """Names ``federable_state`` would withhold, so a caller can LOG what it excluded.

    Silently dropping tensors is what would make this dangerous; a run has to be able to say what
    it withheld and a reader has to be able to audit it.
    """
    return [k for k, v in state.items() if v.dtype != torch.float32]
