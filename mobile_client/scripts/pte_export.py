# mobile_client/scripts/pte_export.py
"""Functional .pte export for the ExecuTorch mobile FL core.

Exports two weight-free functional graphs:
  - Loss graph:  forward(flat_trainable, x, y) -> cross_entropy
  - Infer graph: forward(flat_trainable, x)    -> logits

In both cases the model's *trainable* parameters enter as the single flat input (in
named_parameters() order, matching ZerothOrderEstimator._get_flat_params / C++ getFlatParams)
and *frozen* parameters are baked in as constants. ExecuTorch runs the graph; the C++ FL core
owns and perturbs the flat vector. Validated toolchain: torch 2.12.0 + executorch 1.3.1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.export import export


def trainable_names(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def trainable_flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for n, p in model.named_parameters() if p.requires_grad])


def _unflatten_params(
    base: nn.Module,
    frozen: dict,
    names: list[str],
    shapes: list,
    numel: list[int],
    flat: torch.Tensor,
) -> dict:
    """Reconstruct the full param dict from the flat trainable vector + baked-in frozen params."""
    params = dict(frozen)
    off = 0
    for n, s, k in zip(names, shapes, numel):
        params[n] = flat[off:off + k].reshape(s)
        off += k
    return params


class _FunctionalLoss(nn.Module):
    """forward(flat_trainable, x, y) -> cross_entropy. Trainable params come from flat_trainable;
    frozen params are constants. The base model is hidden in a list so its parameters are NOT
    registered on this wrapper (the exported graph has zero module params)."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self._base = [base]  # list hides base from nn.Module param registration
        self._names = trainable_names(base)
        self._shapes = [base.get_parameter(n).shape for n in self._names]
        self._numel = [base.get_parameter(n).numel() for n in self._names]
        # Frozen params captured as constants (detached, not registered as state).
        self._frozen = {n: p.detach().clone() for n, p in base.named_parameters() if not p.requires_grad}

    def forward(self, flat_trainable: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        params = _unflatten_params(
            self._base[0], self._frozen, self._names, self._shapes, self._numel, flat_trainable
        )
        logits = functional_call(self._base[0], params, (x,))
        return torch.nn.functional.cross_entropy(logits, y)


class _FunctionalInfer(nn.Module):
    """forward(flat_trainable, x) -> logits. Two inputs; no cross-entropy, no y.
    Trainable params come from flat_trainable; frozen params are baked-in constants.
    The base model is hidden in a list so its parameters are NOT registered on this wrapper
    (the exported graph has zero module params)."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self._base = [base]  # list hides base from nn.Module param registration
        self._names = trainable_names(base)
        self._shapes = [base.get_parameter(n).shape for n in self._names]
        self._numel = [base.get_parameter(n).numel() for n in self._names]
        # Frozen params captured as constants (detached, not registered as state).
        self._frozen = {n: p.detach().clone() for n, p in base.named_parameters() if not p.requires_grad}

    def forward(self, flat_trainable: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        params = _unflatten_params(
            self._base[0], self._frozen, self._names, self._shapes, self._numel, flat_trainable
        )
        return functional_call(self._base[0], params, (x,))


def export_functional_pte(model: nn.Module, example_inputs: tuple[torch.Tensor, torch.Tensor]) -> bytes:
    """Return .pte bytes for forward(flat_trainable, x, y) -> cross_entropy."""
    from executorch.exir import to_edge

    model = model.eval()
    wrapper = _FunctionalLoss(model).eval()
    assert sum(p.numel() for p in wrapper.parameters()) == 0, "wrapper must register 0 params"
    x, y = example_inputs
    ex = (trainable_flat(model), x, y)
    ep = export(wrapper, ex)
    return to_edge(ep).to_executorch().buffer


def export_functional_infer_pte(model: nn.Module, example_x: torch.Tensor) -> bytes:
    """Return .pte bytes for forward(flat_trainable, x) -> logits."""
    from executorch.exir import to_edge

    model = model.eval()
    wrapper = _FunctionalInfer(model).eval()
    assert sum(p.numel() for p in wrapper.parameters()) == 0, "wrapper must register 0 params"
    ep = export(wrapper, (trainable_flat(model), example_x))
    return to_edge(ep).to_executorch().buffer
