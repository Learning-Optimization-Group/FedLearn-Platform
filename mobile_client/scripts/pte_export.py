# mobile_client/scripts/pte_export.py
"""Functional .pte export for the ExecuTorch mobile FL core.

Exports a model as a weight-free graph forward(flat_trainable, x, y) -> cross_entropy, where
the model's *trainable* parameters enter as the single flat input (in named_parameters() order,
matching ZerothOrderEstimator._get_flat_params / C++ getFlatParams) and *frozen* parameters are
baked in as constants. ExecuTorch runs the graph; the C++ FL core owns and perturbs the flat
vector. Validated toolchain: torch 2.12.0 + executorch 1.3.1.
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
        params = dict(self._frozen)
        off = 0
        for n, s, k in zip(self._names, self._shapes, self._numel):
            params[n] = flat_trainable[off:off + k].reshape(s)
            off += k
        logits = functional_call(self._base[0], params, (x,))
        return torch.nn.functional.cross_entropy(logits, y)


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
