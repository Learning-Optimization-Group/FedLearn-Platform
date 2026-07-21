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


# --- Trainable (first-order) export -------------------------------------------------------------
# The functional graphs above are weight-FREE (params enter as a flat input) — the C++ core owns and
# perturbs the flat vector, which is why the zeroth-order path needs no autograd. First-order FedAvg
# needs REAL gradients, i.e. the ExecuTorch training extension: params live INSIDE the module and the
# backward pass is captured AS a graph at export time via ``_export_forward_backward``. The resulting
# .pte exposes gradients through ``training_module.named_gradients`` at runtime (Phase B M1b).


class _TrainingGraph(nn.Module):
    """forward(x, y) -> (loss, prediction), with the base model's parameters registered INTERNALLY.

    Frozen params (requires_grad=False) receive no gradient in the captured backward graph, so the
    training module's trainable ``named_parameters`` == the base model's trainable set (same order),
    matching ``estimators.params`` / ``trainable_flat``. The base model is registered as a submodule
    named ``base``, so the exported trainable param names carry a ``base.`` prefix — see
    ``training_trainable_names`` for the exact runtime names + canonical order the C++ side re-maps
    ET's (alphabetically-keyed) ``named_parameters`` map into.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = self.base(x)
        return self.loss(out, y), out.detach().argmax(1)


def training_trainable_names(model: nn.Module) -> list[str]:
    """The trainable parameter names of the training graph, in canonical (base) named_parameters
    order — i.e. ``base.<name>`` for each trainable ``name`` in ``trainable_names(model)``.

    This is the order the flat vector uses; ET's ``TrainingModule::named_parameters`` returns a map
    keyed alphabetically, so the C++ ``getFlatParams``/``setFlatParams`` must project ET's map back
    onto THIS sequence or the flat blocks transpose (the M1 ordering gotcha)."""
    return [f"base.{n}" for n in trainable_names(model)]


def export_trainable_pte(model: nn.Module, example_inputs: tuple[torch.Tensor, torch.Tensor]) -> bytes:
    """Return .pte bytes for a TRAINABLE graph: forward(x, y) -> (cross_entropy, prediction) with a
    captured backward pass. Load it with ET's TrainingModule (execute_forward_backward + optimizer).

    Frozen (requires_grad=False) layers are baked as constants and get no gradient, so only the
    trainable params (``training_trainable_names(model)``) are optimised — matching the framework's
    FedAvg update, which leaves frozen layers fixed."""
    from executorch.exir import to_edge
    from torch.export.experimental import _export_forward_backward

    wrapper = _TrainingGraph(model)
    x, y = example_inputs
    ep = export(wrapper, (x, y), strict=True)
    ep = _export_forward_backward(ep)
    return to_edge(ep).to_executorch().buffer
