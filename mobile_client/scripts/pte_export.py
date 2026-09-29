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

import copy

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.export import Dim, export


def _batch_dim(max_batch: int | None):
    """The example count as an export dimension: dynamic from 1 to ``max_batch``, or static (None).

    A static dimension fixes every program to the example batch it was exported with, so a device dataset of any
    other size is refused at runtime (ExecuTorch NotSupported). A dynamic one lets a device train its own dataset,
    and a final partial minibatch, up to the bound the runtime plans memory for.
    """
    if max_batch is None:
        return None
    if max_batch < 1:
        raise ValueError(f"max_batch must be at least 1, not {max_batch}")
    return Dim("batch", min=1, max=max_batch)


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


def export_functional_pte(model: nn.Module, example_inputs: tuple[torch.Tensor, torch.Tensor],
                          max_batch: int | None = None) -> bytes:
    """Return .pte bytes for forward(flat_trainable, x, y) -> cross_entropy.

    ``max_batch`` makes the example count dynamic (1..max_batch); None keeps it static at the example's."""
    from executorch.exir import to_edge

    model = model.eval()
    wrapper = _FunctionalLoss(model).eval()
    assert sum(p.numel() for p in wrapper.parameters()) == 0, "wrapper must register 0 params"
    x, y = example_inputs
    ex = (trainable_flat(model), x, y)
    batch = _batch_dim(max_batch)
    ep = export(wrapper, ex, dynamic_shapes=None if batch is None else (None, {0: batch}, {0: batch}))
    return to_edge(ep).to_executorch().buffer


def export_functional_infer_pte(model: nn.Module, example_x: torch.Tensor, max_batch: int | None = None) -> bytes:
    """Return .pte bytes for forward(flat_trainable, x) -> logits (``max_batch`` as in export_functional_pte)."""
    from executorch.exir import to_edge

    model = model.eval()
    wrapper = _FunctionalInfer(model).eval()
    assert sum(p.numel() for p in wrapper.parameters()) == 0, "wrapper must register 0 params"
    batch = _batch_dim(max_batch)
    ep = export(wrapper, (trainable_flat(model), example_x),
                dynamic_shapes=None if batch is None else (None, {0: batch}))
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


class _DropoutMaskSlot(nn.Module):
    """Stands in for an nn.Dropout in a mobile training program: multiplies by the mask the step was given."""

    def __init__(self) -> None:
        super().__init__()
        self.mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask


class _MaskedTrainingGraph(nn.Module):
    """forward(x, y, masks) -> (loss, prediction), with every nn.Dropout replaced by multiplication with a mask input.

    The masks (one per dropout layer, in named_modules() order, each shaped like that layer's activation) come from
    the contract's DROPOUT_MASKS_SEEDED_V1 stream, so a device's step is reproducible; torch's own dropout RNG is not.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = copy.deepcopy(base)
        self.slots: list[tuple[str, _DropoutMaskSlot]] = []
        for name, module in list(self.base.named_modules()):
            if isinstance(module, nn.Dropout):
                parent_name, _, child = name.rpartition(".")
                parent = self.base.get_submodule(parent_name) if parent_name else self.base
                slot = _DropoutMaskSlot()
                setattr(parent, child, slot)
                self.slots.append((name, slot))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor, masks: tuple[torch.Tensor, ...]):
        for (_, slot), mask in zip(self.slots, masks):
            slot.mask = mask
        out = self.base(x)
        return self.loss(out, y), out.detach().argmax(1)


def dropout_layers(model: nn.Module) -> list[tuple[str, float]]:
    """The model's dropout layers in forward (named_modules) order, with their rates: what the contract states."""
    return [(name, float(m.p)) for name, m in model.named_modules() if isinstance(m, nn.Dropout)]


def export_masked_trainable_pte(model: nn.Module, example_inputs: tuple[torch.Tensor, torch.Tensor],
                                mask_shapes: list[tuple[int, ...]], max_batch: int | None = None) -> bytes:
    """A trainable graph whose dropout layers take their masks as inputs: forward(x, y, masks).

    ``mask_shapes`` are the per-example activation shapes of the dropout layers, in dropout_layers() order; the batch
    dimension is dynamic like x's when ``max_batch`` is given. Frozen params are baked as in export_trainable_pte.
    """
    from executorch.exir import to_edge
    from torch.export.experimental import _export_forward_backward

    wrapper = _MaskedTrainingGraph(model)
    if len(wrapper.slots) != len(mask_shapes):
        raise ValueError(f"the model has {len(wrapper.slots)} dropout layers, not {len(mask_shapes)}")
    x, y = example_inputs
    masks = tuple(torch.ones(x.shape[0], *shape) for shape in mask_shapes)
    batch = _batch_dim(max_batch)
    dynamic = None if batch is None else ({0: batch}, {0: batch}, tuple({0: batch} for _ in masks))
    ep = export(wrapper, (x, y, masks), strict=True, dynamic_shapes=dynamic)
    ep = _export_forward_backward(ep)
    return to_edge(ep).to_executorch().buffer


def training_trainable_names(model: nn.Module) -> list[str]:
    """The trainable parameter names of the training graph, in canonical (base) named_parameters
    order — i.e. ``base.<name>`` for each trainable ``name`` in ``trainable_names(model)``.

    This is the order the flat vector uses; ET's ``TrainingModule::named_parameters`` returns a map
    keyed alphabetically, so the C++ ``getFlatParams``/``setFlatParams`` must project ET's map back
    onto THIS sequence or the flat blocks transpose (the M1 ordering gotcha)."""
    return [f"base.{n}" for n in trainable_names(model)]


def export_trainable_pte(model: nn.Module, example_inputs: tuple[torch.Tensor, torch.Tensor],
                         max_batch: int | None = None) -> bytes:
    """Return .pte bytes for a TRAINABLE graph: forward(x, y) -> (cross_entropy, prediction) with a
    captured backward pass. Load it with ET's TrainingModule (execute_forward_backward + optimizer).

    Frozen (requires_grad=False) layers are baked as constants and get no gradient, so only the
    trainable params (``training_trainable_names(model)``) are optimised — matching the framework's
    FedAvg update, which leaves frozen layers fixed. ``max_batch`` as in export_functional_pte."""
    from executorch.exir import to_edge
    from torch.export.experimental import _export_forward_backward

    wrapper = _TrainingGraph(model)
    x, y = example_inputs
    batch = _batch_dim(max_batch)
    ep = export(wrapper, (x, y), strict=True,
                dynamic_shapes=None if batch is None else ({0: batch}, {0: batch}))
    ep = _export_forward_backward(ep)
    return to_edge(ep).to_executorch().buffer


def dropout_mask_shapes(model: nn.Module, example_x: torch.Tensor) -> list[tuple[int, ...]]:
    """The per-example activation shape at each dropout layer, in dropout_layers() order: the masks a masked program
    takes, read from one forward pass rather than from the architecture."""
    shapes: list[tuple[int, ...]] = []
    hooks = [m.register_forward_hook(lambda _m, _i, out: shapes.append(tuple(out.shape[1:])))
             for _, m in model.named_modules() if isinstance(m, nn.Dropout)]
    was_training = model.training
    try:
        # Eval mode, so the pass neither draws from torch's RNG (dropout) nor moves BatchNorm running statistics.
        model.eval()
        with torch.no_grad():
            model(example_x[:1])
    finally:
        model.train(was_training)
        for h in hooks:
            h.remove()
    return shapes


PROBE_LEARNING_RATE = 0.1


def probe_batch(step: int, rows: int, width: int, classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The qualification probe's synthetic batch for step 1 or 2: x[i][j] = ((31 i + 7 j + 3 step) mod 17 - 8) / 8,
    labels i mod classes. The native probe builds the same batch.

    Every value is exact in float32. Rows differ from one another (31 is coprime to 17) for up to 17 rows whatever the
    width, and no input is all zeros, so a step moves every layer even of a model whose biases start at zero.
    """
    x = torch.tensor([[((31 * i + 7 * j + 3 * step) % 17 - 8) / 8 for j in range(width)] for i in range(rows)],
                     dtype=torch.float32)
    return x, torch.tensor([i % classes for i in range(rows)], dtype=torch.int64)


def probe_reference(model: nn.Module, rows: int, width: int, classes: int) -> dict:
    """What a trainable program must report on the probe: two SGD steps from its embedded (export-time) weights.

    A masked-dropout program is probed with every mask all ones, so its dropout layers pass activations through
    unchanged; the device does the same. The probe is a property of the artifact, not of a run, so a device can cache
    its result per program digest.
    """
    mask_shapes = dropout_mask_shapes(model, probe_batch(1, rows, width, classes)[0])
    wrapper = _MaskedTrainingGraph(model)
    params = [p for p in wrapper.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=PROBE_LEARNING_RATE)
    losses = []
    for step in (1, 2):
        x, y = probe_batch(step, rows, width, classes)
        masks = tuple(torch.ones(rows, *shape) for shape in mask_shapes)
        opt.zero_grad()
        loss, _ = wrapper(x, y, masks)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return {"rows": rows, "width": width, "classes": classes, "learning_rate": PROBE_LEARNING_RATE,
            "loss_step1": losses[0], "loss_step2": losses[1],
            # ExecuTorch's portable kernels agree with torch to ~1e-7 on these graphs; the tolerance leaves room for
            # other CPUs while rejecting a program that computes something else.
            "loss_tolerance": 1e-4}
