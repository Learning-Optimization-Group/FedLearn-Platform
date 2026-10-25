"""Constrained binary-head ascent using only owned examples and public releases."""
import math

import torch
from torch.nn import functional as F


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def _tensor(value, name):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" \
            or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite floating CPU tensor")
    return value.detach().double().clone()


def ascent_upload(*, weight, bias, owned_x, owned_y, initial_delta, clip_norm,
                  coalition_fraction, seed, released_delta=None, steps=20,
                  restarts=3, step_ratio=.25):
    """Maximize owned-data loss of a mean-update surrogate within one L2 ball.

    The forecast is the previous released delta, held fixed during search.
    Only coalition-owned examples and public model/history may be supplied.
    This is a heuristic candidate, not an optimal attack or engine integration.
    The caller must enforce that information boundary and map the returned
    weight-then-bias vector to the actual upload representation.
    """
    _integer(seed, "seed", 0)
    _integer(steps, "steps")
    _integer(restarts, "restarts")
    for value, name in ((clip_norm, "clip_norm"), (coalition_fraction, "coalition_fraction"),
                        (step_ratio, "step_ratio")):
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if coalition_fraction > 1:
        raise ValueError("coalition_fraction exceeds one")
    weight, bias = _tensor(weight, "weight"), _tensor(bias, "bias")
    x = _tensor(owned_x, "owned_x")
    initial = _tensor(initial_delta, "initial_delta")
    if weight.ndim != 2 or weight.shape[0] != 2 or weight.shape[1] < 1 \
            or bias.shape != (2,) or x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != weight.shape[1]:
        raise ValueError("expected a binary linear head and nonempty matching owned features")
    if not isinstance(owned_y, torch.Tensor) or owned_y.device.type != "cpu" \
            or owned_y.dtype != torch.int64 or owned_y.shape != (x.shape[0],) \
            or int(owned_y.min()) < 0 or int(owned_y.max()) > 1:
        raise ValueError("owned_y must contain matching binary int64 labels")
    y = owned_y.detach().clone()
    parameters = torch.cat((weight.reshape(-1), bias))
    if initial.shape != parameters.shape:
        raise ValueError("initial_delta must match flattened weight then bias")
    history = torch.zeros_like(parameters) if released_delta is None else _tensor(released_delta, "released_delta")
    if history.shape != parameters.shape:
        raise ValueError("released_delta must match flattened parameters")
    base = parameters + (1. - coalition_fraction) * history
    generator = torch.Generator().manual_seed(seed)

    def project(delta):
        norm = torch.linalg.vector_norm(delta)
        return delta * torch.clamp(torch.tensor(float(clip_norm), dtype=torch.float64) / norm.clamp_min(1e-300), max=1.)

    best, best_loss, initial_loss = None, -math.inf, None
    for restart in range(restarts):
        delta = project(initial) if restart == 0 else project(torch.randn(parameters.shape, dtype=torch.float64, generator=generator))
        for step in range(steps + 1):
            delta = delta.detach().requires_grad_(step < steps)
            candidate = base + coalition_fraction * delta
            logits = x @ candidate[:weight.numel()].reshape(weight.shape).T + candidate[weight.numel():]
            loss = F.cross_entropy(logits, y)
            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError("nonfinite attack objective")
            if initial_loss is None:
                initial_loss = value
            if value > best_loss:
                best, best_loss = delta.detach().clone(), value
            if step < steps:
                gradient, = torch.autograd.grad(loss, delta)
                if not torch.isfinite(gradient).all():
                    raise ValueError("nonfinite attack gradient")
                direction = gradient / torch.linalg.vector_norm(gradient).clamp_min(1e-300)
                delta = project(delta.detach() + float(step_ratio) * float(clip_norm) * direction)
    return {"delta": best, "objective_before": initial_loss, "objective_after": best_loss,
            "loss_evaluations": restarts * (steps + 1), "gradient_evaluations": restarts * steps,
            "upload_norm": float(torch.linalg.vector_norm(best)),
            "knowledge": "coalition-owned examples and previously released model/history only",
            "surrogate": "uniform mean; previous released delta forecasts benign contribution"}
