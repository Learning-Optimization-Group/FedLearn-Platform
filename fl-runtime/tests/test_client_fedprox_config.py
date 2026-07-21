"""FR-32: the production client (fl-runtime/client.py, ZOSLClient) now HONORS FedProx — it applies the
proximal-term gradient ``mu * (w - w_global)`` during local training (Li et al. 2020, "Federated
Optimization in Heterogeneous Networks"), instead of REFUSING FedProx as it did under FR-20. FedProx's
entire client-side difference from FedAvg is that proximal term, added to each trainable param's
gradient after loss.backward() and before optimizer.step(), with w_global the round-start snapshot.

This pins (a) the string->int local_epochs coercion (the gRPC config is map<string,string>) and
(b) the proximal-gradient application — mirroring the framework's proven LocalTrainer reference.
"""
import pytest
import torch

import client


def test_local_epochs_coerced_from_proto_string():
    # str->str proto map: local_epochs arrives as a string; range() would crash without coercion.
    assert client._coerce_local_epochs({"local_epochs": "3"}, 1) == 3


def test_local_epochs_default_and_int_passthrough():
    assert client._coerce_local_epochs({}, 5) == 5
    assert client._coerce_local_epochs({"local_epochs": 2}, 1) == 2


def test_local_epochs_rejects_non_numeric():
    with pytest.raises(ValueError):
        client._coerce_local_epochs({"local_epochs": "not-a-number"}, 1)


def _seed_grads(model: torch.nn.Module, grad_value: float) -> torch.nn.Module:
    for p in model.parameters():
        p.grad = torch.full_like(p, grad_value) if p.requires_grad else None
    return model


def test_proximal_gradient_adds_mu_times_weight_minus_global():
    # d/dw (mu/2)||w - w0||^2 = mu*(w - w0). With w0 = w - 1 => (w - w0) = 1, so a zero grad becomes
    # exactly mu. This is the whole FedProx client-side contribution.
    net = torch.nn.Linear(3, 2)
    global_params = [p.detach().clone() - 1.0 for p in net.parameters()]
    _seed_grads(net, 0.0)
    client._apply_proximal_gradient(net, global_params, mu=0.5)
    for p in net.parameters():
        assert torch.allclose(p.grad, torch.full_like(p.grad, 0.5)), "grad must be 0 + mu*(w - w0)"


def test_proximal_gradient_zero_mu_is_noop():
    # FedAvg / FedOpt: mu == 0 => no proximal term, grads untouched (the client trains plainly).
    net = torch.nn.Linear(3, 2)
    global_params = [p.detach().clone() - 1.0 for p in net.parameters()]
    _seed_grads(net, 1.0)
    client._apply_proximal_gradient(net, global_params, mu=0.0)
    for p in net.parameters():
        assert torch.allclose(p.grad, torch.ones_like(p.grad)), "mu=0 must leave grads unchanged"


def test_proximal_gradient_skips_frozen_params_without_grad():
    # Frozen params (requires_grad=False) have grad None; the term must skip them, never crash — e.g.
    # the derived frozen-backbone / LoRA models where only the head trains.
    net = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 2))
    for p in net[1].parameters():
        p.requires_grad_(False)
    global_params = [p.detach().clone() - 1.0 for p in net.parameters()]
    _seed_grads(net, 0.0)
    client._apply_proximal_gradient(net, global_params, mu=0.5)
    assert net[1].weight.grad is None and net[1].bias.grad is None
    assert torch.allclose(net[0].weight.grad, torch.full_like(net[0].weight.grad, 0.5))
