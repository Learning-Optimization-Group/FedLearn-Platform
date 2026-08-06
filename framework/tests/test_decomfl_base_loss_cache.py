"""Forward-difference base-loss caching (DeComFL Algorithm 4, lines 16-21).

Within one local step k the base point x and the batch xi are both fixed — only the perturbation
z varies across the P perturbations. So f(x; xi) is the same number every time and must be
evaluated ONCE per local step, not once per perturbation. The authors' reference implementation
hoists it out of the perturbation loop for the forward-difference method
(https://github.com/ZidongLiu/DeComFL, `pert_minus_loss` computed before `for i in range(num_pert)`).

Recomputing it per perturbation costs 2P forward passes where the algorithm needs P+1 — a 1.9x
waste at P=20 that inflates every measured DeComFL compute/latency number. These tests pin both
halves of the fix: the cost drops to P+1, and the gradient scalars are bit-identical (the base
loss is deterministic under model.eval() + no_grad, so caching is not an approximation).
"""

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from fedlearn.client.decomfl_client import DeComFLClient
from fedlearn.estimators.perturbation import canonical_perturbation
from fedlearn.estimators.zeroth_order import ZerothOrderEstimator


class CountingHead(nn.Module):
    """Linear head that counts forward passes — the quantity the fix is about."""

    def __init__(self, in_features: int = 8, out_features: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return self.fc(x)


def _loader(n: int = 16, in_features: int = 8, batch_size: int = 16):
    torch.manual_seed(0)
    xs = torch.randn(n, in_features)
    ys = torch.randint(0, 2, (n,))
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xs, ys), batch_size=batch_size
    )


def _seeds(K: int, P: int):
    return [[1000 + k * 100 + p for p in range(P)] for k in range(K)]


@pytest.mark.parametrize("K,P", [(1, 1), (1, 10), (1, 20), (5, 10)])
def test_fit_costs_p_plus_one_forward_passes_per_local_step(K, P):
    model = CountingHead()
    client = DeComFLClient(model, _loader(), smoothing_param=1e-3, device="cpu")

    model.forward_calls = 0
    client.fit(
        OrderedDict(),
        {"seeds": _seeds(K, P), "learning_rate": 1e-3, "smoothing_param": 1e-3},
    )

    assert model.forward_calls == K * (P + 1), (
        f"DeComFL fit with K={K}, P={P} used {model.forward_calls} forward passes; the "
        f"forward-difference algorithm needs {K * (P + 1)} (one base loss per local step plus "
        f"one perturbed loss per perturbation). {K * 2 * P} means f(x) is being recomputed "
        f"inside the perturbation loop."
    )


def test_cached_base_loss_gives_bit_identical_scalars():
    """Caching must be a pure speedup: identical scalars, not an approximation."""
    torch.manual_seed(0)
    model = nn.Linear(8, 2)
    est = ZerothOrderEstimator(smoothing_param=1e-3, device="cpu")
    flat = est._get_flat_params(model)
    inputs = torch.randn(16, 8)
    targets = torch.randint(0, 2, (16,))

    base_loss = est.compute_base_loss(model, flat, inputs, targets)

    for p in range(10):
        z = canonical_perturbation(2000 + p, flat.numel())
        recomputed = est.compute_gradient_scalar(model, flat, z, inputs, targets)
        cached = est.compute_gradient_scalar(
            model, flat, z, inputs, targets, base_loss=base_loss
        )
        assert cached == recomputed, (
            f"perturbation {p}: cached base loss changed the gradient scalar "
            f"({cached!r} != {recomputed!r}); it must be bit-identical"
        )


def test_base_loss_is_recomputed_between_local_steps():
    """The cache is only valid WITHIN a step — x advances between steps, so f(x) must be re-read.

    A single base loss reused across all K steps would be a real correctness bug, and would show
    up as K*P+1 forward passes instead of K*(P+1).
    """
    K, P = 4, 3
    model = CountingHead()
    client = DeComFLClient(model, _loader(), smoothing_param=1e-3, device="cpu")

    model.forward_calls = 0
    client.fit(
        OrderedDict(),
        {"seeds": _seeds(K, P), "learning_rate": 1e-3, "smoothing_param": 1e-3},
    )

    assert model.forward_calls != K * P + 1, (
        "base loss appears to be cached ACROSS local steps; x changes at every step so f(x) "
        "must be re-evaluated once per step"
    )
    assert model.forward_calls == K * (P + 1)
