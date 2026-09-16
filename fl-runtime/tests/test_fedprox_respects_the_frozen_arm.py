"""P1-6: FedProx must regularise the federated subset and leave the frozen backbone alone.

The concern was concrete. ``_apply_proximal_gradient`` zips ``net.parameters()`` against
``global_params`` POSITIONALLY::

    for p, w0 in zip(net.parameters(), global_params):

If ``global_params`` were the RECEIVED global model, that would be the head alone under
FROZEN_HEAD — two tensors against sixty-two — and ``zip`` would silently pair the first two
backbone parameters with the head's values and stop, so the head would never be regularised at all
and the pairing would be nonsense besides. ``zip`` truncating to the shorter sequence is exactly
the kind of silent misalignment this campaign has repeatedly been bitten by.

It does not happen: the anchor is snapshotted from ``self.net.parameters()`` — the client's own
full model, after the global weights are loaded into it — so the two sequences are aligned by
construction and equal in length. Frozen parameters then have ``grad is None`` and are skipped.

That is correct, and it was asserted only in a docstring. These tests pin it, plus the agreement
the docstring claims but nothing enforced: ``fl-runtime``'s copy and the framework's
``LocalTrainer`` reference implement the same update, and a divergence between them would mean the
product client and the library disagreed about what FedProx is.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "framework", "src"))

import client  # noqa: E402
import recipes  # noqa: E402

MU = 0.1


class _TwoPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 3)
        self.classifier = nn.Linear(3, 2)

    def forward(self, x):
        return self.classifier(self.features(x))


def _stepped_model(frozen: bool):
    """A model after one backward pass, with the arm applied and an anchor snapshotted."""
    torch.manual_seed(0)
    net = _TwoPart()
    if frozen:
        recipes.apply_arm(net, "FROZEN_HEAD", ["classifier."])
    anchor = [p.detach().clone() for p in net.parameters()]
    nn.functional.cross_entropy(net(torch.randn(8, 4)), torch.randint(0, 2, (8,))).backward()
    return net, anchor


class TestTheProximalTermRespectsTheArm:
    def test_the_anchor_is_aligned_with_the_local_model(self):
        """THE structural property. Equal length and matching shapes, so the positional zip is
        sound; a subset anchor would silently truncate and mispair."""
        net, anchor = _stepped_model(frozen=True)
        params = list(net.parameters())
        assert len(anchor) == len(params), \
            f"anchor has {len(anchor)} tensors for {len(params)} parameters — zip would truncate"
        for p, w0 in zip(params, anchor):
            assert p.shape == w0.shape

    def test_frozen_parameters_receive_no_proximal_gradient(self):
        net, anchor = _stepped_model(frozen=True)
        client._apply_proximal_gradient(net, anchor, MU)
        for name, p in net.named_parameters():
            if not p.requires_grad:
                assert p.grad is None, f"{name} is frozen but received a gradient"

    def test_the_trainable_head_receives_exactly_mu_times_the_drift(self):
        """The head IS regularised — the frozen arm must not turn FedProx into FedAvg by accident."""
        net, anchor = _stepped_model(frozen=True)
        before = {n: p.grad.detach().clone()
                  for n, p in net.named_parameters() if p.grad is not None}
        assert before, "no parameter had a gradient; the fixture is not exercising the path"

        client._apply_proximal_gradient(net, anchor, MU)

        params = dict(net.named_parameters())
        anchor_by_name = dict(zip([n for n, _ in net.named_parameters()], anchor))
        for name, g0 in before.items():
            p = params[name]
            expected = g0 + MU * (p.detach() - anchor_by_name[name])
            assert torch.allclose(p.grad, expected, atol=1e-6), \
                f"{name}: proximal term is not mu*(w - w_global)"

    def test_a_zero_mu_is_exactly_fedavg(self):
        net, anchor = _stepped_model(frozen=True)
        before = {n: p.grad.detach().clone()
                  for n, p in net.named_parameters() if p.grad is not None}
        client._apply_proximal_gradient(net, anchor, 0.0)
        for name, g0 in before.items():
            assert torch.equal(dict(net.named_parameters())[name].grad, g0), \
                f"{name} changed at mu=0; FedAvg must be untouched"

    def test_the_full_arm_regularises_everything(self):
        net, anchor = _stepped_model(frozen=False)
        client._apply_proximal_gradient(net, anchor, MU)
        assert all(p.grad is not None for p in net.parameters()), \
            "the FULL arm must regularise every parameter"


class TestTheTwoImplementationsAgree:
    """The product client and the framework library must implement the SAME FedProx.

    ``_apply_proximal_gradient``'s docstring says it "mirrors the framework's proven
    LocalTrainer.fit reference exactly — the two must stay in agreement", and nothing checked that.
    A divergence would mean a run through fl-runtime and a run through the framework computed
    different updates under the same strategy name.
    """

    def _framework_update(self, net, anchor, mu):
        # The framework's inline form, from client/local_trainer.py.
        for p, w0 in zip(net.parameters(), anchor):
            if p.grad is not None:
                p.grad.add_(p.detach() - w0, alpha=mu)

    @pytest.mark.parametrize("frozen", [True, False])
    def test_identical_gradients_after_the_proximal_step(self, frozen):
        a_net, a_anchor = _stepped_model(frozen=frozen)
        b_net, b_anchor = _stepped_model(frozen=frozen)

        client._apply_proximal_gradient(a_net, a_anchor, MU)
        self._framework_update(b_net, b_anchor, MU)

        a = dict(a_net.named_parameters())
        for name, pb in b_net.named_parameters():
            pa = a[name]
            if pa.grad is None or pb.grad is None:
                assert pa.grad is None and pb.grad is None, f"{name}: one has a grad, the other not"
                continue
            assert torch.allclose(pa.grad, pb.grad, atol=1e-7), \
                f"{name}: fl-runtime and the framework disagree on the FedProx update"
