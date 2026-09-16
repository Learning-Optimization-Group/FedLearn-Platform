"""DA-14 Ph3.0 — activation vertical slice for the DA-11 frozen-backbone contract.

DA-11 shipped three framework modules that are unit-tested in isolation but imported by NO live
path (dormant): estimators.params (the trainable/frozen manifest), backbone.distribution (the
content-addressed frozen-backbone wire), and server.subset_federation (the fail-loud head-only guard
+ non-strict apply). This module proves they COMPOSE end-to-end through ONE real trainable-subset
federation round wired to the ACTUAL FedAvg strategy aggregator (server.strategy.FedAvgAggregator),
not a hand-rolled mean — the piece the existing test_backbone_distribution.py e2e leaves out.

The model is a deliberately backbone-dominant TinyDerived (Linear 256->128 frozen backbone, Linear
128->4 trainable head) so the head-only wire's comms win is measurable, not asserted by fiat.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from fedlearn.backbone.distribution import (
    BackboneKeyMismatch,
    reconstruct_frozen_backbone,
    serialize_backbone,
)
from fedlearn.communication.safetensors_codec import save_safetensors
from fedlearn.estimators.params import frozen_state, trainable_state
from fedlearn.server.strategy import FedAvgAggregator
from fedlearn.server.subset_federation import (
    SubsetDimMismatch,
    apply_trainable_subset,
    expected_trainable_keys,
    guard_client_updates,
)

import pytest


class TinyDerived(nn.Module):
    """A deterministic tiny model with a clearly separable frozen backbone + trainable head, sized so
    the backbone (32 896 params) dominates the head (516 params) — the comms-win demonstrator."""

    def __init__(self, d_in: int = 256, d_hidden: int = 128, n_classes: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Linear(d_in, d_hidden)
        self.head = nn.Linear(d_hidden, n_classes)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - not exercised in the round
        return self.head(torch.relu(self.backbone(x)))


def _build(seed: int, n_classes: int = 4) -> TinyDerived:
    """A TinyDerived whose (random) weights are a deterministic function of ``seed``."""
    torch.manual_seed(seed)
    return TinyDerived(n_classes=n_classes)


def _serialize_state(state: "OrderedDict[str, torch.Tensor]") -> bytes:
    """Serialize an arbitrary float32 state_dict via the SAME deterministic codec serialize_backbone
    uses — so a head-vs-backbone byte comparison is apples-to-apples (same header/data framing)."""
    return save_safetensors([(name, t.cpu().numpy()) for name, t in state.items()])


# ---------------------------------------------------------------------------------------------------
# 1. Backbone-distribution leg: a fresh client reconstructs the reference frozen backbone.
# ---------------------------------------------------------------------------------------------------
def test_backbone_distribution_leg_reconstructs_and_refreezes():
    ref = _build(seed=7)                       # the reference model that owns the backbone
    blob = serialize_backbone(ref)

    fresh = _build(seed=99)                     # a different random init (different backbone + head)
    # Precondition: the fresh backbone genuinely differs before reconstruction (nothing to prove
    # otherwise).
    assert not torch.equal(fresh.backbone.weight.detach(), ref.backbone.weight.detach())

    returned = reconstruct_frozen_backbone(fresh, blob)
    assert returned is fresh                    # loads in place, returns the same module

    # Backbone is now byte-identical to the reference...
    assert torch.equal(fresh.backbone.weight.detach(), ref.backbone.weight.detach())
    assert torch.equal(fresh.backbone.bias.detach(), ref.backbone.bias.detach())
    # ...and re-frozen.
    assert fresh.backbone.weight.requires_grad is False
    assert fresh.backbone.bias.requires_grad is False
    # The head is untouched and still trainable — the only federated subset.
    assert fresh.head.weight.requires_grad is True
    assert fresh.head.bias.requires_grad is True
    # The declared frozen layout is exactly the backbone keys.
    assert set(frozen_state(fresh).keys()) == {"backbone.weight", "backbone.bias"}


# ---------------------------------------------------------------------------------------------------
# 2. Head-only wire payload: the trainable subset is the head, and it is far smaller than the backbone.
# ---------------------------------------------------------------------------------------------------
def test_head_only_payload_is_far_smaller_than_backbone():
    model = _build(seed=0)

    # The wire payload is the head only — never the backbone.
    assert set(trainable_state(model).keys()) == {"head.weight", "head.bias"}
    assert set(expected_trainable_keys(model)) == {"head.weight", "head.bias"}

    head_bytes = len(_serialize_state(trainable_state(model)))
    backbone_bytes = len(serialize_backbone(model))
    ratio = backbone_bytes / head_bytes
    print(f"\n[DA-14 Ph3.0] backbone={backbone_bytes} B  head={head_bytes} B  ratio={ratio:.1f}x")

    # The head is an order of magnitude smaller than the backbone (comms win of freezing the backbone).
    assert head_bytes * 10 < backbone_bytes


# ---------------------------------------------------------------------------------------------------
# 3. Federation round: two clients, real FedAvgAggregator, frozen backbone survives.
# ---------------------------------------------------------------------------------------------------
def test_one_round_head_federation_via_fedavg_preserves_frozen_backbone():
    # Server registers a frozen backbone; every participant reconstructs the SAME one.
    ref = _build(seed=7)
    blob = serialize_backbone(ref)

    def participant(seed: int) -> TinyDerived:
        net = _build(seed=seed)                 # fresh head + wrong backbone until reconstruct
        reconstruct_frozen_backbone(net, blob)
        return net

    c0, c1, server_model = participant(11), participant(22), participant(33)

    # Each client sets a DISTINCT head, then exports its head-only trainable subset.
    with torch.no_grad():
        c0.head.weight.fill_(0.25)
        c0.head.bias.fill_(0.1)
        c1.head.weight.fill_(0.75)
        c1.head.bias.fill_(0.3)
    u0, u1 = trainable_state(c0), trainable_state(c1)

    # Per-client fail-loud guard passes BEFORE aggregation (validated against the server layout).
    guard_client_updates([u0, u1], server_model)

    # Snapshot what the mean SHOULD be (aggregate() clears client dicts as it consumes them) and the
    # frozen backbone the round must leave intact.
    expected_head_w = (u0["head.weight"].clone() + u1["head.weight"].clone()) / 2
    expected_head_b = (u0["head.bias"].clone() + u1["head.bias"].clone()) / 2
    backbone_w_before = server_model.backbone.weight.detach().clone()
    backbone_b_before = server_model.backbone.bias.detach().clone()

    # The ACTUAL FedAvg strategy aggregator — equal num_examples => simple mean.
    agg = FedAvgAggregator().aggregate([("c0", u0, 10), ("c1", u1, 10)])
    assert set(agg.keys()) == {"head.weight", "head.bias"}   # head keys only, no backbone leaked in

    apply_trainable_subset(server_model, agg)

    # Head is the averaged head...
    assert torch.allclose(server_model.head.weight.detach(), expected_head_w)
    assert torch.allclose(server_model.head.bias.detach(), expected_head_b)
    # ...and the frozen backbone is byte-identical before and after (it never rode the wire).
    assert torch.equal(server_model.backbone.weight.detach(), backbone_w_before)
    assert torch.equal(server_model.backbone.bias.detach(), backbone_b_before)
    assert server_model.backbone.weight.requires_grad is False


# ---------------------------------------------------------------------------------------------------
# 4. Fail-loud: a wrong-shape head and a truncated backbone blob are both rejected, not silently loaded.
# ---------------------------------------------------------------------------------------------------
def test_guard_rejects_wrong_shape_head_client():
    server_model = _build(seed=7, n_classes=4)          # server expects a 4-class head
    bad_client = _build(seed=7, n_classes=5)            # 5-class head -> head.* shapes differ
    bad_update = trainable_state(bad_client)
    with pytest.raises(SubsetDimMismatch):
        guard_client_updates([bad_update], server_model)


def test_reconstruct_rejects_truncated_backbone_blob():
    model = _build(seed=7)
    # A blob missing backbone.bias — a truncated/wrong artifact, not the model's declared frozen layout.
    truncated = save_safetensors(
        [("backbone.weight", model.backbone.weight.detach().cpu().numpy())]
    )
    with pytest.raises(BackboneKeyMismatch):
        reconstruct_frozen_backbone(model, truncated)


# ---------------------------------------------------------------------------------------------------
# 5. Wire-through: the SAME round survives the REAL unary proto transport (the one that scrambles key
#    order). The other round test averages in-memory dicts still in named_parameters order, so it
#    cannot catch a transport that reorders keys; a small head takes the unary path
#    (_submit_update_unary -> parameters_to_proto), whose map<string,Tensor> iterates unordered.
# ---------------------------------------------------------------------------------------------------
def test_one_round_head_federation_survives_unary_proto_wire():
    from fedlearn.communication.serializer import parameters_to_proto, proto_to_parameters

    ref = _build(seed=7)
    blob = serialize_backbone(ref)

    def participant(seed: int) -> TinyDerived:
        net = _build(seed=seed)
        reconstruct_frozen_backbone(net, blob)
        return net

    c0, c1, server_model = participant(11), participant(22), participant(33)
    with torch.no_grad():
        c0.head.weight.fill_(0.25); c0.head.bias.fill_(0.1)
        c1.head.weight.fill_(0.75); c1.head.bias.fill_(0.3)

    # Each client's head-only update round-trips through the ACTUAL unary proto wire before the server
    # ever sees it — exactly what _submit_update_unary does. The received dicts come back in protobuf
    # map order, NOT named_parameters order.
    def over_the_wire(state):
        recv, _ = proto_to_parameters(parameters_to_proto(state, num_examples=10))
        return recv

    u0 = over_the_wire(trainable_state(c0))
    u1 = over_the_wire(trainable_state(c1))

    expected_head_w = (u0["head.weight"].clone() + u1["head.weight"].clone()) / 2
    expected_head_b = (u0["head.bias"].clone() + u1["head.bias"].clone()) / 2
    backbone_w_before = server_model.backbone.weight.detach().clone()

    # The guard must accept the reordered updates (pre-fix it raised SubsetDimMismatch here, wedging
    # the round); aggregation is by-name so the scrambled order is harmless.
    guard_client_updates([u0, u1], server_model)
    agg = FedAvgAggregator().aggregate([("c0", u0, 10), ("c1", u1, 10)])
    assert set(agg.keys()) == {"head.weight", "head.bias"}
    apply_trainable_subset(server_model, agg)

    assert torch.allclose(server_model.head.weight.detach(), expected_head_w)
    assert torch.allclose(server_model.head.bias.detach(), expected_head_b)
    # Frozen backbone untouched — it never rode the wire.
    assert torch.equal(server_model.backbone.weight.detach(), backbone_w_before)
    assert server_model.backbone.weight.requires_grad is False
