import torch
from tests.fixtures.tiny_frozen_model import build_tiny_frozen_net


def test_tiny_frozen_net_has_frozen_backbone_and_trainable_head():
    net = build_tiny_frozen_net(seed=0)
    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    frozen = {n for n, p in net.named_parameters() if not p.requires_grad}
    assert trainable == {"head.weight", "head.bias"}, trainable
    assert all(n.startswith("backbone.") for n in frozen), frozen
    assert frozen, "backbone must contribute frozen params"


import pytest
from fedlearn.server.subset_federation import (
    expected_trainable_keys, validate_subset_update, SubsetDimMismatch,
)


def test_expected_keys_are_trainable_only_in_order():
    net = build_tiny_frozen_net(seed=0)
    assert expected_trainable_keys(net) == ["head.weight", "head.bias"]


def test_guard_accepts_matching_keys_and_rejects_mismatch():
    net = build_tiny_frozen_net(seed=0)
    expected = expected_trainable_keys(net)
    validate_subset_update(["head.weight", "head.bias"], expected)  # no raise
    with pytest.raises(SubsetDimMismatch):
        validate_subset_update(["head.weight"], expected)            # missing key
    with pytest.raises(SubsetDimMismatch):
        validate_subset_update(["backbone.0.weight", "head.weight", "head.bias"], expected)  # extra/frozen key


from collections import OrderedDict
from fedlearn.server.subset_federation import apply_trainable_subset


def test_apply_subset_updates_head_keeps_backbone_and_rejects_mismatch():
    net = build_tiny_frozen_net(seed=0)
    backbone_before = {n: p.clone() for n, p in net.named_parameters() if not p.requires_grad}
    new_head = OrderedDict([
        ("head.weight", torch.ones_like(net.head.weight)),
        ("head.bias", torch.full_like(net.head.bias, 2.0)),
    ])
    apply_trainable_subset(net, new_head)
    assert torch.equal(net.head.weight, torch.ones_like(net.head.weight))
    assert torch.equal(net.head.bias, torch.full_like(net.head.bias, 2.0))
    for n, p in net.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p, backbone_before[n]), f"frozen {n} changed"
    with pytest.raises(SubsetDimMismatch):
        apply_trainable_subset(net, OrderedDict([("head.weight", net.head.weight.clone())]))


from fedlearn.estimators.params import trainable_state, num_trainable
from fedlearn.server.strategy import FedAvgAggregator


def test_vertical_slice_payload_is_head_only_and_round_averages():
    server = build_tiny_frozen_net(seed=0)
    expected = expected_trainable_keys(server)

    # Two clients start from the server's frozen backbone; each has a different head.
    def client_update(head_fill, num_examples):
        net = build_tiny_frozen_net(seed=0)
        with torch.no_grad():
            net.head.weight.fill_(head_fill); net.head.bias.fill_(head_fill)
        payload = trainable_state(net)                       # the wire payload
        # (a) payload is HEAD-ONLY — the frozen backbone is NOT on the wire
        assert list(payload.keys()) == expected
        assert all(not k.startswith("backbone.") for k in payload)
        assert sum(t.numel() for t in payload.values()) == num_trainable(net)  # << full model
        return ("cid", payload, num_examples)

    updates = [client_update(1.0, 10), client_update(3.0, 10)]
    aggregated = FedAvgAggregator().aggregate(updates)       # averages the subset only
    validate_subset_update(list(aggregated.keys()), expected)
    apply_trainable_subset(server, aggregated)               # reconstruct on the frozen backbone

    # equal weights -> head averages to 2.0; backbone untouched (still frozen, non-trainable)
    assert torch.allclose(server.head.weight, torch.full_like(server.head.weight, 2.0))
    assert torch.allclose(server.head.bias, torch.full_like(server.head.bias, 2.0))
    assert not any(p.requires_grad for n, p in server.named_parameters() if n.startswith("backbone."))
