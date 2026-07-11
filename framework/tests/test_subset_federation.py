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
