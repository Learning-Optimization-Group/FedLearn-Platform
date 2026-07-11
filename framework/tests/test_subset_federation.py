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
    good = OrderedDict([
        ("head.weight", torch.zeros_like(net.head.weight)),
        ("head.bias", torch.zeros_like(net.head.bias)),
    ])
    validate_subset_update(good, net)  # no raise
    with pytest.raises(SubsetDimMismatch):
        validate_subset_update(OrderedDict([("head.weight", torch.zeros_like(net.head.weight))]), net)  # missing key
    with pytest.raises(SubsetDimMismatch):
        validate_subset_update(OrderedDict([
            ("backbone.0.weight", torch.zeros_like(net.backbone[0].weight)),
            ("head.weight", torch.zeros_like(net.head.weight)),
            ("head.bias", torch.zeros_like(net.head.bias)),
        ]), net)  # extra/frozen key


def test_guard_rejects_same_keys_different_order():
    """Minor: cardinality alone isn't enough -- a REORDERED same-key-set update must also be
    rejected, since the wire layout is order-sensitive (estimators.params.trainable_state order)."""
    net = build_tiny_frozen_net(seed=0)
    reordered = OrderedDict([
        ("head.bias", torch.zeros_like(net.head.bias)),
        ("head.weight", torch.zeros_like(net.head.weight)),
    ])
    with pytest.raises(SubsetDimMismatch):
        validate_subset_update(reordered, net)


def test_guard_rejects_same_keys_wrong_shape_with_typed_error():
    """FINDING 2: a same-key update with a WRONG SHAPE (e.g. a differently-sized head from a
    misconfigured client) must raise the typed SubsetDimMismatch -- not fall through to a raw
    untyped RuntimeError out of load_state_dict."""
    net = build_tiny_frozen_net(seed=0)  # head.weight is [3, 2], head.bias is [3]
    wrong_shape = OrderedDict([
        ("head.weight", torch.zeros(5, 2)),   # server expects [3, 2]
        ("head.bias", torch.zeros(3)),
    ])
    with pytest.raises(SubsetDimMismatch):
        validate_subset_update(wrong_shape, net)


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
from fedlearn.server.subset_federation import guard_client_updates


def test_guard_client_updates_rejects_second_client_missing_a_key():
    """FINDING 1: FedAvgAggregator.aggregate() builds its key-set from the FIRST client and
    silently skips (`if key in params`) any key a LATER client lacks -- no error. The guard must
    therefore run on each client's RAW payload before aggregation, so a non-first client missing a
    trainable key is REJECTED, never silently averaged."""
    server = build_tiny_frozen_net(seed=0)
    good = trainable_state(build_tiny_frozen_net(seed=0))
    bad_missing_bias = OrderedDict([("head.weight", good["head.weight"].clone())])  # no head.bias
    with pytest.raises(SubsetDimMismatch):
        guard_client_updates([good, bad_missing_bias], server)


def test_guard_client_updates_rejects_second_client_extra_key():
    """FINDING 1, extra-key variant: a later client sending a key the server doesn't expect (e.g.
    a frozen backbone param sneaking onto the wire) must also be rejected pre-aggregation."""
    server = build_tiny_frozen_net(seed=0)
    good = trainable_state(build_tiny_frozen_net(seed=0))
    bad_extra_key = OrderedDict(good)
    bad_extra_key["backbone.0.weight"] = torch.zeros_like(server.backbone[0].weight)
    with pytest.raises(SubsetDimMismatch):
        guard_client_updates([good, bad_extra_key], server)


def test_guard_client_updates_accepts_all_matching_clients():
    server = build_tiny_frozen_net(seed=0)
    a = trainable_state(build_tiny_frozen_net(seed=0))
    b = trainable_state(build_tiny_frozen_net(seed=0))
    guard_client_updates([a, b], server)  # no raise


def test_vertical_slice_payload_is_head_only_and_round_averages():
    server = build_tiny_frozen_net(seed=0)
    expected = expected_trainable_keys(server)

    # Two clients start from the server's frozen backbone; each has a different head.
    def client_update(cid, head_fill, num_examples):
        net = build_tiny_frozen_net(seed=0)
        with torch.no_grad():
            net.head.weight.fill_(head_fill); net.head.bias.fill_(head_fill)
        payload = trainable_state(net)                       # the wire payload
        # (a) payload is HEAD-ONLY — the frozen backbone is NOT on the wire
        assert list(payload.keys()) == expected
        assert all(not k.startswith("backbone.") for k in payload)
        assert sum(t.numel() for t in payload.values()) == num_trainable(net)  # << full model
        return (cid, payload, num_examples)

    updates = [client_update("client-a", 1.0, 10), client_update("client-b", 3.0, 10)]
    # Per-client guard runs BEFORE aggregation (FINDING 1) -- this is the real fail-loud check;
    # validating the AGGREGATED output afterward (the old assertion here) is redundant with what
    # apply_trainable_subset already does internally, and can never catch a non-first client's
    # bad payload since FedAvgAggregator's key-set always mirrors the FIRST client.
    guard_client_updates([payload for _, payload, _ in updates], server)
    aggregated = FedAvgAggregator().aggregate(updates)       # averages the subset only
    apply_trainable_subset(server, aggregated)               # reconstruct on the frozen backbone

    # equal weights -> head averages to 2.0; backbone untouched (still frozen, non-trainable)
    assert torch.allclose(server.head.weight, torch.full_like(server.head.weight, 2.0))
    assert torch.allclose(server.head.bias, torch.full_like(server.head.bias, 2.0))
    assert not any(p.requires_grad for n, p in server.named_parameters() if n.startswith("backbone."))
