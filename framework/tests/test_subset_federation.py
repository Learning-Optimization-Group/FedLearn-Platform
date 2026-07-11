import torch
from tests.fixtures.tiny_frozen_model import build_tiny_frozen_net


def test_tiny_frozen_net_has_frozen_backbone_and_trainable_head():
    net = build_tiny_frozen_net(seed=0)
    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    frozen = {n for n, p in net.named_parameters() if not p.requires_grad}
    assert trainable == {"head.weight", "head.bias"}, trainable
    assert all(n.startswith("backbone.") for n in frozen), frozen
    assert frozen, "backbone must contribute frozen params"
