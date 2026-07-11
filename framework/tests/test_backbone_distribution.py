from collections import OrderedDict

import torch
import torch.nn as nn

from fedlearn.estimators.params import frozen_state, trainable_state
from tests.fixtures.tiny_frozen_model import build_tiny_frozen_net


def test_frozen_state_is_backbone_only_and_disjoint_from_trainable():
    net = build_tiny_frozen_net(seed=0)
    frozen = frozen_state(net)
    trainable = trainable_state(net)
    # Backbone conv params are frozen; the head is trainable.
    assert set(frozen.keys()) == {"backbone.0.weight", "backbone.0.bias"}
    assert set(trainable.keys()) == {"head.weight", "head.bias"}
    # Disjoint, and together they cover every parameter.
    assert set(frozen) & set(trainable) == set()
    all_params = {name for name, _ in net.named_parameters()}
    assert set(frozen) | set(trainable) == all_params
    # Values match the live tensors and are detached clones (mutation-safe).
    for name, p in net.named_parameters():
        if name in frozen:
            assert torch.equal(frozen[name], p.detach())
    assert frozen["backbone.0.weight"].requires_grad is False


def test_frozen_state_includes_float_buffers_excludes_integer_buffers():
    class BNBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(2)     # running_mean/var (float) + num_batches_tracked (int64)
            self.head = nn.Linear(2, 3)
        def forward(self, x):
            return self.head(self.bn(x).mean(dim=(2, 3)))

    net = BNBackbone()
    for p in net.bn.parameters():
        p.requires_grad_(False)
    frozen = frozen_state(net)
    # Float buffers are in; the int64 num_batches_tracked is out (not F32-wire, unused in eval).
    assert "bn.running_mean" in frozen
    assert "bn.running_var" in frozen
    assert "bn.num_batches_tracked" not in frozen
    # Frozen bn weight/bias present; trainable head absent.
    assert "bn.weight" in frozen and "bn.bias" in frozen
    assert "head.weight" not in frozen
    # Every emitted tensor is float dtype.
    assert all(t.is_floating_point() for t in frozen.values())


def test_frozen_state_preserves_named_order():
    net = build_tiny_frozen_net(seed=0)
    frozen = frozen_state(net)
    # Params emitted in named_parameters() order (weight before bias for the conv).
    assert list(frozen.keys()) == ["backbone.0.weight", "backbone.0.bias"]
    assert isinstance(frozen, OrderedDict)
