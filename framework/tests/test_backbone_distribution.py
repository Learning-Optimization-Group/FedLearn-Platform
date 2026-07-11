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


import hashlib

from fedlearn.backbone.distribution import serialize_backbone, backbone_sha256
from fedlearn.communication.safetensors_codec import load_safetensors


def test_serialize_backbone_is_byte_deterministic_and_content_addressed():
    net_a = build_tiny_frozen_net(seed=0)
    net_b = build_tiny_frozen_net(seed=0)  # same seed -> identical frozen backbone
    blob_a = serialize_backbone(net_a)
    blob_b = serialize_backbone(net_b)
    assert blob_a == blob_b                      # byte-identical
    assert backbone_sha256(blob_a) == backbone_sha256(blob_b)
    assert backbone_sha256(blob_a) == hashlib.sha256(blob_a).hexdigest()


def test_serialize_backbone_differs_when_frozen_weights_differ():
    net0 = build_tiny_frozen_net(seed=0)
    net1 = build_tiny_frozen_net(seed=1)  # different frozen backbone
    assert backbone_sha256(serialize_backbone(net0)) != backbone_sha256(serialize_backbone(net1))


def test_serialize_backbone_roundtrips_to_frozen_tensors():
    net = build_tiny_frozen_net(seed=0)
    tensors, meta = load_safetensors(serialize_backbone(net))
    names = [n for n, _ in tensors]
    assert names == ["backbone.0.weight", "backbone.0.bias"]  # frozen only, in order


import pytest

from fedlearn.backbone.distribution import BackboneCache, BackboneIntegrityError


def test_cache_fetches_once_then_serves_from_disk(tmp_path):
    net = build_tiny_frozen_net(seed=0)
    blob = serialize_backbone(net)
    sha = backbone_sha256(blob)
    cache = BackboneCache(tmp_path)
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return blob

    p1 = cache.get_or_fetch(sha, fetch)
    assert p1.exists() and p1.read_bytes() == blob
    assert calls["n"] == 1
    p2 = cache.get_or_fetch(sha, fetch)   # hit -> fetch NOT called again
    assert p2 == p1
    assert calls["n"] == 1


def test_cache_rejects_hash_mismatch_and_writes_nothing(tmp_path):
    net = build_tiny_frozen_net(seed=0)
    blob = serialize_backbone(net)
    wrong_sha = backbone_sha256(serialize_backbone(build_tiny_frozen_net(seed=1)))
    cache = BackboneCache(tmp_path)
    with pytest.raises(BackboneIntegrityError):
        cache.get_or_fetch(wrong_sha, lambda: blob)  # bytes hash != requested key
    assert not (tmp_path / wrong_sha).exists()        # nothing partial left behind
    assert list(tmp_path.iterdir()) == []


def test_cache_self_heals_a_corrupted_cache_file(tmp_path):
    net = build_tiny_frozen_net(seed=0)
    blob = serialize_backbone(net)
    sha = backbone_sha256(blob)
    cache = BackboneCache(tmp_path)
    (tmp_path / sha).write_bytes(b"corrupted-not-the-backbone")  # wrong bytes on disk
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return blob

    p = cache.get_or_fetch(sha, fetch)   # detects bad on-disk bytes, re-fetches, overwrites
    assert p.read_bytes() == blob
    assert calls["n"] == 1
