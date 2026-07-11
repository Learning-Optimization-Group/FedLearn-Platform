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


from fedlearn.backbone.distribution import reconstruct_frozen_backbone, BackboneKeyMismatch


def test_reconstruct_loads_backbone_freezes_it_and_leaves_head_trainable():
    source = build_tiny_frozen_net(seed=7)          # the registered backbone
    blob = serialize_backbone(source)
    target = build_tiny_frozen_net(seed=99)         # a fresh client net, different backbone weights
    # Precondition: the two nets' backbones differ before reconstruction.
    assert not torch.equal(
        dict(target.named_parameters())["backbone.0.weight"].detach(),
        dict(source.named_parameters())["backbone.0.weight"].detach(),
    )
    reconstruct_frozen_backbone(target, blob)
    # Backbone now byte-identical to the source; frozen.
    for name, p in target.named_parameters():
        if name.startswith("backbone."):
            assert torch.equal(p.detach(), dict(source.named_parameters())[name].detach())
            assert p.requires_grad is False
    # Head untouched + still trainable; only the head is the federated (trainable) subset.
    assert dict(target.named_parameters())["head.weight"].requires_grad is True
    assert set(trainable_state(target).keys()) == {"head.weight", "head.bias"}


def test_reconstruct_rejects_unexpected_key():
    target = build_tiny_frozen_net(seed=0)
    # A blob carrying a key the model's frozen layout does not declare.
    from fedlearn.communication.safetensors_codec import save_safetensors
    import numpy as np
    bad = save_safetensors([("backbone.0.weight", np.zeros((2, 1, 3, 3), dtype="<f4")),
                            ("backbone.0.bias", np.zeros((2,), dtype="<f4")),
                            ("backbone.ghost", np.zeros((1,), dtype="<f4"))])
    with pytest.raises(BackboneKeyMismatch):
        reconstruct_frozen_backbone(target, bad)


def test_reconstruct_rejects_missing_key():
    target = build_tiny_frozen_net(seed=0)
    from fedlearn.communication.safetensors_codec import save_safetensors
    import numpy as np
    incomplete = save_safetensors([("backbone.0.weight", np.zeros((2, 1, 3, 3), dtype="<f4"))])  # missing bias
    with pytest.raises(BackboneKeyMismatch):
        reconstruct_frozen_backbone(target, incomplete)


from fedlearn.server.subset_federation import (
    expected_trainable_keys,
    guard_client_updates,
    apply_trainable_subset,
)


def _average_subset(payloads):
    keys = list(payloads[0].keys())
    out = OrderedDict()
    for k in keys:
        out[k] = torch.stack([p[k] for p in payloads], dim=0).mean(dim=0)
    return out


def test_fetched_backbone_model_federates_head_only_and_backbone_survives(tmp_path):
    # 1. "Server" registers a frozen backbone as content-addressed bytes.
    server_net = build_tiny_frozen_net(seed=7)
    blob = serialize_backbone(server_net)
    sha = backbone_sha256(blob)
    store = {sha: blob}  # stand-in for the Java BASE_REF blob store (Phase 2B wires the real fetch)

    # 2. Each client fetches (verify + cache) and reconstructs a frozen-backbone model.
    def make_client(seed):
        cache = BackboneCache(tmp_path / f"c{seed}")
        path = cache.get_or_fetch(sha, lambda: store[sha])
        net = build_tiny_frozen_net(seed=seed)  # fresh head, wrong backbone until reconstruct
        reconstruct_frozen_backbone(net, path.read_bytes())
        return net

    clients = [make_client(11), make_client(22)]

    # 3. The federated wire payload is the HEAD ONLY (backbone excluded).
    payloads = []
    for net in clients:
        keys = expected_trainable_keys(net)
        assert keys == ["head.weight", "head.bias"]           # no backbone.* on the wire
        payloads.append(trainable_state(net))
    guard_client_updates(payloads, clients[0])                # Phase-1 fail-loud guard, per client

    # 4. Aggregate the head and load it back non-strict onto a fresh reconstructed model.
    aggregated = _average_subset(payloads)
    global_net = make_client(33)
    backbone_before = serialize_backbone(global_net)
    apply_trainable_subset(global_net, aggregated)

    # The head was averaged; the frozen backbone is byte-identical before and after (unchanged),
    # and still hashes to the originally-registered content address.
    for k, v in aggregated.items():
        assert torch.equal(dict(global_net.named_parameters())[k].detach(), v)
    assert serialize_backbone(global_net) == backbone_before
    assert backbone_sha256(serialize_backbone(global_net)) == sha
