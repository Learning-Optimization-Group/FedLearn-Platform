"""DA-14 Ph3.0 (recipe side) — a frozen-backbone derived-model builder in the recipe registry.

Activates the DA-11 partial-load path (fedlearn.backbone.distribution.reconstruct_frozen_backbone)
at the recipe layer: build_frozen_backbone_model builds a fresh model, loads a content-addressed
frozen backbone onto it (non-strict) and re-freezes it, leaving only the head trainable (the
federated subset). Exposed as a NON-CATALOG demo recipe FROZEN_DEMO — dispatchable via get_recipe
but excluded from --describe / the project picker (superseded by the real derivation recipe, Ph3.3).
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402

# The advertised catalog must NOT change — FROZEN_DEMO is dispatchable but not selectable (SE-10).
CATALOG_KEYS = ["PNEUMONIA_CNN", "CNN", "MLP", "TRANSFORMER", "LLM_LORA", "TINYNET_GOLDEN"]


def test_frozen_demo_is_dispatchable_but_not_catalogued():
    r = recipes.get_recipe("FROZEN_DEMO")
    assert r.key == "FROZEN_DEMO"
    assert recipes.is_recipe("FROZEN_DEMO") is True
    # Excluded from the advertised catalog / picker (SE-10 preserved).
    assert "FROZEN_DEMO" not in [x["key"] for x in recipes.describe()]
    assert "FROZEN_DEMO" not in recipes.catalog_keys()
    assert [x["key"] for x in recipes.describe()] == CATALOG_KEYS


def test_frozen_demo_builds_frozen_backbone_and_trainable_head():
    from fedlearn.estimators.params import frozen_state, trainable_state
    model = recipes.get_recipe("FROZEN_DEMO").build_model(device="cpu")
    # Backbone frozen, head trainable — the only federated subset is the head.
    assert model.backbone.weight.requires_grad is False
    assert model.backbone.bias.requires_grad is False
    assert model.head.weight.requires_grad is True
    assert model.head.bias.requires_grad is True
    assert set(trainable_state(model).keys()) == {"head.weight", "head.bias"}
    assert set(frozen_state(model).keys()) == {"backbone.weight", "backbone.bias"}
    # Head width follows the recipe's class list.
    assert model.head.weight.shape[0] == len(recipes.get_recipe("FROZEN_DEMO").classes)


def test_frozen_demo_backbone_roundtrips_via_serialize_reconstruct():
    """A second model built from serialize_backbone(first) carries the first's frozen backbone
    byte-for-byte (the content-addressed BASE_REF path), with its head still trainable."""
    from fedlearn.backbone.distribution import serialize_backbone
    ref = recipes.build_frozen_backbone_model(num_classes=3)
    blob = serialize_backbone(ref)
    other = recipes.build_frozen_backbone_model(num_classes=3, backbone_bytes=blob)
    assert torch.equal(other.backbone.weight.detach(), ref.backbone.weight.detach())
    assert torch.equal(other.backbone.bias.detach(), ref.backbone.bias.detach())
    assert other.backbone.weight.requires_grad is False
    assert other.head.weight.requires_grad is True


def test_recipe_derived_model_is_a_valid_subset_federation_participant():
    """The bridge: a model built by the fl-runtime recipe (build_frozen_backbone_model over a shared
    BASE_REF blob) is a valid participant in the framework's trainable-subset contract end to end —
    its wire payload is head-only, two clients guard + aggregate via the real FedAvgAggregator, and
    apply_trainable_subset preserves the shared frozen backbone. Guards the seam between the recipe
    (this repo) and the DA-11 framework modules that Ph3.0 activates."""
    from fedlearn.backbone.distribution import serialize_backbone
    from fedlearn.estimators.params import trainable_state
    from fedlearn.server.strategy import FedAvgAggregator
    from fedlearn.server.subset_federation import apply_trainable_subset, guard_client_updates

    # One shared frozen backbone (a BASE_REF); every participant reconstructs the SAME one.
    ref = recipes.build_frozen_backbone_model(num_classes=3)
    blob = serialize_backbone(ref)
    c0 = recipes.build_frozen_backbone_model(num_classes=3, backbone_bytes=blob)
    c1 = recipes.build_frozen_backbone_model(num_classes=3, backbone_bytes=blob)
    server = recipes.build_frozen_backbone_model(num_classes=3, backbone_bytes=blob)

    with torch.no_grad():
        c0.head.weight.fill_(0.25); c0.head.bias.fill_(0.1)
        c1.head.weight.fill_(0.75); c1.head.bias.fill_(0.3)
    u0, u1 = trainable_state(c0), trainable_state(c1)
    assert set(u0.keys()) == {"head.weight", "head.bias"}          # head-only wire

    guard_client_updates([u0, u1], server)                        # fail-loud guard passes
    expected_head_w = (u0["head.weight"].clone() + u1["head.weight"].clone()) / 2
    backbone_before = server.backbone.weight.detach().clone()

    agg = FedAvgAggregator().aggregate([("c0", u0, 10), ("c1", u1, 10)])
    assert set(agg.keys()) == {"head.weight", "head.bias"}
    apply_trainable_subset(server, agg)

    assert torch.allclose(server.head.weight.detach(), expected_head_w)   # head is the average
    assert torch.equal(server.backbone.weight.detach(), backbone_before)  # frozen backbone survives
    assert torch.equal(server.backbone.weight.detach(), ref.backbone.weight.detach())  # == the shared base
