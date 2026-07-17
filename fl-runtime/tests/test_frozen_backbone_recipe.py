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
