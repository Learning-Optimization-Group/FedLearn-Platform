"""DA-14 Phase 0 — registry skeleton + safety net.

Makes the recipe registry resolve EVERY dispatched key (fixing the BLOOD_CNN latent crash: it is
dispatched by init_model.py but was absent from the registry, so get_recipe raised), while keeping
the advertised --describe catalog byte-stable (SE-10: BLOOD_CNN needs medmnist and must stay
un-advertised). The --describe golden + the introspection contract are the safety net that lets the
Phase-1/2 if/elif→registry dispatch collapse proceed without silently drifting the catalog.
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402

# The exactly-six advertised catalog keys, in order (the --describe golden).
CATALOG_KEYS = ["PNEUMONIA_CNN", "CNN", "MLP", "TRANSFORMER", "LLM_LORA", "TINYNET_GOLDEN"]


def test_get_recipe_blood_cnn_does_not_raise():
    """BLOOD_CNN is dispatched (init_model.py calls get_recipe('BLOOD_CNN')) but was not registered,
    so the call raised ValueError — a latent crash. It must now resolve to a functional recipe."""
    r = recipes.get_recipe("BLOOD_CNN")
    assert r.key == "BLOOD_CNN"
    assert r.is_functional is True


def test_blood_cnn_is_registered_but_not_advertised():
    """SE-10 preserved: BLOOD_CNN is resolvable (is_recipe) but excluded from the --describe catalog
    / project-creation picker — dispatchable, not selectable."""
    assert recipes.is_recipe("BLOOD_CNN") is True
    assert "BLOOD_CNN" not in [r["key"] for r in recipes.describe()]


def test_describe_catalog_golden_is_exactly_the_six_advertised_recipes():
    """--describe golden: the advertised catalog is exactly these six keys, in this order."""
    assert [r["key"] for r in recipes.describe()] == CATALOG_KEYS


def test_describe_subprocess_is_torch_free_and_stable():
    """--describe runs as a cheap torch-free subprocess (the backend serves it at GET
    /api/model-recipes) and emits the same catalog keys."""
    out = subprocess.run(
        [sys.executable, "recipes.py", "--describe"],
        cwd=os.path.dirname(os.path.abspath(recipes.__file__)),
        capture_output=True, text=True, check=True,
    )
    assert [r["key"] for r in json.loads(out.stdout)] == CATALOG_KEYS


@pytest.mark.parametrize("key", CATALOG_KEYS + ["BLOOD_CNN"])
def test_every_registered_recipe_is_introspectable(key):
    """Contract: every registered recipe resolves and exposes consistent, non-empty metadata."""
    r = recipes.get_recipe(key)
    assert r.key == key
    assert isinstance(r.classes, list) and len(r.classes) >= 1
    assert isinstance(r.is_functional, bool)


def test_tinynet_golden_builds_a_model_on_cpu_via_the_registry():
    """The cheapest functional recipe builds a real model through the registry — no if/elif at the
    call site. (torch-only; BLOOD_CNN/LLM build is exercised elsewhere where their deps exist.)"""
    import torch
    model = recipes.get_recipe("TINYNET_GOLDEN").build_model(device="cpu")
    assert isinstance(model, torch.nn.Module)


# --- DA-14 Phase 1: collapse the CNN/MLP model construction onto the registry ------------------
# State-key goldens captured from the legacy init_model.get_model() build. The registry build must
# produce byte-identical state-dict keys, so init_model can delegate without changing any model.
CNN_GOLDEN_KEYS = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
                   "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]
MLP_GOLDEN_KEYS = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]


def test_registry_builds_cnn_with_golden_state_keys():
    """CNN must build via recipe.build_model (was inline in init_model.py + NotImplementedError in
    the registry), with keys byte-identical to the legacy CnnNet."""
    model = recipes.get_recipe("CNN").build_model(device="cpu")
    assert list(model.state_dict().keys()) == CNN_GOLDEN_KEYS


def test_registry_builds_mlp_with_golden_state_keys():
    model = recipes.get_recipe("MLP").build_model(device="cpu")
    assert list(model.state_dict().keys()) == MLP_GOLDEN_KEYS


def test_init_model_delegates_cnn_and_mlp_to_the_registry():
    """The collapse is behavior-preserving: init_model.get_model and the registry build the same
    architecture (same state-dict keys) for CNN and MLP."""
    import init_model
    for mtype, name, golden in (("CNN", "net", CNN_GOLDEN_KEYS), ("MLP", "ecg_mlp", MLP_GOLDEN_KEYS)):
        legacy = init_model.get_model(mtype, name, "cpu")
        assert list(legacy.state_dict().keys()) == golden


# --- DA-14 Phase 1: collapse the TRANSFORMER model construction onto the registry (FR-29) -------
# The TRANSFORMER recipe is the opt-125m sequence-classification model. Its head width is the single
# num_labels authority: len(classes)=3 (CB). The legacy sites disagreed — init_model.py hardcoded
# num_labels=3, but client.py built num_labels=config["num_classes"] (cb->3 loads the 3-class wire;
# sst2->2 can NEVER strict-load the 3-class global model init_model produces — a latent crash).
# Collapsing every site onto build_model('TRANSFORMER') unifies on 3: byte-identical on the cb path,
# corrective on the already-broken sst2 path.
def test_transformer_recipe_classes_is_the_num_labels_authority():
    """The recipe's class list is the single source for the opt-125m SEQ_CLS head width (FR-29)."""
    r = recipes.get_recipe("TRANSFORMER")
    assert r.classes == ["entailment", "contradiction", "neutral"]
    assert len(r.classes) == 3


@pytest.mark.slow  # loads facebook/opt-125m from the HF cache (~250MB); deselected in CI (-m "not slow")
def test_registry_builds_transformer_with_num_labels_from_classes():
    """build_model('TRANSFORMER') constructs opt-125m with head width == len(classes) (=3), and is
    byte-identical (same state-dict keys, same head shape) to a legacy from_pretrained(num_labels=3)."""
    from transformers import AutoModelForSequenceClassification
    r = recipes.get_recipe("TRANSFORMER")
    model = r.build_model(device="cpu")
    assert model.score.weight.shape[0] == len(r.classes) == 3
    legacy = AutoModelForSequenceClassification.from_pretrained(
        "facebook/opt-125m", num_labels=3, use_safetensors=True)
    assert list(model.state_dict().keys()) == list(legacy.state_dict().keys())
    assert model.score.weight.shape == legacy.score.weight.shape


@pytest.mark.slow  # loads facebook/opt-125m from the HF cache; deselected in CI
def test_init_model_delegates_transformer_to_the_registry():
    """The server-init collapse is behavior-preserving: init_model.get_model('TRANSFORMER', ...) and
    the registry build the same architecture (byte-identical state-dict keys, same head shape)."""
    import init_model
    legacy = init_model.get_model("TRANSFORMER", "opt-125m", "cpu")
    viareg = recipes.get_recipe("TRANSFORMER").build_model(device="cpu")
    assert list(legacy.state_dict().keys()) == list(viareg.state_dict().keys())
    assert legacy.score.weight.shape == viareg.score.weight.shape
