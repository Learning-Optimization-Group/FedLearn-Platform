"""DA-14 Phase 1 (Target B) — infer.py build_model dispatch collapses onto the recipe registry.

infer.build_model built CNN (architecture.cnn.net.Net), MLP (models.ecg_mlp.ECGModel) and
TRANSFORMER (opt-125m) inline. This pins that delegating to recipes.get_recipe(...).build_model()
is byte-identical: same state-dict keys (so a saved run's weights still strict-load), same class
labels / input kind, and — the delegation proof — the returned module is now the registry's class
(models.CnnNet), not infer's private architecture.cnn.net.Net.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import infer  # noqa: E402
import recipes  # noqa: E402

# Goldens captured from the legacy inline builds (kept in sync with test_recipes_registry.py).
CNN_GOLDEN_KEYS = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
                   "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]
MLP_GOLDEN_KEYS = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
ECG_CLASSES = ["Normal", "Abnormal"]


def test_infer_cnn_returns_registry_model_with_golden_keys():
    net, classes, kind, transform = infer.build_model("CNN", "net")
    assert list(net.state_dict().keys()) == CNN_GOLDEN_KEYS
    assert classes == CIFAR10_CLASSES
    assert kind == "image"
    assert transform is None
    # Delegation proof: the module is the registry's CnnNet, not infer's private Net.
    from models import CnnNet
    assert isinstance(net, CnnNet)


def test_infer_mlp_returns_registry_model_with_golden_keys():
    net, classes, kind, transform = infer.build_model("MLP", "ecg_mlp")
    assert list(net.state_dict().keys()) == MLP_GOLDEN_KEYS
    assert classes == ECG_CLASSES
    assert kind == "vector"
    assert transform is None
    from models.ecg_mlp import ECGModel
    assert isinstance(net, ECGModel)


@pytest.mark.parametrize("mt, name, golden", [("CNN", "net", CNN_GOLDEN_KEYS),
                                              ("MLP", "ecg_mlp", MLP_GOLDEN_KEYS)])
def test_infer_model_strict_loads_weights_saved_from_the_registry_build(tmp_path, mt, name, golden):
    """A run's weights (saved from the registry build, __DOT__-mangled like init_model does) must
    strict-load into infer's reconstructed model — proving the arch matches byte-for-byte."""
    saved = recipes.get_recipe(mt).build_model("cpu")
    mangled = {k.replace(".", "__DOT__"): v.detach().cpu().numpy()
               for k, v in saved.state_dict().items()}
    npz_path = os.path.join(tmp_path, "model.npz")
    np.savez(npz_path, **mangled)

    net = infer.build_model(mt, name)[0]
    state = infer.decode_npz(npz_path)
    net.load_state_dict(state, strict=True)  # raises on any key/shape drift
    assert list(state.keys()) == golden


def test_infer_build_model_is_registry_driven(monkeypatch):
    """DA-14 Ph3.1: infer.build_model routes ANY recipe key through recipe.build_for_inference()
    with no dedicated if/elif branch — a new recipe needs no infer edit. Before the collapse an
    unbranched key hit the terminal 'Unsupported model type' raise."""
    import torch.nn as nn
    real_get_recipe = recipes.get_recipe

    class _FakeRecipe:
        key = "FAKE_IMG"
        classes = ["a", "b"]
        input_kind = "image"

        def build_for_inference(self, model_name=None, task_type="SEQ_CLASSIFICATION"):
            return nn.Linear(2, 2), self.classes, self.input_kind, None

    monkeypatch.setattr(recipes, "get_recipe",
                        lambda k: _FakeRecipe() if str(k).upper() == "FAKE_IMG" else real_get_recipe(k))
    net, classes, kind, transform = infer.build_model("FAKE_IMG", "x")
    assert isinstance(net, nn.Linear)
    assert classes == ["a", "b"] and kind == "image" and transform is None


@pytest.mark.slow  # loads facebook/opt-125m from the HF cache (~250MB); deselected in CI
def test_infer_transformer_delegates_model_and_tokenizer_to_the_registry():
    net, classes, kind, tok = infer.build_model("TRANSFORMER", "opt-125m")
    assert net.score.weight.shape[0] == 3
    assert classes == ["entailment", "contradiction", "neutral"]
    assert kind == "text"
    # tokenizer came through with a pad token, and the net was wired to it (needed for padding).
    assert tok.pad_token_id is not None
    assert net.config.pad_token_id == tok.pad_token_id
