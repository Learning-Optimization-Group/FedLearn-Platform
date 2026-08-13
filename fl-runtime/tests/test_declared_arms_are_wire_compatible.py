"""A recipe must not declare an arm whose payload cannot cross the wire.

Third instance of one defect class, so this test targets the class rather than the instance:

1. ``CNN`` declared FROZEN_HEAD with prefix ``classifier.`` — a module ``CnnNet`` does not have.
   Caught by building the model (``test_declared_prefixes_match_real_parameters``).
2. ``CIFAR_RESNET18`` declared FULL, and FULL cannot be federated at all: the safetensors wire
   accepts float32 only, and every BatchNorm module carries an int64 ``num_batches_tracked``.
   Building the model does NOT catch this — the model is fine; the *payload* is not.

The float32 restriction is deliberate, not an oversight: the wire has to be decodable by the
libtorch-free mobile C++ client, so other dtypes raise rather than silently coerce. That makes
"can this arm's federated set actually be serialised?" a real precondition on a declaration, and a
distinct one from "does this arm's prefix match a parameter".

Every catalog recipe except CIFAR_RESNET18 is BatchNorm-free, which is why this went unnoticed
until the first ResNet recipe.
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "framework", "src"))

import recipes  # noqa: E402

# TRANSFORMER/LLM_LORA build multi-hundred-MB HuggingFace models; their wire path is exercised by
# their own suites. Excluded here for build cost, not because they are exempt from the rule.
_HEAVY = {"TRANSFORMER", "LLM_LORA"}
CHECKABLE = [m["key"] for m in recipes.RECIPE_METADATA if m["key"] not in _HEAVY]


def _federated_tensors(recipe_key, arm):
    """Exactly what would go on the wire for this (recipe, arm)."""
    from fedlearn.estimators.params import trainable_state

    net = recipes.get_recipe(recipe_key).build_model("cpu")
    prefixes = recipes.trainable_prefixes(recipe_key, arm)
    recipes.apply_arm(net, arm, prefixes)
    if prefixes is None:
        return net.state_dict()          # FULL federates the whole state_dict, buffers included
    return trainable_state(net)


@pytest.mark.slow
@pytest.mark.parametrize("recipe_key", CHECKABLE)
def test_every_declared_arm_can_be_serialised(recipe_key):
    """THE guard. A declared arm whose payload the wire rejects is an option that always fails."""
    for arm in recipes._METADATA_BY_KEY[recipe_key]["supported_arms"]:
        bad = {n: str(t.dtype) for n, t in _federated_tensors(recipe_key, arm).items()
               if t.dtype != torch.float32}
        assert not bad, (
            f"{recipe_key} declares arm {arm!r}, but its federated payload contains non-float32 "
            f"tensors the safetensors wire rejects: {dict(list(bad.items())[:3])}"
            f"{' ...' if len(bad) > 3 else ''}. Either the arm must not be declared, or the wire "
            f"format has to change — and that is a cross-language decision, since the mobile C++ "
            f"client decodes float32 without libtorch.")


@pytest.mark.slow
def test_the_resnet_recipe_can_now_run_full():
    """CIFAR_RESNET18's FULL arm was blocked and is now unblocked.

    This test previously asserted the OPPOSITE — that FULL must not be declared, because int64
    num_batches_tracked made the payload unserialisable. Excluding non-float32 tensors from the
    federated set removed that limit, so the assertion inverts rather than being deleted: the
    recipe must offer FULL, and the withheld tensors must be exactly the int64 counters.
    """
    meta = recipes._METADATA_BY_KEY["CIFAR_RESNET18"]
    assert "FULL" in meta["supported_arms"]
    assert not meta.get("full_arm_unsupported_reason"), \
        "the recipe still carries a reason it cannot run FULL, but it can"

    from fedlearn.estimators.params import non_federable_names
    net = recipes.get_recipe("CIFAR_RESNET18").build_model("cpu")
    withheld = non_federable_names(net.state_dict())
    assert withheld, "expected the int64 BatchNorm counters to be withheld"
    assert all(n.endswith("num_batches_tracked") for n in withheld), \
        f"something other than a batch counter is being withheld: {withheld[:5]}"


def test_recipes_that_support_full_still_do():
    """The restriction must be confined to the one recipe that needs it."""
    for key in ("PNEUMONIA_CNN", "CNN", "MLP", "TINYNET_GOLDEN"):
        assert "FULL" in recipes._METADATA_BY_KEY[key]["supported_arms"], \
            f"{key} lost its FULL arm"
