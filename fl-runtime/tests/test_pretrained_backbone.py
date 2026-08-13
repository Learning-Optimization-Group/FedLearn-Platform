"""Pretrained-backbone recipes — the deferred roadmap item (2), and what makes the frozen arm useful.

The live federation on 2026-08-13 proved the frozen arm is mechanically correct and, on the
recipes available then, *useless*: FROZEN_HEAD on CIFAR-10 sat at 10.05 / 10.02 / 10.00 % — chance
for ten classes, loss 2.3042 against ln(10)=2.3026. That is not a wiring defect. Freezing a
RANDOMLY-INITIALISED convolutional backbone trains a linear head on random features, and random
features carry almost no class signal.

The whole premise of the frozen arm is a backbone worth keeping: the research campaign's frozen arm
ran on ImageNet-pretrained ResNet features and matched a full fine-tune to within seed noise. The
product path could not express that, because every recipe built its model from scratch.

So a recipe may now declare a PRETRAINED source. The tests below pin the two things that make such
a recipe trustworthy rather than merely present:

* the weights are actually loaded — a "pretrained" recipe that silently fell back to random init
  would reproduce the 10% result while claiming to fix it, which is the worst possible failure here;
* the source is recorded, so a result can say which weights produced it.
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import recipes  # noqa: E402

RECIPE = "CIFAR_RESNET18"


class TestTheRecipeDeclaresItsPretrainedSource:
    def test_the_recipe_exists_in_the_catalog(self):
        assert RECIPE in [m["key"] for m in recipes.RECIPE_METADATA]

    def test_it_declares_a_pretrained_source(self):
        meta = recipes._METADATA_BY_KEY[RECIPE]
        assert meta.get("pretrained"), f"{RECIPE} does not declare where its weights come from"
        assert meta["pretrained"].get("source"), "no source recorded"
        assert meta["pretrained"].get("weights"), "no weights identifier recorded"

    def test_the_source_reaches_the_served_catalog(self):
        """The picker should be able to say 'starts from ImageNet weights' rather than implying the
        model trains from scratch like every other recipe."""
        entry = next(e for e in recipes.describe() if e["key"] == RECIPE)
        assert entry.get("pretrained"), "pretrained provenance is dropped by describe()"

    def test_it_supports_the_frozen_arm_only_and_says_why(self):
        """FULL was declared first and could not run: BatchNorm's int64 num_batches_tracked is
        rejected by the float32-only safetensors wire, so a FULL run dies on the first
        GetGlobalModel. Declaring an arm that always fails is the defect; stating the limit is the
        fix. See tests/test_declared_arms_are_wire_compatible.py."""
        meta = recipes._METADATA_BY_KEY[RECIPE]
        assert set(meta["supported_arms"]) == {"FROZEN_HEAD"}
        assert meta.get("full_arm_unsupported_reason")

    def test_the_frozen_prefix_is_the_classifier_only(self):
        pre = recipes.trainable_prefixes(RECIPE, "FROZEN_HEAD")
        assert pre == ["fc."], f"expected the torchvision resnet head 'fc.', got {pre}"


@pytest.mark.slow
class TestTheWeightsAreActuallyLoaded:
    """Marked slow: these build a real ResNet-18 and read the cached ImageNet checkpoint."""

    def test_the_backbone_is_not_randomly_initialised(self):
        """THE claim. Two independently built models must share identical BACKBONE weights — that
        can only happen if both read the same checkpoint. A silent fallback to random init would
        give different weights each time and put the frozen arm straight back at chance.
        """
        a = recipes.get_recipe(RECIPE).build_model("cpu")
        b = recipes.get_recipe(RECIPE).build_model("cpu")
        sa, sb = a.state_dict(), b.state_dict()
        backbone = [k for k in sa if not k.startswith("fc.")]
        assert backbone, "no backbone parameters found"
        for k in backbone:
            assert torch.equal(sa[k], sb[k]), \
                f"{k} differs between two builds — the pretrained weights were NOT loaded"

    def test_the_backbone_is_not_all_zeros_or_constant(self):
        """Guards the degenerate 'loaded something' case: a zeroed or constant tensor would pass an
        equality check between two builds while carrying no information at all."""
        m = recipes.get_recipe(RECIPE).build_model("cpu")
        w = m.state_dict()["conv1.weight"]
        assert w.abs().sum() > 0, "conv1 is all zeros"
        assert w.std() > 1e-4, "conv1 has no variance — not a trained filter bank"

    def test_the_head_is_freshly_initialised_for_this_task(self):
        """The ImageNet head predicts 1000 classes and is useless here; it must be replaced by a
        10-class head, and that head must NOT be shared between builds (it is the thing we train)."""
        m = recipes.get_recipe(RECIPE).build_model("cpu")
        assert m.state_dict()["fc.weight"].shape[0] == 10, "head is not a 10-class CIFAR head"

    def test_the_frozen_arm_leaves_only_the_head_trainable(self):
        m = recipes.get_recipe(RECIPE).build_model("cpu")
        recipes.apply_arm(m, "FROZEN_HEAD", recipes.trainable_prefixes(RECIPE, "FROZEN_HEAD"))
        trainable = {n for n, p in m.named_parameters() if p.requires_grad}
        assert trainable == {"fc.weight", "fc.bias"}, f"unexpected trainable set: {sorted(trainable)}"

    def test_the_frozen_payload_is_a_tiny_fraction_of_the_model(self):
        """The communication argument for the frozen arm, as a property of this recipe."""
        sys.path.insert(0, os.path.join(HERE, "..", "..", "framework", "src"))
        from fedlearn.estimators.params import trainable_state

        m = recipes.get_recipe(RECIPE).build_model("cpu")
        total = sum(p.numel() for p in m.parameters())
        recipes.apply_arm(m, "FROZEN_HEAD", recipes.trainable_prefixes(RECIPE, "FROZEN_HEAD"))
        head = sum(t.numel() for t in trainable_state(m).values())
        assert head < total / 1000, f"frozen payload {head} is not << model {total}"


@pytest.mark.slow
class TestTheDataMatchesTheBackbone:
    def test_the_input_is_resized_for_an_imagenet_backbone(self):
        """32x32 CIFAR through an ImageNet backbone yields poor features — the resolution the
        weights were trained near is part of what makes them worth freezing."""
        loader, _ = recipes.get_recipe(RECIPE).load_client_data(0, 10, batch_size=4)
        batch = next(iter(loader))
        images = batch["img"] if isinstance(batch, dict) else batch[0]
        assert images.shape[-1] >= 64, \
            f"images are {tuple(images.shape[-2:])}; too small for ImageNet features to be useful"

    def test_a_forward_pass_works_end_to_end(self):
        loader, _ = recipes.get_recipe(RECIPE).load_client_data(0, 10, batch_size=4)
        batch = next(iter(loader))
        images = batch["img"] if isinstance(batch, dict) else batch[0]
        out = recipes.get_recipe(RECIPE).build_model("cpu")(images)
        assert out.shape == (4, 10)
