"""P1: the training arm (frozen-head vs full fine-tune) as a first-class, declared concept.

Written before the implementation (TDD).

Today the arm is not selected — it is *inferred from the recipe key*. `fl-runtime/client.py:44`
declares a module-global ``USE_DERIVED = False`` which line 945 sets via
``USE_DERIVED = (mt == "FROZEN_DEMO")``. Four consequences, all of which these tests close:

1. **Every recipe has exactly one hard-coded arm.** PNEUMONIA_CNN cannot be run frozen and full
   as two arms of one comparison through the product path — which is precisely the comparison
   `research/results/frozen-backbone/` (177 result files) is built on. The science is far ahead
   of the product.
2. **The choice is not persisted**, so it cannot be queried, audited, or attached to a result.
3. **`FROZEN_DEMO` lives in `_NONCATALOG_METADATA`**, so it never reaches ``--describe`` or the
   project-creation picker. The frozen arm is developer-only.
4. **A result cannot say which arm produced it.** Commit `21699bc` — *"frozen arm silently
   mislabelled its backbone, risking cell overwrites"* — is exactly this bug class: when the arm
   is implicit, two different experiments write the same cell.

The design under test: a recipe *declares* which arms it supports and, per arm, which module
prefixes stay trainable. `trainable_state()` is then driven by that declaration rather than by a
key comparison, which generalises the frozen path from one demo recipe to all of them.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402


# --------------------------------------------------------------------------------------
# 1. Recipes DECLARE their arms
# --------------------------------------------------------------------------------------

class TestRecipesDeclareArms:
    def test_every_catalog_recipe_declares_supported_arms(self):
        """A recipe that does not say which arms it supports cannot be validated at creation."""
        for meta in recipes.RECIPE_METADATA:
            assert "supported_arms" in meta, f"{meta['key']} declares no supported_arms"
            assert meta["supported_arms"], f"{meta['key']} supports no arms"
            for arm in meta["supported_arms"]:
                assert arm in recipes.TRAINING_ARMS, f"{meta['key']}: unknown arm {arm!r}"

    def test_full_is_always_supported(self):
        """Every recipe can be trained end-to-end; FROZEN_HEAD is the optional one."""
        for meta in recipes.RECIPE_METADATA:
            assert "FULL" in meta["supported_arms"], f"{meta['key']} cannot run FULL"

    def test_a_frozen_capable_recipe_declares_its_trainable_prefixes(self):
        """FROZEN_HEAD is meaningless without saying WHICH module stays trainable."""
        frozen = [m for m in recipes.RECIPE_METADATA if "FROZEN_HEAD" in m["supported_arms"]]
        assert frozen, "no catalog recipe supports FROZEN_HEAD — the frozen arm is still dev-only"
        for meta in frozen:
            spec = meta.get("trainable_spec", {})
            assert spec.get("FROZEN_HEAD"), f"{meta['key']}: FROZEN_HEAD has no trainable prefixes"
            assert isinstance(spec["FROZEN_HEAD"], (list, tuple))

    def test_full_arm_needs_no_prefixes(self):
        """FULL means 'everything', which is expressed as None rather than a list of every module."""
        for meta in recipes.RECIPE_METADATA:
            spec = meta.get("trainable_spec", {})
            if "FULL" in spec:
                assert spec["FULL"] is None, f"{meta['key']}: FULL should be None (= all trainable)"

    def test_the_catalog_is_served_with_the_arms(self):
        """The picker cannot offer an arm the backend never sends it."""
        for entry in recipes.describe():
            assert "supported_arms" in entry, f"{entry['key']} loses its arms in describe()"


# --------------------------------------------------------------------------------------
# 2. Applying an arm
# --------------------------------------------------------------------------------------

class _TwoPart(nn.Module):
    """Stand-in with an obvious backbone/head split."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
        self.classifier = nn.Linear(4, 2)

    def forward(self, x):
        return self.classifier(self.features(x))


class TestApplyArm:
    def test_full_leaves_everything_trainable(self):
        m = _TwoPart()
        recipes.apply_arm(m, "FULL", trainable_prefixes=None)
        assert all(p.requires_grad for p in m.parameters())

    def test_frozen_head_freezes_everything_outside_the_prefixes(self):
        m = _TwoPart()
        recipes.apply_arm(m, "FROZEN_HEAD", trainable_prefixes=["classifier."])
        for name, p in m.named_parameters():
            expected = name.startswith("classifier.")
            assert p.requires_grad is expected, f"{name}: requires_grad={p.requires_grad}"

    def test_frozen_head_leaves_a_nonempty_trainable_set(self):
        """A prefix that matches nothing silently freezes the whole model and trains nothing."""
        m = _TwoPart()
        with pytest.raises(ValueError, match="no parameter"):
            recipes.apply_arm(m, "FROZEN_HEAD", trainable_prefixes=["nonexistent."])

    def test_applying_an_arm_is_idempotent(self):
        m = _TwoPart()
        recipes.apply_arm(m, "FROZEN_HEAD", trainable_prefixes=["classifier."])
        before = {n: p.requires_grad for n, p in m.named_parameters()}
        recipes.apply_arm(m, "FROZEN_HEAD", trainable_prefixes=["classifier."])
        assert {n: p.requires_grad for n, p in m.named_parameters()} == before

    def test_switching_back_to_full_unfreezes(self):
        """A process that ran a frozen arm must not leak frozen state into a later FULL run."""
        m = _TwoPart()
        recipes.apply_arm(m, "FROZEN_HEAD", trainable_prefixes=["classifier."])
        recipes.apply_arm(m, "FULL", trainable_prefixes=None)
        assert all(p.requires_grad for p in m.parameters())

    def test_unknown_arm_is_rejected(self):
        m = _TwoPart()
        with pytest.raises(ValueError, match="unknown arm"):
            recipes.apply_arm(m, "SEMI_FROZEN", trainable_prefixes=None)


# --------------------------------------------------------------------------------------
# 3. The federated payload follows the arm
# --------------------------------------------------------------------------------------

class TestFederatedPayloadFollowsTheArm:
    def test_frozen_head_federates_only_the_head(self):
        """This is the whole point: the frozen arm's wire payload is the head, not the model."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                        "framework", "src"))
        from fedlearn.estimators.params import trainable_state

        m = _TwoPart()
        recipes.apply_arm(m, "FROZEN_HEAD", trainable_prefixes=["classifier."])
        keys = list(trainable_state(m))
        assert keys == ["classifier.weight", "classifier.bias"]

    def test_full_federates_everything(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                        "framework", "src"))
        from fedlearn.estimators.params import trainable_state

        m = _TwoPart()
        recipes.apply_arm(m, "FULL", trainable_prefixes=None)
        assert len(trainable_state(m)) == len(list(m.named_parameters()))

    def test_the_frozen_payload_is_dramatically_smaller(self):
        """The measured claim the frozen arm exists for, asserted as a property."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                        "framework", "src"))
        from fedlearn.estimators.params import trainable_state

        m = _TwoPart()
        recipes.apply_arm(m, "FULL", trainable_prefixes=None)
        full = sum(t.numel() for t in trainable_state(m).values())
        recipes.apply_arm(m, "FROZEN_HEAD", trainable_prefixes=["classifier."])
        head = sum(t.numel() for t in trainable_state(m).values())
        assert head < full / 5, f"frozen payload {head} is not much smaller than full {full}"


# --------------------------------------------------------------------------------------
# 4. Validation at the point of choice
# --------------------------------------------------------------------------------------

class TestArmValidation:
    def test_an_unsupported_arm_is_rejected_for_the_recipe(self):
        """Rejected at creation, not at spawn — a bad arm must not reach the FL server."""
        key = recipes.RECIPE_METADATA[0]["key"]
        with pytest.raises(ValueError, match="does not support"):
            recipes.validate_arm(key, "NO_SUCH_ARM")

    def test_a_supported_arm_is_accepted(self):
        for meta in recipes.RECIPE_METADATA:
            for arm in meta["supported_arms"]:
                assert recipes.validate_arm(meta["key"], arm) == arm

    def test_the_default_arm_is_full(self):
        """An omitted arm must mean FULL, so existing projects keep their behaviour."""
        for meta in recipes.RECIPE_METADATA:
            assert recipes.validate_arm(meta["key"], None) == "FULL"

    def test_resolving_prefixes_for_an_arm(self):
        """The runtime asks the recipe what to freeze; it does not hard-code a key comparison."""
        frozen = [m for m in recipes.RECIPE_METADATA if "FROZEN_HEAD" in m["supported_arms"]][0]
        assert recipes.trainable_prefixes(frozen["key"], "FULL") is None
        assert recipes.trainable_prefixes(frozen["key"], "FROZEN_HEAD")


# --------------------------------------------------------------------------------------
# 5. The provenance guard — cf. commit 21699bc
# --------------------------------------------------------------------------------------

class TestArmProvenance:
    def test_arm_is_recorded_in_a_result_stamp(self):
        """A result that cannot say which arm produced it can be silently compared across arms.

        Commit 21699bc — "frozen arm silently mislabelled its backbone, risking cell overwrites"
        — is this bug class. The stamp is the fix, and it must carry the prefixes too: two runs
        can share an arm NAME while freezing different modules.
        """
        frozen = [m for m in recipes.RECIPE_METADATA if "FROZEN_HEAD" in m["supported_arms"]][0]
        stamp = recipes.arm_stamp(frozen["key"], "FROZEN_HEAD")
        assert stamp["recipe"] == frozen["key"]
        assert stamp["arm"] == "FROZEN_HEAD"
        assert stamp["trainable_prefixes"]
        import json
        json.dumps(stamp)          # must land in a result's meta block

    def test_stamps_differ_across_arms_of_the_same_recipe(self):
        frozen = [m for m in recipes.RECIPE_METADATA if "FROZEN_HEAD" in m["supported_arms"]][0]
        a = recipes.arm_stamp(frozen["key"], "FULL")
        b = recipes.arm_stamp(frozen["key"], "FROZEN_HEAD")
        assert a != b, "the two arms of one recipe must not produce identical provenance"
