"""OvA-LP as a third training arm — and the arm abstraction it forces open.

Source, as recorded in this repo (`docs/artifacts/fedlearn-competitive-roadmap.html`):

    OvA-LP: A Simple and Efficient Framework for Federated Learning on Non-IID Data.
    arXiv:2511.05028. Frozen encoder + one-vs-all heads + two-stage schedule.

WHAT IS AND IS NOT IMPLEMENTED HERE
-----------------------------------
Implemented: the frozen encoder (the existing FROZEN_HEAD mechanism) and the **one-vs-all
objective** — C independent binary classifiers instead of a single softmax over C classes.

NOT implemented: the **two-stage schedule**. The repo records that the paper has one; it does not
record what it is, and this was built without reading the paper. Inventing a schedule and calling
it OvA-LP would misattribute a design to a citation, so the arm implements the two components that
are unambiguous from the recorded description and declares the third missing.

WHY THIS FORCED A CHANGE TO THE ARM CONCEPT
-------------------------------------------
P1 defined an arm as "which parameters train" (``trainable_spec`` prefixes). OvA-LP trains exactly
the same parameters as FROZEN_HEAD — the head — and differs in the OBJECTIVE. So the abstraction
had to widen from "which parameters" to "which parameters, under what objective".

That widening is the point of the mechanism, not incidental: under a softmax, every class's logit
is coupled through the normalisation term, so a client holding none of class k still pushes class
k's weights. Independent per-class binary objectives remove that coupling, which is the paper's
stated argument for suppressing client drift at its source under extreme non-IID.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "framework", "src"))

import recipes  # noqa: E402

ARM = "OVA_LP"


class TestTheArmIsDeclared:
    def test_it_is_a_known_arm(self):
        assert ARM in recipes.TRAINING_ARMS

    def test_only_recipes_with_a_frozen_encoder_offer_it(self):
        """OvA-LP is linear probing on a FROZEN encoder. A recipe that cannot freeze cannot offer
        it, so the declaration must track FROZEN_HEAD support rather than being set independently."""
        for meta in recipes.RECIPE_METADATA:
            if ARM in meta["supported_arms"]:
                assert "FROZEN_HEAD" in meta["supported_arms"], (
                    f"{meta['key']} offers {ARM} without FROZEN_HEAD; OvA-LP is linear probing on "
                    f"a frozen encoder, so it needs the same freezing capability")

    def test_it_trains_the_same_parameters_as_the_frozen_arm(self):
        """The distinguishing feature is the OBJECTIVE, not the parameter subset. If these ever
        diverged, the comparison against FROZEN_HEAD would stop being controlled."""
        for meta in recipes.RECIPE_METADATA:
            if ARM not in meta["supported_arms"]:
                continue
            assert (recipes.trainable_prefixes(meta["key"], ARM)
                    == recipes.trainable_prefixes(meta["key"], "FROZEN_HEAD")), \
                f"{meta['key']}: {ARM} and FROZEN_HEAD must train the same parameters"

    def test_at_least_one_recipe_offers_it(self):
        assert any(ARM in m["supported_arms"] for m in recipes.RECIPE_METADATA), \
            "no recipe offers OvA-LP — the arm would be unreachable"


class TestTheObjectiveIsPartOfTheArm:
    def test_the_default_objective_is_cross_entropy(self):
        """Every pre-existing arm keeps softmax cross-entropy, so no existing run changes."""
        assert recipes.arm_objective("CIFAR_RESNET18", "FULL") == "cross_entropy"
        assert recipes.arm_objective("CIFAR_RESNET18", "FROZEN_HEAD") == "cross_entropy"

    def test_ova_lp_declares_the_one_vs_all_objective(self):
        assert recipes.arm_objective("CIFAR_RESNET18", ARM) == "one_vs_all"

    def test_an_unknown_arm_has_no_objective(self):
        with pytest.raises(ValueError):
            recipes.arm_objective("CIFAR_RESNET18", "NO_SUCH_ARM")

    def test_the_objective_reaches_the_arm_stamp(self):
        """Provenance: two runs that train the same parameters under different objectives are
        different experiments, and a result must be able to say which it was."""
        stamp = recipes.arm_stamp("CIFAR_RESNET18", ARM)
        assert stamp["arm"] == ARM
        assert stamp["objective"] == "one_vs_all"
        assert recipes.arm_stamp("CIFAR_RESNET18", "FROZEN_HEAD")["objective"] == "cross_entropy"

    def test_stamps_differ_between_the_two_frozen_arms(self):
        """They share a parameter subset, so without the objective their provenance would be
        identical — the cell-overwrite hazard of 21699bc, one level subtler."""
        assert (recipes.arm_stamp("CIFAR_RESNET18", ARM)
                != recipes.arm_stamp("CIFAR_RESNET18", "FROZEN_HEAD"))


class TestTheLossImplementsOneVsAll:
    def _loss(self, objective):
        return recipes.build_criterion(objective)

    def test_cross_entropy_is_unchanged(self):
        assert isinstance(self._loss("cross_entropy"), nn.CrossEntropyLoss)

    def test_one_vs_all_uses_independent_per_class_binary_objectives(self):
        """THE mechanism. Under softmax, raising one class's logit lowers every other class's loss
        contribution through the normalisation term. Under one-vs-all it must not: each class is
        its own binary problem."""
        crit = self._loss("one_vs_all")
        logits = torch.zeros(1, 4, requires_grad=True)
        targets = torch.tensor([1])

        loss = crit(logits, targets)
        loss.backward()
        g = logits.grad[0]

        assert g[1] < 0, "the true class's logit should be pushed up"
        assert all(g[k] > 0 for k in (0, 2, 3)), "each negative class should be pushed down"
        # Independence: the three negative classes are symmetric and identical here, and their
        # gradient magnitude does not depend on how many classes exist (softmax's would).
        assert torch.allclose(g[0], g[2]) and torch.allclose(g[0], g[3])

    def test_one_vs_all_gradient_is_independent_of_class_count(self):
        """The non-IID argument in one assertion: a class's update must not depend on the other
        classes present. Softmax cross-entropy fails this; one-vs-all passes it."""
        crit = self._loss("one_vs_all")
        grads = []
        for c in (4, 10):
            logits = torch.zeros(1, c, requires_grad=True)
            crit(logits, torch.tensor([1])).backward()
            grads.append(logits.grad[0, 1].item())
        assert grads[0] == pytest.approx(grads[1]), \
            "the true class's gradient changed with the number of classes — that is softmax coupling"

    def test_softmax_by_contrast_does_couple(self):
        """Control, so the test above is known to be measuring something real."""
        crit = self._loss("cross_entropy")
        grads = []
        for c in (4, 10):
            logits = torch.zeros(1, c, requires_grad=True)
            crit(logits, torch.tensor([1])).backward()
            grads.append(logits.grad[0, 1].item())
        assert grads[0] != pytest.approx(grads[1]), \
            "control failed: softmax showed no class-count coupling, so the OvA test proves nothing"

    def test_predictions_are_still_argmax(self):
        """Inference must not change: the head still emits per-class scores, so argmax is valid and
        the accuracy metric stays comparable with the other arms."""
        crit = self._loss("one_vs_all")
        logits = torch.tensor([[-2.0, 3.0, 0.5]])
        assert crit(logits, torch.tensor([1])) < crit(logits, torch.tensor([0]))


class TestTheClientTreatsItAsASubsetArm:
    def test_ova_lp_is_a_subset_arm_by_the_same_test_as_frozen_head(self):
        """USE_DERIVED gates the subset-federation path -- what the client uploads, and whether the
        arm is applied at all. It was `TRAINING_ARM == "FROZEN_HEAD"`, a NAME comparison, which
        silently excluded OVA_LP: the arm would have trained the whole model under a one-vs-all
        loss and uploaded every parameter to a server expecting the head."""
        for arm in ("FROZEN_HEAD", "OVA_LP"):
            assert recipes.trainable_prefixes("CIFAR_RESNET18", arm) is not None, \
                f"{arm} must be recognised as a subset arm"
        assert recipes.trainable_prefixes("CIFAR_RESNET18", "FULL") is None

    def test_the_client_does_not_compare_arm_names_to_decide_this(self):
        import ast

        src = ast.parse(open(os.path.join(HERE, "..", "client.py")).read())
        for node in ast.walk(src):
            if not isinstance(node, ast.Assign):
                continue
            if not any(getattr(tgt, "id", None) == "USE_DERIVED" for tgt in node.targets):
                continue
            code = ast.unparse(node.value)
            assert "FROZEN_HEAD" not in code, (
                f"USE_DERIVED is set by comparing arm names ({code}), which silently excludes any "
                f"new subset arm. Derive it from trainable_prefixes instead.")


class TestTheTwoStageScheduleIsNotClaimed:
    def test_the_recipe_records_what_is_unimplemented(self):
        """The paper describes three components; two are implemented. Silently shipping a partial
        method under the paper's name would misattribute a design to a citation."""
        meta = recipes._METADATA_BY_KEY["CIFAR_RESNET18"]
        spec = meta.get("arm_notes", {}).get(ARM, "")
        assert "two-stage" in spec.lower(), \
            "the recipe does not record that OvA-LP's two-stage schedule is unimplemented"
