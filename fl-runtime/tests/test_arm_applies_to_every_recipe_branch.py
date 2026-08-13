"""The arm must be applied whichever branch built the model.

Found by running PNEUMONIA_CNN live under FROZEN_HEAD. The run completed, reported itself as
FROZEN_HEAD, and produced a saved model whose backbone had **changed**::

    BACKBONE (features.*) after a FROZEN_HEAD run: CHANGED — it was NOT frozen
      max|delta| backbone = 6.505e-01

A full fine-tune wearing the frozen label. That is commit ``21699bc``'s bug — "frozen arm silently
mislabelled" — reproduced at the product level, and worse than a mislabel: the eval card, the
project row and the run's provenance all say FROZEN_HEAD.

The cause is branch ordering. The client builds its model through an if/elif chain, and
``apply_arm`` was called only inside the ``USE_DERIVED`` branch. ``USE_PNEUMONIA`` is tested first,
so a pneumonia model was built and the arm never applied. ``USE_LLM_LORA`` and ``USE_LLM`` sit
ahead of it too.

The arm is a CROSS-CUTTING property: it says which parameters train, for whatever model was built.
Applying it inside one branch of a build chain makes correctness depend on branch order, which is
exactly how this stayed invisible — the CIFAR recipes reach the `USE_DERIVED` branch, so they froze
correctly, and only a recipe with its own earlier branch exposed it.
"""

import os
import sys

import pytest
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "framework", "src"))

import client  # noqa: E402
import recipes  # noqa: E402


@pytest.fixture
def restore_globals():
    saved = {k: getattr(client, k) for k in
             ("USE_LLM", "USE_MLP", "USE_PNEUMONIA", "USE_LLM_LORA", "USE_DERIVED",
              "TRAINING_ARM", "MODEL_TYPE")}
    yield
    for k, v in saved.items():
        setattr(client, k, v)


def _configure(monkeypatch, model_type, arm):
    for flag in ("USE_LLM", "USE_MLP", "USE_PNEUMONIA", "USE_LLM_LORA"):
        monkeypatch.setattr(client, flag, False, raising=False)
    monkeypatch.setattr(client, "USE_PNEUMONIA", model_type == "PNEUMONIA_CNN", raising=False)
    monkeypatch.setattr(client, "MODEL_TYPE", model_type, raising=False)
    monkeypatch.setattr(client, "TRAINING_ARM", arm, raising=False)
    monkeypatch.setattr(client, "USE_DERIVED", arm == "FROZEN_HEAD", raising=False)


class TestThereIsOneUploadPath:
    """What a client federates must be decided in ONE place.

    `fit()` forked: subset arms returned `get_parameters()` and the FULL arm returned
    `self.net.state_dict()` raw. That second path bypassed the wire filter, so a BatchNorm model's
    int64 `num_batches_tracked` re-entered through the round-1 aggregate and the round-2 download
    failed to serialise — after round 1 had reported success, which is the worst shape for a bug.

    Third time this session that a second path around a rule caused a defect: the arm applied in
    one build branch, the dataset chosen from the arm, and now the payload built two ways.
    """

    def test_fit_returns_the_payload_only_via_get_parameters(self):
        import ast

        src = ast.parse(open(os.path.join(HERE, "..", "client.py")).read())
        fits = [n for n in ast.walk(src)
                if isinstance(n, ast.FunctionDef) and n.name == "fit"]
        assert fits, "client.fit() not found"

        offenders = []
        for fn in fits:
            for node in ast.walk(fn):
                if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Tuple):
                    continue
                payload = node.value.elts[0] if node.value.elts else None
                unparsed = ast.unparse(payload) if payload is not None else ""
                if "get_parameters" not in unparsed:
                    offenders.append(f"line {node.lineno}: returns {unparsed[:60]}")
        assert not offenders, (
            "client.fit() builds its upload payload somewhere other than get_parameters():\n  "
            + "\n  ".join(offenders)
            + "\nget_parameters() owns what this client federates for every arm; a second path "
              "around it bypasses the wire filter.")


class TestTheArmSurvivesTheBuildChain:
    @pytest.mark.slow
    def test_pneumonia_frozen_head_freezes_its_backbone(self, monkeypatch, restore_globals):
        """THE regression. PNEUMONIA_CNN has its own build branch ahead of the arm's, so this is
        the recipe on which the arm was silently dropped."""
        _configure(monkeypatch, "PNEUMONIA_CNN", "FROZEN_HEAD")
        net = recipes.get_recipe("PNEUMONIA_CNN").build_model("cpu")
        client.apply_declared_arm(net)

        trainable = {n for n, p in net.named_parameters() if p.requires_grad}
        frozen = {n for n, p in net.named_parameters() if not p.requires_grad}
        assert trainable, "nothing is trainable"
        assert frozen, "NOTHING was frozen — the arm was dropped"
        assert all(n.startswith("classifier.") for n in trainable), \
            f"non-head parameters are trainable under FROZEN_HEAD: {sorted(trainable - {n for n in trainable if n.startswith('classifier.')})}"
        assert all(n.startswith("features.") for n in frozen)

    @pytest.mark.slow
    def test_a_full_arm_leaves_everything_trainable(self, monkeypatch, restore_globals):
        _configure(monkeypatch, "PNEUMONIA_CNN", "FULL")
        net = recipes.get_recipe("PNEUMONIA_CNN").build_model("cpu")
        client.apply_declared_arm(net)
        assert all(p.requires_grad for p in net.parameters())

    def test_applying_the_arm_is_independent_of_the_build_branch(self):
        """The structural guard: apply_arm must not be reachable only from one branch of the build
        chain. Expressed as 'the client exposes a single entry point that the build chain calls
        unconditionally', because branch order is what made this invisible."""
        assert hasattr(client, "apply_declared_arm"), \
            "the arm is applied inline in a build branch; it must be a cross-cutting step"

    @pytest.mark.slow
    def test_the_frozen_payload_is_the_declared_subset(self, monkeypatch, restore_globals):
        """What actually goes on the wire has to match the declaration, since that is what the
        server filtered its own parameters to."""
        from fedlearn.estimators.params import trainable_state

        _configure(monkeypatch, "PNEUMONIA_CNN", "FROZEN_HEAD")
        net = recipes.get_recipe("PNEUMONIA_CNN").build_model("cpu")
        client.apply_declared_arm(net)
        keys = set(trainable_state(net))
        expected = {n for n, _ in net.named_parameters() if n.startswith("classifier.")}
        assert keys == expected, f"payload {sorted(keys)} != declared subset {sorted(expected)}"
