"""P1-3: the arm rides on every result, and cannot silently collide with another arm's result.

Written before the implementation (TDD).

P1-2 made the arm a persisted property of a project. That alone does not fix commit `21699bc` —
*"frozen arm silently mislabelled its backbone, risking cell overwrites"*. The overwrite happens
downstream, when results are **keyed**: if the identity of a result does not include the arm, then
two arms of the same recipe produce the same key and the second silently replaces the first. A
warning does not fix that; only putting the arm in the key does.

So this pins two things:

1. **The eval card carries the arm** — the production result record must be able to answer "which
   arm produced this?" without consulting the project row, because a card outlives and travels
   independently of it (it is attached to a registered model artifact).
The second half of this concern — that two cells cannot silently overwrite each other — is NOT
here. It belongs at the point where cells are actually written, which is the research harness's
``_emit_run``; see ``framework/tests/test_emit_run_overwrite_guard.py``. A cell key defined in
``recipes.py`` had no caller: ``fl-runtime`` does not write per-configuration result files, and the
harness that does has always led its filename with the arm.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402


class _Args:
    """Minimal stand-in for the argparse namespace build_eval_card reads."""

    def __init__(self, **kw):
        self.model_type = kw.get("model_type", "PNEUMONIA_CNN")
        self.strategy = kw.get("strategy", "FedAvg")
        self.num_rounds = kw.get("num_rounds", 5)
        self.seed = kw.get("seed", 0)
        self.training_arm = kw.get("training_arm", "FULL")


def _card(**kw):
    import fl_server
    return json.loads(fl_server.build_eval_card(_Args(**kw), [(1, {"loss": 0.1, "accuracy": 0.9})]))


# --------------------------------------------------------------------------------------
# 1. The eval card carries the arm
# --------------------------------------------------------------------------------------

class TestEvalCardCarriesTheArm:
    def test_card_records_the_arm(self):
        assert _card(training_arm="FROZEN_HEAD")["training_arm"] == "FROZEN_HEAD"

    def test_card_records_full_explicitly(self):
        """Absence must not be how FULL is expressed: a reader cannot distinguish 'FULL' from
        'written by a version that did not record the arm'."""
        assert _card(training_arm="FULL")["training_arm"] == "FULL"

    def test_card_records_the_trainable_prefixes(self):
        """Two runs can share an arm NAME while freezing different modules. The name alone is not
        provenance; the prefixes are what make the claim checkable."""
        card = _card(training_arm="FROZEN_HEAD")
        assert card["trainable_prefixes"] == recipes.trainable_prefixes(
            "PNEUMONIA_CNN", "FROZEN_HEAD")

    def test_full_arm_records_null_prefixes_not_a_missing_key(self):
        card = _card(training_arm="FULL")
        assert "trainable_prefixes" in card
        assert card["trainable_prefixes"] is None

    def test_the_card_is_still_valid_json_with_every_prior_field(self):
        """The card is consumed by the backend upload gate; adding fields must not drop any."""
        card = _card(training_arm="FROZEN_HEAD")
        for k in ("recipe_key", "strategy", "rounds", "final_loss", "final_accuracy",
                  "torch_version", "seed", "framework"):
            assert k in card, f"eval card lost {k!r}"
