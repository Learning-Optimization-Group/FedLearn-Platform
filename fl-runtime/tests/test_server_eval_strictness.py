"""Server-side evaluation must load a SUBSET-federated global model non-strictly.

The fourth defect this live FROZEN_HEAD run surfaced, and the one that stopped a completed round
from being evaluated. A client trained and submitted successfully; then::

    RuntimeError: Error(s) in loading state_dict for CnnNet:
        Missing key(s) in state_dict: "conv1.weight", ..., "fc2.bias"

The server's global model is head-only — correct, that is the whole point of the arm — but
`server_side_evaluate` loaded it with::

    _strict = args.model_type.upper() != 'TINYNET_GOLDEN'

which hardcodes the ONE recipe that historically federated a subset. Under P1, any recipe running
FROZEN_HEAD federates a subset, so strictness has to follow *whether the federated set is a
subset*, not a recipe name.

`recipes.trainable_prefixes(recipe, arm) is not None` is exactly that predicate, so the decision is
derived from it rather than from a growing list of special-cased keys.
"""

import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

import fl_server  # noqa: E402
import recipes  # noqa: E402


class TestEvaluationLoadStrictness:
    def test_a_frozen_arm_loads_non_strictly(self):
        """THE regression: the frozen global model lacks the backbone keys by design."""
        assert fl_server.evaluation_load_is_strict("CNN", "FROZEN_HEAD") is False

    def test_a_full_arm_still_loads_strictly(self):
        """Strictness is a real guard for the full arm — it catches a genuinely malformed payload,
        and must not be relaxed globally just to accommodate the frozen arm."""
        assert fl_server.evaluation_load_is_strict("CNN", "FULL") is True

    def test_the_golden_demo_stays_non_strict(self):
        """TINYNET_GOLDEN syncs only its 25 trainable fc1 params; fc2 lives only in the fresh net.
        It was the original reason for the exception and must keep working."""
        assert fl_server.evaluation_load_is_strict("TINYNET_GOLDEN", "FULL") is False

    def test_an_omitted_arm_means_full(self):
        assert fl_server.evaluation_load_is_strict("CNN", None) is True

    def test_wire_withheld_tensors_force_a_non_strict_load(self):
        """Unblocking BatchNorm models for the FULL arm means the global model legitimately lacks
        their int64 num_batches_tracked. A strict load would then fail on exactly the keys the wire
        was told not to carry, so strictness has to know about the wire filter as well as the arm."""
        assert fl_server.evaluation_load_is_strict("CNN", "FULL", withheld=0) is True
        assert fl_server.evaluation_load_is_strict("CNN", "FULL", withheld=20) is False

    @pytest.mark.parametrize(
        "recipe_key", [m["key"] for m in recipes.RECIPE_METADATA
                       if len(m.get("supported_arms", ())) > 1])
    def test_every_dual_arm_recipe_is_non_strict_when_frozen(self, recipe_key):
        """Derived from the arm, so a recipe that gains FROZEN_HEAD later is covered automatically
        rather than needing another name added to a special-case list."""
        assert fl_server.evaluation_load_is_strict(recipe_key, "FROZEN_HEAD") is False
        assert fl_server.evaluation_load_is_strict(recipe_key, "FULL") is True
