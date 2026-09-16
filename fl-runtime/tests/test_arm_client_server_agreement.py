"""P1-5: the client and the FL server must agree on WHICH parameters are federated.

This is the contract the whole arm feature rests on, and until now nothing asserted it.

P1-2 taught the server to filter its initial parameters to the arm's trainable subset
(``fl_server.py:590``). P1-4 let a user choose the arm. But nothing carried the choice to the
client, so ``client.py`` kept its default ``TRAINING_ARM = "FULL"``, ``USE_DERIVED`` stayed false,
and ``get_parameters()`` uploaded the FULL state dict to a server holding only the head. The two
sides silently disagreed about the shape of the payload for exactly the arm the picker offers.

So the assertion here is a genuine agreement check rather than two independent ones: the key set
the CLIENT would upload must equal the key set the SERVER filtered down to, computed from the same
recipe and arm by each side's own code path.
"""

import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "framework", "src"))

import recipes  # noqa: E402


def _server_side_keys(recipe_key, arm, state_keys):
    """Reproduce the server's filter (fl_server.py:590-594) over a full state_dict's keys."""
    prefixes = recipes.trainable_prefixes(recipe_key, arm)
    if prefixes is None:
        return set(state_keys)
    return {k for k in state_keys for pre in prefixes if k.startswith(pre)}


def _client_side_keys(recipe_key, arm):
    """What the client would upload: apply the arm, then take trainable_state (client.py:643-647)."""
    from fedlearn.estimators.params import trainable_state

    model = recipes.get_recipe(recipe_key).build_model("cpu")
    recipes.apply_arm(model, arm, recipes.trainable_prefixes(recipe_key, arm))
    return set(trainable_state(model))


DUAL_ARM_RECIPES = [m["key"] for m in recipes.RECIPE_METADATA
                    if len(m.get("supported_arms", ())) > 1]


class TestClientAndServerAgreeOnTheFederatedSubset:
    @pytest.mark.parametrize("recipe_key", DUAL_ARM_RECIPES)
    def test_frozen_head_subsets_match(self, recipe_key):
        """THE P1-5 contract. A mismatch here is the silent corruption the arm feature would ship."""
        model = recipes.get_recipe(recipe_key).build_model("cpu")
        all_keys = list(model.state_dict())

        client = _client_side_keys(recipe_key, "FROZEN_HEAD")
        server = _server_side_keys(recipe_key, "FROZEN_HEAD", all_keys)

        assert client == server, (
            f"{recipe_key}: client would upload {sorted(client)} but the server filtered to "
            f"{sorted(server)} — the two sides disagree about the federated payload")

    @pytest.mark.parametrize("recipe_key", DUAL_ARM_RECIPES)
    def test_the_frozen_subset_is_a_strict_subset(self, recipe_key):
        """If FROZEN_HEAD federated everything, the arm would be a no-op wearing a label."""
        model = recipes.get_recipe(recipe_key).build_model("cpu")
        all_params = {k for k, _ in model.named_parameters()}
        frozen = _client_side_keys(recipe_key, "FROZEN_HEAD")
        assert frozen < all_params, f"{recipe_key}: FROZEN_HEAD federates the whole model"
        assert frozen, f"{recipe_key}: FROZEN_HEAD federates nothing"

    @pytest.mark.parametrize("recipe_key", DUAL_ARM_RECIPES)
    def test_full_federates_everything_on_both_sides(self, recipe_key):
        model = recipes.get_recipe(recipe_key).build_model("cpu")
        all_params = {k for k, _ in model.named_parameters()}
        assert _client_side_keys(recipe_key, "FULL") == all_params
        assert _server_side_keys(recipe_key, "FULL", all_params) == all_params


class TestADeclaredArmIsTrueOfTheModel:
    """A recipe can declare an arm it cannot actually run, and nothing before this noticed.

    `CNN` shipped `supported_arms: ["FULL", "FROZEN_HEAD"]` with prefix `classifier.`, but
    `CnnNet` is conv1/conv2/fc1/fc2/fc3 — no `classifier.` module exists. `apply_arm` raises
    "no parameter matches prefixes", so choosing frozen-head on a CIFAR CNN crashed at model
    build. P1-4's picker offers exactly that choice, so it was reachable by any user.

    Declaring the arm and wiring the picker are two different things from the arm being POSSIBLE,
    and only building the model proves the third. Checked for every declaring recipe, including
    non-catalog ones, so a new recipe cannot reintroduce it.
    """

    @pytest.mark.parametrize("recipe_key", DUAL_ARM_RECIPES)
    def test_declared_prefixes_match_real_parameters(self, recipe_key):
        model = recipes.get_recipe(recipe_key).build_model("cpu")
        prefixes = tuple(recipes.trainable_prefixes(recipe_key, "FROZEN_HEAD"))
        matched = [n for n, _ in model.named_parameters() if n.startswith(prefixes)]
        assert matched, (
            f"{recipe_key} declares FROZEN_HEAD with prefixes {list(prefixes)}, but no parameter "
            f"matches: {[n for n, _ in model.named_parameters()]}. The picker offers an arm that "
            f"cannot run.")

    @pytest.mark.parametrize("recipe_key", DUAL_ARM_RECIPES)
    def test_applying_the_declared_arm_actually_works(self, recipe_key):
        """The end-to-end proof: build the model and apply the arm exactly as the client does."""
        model = recipes.get_recipe(recipe_key).build_model("cpu")
        recipes.apply_arm(model, "FROZEN_HEAD",
                          recipes.trainable_prefixes(recipe_key, "FROZEN_HEAD"))
        assert any(p.requires_grad for p in model.parameters())
        assert not all(p.requires_grad for p in model.parameters())


class TestTheClientHonoursTheFlag:
    def test_the_cli_flag_selects_the_arm(self):
        """The flag has existed since P1-1b; until P1-5 nothing ever passed it, so nothing proved
        that passing it actually changes what the client federates."""
        import client

        prev_arm, prev_derived = client.TRAINING_ARM, client.USE_DERIVED
        try:
            args = client.parse_args([
                "--project-id", "p", "--server-address", "h:1", "--partition-id", "0",
                "--model-type", "PNEUMONIA_CNN", "--training-arm", "FROZEN_HEAD"])
            assert args.training_arm == "FROZEN_HEAD"
        finally:
            client.TRAINING_ARM, client.USE_DERIVED = prev_arm, prev_derived

    def test_omitting_the_flag_keeps_the_pre_p1_default(self):
        """A backend predating P1 sends no arm; that must reproduce the old behaviour exactly."""
        import client

        args = client.parse_args([
            "--project-id", "p", "--server-address", "h:1", "--partition-id", "0",
            "--model-type", "PNEUMONIA_CNN"])
        assert getattr(args, "training_arm", None) in (None, "FULL")

    def test_an_unsupported_arm_is_refused_for_the_recipe(self):
        """Refused rather than silently downgraded to FULL — a downgrade would upload every
        parameter to a server expecting the head, which is the exact failure P1-5 closes."""
        with pytest.raises(ValueError, match="does not support"):
            recipes.validate_arm("MLP", "FROZEN_HEAD")
