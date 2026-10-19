import copy
import json

import numpy as np
import pytest

pytest.importorskip("xgboost")

from fedlearn.tree.model import initial_model, train_delta, merge_trees, validate_model, load_model


@pytest.fixture
def rows():
    x = np.random.default_rng(42).normal(size=(96, 4)).astype("float32")
    return x, (x[:, 0] + x[:, 1] > 0).astype("float32")


def test_two_client_rounds_preserve_old_trees_and_add_deltas_once(rows):
    parent = initial_model()
    x, y = rows
    previous = []
    for round_number in range(1, 4):
        a = train_delta(parent, x[:48], y[:48], cohort_size=2, seed=round_number)
        b = train_delta(parent, x[48:], y[48:], cohort_size=2, seed=round_number)
        assert len(validate_model(a, expected_trees=1)["learner"]["gradient_booster"]["model"]["trees"]) == 1
        combined = merge_trees(parent, [("b", b), ("a", a)])
        assert combined == merge_trees(parent, [("a", a), ("b", b)])
        trees = json.loads(combined)["learner"]["gradient_booster"]["model"]["trees"]
        assert trees[:len(previous)] == previous
        assert len(trees) == round_number * 2
        previous, parent = copy.deepcopy(trees), combined
    booster = load_model(parent)
    assert booster.num_boosted_rounds() == 6
    prediction = booster.inplace_predict(x)
    assert np.isfinite(prediction).all()
    assert (prediction >= .5).astype(int).tolist() != [1] * len(x)
    assert np.mean((prediction >= .5) == y) > .8
    ubj = booster.save_raw(raw_format="ubj")
    import xgboost as xgb
    restored = xgb.Booster()
    restored.load_model(ubj)
    np.testing.assert_array_equal(restored.inplace_predict(x), prediction)


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(version=[3, 0, 0]),
    lambda d: d.update(version=[3, 4, True]),
    lambda d: d["learner"].update(feature_names=["f1", "f0", "f2", "f3"]),
    lambda d: d["learner"]["objective"].update(name="reg:squarederror"),
    lambda d: d["learner"]["learner_model_param"].update(base_score="[8E-1]"),
    lambda d: d["learner"]["gradient_booster"]["model"].update(tree_info=[1]),
    lambda d: d["learner"]["gradient_booster"]["model"].update(tree_info=[False]),
    lambda d: d["learner"]["gradient_booster"]["model"].update(iteration_indptr=[False, True]),
    lambda d: d["learner"]["gradient_booster"]["model"].update(iteration_indptr=[0, 2]),
    lambda d: d["learner"]["gradient_booster"]["model"]["trees"][0].update(id=7),
    lambda d: d["learner"]["gradient_booster"]["model"]["trees"][0]["split_conditions"].__setitem__(0, float("nan")),
    lambda d: d["learner"]["gradient_booster"]["model"]["trees"][0]["left_children"].__setitem__(0, 0),
    lambda d: d["learner"]["gradient_booster"]["model"]["trees"][0]["split_indices"].__setitem__(0, 4),
    lambda d: d["learner"]["gradient_booster"]["model"]["trees"][0]["parents"].__setitem__(0, 0),
    lambda d: d["learner"]["gradient_booster"]["model"]["trees"][0]["tree_param"].update(num_nodes="1000000"),
])
def test_malformed_contributions_are_rejected_before_native_loading(rows, mutation):
    x, y = rows
    delta = json.loads(train_delta(initial_model(), x, y, cohort_size=2, seed=1))
    mutation(delta)
    with pytest.raises(ValueError):
        validate_model(json.dumps(delta).encode(), expected_trees=1)


def test_full_parent_cannot_be_submitted_as_delta_and_duplicate_ids_are_refused(rows):
    x, y = rows
    delta = train_delta(initial_model(), x, y, cohort_size=2, seed=1)
    parent = merge_trees(initial_model(), [("a", delta), ("b", delta)])
    with pytest.raises(ValueError):
        merge_trees(parent, [("a", parent)])
    with pytest.raises(ValueError):
        merge_trees(initial_model(), [("a", delta), ("a", delta)])


def test_nonfinite_data_and_wrong_feature_width_are_refused(rows):
    x, y = rows
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        train_delta(initial_model(), x, y, cohort_size=2, seed=1)
    with pytest.raises(ValueError):
        train_delta(initial_model(), x[:, :3], y, cohort_size=2, seed=1)
