"""Final-study inference checks; every cohort and prediction here is synthetic."""
import importlib

import numpy as np
import pytest
from scipy.stats import t
from sklearn.metrics import roc_auc_score


def module():
    return importlib.import_module("finalist_inference")


def fixture():
    # Unequal cluster sizes, mixed-class clusters, ties, and two distinct heads.
    labels = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    patients = np.array(["c", "c", "a", "a", "a", "b", "d", "d"])
    left = np.array([[.1, .9, .5, .5, .3, .8, .7, .6],
                     [.5, .5, .3, .1, .8, .7, .9, .2]])
    right = np.array([[.6, .2, .5, .5, .8, .1, .4, .7],
                      [.1, .9, .5, .5, .3, .8, .7, .6]])
    return labels, patients, left, right


def reference_bootstrap(labels, patients, left, right, *, draws, seed, confidence=.95):
    """Slow independent reference: materialize entire sampled image clusters."""
    rng = np.random.default_rng(seed)
    clusters = [np.flatnonzero(patients == patient) for patient in sorted(set(patients))]
    effects = []
    invalid = 0
    while len(effects) < draws:
        rows = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
        if len(np.unique(labels[rows])) != 2:
            invalid += 1
            continue
        effects.append(np.mean([
            roc_auc_score(labels[rows], a[rows]) - roc_auc_score(labels[rows], b[rows])
            for a, b in zip(left, right)
        ]))
    alpha = (1 - confidence) / 2
    return np.quantile(effects, [alpha, 1 - alpha]), invalid


def test_paired_seed_interval_matches_scipy_and_preserves_paired_effects():
    effects = {11: .04, 12: -.01, 13: .02, 14: .06, 15: .03}
    result = module().paired_seed_interval(effects, confidence=.9)
    values = np.array(list(effects.values()))
    point, sd = np.mean(values), np.std(values, ddof=1)
    half = t.ppf(.95, 4) * sd / np.sqrt(5)
    assert result["n"] == 5 and result["df"] == 4
    assert result["point"] == pytest.approx(point)
    assert result["sample_sd"] == pytest.approx(sd)
    assert result["se"] == pytest.approx(sd / np.sqrt(5))
    assert [result["ci_low"], result["ci_high"]] == pytest.approx([point - half, point + half])
    assert result["seed_effects"] == effects
    assert "conditional on fixed evaluation cohort" in result["scope"]
    assert "not population/joint interval" in result["scope"]
    result["seed_effects"][11] = 999
    assert effects[11] == .04


def test_constant_seed_effects_have_zero_width_interval():
    result = module().paired_seed_interval({1: .125, 2: .125})
    assert result["ci_low"] == result["ci_high"] == .125
    assert result["sample_sd"] == result["se"] == 0


@pytest.mark.parametrize("effects", [{}, {1: .1}, {1: .1, 2: np.nan},
                                    {1: np.inf, 2: .2}, {1: True, 2: .2},
                                    {1: ".2", 2: .1}, [(1, .1), (1, .2)]])
def test_seed_interval_rejects_insufficient_or_nonfinite_effects(effects):
    with pytest.raises(ValueError):
        module().paired_seed_interval(effects)


@pytest.mark.parametrize("confidence", [0, 1, -1, np.nan, np.inf, True, ".95"])
def test_intervals_reject_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        module().paired_seed_interval({1: .1, 2: .2}, confidence=confidence)
    with pytest.raises(ValueError):
        module().conditional_patient_interval(*fixture(), draws=5, confidence=confidence)


def test_weighted_auc_matches_hand_ties_and_sklearn_weights():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0., .5, .5, 1.])
    weights = np.array([2., 3., 4., 5.])
    # 3*(2 + .5*4) + 5*(2 + 4) = 42 concordant mass / (8*6).
    assert module().weighted_auc(labels, scores, weights) == pytest.approx(42 / 48)
    rng = np.random.default_rng(7)
    for _ in range(15):
        labels = rng.integers(2, size=31)
        scores = rng.integers(5, size=31).astype(float)
        weights = rng.integers(4, size=31).astype(float)
        assert module().weighted_auc(labels, scores, weights) == pytest.approx(
            roc_auc_score(labels, scores, sample_weight=weights), abs=1e-14)


def test_patient_interval_uses_shared_cluster_samples_and_image_level_point():
    inputs = fixture()
    result = module().conditional_patient_interval(*inputs, draws=127, seed=81, confidence=.8)
    expected, invalid = reference_bootstrap(*inputs, draws=127, seed=81, confidence=.8)
    point = np.mean([roc_auc_score(inputs[0], a) - roc_auc_score(inputs[0], b)
                     for a, b in zip(inputs[2], inputs[3])])
    assert result["point"] == pytest.approx(point)
    assert [result["ci_low"], result["ci_high"]] == pytest.approx(expected)
    assert result["invalid_draws"] == invalid
    assert (result["draws"], result["n_heads"], result["n_images"], result["n_patients"]) == (127, 2, 8, 4)
    assert "conditional on realized heads" in result["scope"]
    assert "not training-seed or joint uncertainty" in result["scope"]
    assert set(result) == {"point", "ci_low", "ci_high", "draws", "invalid_draws",
                           "n_heads", "n_images", "n_patients", "scope"}


def test_identical_heads_have_exact_zero_gain_under_shared_draws():
    labels, patients, left, _ = fixture()
    result = module().conditional_patient_interval(labels, patients, left, left.copy(), draws=73, seed=2)
    assert result["point"] == result["ci_low"] == result["ci_high"] == 0


def test_cluster_bootstrap_is_row_order_invariant_reproducible_and_does_not_mutate():
    inputs = fixture()
    originals = [x.copy() for x in inputs]
    np.random.seed(981)
    global_state = np.random.get_state()
    first = module().conditional_patient_interval(*inputs, draws=59, seed=413)
    second = module().conditional_patient_interval(*inputs, draws=59, seed=413)
    permutation = np.array([6, 2, 4, 0, 7, 1, 5, 3])
    shuffled = module().conditional_patient_interval(
        inputs[0][permutation], inputs[1][permutation],
        inputs[2][:, permutation], inputs[3][:, permutation], draws=59, seed=413)
    assert first == second
    assert [shuffled[k] for k in ("point", "ci_low", "ci_high")] == pytest.approx(
        [first[k] for k in ("point", "ci_low", "ci_high")])
    current = np.random.get_state()
    assert current[0] == global_state[0]
    np.testing.assert_array_equal(current[1], global_state[1])
    assert current[2:] == global_state[2:]
    for actual, original in zip(inputs, originals):
        np.testing.assert_array_equal(actual, original)


def test_single_class_draws_are_counted_and_attempt_guard_fails_closed():
    labels = np.array([0, 1])
    patients = np.array(["a", "b"])
    left, right = np.array([[0., 1.]]), np.array([[1., 0.]])
    expected, invalid = reference_bootstrap(labels, patients, left, right, draws=30, seed=4)
    result = module().conditional_patient_interval(labels, patients, left, right,
                                                    draws=30, seed=4, max_attempts=300)
    assert invalid > 0 and result["invalid_draws"] == invalid
    assert [result["ci_low"], result["ci_high"]] == pytest.approx(expected)
    with pytest.raises(RuntimeError, match="max_attempts"):
        module().conditional_patient_interval(labels, patients, left, right,
                                               draws=30, seed=4, max_attempts=30)


@pytest.mark.parametrize("change", ["left_1d", "right_shape", "no_heads", "no_images",
    "label_length", "label_2d", "one_class", "nonbinary", "label_nan", "patient_length",
    "patient_2d", "patient_none", "patient_nan", "patient_blank", "left_inf", "right_nan"])
def test_patient_interval_rejects_invalid_inputs(change):
    labels, patients, left, right = fixture()
    patients = patients.astype(object)
    if change == "left_1d": left = left[0]
    elif change == "right_shape": right = right[:1]
    elif change == "no_heads": left, right = left[:0], right[:0]
    elif change == "no_images": labels, patients, left, right = labels[:0], patients[:0], left[:, :0], right[:, :0]
    elif change == "label_length": labels = labels[:-1]
    elif change == "label_2d": labels = labels[:, None]
    elif change == "one_class": labels = np.zeros_like(labels)
    elif change == "nonbinary": labels[0] = 2
    elif change == "label_nan": labels = labels.astype(float); labels[0] = np.nan
    elif change == "patient_length": patients = patients[:-1]
    elif change == "patient_2d": patients = patients[:, None]
    elif change == "patient_none": patients[0] = None
    elif change == "patient_nan": patients[0] = np.nan
    elif change == "patient_blank": patients[0] = " "
    elif change == "left_inf": left[0, 0] = np.inf
    else: right[0, 0] = np.nan
    with pytest.raises(ValueError):
        module().conditional_patient_interval(labels, patients, left, right, draws=5)


@pytest.mark.parametrize("kwargs", [{"draws": 0}, {"draws": -1}, {"draws": 1.5},
    {"draws": True}, {"draws": 3, "max_attempts": 2}, {"max_attempts": 1.5},
    {"max_attempts": True}, {"seed": -1}, {"seed": True}, {"seed": np.random.default_rng(1)}])
def test_patient_interval_rejects_invalid_sampling_budget_or_rng_seed(kwargs):
    with pytest.raises(ValueError):
        module().conditional_patient_interval(*fixture(), **kwargs)


@pytest.mark.parametrize("labels,scores,weights", [([], [], []), ([0, 0], [0., 1.], [1., 1.]),
    ([0, 1], [0., 1.], [0., 1.]), ([0, 1], [0., 1.], [1., -1.]),
    ([0, 1], [0., np.inf], [1., 1.]), ([0, 1], [0., 1.], [1., np.nan]),
    ([0, 2], [0., 1.], [1., 1.]), ([0, 1], [0.], [1., 1.]),
    ([0, 1], [0., 1.], [1.])])
def test_weighted_auc_rejects_invalid_mass_or_shapes(labels, scores, weights):
    with pytest.raises(ValueError):
        module().weighted_auc(labels, scores, weights)
