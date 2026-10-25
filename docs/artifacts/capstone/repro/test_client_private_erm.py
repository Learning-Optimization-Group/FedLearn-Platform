"""Generated-data checks for scalar client-average output perturbation."""
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit

import client_private_erm as erm


def fit(x, y, partitions, **kwargs):
    options = dict(row_bound=2., regularization=.4, steps=100)
    options.update(kwargs)
    return erm.fit_client_private_erm(x, y, partitions, **options)


def test_one_step_uses_equal_client_means_not_equal_rows():
    # At zero, the positive singleton contributes -1/2 and the negative
    # three-row client contributes +1/2 to the intercept: they cancel.
    result = fit([[2.], [0.], [0.], [0.]], [1, 0, 0, 0], [[0], [1, 2, 3]], steps=1)
    np.testing.assert_allclose(result['theta'], [.5 / 1.65, 0.], atol=1e-15)
    assert result['theta'].dtype == np.float64
    assert result['research_diagnostics']['steps_completed'] == 1


def test_fixed_solver_matches_independent_scipy_optimum_and_gradient():
    rng = np.random.default_rng(72)
    x = rng.normal(scale=.3, size=(19, 3))
    y = rng.integers(0, 2, size=19)
    parts = [np.arange(0, 2), np.arange(2, 9), np.arange(9, 19)]
    augmented = np.column_stack((x, np.ones(len(x))))
    weights = np.zeros(len(x))
    for ids in parts:
        weights[ids] = 1. / (3 * len(ids))

    def objective(theta):
        logits = augmented @ theta
        return weights @ (np.logaddexp(0., logits) - y * logits) + .2 * (theta @ theta)

    def gradient(theta):
        return augmented.T @ (weights * (expit(augmented @ theta) - y)) + .4 * theta

    reference = minimize(objective, np.zeros(4), jac=gradient, method='BFGS', tol=1e-11)
    result = fit(x, y, parts, steps=300)
    np.testing.assert_allclose(result['theta'], reference.x, atol=1e-8)
    diag = result['research_diagnostics']
    assert diag['objective'] == pytest.approx(objective(result['theta']), abs=1e-14)
    assert diag['gradient_norm'] == pytest.approx(np.linalg.norm(gradient(result['theta'])), abs=1e-14)
    assert diag['privacy_status'] == 'NOT_DP_OUTPUTS'


def test_intercept_is_regularized_and_zero_features_stay_zero():
    result = fit(np.zeros((4, 2)), np.ones(4), [[0], [1, 2, 3]], steps=200)
    theta = result['theta']
    np.testing.assert_array_equal(theta[:2], [0., 0.])
    assert expit(theta[-1]) - 1. + .4 * theta[-1] == pytest.approx(0., abs=1e-14)
    assert 0. < theta[-1] < 2.5


def test_row_clip_happens_before_intercept_and_does_not_mutate_inputs():
    x = np.array([[3., 4.], [0., 0.], [0., -2.]], dtype=np.float32)
    y = np.array([1, 0, 1])
    parts = [np.array([0]), np.array([1, 2])]
    saved = (x.copy(), y.copy(), [p.copy() for p in parts])
    result = fit(x, y, parts, row_bound=1., steps=3)
    manual = fit([[.6, .8], [0., 0.], [0., -1.]], y, parts, row_bound=1., steps=3)
    np.testing.assert_allclose(result['theta'], manual['theta'], atol=1e-15)
    assert result['research_diagnostics']['feature_clip_rate'] == pytest.approx(2 / 3)
    np.testing.assert_array_equal(x, saved[0])
    np.testing.assert_array_equal(y, saved[1])
    for actual, original in zip(parts, saved[2]):
        np.testing.assert_array_equal(actual, original)


def test_clipping_handles_extreme_finite_rows_without_overflow():
    result = fit([[1e308, 1e308], [0., 0.]], [1, 0], [[0], [1]], row_bound=1., steps=1)
    manual = fit([[2 ** -.5, 2 ** -.5], [0., 0.]], [1, 0], [[0], [1]], row_bound=1., steps=1)
    np.testing.assert_allclose(result['theta'], manual['theta'], atol=1e-15)


def test_bound_values_step_selection_and_public_maximum_guard():
    # B=sqrt(3), L=2, lambda=1 => beta=2, q=1/2.
    bound = erm.solver_sensitivity_bound(row_bound=np.sqrt(3.), regularization=1., num_clients=4, steps=3)
    assert bound['augmented_row_bound'] == pytest.approx(2.)
    assert bound['beta'] == pytest.approx(2.)
    assert bound['q'] == pytest.approx(.5)
    assert bound['exact_minimizer_sensitivity'] == pytest.approx(1.)
    assert bound['solver_residual_sensitivity'] == pytest.approx(.5)
    assert bound['sensitivity'] == pytest.approx(1.5)
    chosen = erm.choose_solver_steps(row_bound=np.sqrt(3.), regularization=1., num_clients=4, residual_ratio=.125, max_steps=10)
    assert chosen['steps'] == 5
    assert chosen['solver_residual_sensitivity'] <= .125 * chosen['exact_minimizer_sensitivity'] * (1 + 1e-14)
    with pytest.raises(ValueError, match='max_steps'):
        erm.choose_solver_steps(row_bound=10., regularization=1e-10, num_clients=20, residual_ratio=.01, max_steps=100)


def test_bound_scales_with_clients_and_includes_zero_step_residual():
    one = erm.solver_sensitivity_bound(row_bound=0., regularization=2., num_clients=1, steps=0)
    five = erm.solver_sensitivity_bound(row_bound=0., regularization=2., num_clients=5, steps=0)
    assert one['exact_minimizer_sensitivity'] == 1.
    assert one['solver_residual_sensitivity'] == 1.
    assert five['exact_minimizer_sensitivity'] == .2
    assert five['solver_residual_sensitivity'] == 1.
    result = fit([[0.]], [1], [[0]], row_bound=0., steps=0)
    np.testing.assert_array_equal(result['theta'], [0., 0.])
    assert result['research_diagnostics']['steps_completed'] == 0


def test_adjacent_client_solver_distance_is_below_bound_illustrative_only():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(14, 2))
    y = rng.integers(0, 2, size=14)
    parts = [list(range(0, 3)), list(range(3, 10)), list(range(10, 14))]
    for steps in [0, 1, 7, 150]:
        original = fit(x, y, parts, steps=steps)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[:3] = -5 * changed_x[:3]
        changed_y[:3] = 1 - changed_y[:3]
        adjacent = fit(changed_x, changed_y, parts, steps=steps)
        distance = np.linalg.norm(original['theta'] - adjacent['theta'])
        assert distance <= original['public_bound']['sensitivity'] + 1e-12


def test_fixed_iteration_budget_even_when_initial_gradient_is_zero(monkeypatch):
    calls = 0
    original = erm._objective_gradient

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(erm, '_objective_gradient', counted)
    result = fit([[0.], [0.]], [0, 1], [[0], [1]], steps=17)
    np.testing.assert_array_equal(result['theta'], [0., 0.])
    assert result['research_diagnostics']['steps_completed'] == 17
    # One final gradient evaluation is a research diagnostic, never a stop rule.
    assert calls == 18


def test_noise_is_one_local_vector_draw_with_the_stated_scale(monkeypatch):
    theta = np.array([.2, -.1, 0.])
    original = theta.copy()
    expected = theta + np.random.default_rng(91).normal(0., 1.2, size=theta.shape)
    real_default_rng = np.random.default_rng
    calls = []

    class TracedGenerator:
        def __init__(self, seed):
            self.generator = real_default_rng(seed)

        def normal(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self.generator.normal(*args, **kwargs)

    monkeypatch.setattr(erm.np.random, 'default_rng', TracedGenerator)
    np.random.seed(5)
    state = np.random.get_state()
    released = erm.release_gaussian(theta, sensitivity=.4, noise_multiplier=3., seed=91)
    np.testing.assert_allclose(released['theta'], expected, atol=1e-15)
    assert len(calls) == 1
    assert released['noise_std'] == pytest.approx(1.2)
    assert released['noise_multiplier'] == 3.
    np.testing.assert_array_equal(theta, original)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_releases_reproduce_zero_control_and_have_actual_requested_variance():
    theta = np.zeros(100_000)
    result = erm.release_gaussian(theta, sensitivity=.3, noise_multiplier=2., seed=87)
    assert np.std(result['theta']) == pytest.approx(.6, rel=.015)
    repeat = erm.release_gaussian(theta, sensitivity=.3, noise_multiplier=2., seed=87)
    np.testing.assert_array_equal(result['theta'], repeat['theta'])
    doubled = erm.release_gaussian(theta, sensitivity=.6, noise_multiplier=2., seed=87)
    np.testing.assert_array_equal(doubled['theta'], 2 * result['theta'])
    clean = erm.release_gaussian(theta, sensitivity=.3, noise_multiplier=0., seed=87)
    np.testing.assert_array_equal(clean['theta'], theta)
    assert clean['theta'] is not theta


@pytest.mark.parametrize('parts', [[], [[]], [[0], []], [[0], [0, 1]], [[0]], [[0, 2]], [[-1], [0]], [[0.], [1.]], [[True], [False]], [[0, 0], [1]], [[0], [[1]]]])
def test_rejects_empty_overlapping_nonexhaustive_or_noninteger_partitions(parts):
    with pytest.raises(ValueError):
        fit([[0.], [1.]], [0, 1], parts)


@pytest.mark.parametrize('x,y', [([[np.nan], [1.]], [0, 1]), ([[np.inf], [1.]], [0, 1]), ([[0.], [1.]], [0, np.nan]), ([[0.], [1.]], [0, 2]), ([[0.], [1.]], [[0], [1]]), ([0., 1.], [0, 1]), ([], []), ([[0.], [1.]], [0])])
def test_rejects_nonfinite_features_nonbinary_labels_and_wrong_shapes(x, y):
    with pytest.raises(ValueError):
        fit(x, y, [[0], [1]])


@pytest.mark.parametrize('options', [dict(row_bound=-1.), dict(row_bound=np.inf), dict(regularization=0.), dict(regularization=np.nan), dict(steps=-1), dict(steps=True), dict(steps=1.5), dict(steps=101, max_steps=100)])
def test_rejects_invalid_public_solver_configuration(options):
    with pytest.raises(ValueError):
        fit([[0.]], [1], [[0]], **options)


@pytest.mark.parametrize('options', [dict(sensitivity=-1.), dict(sensitivity=np.nan), dict(noise_multiplier=-1.), dict(noise_multiplier=np.inf), dict(seed=True), dict(seed=-1), dict(sensitivity=1e308, noise_multiplier=1e308)])
def test_rejects_invalid_release_configuration(options):
    args = dict(sensitivity=1., noise_multiplier=1., seed=1)
    args.update(options)
    with pytest.raises(ValueError):
        erm.release_gaussian([0., 1.], **args)


@pytest.mark.parametrize('theta', [[np.nan], [np.inf], [[0.]], []])
def test_release_rejects_malformed_or_nonfinite_vectors(theta):
    with pytest.raises(ValueError):
        erm.release_gaussian(theta, sensitivity=1., noise_multiplier=1., seed=1)


def test_step_selection_checks_residual_after_logarithmic_rounding():
    # A floating-point logarithm can round the required integer down exactly
    # at the boundary. The returned public bound must still meet the target.
    chosen = erm.choose_solver_steps(
        row_bound=2.325608691903514, regularization=8.213219970431846,
        num_clients=92, residual_ratio=3.812319632186748e-12, max_steps=30,
    )
    assert chosen['solver_residual_sensitivity'] <= 3.812319632186748e-12 * chosen['exact_minimizer_sensitivity']
    with pytest.raises(ValueError, match='max_steps'):
        erm.choose_solver_steps(
            row_bound=2.325608691903514, regularization=8.213219970431846,
            num_clients=92, residual_ratio=3.812319632186748e-12, max_steps=17,
        )


def test_public_row_transform_matches_fit_and_preserves_evaluation_inputs():
    x = np.array([[3., 4.], [0., -2.], [0., 0.]])
    saved = x.copy()
    bounded = erm.bound_rows(x, 1.)
    np.testing.assert_array_equal(x, saved)
    np.testing.assert_allclose(bounded, [[.6, .8], [0., -1.], [0., 0.]], atol=1e-15)
    assert bounded.dtype == np.float64
    assert not np.shares_memory(x, bounded)
    expected = fit(bounded, [1, 0, 1], [[0], [1, 2]], row_bound=1., steps=7)
    actual = fit(x, [1, 0, 1], [[0], [1, 2]], row_bound=1., steps=7)
    np.testing.assert_array_equal(actual['theta'], expected['theta'])


@pytest.mark.parametrize('x,bound', [([[np.nan]], 1.), ([[np.inf]], 1.), ([1.], 1.), ([], 1.), ([[1.]], -1.), ([[1.]], np.nan)])
def test_public_row_transform_validates_inputs(x, bound):
    with pytest.raises(ValueError):
        erm.bound_rows(x, bound)


def test_bounded_objective_and_solver_remain_warning_free_after_torch_training():
    # Some macOS BLAS paths report stale FP flags after Torch work even for
    # zero logits. Exercise that transition; compare against scalar formulas.
    import warnings
    import torch

    torch.set_num_threads(1)
    x = np.zeros((8, 512), dtype=np.float64)
    x[:, 0] = [-1., 1.] * 4
    y = np.array([0, 1] * 4)
    with torch.random.fork_rng():
        torch.manual_seed(53)
        model = torch.nn.Linear(512, 2).double()
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        for _ in range(5):
            loss = torch.nn.functional.cross_entropy(model(torch.from_numpy(x)), torch.from_numpy(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    augmented = np.column_stack((x, np.ones(8)))
    theta = np.zeros(513)
    theta[0] = .5
    with warnings.catch_warnings(), np.errstate(divide='raise', over='raise', invalid='raise'):
        warnings.simplefilter('error', RuntimeWarning)
        objective, gradient = erm._objective_gradient(theta, augmented, y, np.full(8, .125), .1)
        budget = erm.choose_solver_steps(row_bound=1., regularization=.1, num_clients=2, residual_ratio=.01)
        result = fit(x, y, [list(range(4)), list(range(4, 8))], row_bound=1., regularization=.1, steps=budget['steps'])
    assert objective == pytest.approx(np.logaddexp(0., -.5) + .05 * .5 ** 2, abs=1e-15)
    assert gradient[0] == pytest.approx(.05 - expit(-.5), abs=1e-15)
    np.testing.assert_allclose(gradient[1:], 0., atol=1e-15)
    assert np.isfinite(result['theta']).all()
    weight = result['theta'][0]
    assert result['research_diagnostics']['objective'] == pytest.approx(np.logaddexp(0., -weight) + .05 * weight ** 2, abs=1e-15)
    assert result['research_diagnostics']['gradient_norm'] == pytest.approx(abs(.1 * weight - expit(-weight)), abs=1e-15)
