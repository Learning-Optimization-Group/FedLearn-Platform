"""Scalar-logit client-average ERM and one final Gaussian perturbation.

The coordinator receives all rows. This is a trusted-coordinator research
comparator, not a distributed protocol or a whole-program privacy guarantee.
Only the final perturbed vector is a candidate private output. Raw fits and
research diagnostics are NOT DP outputs. Public cohort membership, B, lambda,
T and the supplied Gaussian multiplier must be frozen independently of data.
The finite-solver sensitivity argument is in
../phase-3-mechanism-preparation-2026-10-25.md and assumes exact arithmetic.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


def _nonnegative_integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _finite_scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real scalar')
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return value


def solver_sensitivity_bound(*, row_bound, regularization, num_clients, steps):
    """Return public constants and replace-one-client sensitivity components.

    ``steps`` is T, ``regularization`` is lambda, and ``row_bound`` is B.
    Every client must remain nonempty in both neighboring datasets; row counts
    may differ. The scalar logit has d weights and one regularized intercept.
    """
    b = _finite_scalar(row_bound, 'row_bound')
    lam = _finite_scalar(regularization, 'regularization', positive=True)
    n = _nonnegative_integer(num_clients, 'num_clients', minimum=1)
    t = _nonnegative_integer(steps, 'steps')
    l = math.hypot(b, 1.)
    curvature = (b * b + 1.) / 4.
    beta = lam + curvature
    radius = l / lam
    if not all(math.isfinite(v) for v in (curvature, beta, radius, 2 * radius)):
        raise ValueError('public constants exceed float64 range')
    # Algebraically 1-lambda/beta; this avoids cancellation when q is small.
    q = curvature / beta
    exact = 2 * radius / n
    residual = 2 * radius * q ** t
    sensitivity = exact + residual
    if not math.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError('public sensitivity is outside float64 range')
    return {
        'row_bound': b, 'regularization': lam, 'num_clients': n, 'steps': t,
        'augmented_row_bound': l, 'beta': beta, 'q': q,
        'exact_minimizer_sensitivity': exact,
        'solver_residual_sensitivity': residual, 'sensitivity': sensitivity,
    }


def choose_solver_steps(*, row_bound, regularization, num_clients, residual_ratio, max_steps=100_000):
    """Choose public T with residual <= residual_ratio * exact bound.

    The analytic budget check precedes optimization. An infeasible public
    max_steps raises rather than silently returning an under-solved fit.
    No loss, gradient, private row count or convergence statistic chooses T.
    """
    ratio = _finite_scalar(residual_ratio, 'residual_ratio', positive=True)
    maximum = _nonnegative_integer(max_steps, 'max_steps')
    initial = solver_sensitivity_bound(
        row_bound=row_bound, regularization=regularization,
        num_clients=num_clients, steps=0,
    )
    target_log = math.log(ratio) - math.log(initial['num_clients'])
    if target_log >= 0:
        steps = 0
    else:
        q = initial['q']
        if q >= 1.:
            raise ValueError('max_steps cannot meet the requested residual ratio at float64 precision')
        steps = 1 if q == 0. else math.ceil(target_log / math.log(q))
    if steps > maximum:
        raise ValueError('required public iteration budget exceeds max_steps')
    bound = solver_sensitivity_bound(
        row_bound=row_bound, regularization=regularization,
        num_clients=num_clients, steps=steps,
    )
    # Logarithmic rounding can land just below the requested boundary. This
    # correction depends solely on public constants, never fitted gradients.
    while bound['solver_residual_sensitivity'] > ratio * bound['exact_minimizer_sensitivity']:
        steps += 1
        if steps > maximum:
            raise ValueError('required public iteration budget exceeds max_steps')
        bound = solver_sensitivity_bound(
            row_bound=row_bound, regularization=regularization,
            num_clients=num_clients, steps=steps,
        )
    bound['residual_ratio'] = ratio
    return bound


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'biuf':
        raise ValueError(f'{name} must contain real numeric values')
    array = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain only finite values')
    return array


def _bound_rows_and_rate(x, row_bound):
    b = _finite_scalar(row_bound, 'row_bound')
    features = _real_array(x, 'x')
    if features.ndim != 2 or not features.shape[0] or not features.shape[1]:
        raise ValueError('x must have nonempty shape (rows, features)')
    # Compute normalized directions first so even finite rows near float64's
    # maximum can be clipped without overflowing their squared norm.
    largest = np.max(np.abs(features), axis=1)
    directions = features / np.where(largest == 0., 1., largest)[:, None]
    lengths = np.sqrt(np.sum(directions * directions, axis=1))
    threshold = b / np.where(lengths == 0., 1., lengths)
    clipped = largest > threshold
    features[clipped] = directions[clipped] * threshold[clipped, None]
    return features, float(np.mean(clipped))


def bound_rows(x, row_bound):
    """Return a float64 copy with the exact feature-row transform used in fit.

    Accepts a finite, nonempty (rows, features) numeric matrix and public B>=0.
    Use this same transform before evaluating scalar margins. No intercept is
    appended here, and the input matrix is never mutated.
    """
    return _bound_rows_and_rate(x, row_bound)[0]


def _validated_rows(x, y, partitions, row_bound):
    features, clip_rate = _bound_rows_and_rate(x, row_bound)
    labels = _real_array(y, 'y')
    if labels.shape != (features.shape[0],) or not np.isin(labels, [0., 1.]).all():
        raise ValueError('y must be a binary vector with one label per row')
    if not isinstance(partitions, (list, tuple)) or not partitions:
        raise ValueError('partitions must be a fixed nonempty list or tuple of clients')
    seen = np.zeros(len(labels), dtype=bool)
    row_weights = np.empty(len(labels), dtype=np.float64)
    for part in partitions:
        ids = np.asarray(part)
        if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in 'iu':
            raise ValueError('each client must have a nonempty integer index vector')
        if np.any(ids < 0) or np.any(ids >= len(labels)) or len(np.unique(ids)) != len(ids):
            raise ValueError('client indices must be unique and in range')
        if seen[ids].any():
            raise ValueError('client partitions must be disjoint')
        seen[ids] = True
        row_weights[ids] = 1. / (len(partitions) * len(ids))
    if not seen.all():
        raise ValueError('client partitions must exhaust every row')

    augmented = np.column_stack((features, np.ones(len(labels))))
    return augmented, labels, row_weights, clip_rate


def _objective_gradient(theta, augmented, labels, row_weights, regularization):
    # Explicit non-BLAS contractions avoid stale FP-status warnings observed
    # from macOS matmul after Torch training, including with all-zero theta.
    logits = np.einsum('ij,j->i', augmented, theta, optimize=False)
    probabilities = np.exp(-np.logaddexp(0., -logits))
    gradient = np.einsum(
        'ij,i->j', augmented, row_weights * (probabilities - labels), optimize=False,
    ) + regularization * theta
    loss = np.sum(row_weights * np.logaddexp(0., (1. - 2. * labels) * logits))
    return float(loss + .5 * regularization * np.sum(theta * theta)), gradient


def fit_client_private_erm(x, y, partitions, *, row_bound, regularization, steps, max_steps=100_000):
    """Fit exactly T full-gradient steps from zero in CPU NumPy float64.

    Inputs: x array-like (m,d), y binary array-like (m,), partitions list/tuple
    of nonempty integer row-index vectors, disjoint and exhaustive. Each
    client's mean loss receives weight 1/n irrespective of its row count.
    B clips feature rows before adding an intercept; lambda regularizes every
    coordinate. The step size is exactly 1/(lambda+(B**2+1)/4).

    Returns dict(theta ndarray[d+1], public_bound dict, research_diagnostics
    dict). theta stores weights followed by the intercept. Neither raw theta
    nor any research diagnostic is a DP release. No convergence-based stopping,
    rejection or retry occurs; one final gradient evaluation is diagnostic.
    """
    maximum = _nonnegative_integer(max_steps, 'max_steps')
    t = _nonnegative_integer(steps, 'steps')
    if t > maximum:
        raise ValueError('steps exceeds max_steps')
    if not isinstance(partitions, (list, tuple)) or not partitions:
        raise ValueError('partitions must be a fixed nonempty list or tuple of clients')
    bound = solver_sensitivity_bound(
        row_bound=row_bound, regularization=regularization,
        num_clients=len(partitions), steps=t,
    )
    augmented, labels, weights, clip_rate = _validated_rows(x, y, partitions, bound['row_bound'])
    theta = np.zeros(augmented.shape[1], dtype=np.float64)
    for _ in range(t):
        _, gradient = _objective_gradient(theta, augmented, labels, weights, bound['regularization'])
        theta -= gradient / bound['beta']
    objective, gradient = _objective_gradient(theta, augmented, labels, weights, bound['regularization'])
    return {
        'theta': theta,
        'public_bound': bound,
        'research_diagnostics': {
            'privacy_status': 'NOT_DP_OUTPUTS', 'feature_clip_rate': clip_rate,
            'objective': objective, 'gradient_norm': float(np.linalg.norm(gradient)),
            'steps_completed': t,
        },
    }


def release_gaussian(theta, *, sensitivity, noise_multiplier, seed):
    """Add one isotropic Gaussian vector with sigma=noise_multiplier*sensitivity.

    The multiplier is z, NOT epsilon. The caller must calibrate z with the
    intended one-release accountant and use the matching solver bound. A local
    seeded NumPy Generator preserves global RNG state. Research seeds support
    reproducibility and do not supply production secret randomness. z=0 is an
    explicitly nonprivate matched control. No intermediate state is released.
    Returns dict(theta ndarray, noise_std float, sensitivity, noise_multiplier).
    """
    vector = _real_array(theta, 'theta')
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError('theta must be a nonempty vector')
    delta = _finite_scalar(sensitivity, 'sensitivity')
    z = _finite_scalar(noise_multiplier, 'noise_multiplier')
    seed = _nonnegative_integer(seed, 'seed')
    sigma = delta * z
    if not math.isfinite(sigma):
        raise ValueError('noise standard deviation exceeds float64 range')
    noise = np.random.default_rng(seed).normal(0., sigma, size=vector.shape)
    return {
        'theta': vector + noise, 'noise_std': sigma,
        'sensitivity': delta, 'noise_multiplier': z,
    }
