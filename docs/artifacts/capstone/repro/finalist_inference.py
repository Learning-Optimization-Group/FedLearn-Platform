"""Separate seed and conditional patient intervals for frozen final-study contrasts.

All gains are left AUROC minus right AUROC. Patient intervals resample whole
patients and retain image-level weighting; they condition on the supplied heads.
Neither interval combines training-seed and evaluation-cohort uncertainty.
This module performs no file access, fitting, or prediction export.
"""
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from scipy.stats import t


def _confidence(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or not 0 < value < 1):
        raise ValueError("confidence must be finite and strictly between zero and one")
    return float(value)


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def paired_seed_interval(effects, confidence=.95):
    """Approximate two-sided Student t interval over independent paired seeds.

    Supply one already-paired scalar effect per distinct training seed. The
    evaluation cohort must be held fixed across those effects. Pair construction
    and independence of the training seeds are the caller's responsibility.
    """
    confidence = _confidence(confidence)
    if not isinstance(effects, Mapping) or len(effects) < 2:
        raise ValueError("effects must map at least two distinct seeds to finite effects")
    copied = {}
    for seed, value in effects.items():
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value)):
            raise ValueError("seed effects must be finite real numbers")
        copied[seed] = float(value)
    if len(copied) != len(effects) or len(copied) < 2:
        raise ValueError("effects must contain distinct seed entries")
    values = np.asarray(list(copied.values()), dtype=float)
    n = len(values)
    point = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(n)
    half = float(t.ppf(.5 + confidence / 2, n - 1) * se)
    if not np.all(np.isfinite([point, sd, se, half])):
        raise ValueError("seed interval arithmetic overflowed")
    return dict(n=n, point=point, sample_sd=sd, se=float(se),
                ci_low=point - half, ci_high=point + half, df=n - 1,
                seed_effects=copied,
                scope="training-seed interval conditional on fixed evaluation cohort; "
                      "approximate t; not population/joint interval")


def _finite_array(values, name):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _labels(values):
    result = _finite_array(values, "labels")
    if (result.ndim != 1 or result.size == 0 or not np.all((result == 0) | (result == 1))
            or np.unique(result).size != 2):
        raise ValueError("labels must be one-dimensional binary labels containing both classes")
    return result


class _RankedAUC:
    """A fixed head's ascending rank and tie groups, reused across resamples."""

    def __init__(self, labels, scores):
        self.order = np.argsort(scores, kind="stable")
        ordered = scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        self.positive = labels[self.order] == 1

    def evaluate(self, weights):
        ordered_weights = weights[self.order]
        positive = np.add.reduceat(ordered_weights * self.positive, self.starts)
        negative = np.add.reduceat(ordered_weights * ~self.positive, self.starts)
        total_positive, total_negative = positive.sum(), negative.sum()
        if total_positive <= 0 or total_negative <= 0:
            raise ValueError("weighted AUROC requires positive mass in both classes")
        previous_negative = np.cumsum(negative) - negative
        # Normalize each class first to avoid an unnecessary product of masses.
        return float(np.sum((positive / total_positive)
                            * ((previous_negative + .5 * negative) / total_negative)))


def weighted_auc(labels, scores, sample_weight=None):
    """Image-level, tie-aware AUROC with nonnegative per-image sample weights."""
    labels = _labels(labels)
    scores = _finite_array(scores, "scores")
    if scores.shape != labels.shape:
        raise ValueError("scores must have one entry per image")
    weights = (np.ones(labels.size) if sample_weight is None
               else _finite_array(sample_weight, "sample_weight"))
    if weights.shape != labels.shape or np.any(weights < 0):
        raise ValueError("sample_weight must be nonnegative with one entry per image")
    # AUROC is unchanged by a common weight scale. Keep finite input masses safe.
    maximum = float(weights.max())
    if maximum <= 0:
        raise ValueError("weighted AUROC requires positive mass in both classes")
    return _RankedAUC(labels, scores).evaluate(weights / maximum)


def _patient_indices(patient_ids, n_images):
    patients = np.asarray(patient_ids, dtype=object)
    if patients.shape != (n_images,):
        raise ValueError("patient_ids must be one-dimensional with one ID per image")
    unique = {}
    for value in patients:
        if (value is None or (isinstance(value, str) and not value.strip())
                or (isinstance(value, Real) and not np.isfinite(value))):
            raise ValueError("patient_ids must be nonmissing finite hashable IDs")
        try:
            if value != value:
                raise ValueError("patient_ids must be nonmissing")
            unique[value] = None
        except (TypeError, ValueError) as error:
            raise ValueError("patient_ids must be nonmissing finite hashable IDs") from error
    # Canonical order makes a fixed random seed independent of image row order.
    ordered = sorted(unique, key=lambda value: (type(value).__module__,
                                               type(value).__qualname__, value)
                     if isinstance(value, (str, Real)) else
                     (type(value).__module__, type(value).__qualname__, repr(value)))
    lookup = {value: index for index, value in enumerate(ordered)}
    return np.asarray([lookup[value] for value in patients], dtype=np.intp), len(ordered)


def conditional_patient_interval(labels, patient_ids, margins_left, margins_right, *,
                                 draws=2000, seed=0, confidence=.95, max_attempts=None):
    """Percentile patient-cluster bootstrap conditional on matched realized heads.

    Margin arrays have shape (heads, images), with matching head and image order
    on both sides. Each replicate samples n_patients patients with replacement;
    each sampled copy contributes every image in that patient's cluster. The
    resulting multiplicities are shared across every head and both conditions.
    The point is the mean paired gain in ordinary image-level AUROC, with no
    inverse cluster-size weighting. Single-class draws are skipped and counted.

    ``draws`` counts valid replicates. ``max_attempts`` bounds total attempted
    replicates (default 10 * draws, at least 100). Exhaustion raises RuntimeError
    instead of returning an interval based on a smaller, undeclared budget.
    Only aggregate research-analysis results are returned; this is not a private
    release mechanism or a population, joint, or clinical confirmation interval.
    """
    confidence = _confidence(confidence)
    draws = _integer(draws, "draws", 1)
    seed = _integer(seed, "seed", 0)
    max_attempts = (max(100, 10 * draws) if max_attempts is None
                    else _integer(max_attempts, "max_attempts", draws))
    labels = _labels(labels)
    left = _finite_array(margins_left, "margins_left")
    right = _finite_array(margins_right, "margins_right")
    if (left.ndim != 2 or left.shape != right.shape or left.shape[0] < 1
            or left.shape[1] != labels.size):
        raise ValueError("margins must have matching nonempty (heads, images) shapes")
    patient_index, n_patients = _patient_indices(patient_ids, labels.size)
    ranked = [(_RankedAUC(labels, a), _RankedAUC(labels, b)) for a, b in zip(left, right)]

    def mean_gain(weights):
        return float(np.mean([a.evaluate(weights) - b.evaluate(weights) for a, b in ranked]))

    point = mean_gain(np.ones(labels.size))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws)
    valid, invalid = 0, 0
    positive = labels == 1
    for _ in range(max_attempts):
        sampled = rng.integers(n_patients, size=n_patients)
        weights = np.bincount(sampled, minlength=n_patients)[patient_index]
        if weights[positive].sum() == 0 or weights[~positive].sum() == 0:
            invalid += 1
            continue
        bootstrap[valid] = mean_gain(weights)
        valid += 1
        if valid == draws:
            break
    if valid != draws:
        raise RuntimeError(f"max_attempts exhausted: {valid} valid and {invalid} invalid draws; "
                           f"{draws} valid draws required")
    alpha = (1 - confidence) / 2
    low, high = np.quantile(bootstrap, [alpha, 1 - alpha])
    return dict(point=point, ci_low=float(low), ci_high=float(high), draws=draws,
                invalid_draws=invalid, n_heads=left.shape[0], n_images=labels.size,
                n_patients=n_patients,
                scope="patient-cluster percentile interval conditional on realized heads; "
                      "image-level AUROC; not training-seed or joint uncertainty")
