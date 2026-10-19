"""Validation-selected thresholds must retain tied positives and reject invalid scores."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest


def calibration_module():
    path = Path(__file__).resolve().parents[2] / 'research/benchmarks/nih_calibration.py'
    spec = importlib.util.spec_from_file_location('nih_calibration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_threshold_achieves_declared_sensitivity_including_ties():
    module = calibration_module()
    labels = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([.2, .5, .5, .9, .4, .8])
    threshold = module.validation_threshold(labels, scores, .75)
    assert .4 < threshold < .5
    metrics = module.threshold_metrics(labels, scores, threshold)
    assert metrics['true_positive'] == 3
    assert metrics['false_negative'] == 1
    assert metrics['false_positive'] == 1
    assert metrics['true_negative'] == 1
    assert metrics['sensitivity'] == .75


@pytest.mark.parametrize('scores', [[np.nan, .2], [1.1, .2], [-.1, .2]])
def test_validation_threshold_refuses_invalid_probabilities(scores):
    with pytest.raises(ValueError, match='probabilities'):
        calibration_module().validation_threshold([0, 1], scores, .8)


def test_validation_threshold_refuses_a_cohort_without_positive_labels():
    with pytest.raises(ValueError, match='positive'):
        calibration_module().validation_threshold([0, 0], [.2, .3], .8)
