"""Learning controls distinguish optimizer lifetime and reject data leakage."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

spec = importlib.util.spec_from_file_location(
    'nih_learning', Path(__file__).resolve().parents[2] / 'research/benchmarks/nih_learning.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_binary_metrics_expose_collapsed_predictions():
    result = module.binary_metrics(np.array([0, 0, 0, 1]), np.array([.9, .9, .9, .9]))
    assert result['accuracy'] == .25
    assert result['balanced_accuracy'] == .5
    assert result['majority_baseline_accuracy'] == .75
    assert result['confusion'] == {'tn': 0, 'fp': 3, 'fn': 0, 'tp': 1}
    assert result['predicted_positive_fraction'] == 1


@pytest.mark.parametrize('probabilities', [[.2, float('nan')], [-.1, .8], [.2, 1.1]])
def test_binary_metrics_reject_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        module.binary_metrics(np.array([0, 1]), np.array(probabilities))


def test_binding_rejects_cross_split_patient_or_image_overlap():
    good = [{'patients': {'1'}, 'images': {'a'}}, {'patients': {'2'}, 'images': {'b'}},
            {'patients': {'3'}, 'images': {'c'}}]
    module.require_disjoint(good)
    with pytest.raises(ValueError, match='patient'):
        module.require_disjoint([good[0], {'patients': {'1'}, 'images': {'b'}}])
    with pytest.raises(ValueError, match='image'):
        module.require_disjoint([good[0], {'patients': {'2'}, 'images': {'a'}}])


def test_optimizer_lifetime_changes_training_and_seed_reproduces_it():
    x = torch.tensor([[0., 1.], [1., 0.], [2., 1.], [1., 2.]])
    ds = TensorDataset(x, torch.tensor([0, 1, 1, 0]))
    torch.manual_seed(3)
    initial = torch.nn.Linear(2, 2).state_dict()

    def run(reset):
        net = torch.nn.Linear(2, 2)
        net.load_state_dict(initial)
        return module.train_control(net, ds, ds, epochs=3, learning_rate=.03,
                                    seed=7, batch_size=2, reset_optimizer_each_epoch=reset)

    persistent, repeat, reset = run(False), run(False), run(True)
    assert persistent['epochs'] == repeat['epochs']
    assert persistent['final_state_sha256'] == repeat['final_state_sha256']
    assert persistent['final_state_sha256'] != reset['final_state_sha256']
    assert len(persistent['epochs']) == 3
    assert all(np.isfinite(e['training_loss']) for e in persistent['epochs'])
    assert persistent['optimizer_lifetime'] == 'persistent'
    assert reset['optimizer_lifetime'] == 'reset_each_epoch'
