"""A10 holdout evaluation: frozen calibration, metric definitions and holdout-only label loading."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[3] / 'framework' / 'src'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class HoldoutEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.module = load('evaluate_effusion_holdout')

    def test_threshold_is_frozen_from_selection_and_never_refit_on_the_holdout(self):
        rng = np.random.default_rng(0)
        y = (rng.random(4000) < 0.15).astype(int); margin = 2.0 * y + rng.normal(size=4000)
        frozen = self.module.freeze(y, margin)
        p = self.module.probabilities(margin, frozen['platt'])
        self.assertGreaterEqual(((p >= frozen['threshold']) & (y == 1)).sum() / y.sum(), 0.9)
        patients = np.arange(4000).astype(str)
        first = self.module.holdout_metrics(y, margin, frozen, patients)
        shuffled = self.module.holdout_metrics(rng.permutation(y), margin, frozen, patients)
        self.assertEqual(first['threshold'], shuffled['threshold'])
        self.assertEqual(first['threshold'], frozen['threshold'])

    def test_ece_is_near_zero_when_calibrated_and_large_when_not(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0.01, 0.99, 50_000); y = (rng.random(50_000) < p).astype(int)
        self.assertLess(self.module.ece(y, p), 0.02)
        self.assertAlmostEqual(self.module.ece(np.zeros(1000, int), np.full(1000, 0.9)), 0.9, places=6)

    def test_calibration_slope_is_one_for_calibrated_and_half_for_overconfident(self):
        rng = np.random.default_rng(2)
        t = rng.normal(scale=2.0, size=40_000); y = (rng.random(40_000) < 1 / (1 + np.exp(-t))).astype(int)
        intercept, slope = self.module.calibration_fit(y, 1 / (1 + np.exp(-t)))
        self.assertAlmostEqual(slope, 1.0, delta=0.06); self.assertAlmostEqual(intercept, 0.0, delta=0.06)
        _, overconfident = self.module.calibration_fit(y, 1 / (1 + np.exp(-2 * t)))
        self.assertAlmostEqual(overconfident, 0.5, delta=0.04)

    def test_federated_margin_matches_the_engine_linear_head_forward(self):
        import torch
        from fedlearn.simulation.workloads import LinearHead
        rng = np.random.default_rng(3)
        z = rng.normal(size=(50, 512)).astype(np.float32)
        head = {'head.weight': rng.normal(size=(2, 512)).tolist(), 'head.bias': rng.normal(size=2).tolist()}
        model = LinearHead(512, 2)
        model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in head.items()})
        with torch.no_grad():
            logits = model(torch.from_numpy(z)).to(torch.float64)
        expected = (logits[:, 1] - logits[:, 0]).numpy()
        self.assertTrue(np.array_equal(self.module.federated_margin(z, head), expected))

    def test_holdout_labels_cover_exactly_the_locked_holdout(self):
        labels = load('test_effusion_labels')
        with tempfile.TemporaryDirectory() as directory:
            cache, lock, _, pins = labels.fixture(directory, lambda p, i: 'Effusion' if p % 3 == 0 else 'No Finding')
            result = load('effusion_labels').holdout_labels(cache, lock, pins=pins)
            holdout = [line.split(',')[0] for line in (lock / 'holdout.csv').read_text().splitlines()[1:]]
            self.assertEqual([r['image_index'] for r in result['rows']], holdout)
            self.assertEqual(result['y'].tolist(), [int(int(i[:8]) % 3 == 0) for i in holdout])


if __name__ == '__main__':
    unittest.main()
