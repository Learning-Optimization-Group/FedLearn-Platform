"""Protocol v1 gate core: fixed grid, selection rule, patient bootstrap, operating point, gate."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).with_name('fit_effusion_gate.py')


def load():
    spec = importlib.util.spec_from_file_location('fit_effusion_gate', SCRIPT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def synthetic(seed, informative):
    rng = np.random.default_rng(seed)
    patients = np.repeat(np.arange(600), 2)
    y = (rng.random(600) < 0.2).repeat(2).astype(np.int64)
    x = rng.normal(size=(1200, 6)).astype(np.float32)
    if informative:
        x[:, 0] += 3.0 * y
    return x, y, patients.astype(str)


class GateCoreTests(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def test_grid_is_exactly_the_protocol_grid(self):
        self.assertEqual(self.module.GRID, [(c, w) for c in (0.001, 0.01, 0.1, 1.0, 10.0) for w in (None, 'balanced')])

    def test_selection_prefers_auroc_then_smaller_c_and_skips_unconverged(self):
        fits = [dict(C=1.0, classWeight=None, converged=True, finite=True, selectionAuroc=0.81),
                dict(C=0.1, classWeight='balanced', converged=True, finite=True, selectionAuroc=0.81),
                dict(C=10.0, classWeight=None, converged=False, finite=True, selectionAuroc=0.95),
                dict(C=0.001, classWeight=None, converged=True, finite=False, selectionAuroc=0.99)]
        self.assertEqual(self.module.select(fits), 1)

    def test_operating_point_is_highest_threshold_reaching_target_sensitivity(self):
        y = np.array([1] * 10 + [0] * 10); scores = np.arange(20, dtype=float)[::-1]
        point = self.module.operating_point(y, scores, 0.9)
        self.assertEqual(point['threshold'], 11.0)
        self.assertEqual((point['sensitivity'], point['specificity']), (0.9, 1.0))

    def test_cluster_bootstrap_resamples_whole_patients_deterministically(self):
        patients = np.array(['a', 'a', 'b', 'c', 'c', 'c'])
        first = list(self.module.cluster_bootstrap(patients, reps=5, seed=7))
        second = list(self.module.cluster_bootstrap(patients, reps=5, seed=7))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))
        for index in first:
            counts = {p: int((patients[index] == p).sum()) for p in 'abc'}
            self.assertEqual(counts['a'] % 2, 0); self.assertEqual(counts['c'] % 3, 0)

    def test_standardiser_uses_training_statistics_only(self):
        train = np.array([[0.0, 5.0], [2.0, 5.0]], np.float32); other = np.array([[100.0, 7.0]], np.float32)
        mean, std = self.module.standardiser(train)
        self.assertEqual(mean.tolist(), [1.0, 5.0]); self.assertEqual(std.tolist(), [1.0, 1.0])
        self.assertEqual(((other - mean) / std).tolist(), [[99.0, 2.0]])

    def test_gate_passes_only_for_informative_candidate(self):
        train_inf, train_noise = synthetic(1, True), synthetic(2, False)
        sel_inf, sel_noise = synthetic(3, True), synthetic(4, False)
        y_train, y_sel = train_inf[1], sel_inf[1]
        noise_train = train_noise[0]; noise_sel = sel_noise[0]
        result = self.module.run_gate(
            {'R0': (noise_train, noise_sel), 'R1': (train_inf[0], sel_inf[0])},
            y_train, y_sel, sel_inf[2], reps=200, seed=11)
        self.assertEqual(len(result['fits']), 20)
        self.assertTrue(all(f['finite'] for f in result['fits']))
        self.assertLess(result['blasCrossCheckMaxAbsDiff'], 1e-9)
        self.assertTrue(result['gate']['passed'])
        reverse = self.module.run_gate(
            {'R0': (train_inf[0], sel_inf[0]), 'R1': (noise_train, noise_sel)},
            y_train, y_sel, sel_inf[2], reps=200, seed=11)
        self.assertFalse(reverse['gate']['passed'])
        self.assertFalse(reverse['gate']['criteria']['deltaAurocIntervalAboveZero'])


if __name__ == '__main__':
    unittest.main()
