"""A4 central-DP sweep: grid, engine privacy wiring and the width contrast."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[3] / 'framework' / 'src'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class DpSweepTests(unittest.TestCase):
    def setUp(self):
        self.module = load('run_effusion_dp')

    def test_grid_is_exactly_the_a4_grid(self):
        cells = self.module.cells()
        self.assertEqual(len(cells), 40)
        self.assertEqual({c['arm'] for c in cells}, {'R0', 'R1'})
        self.assertEqual(sorted({c['epsilon'] for c in cells}, key=lambda e: (e is not None, e)), [None, 1.0, 4.0, 8.0])
        self.assertEqual(sorted({c['seed'] for c in cells}), list(range(20261101, 20261106)))
        self.assertIn({'arm': 'R1', 'epsilon': 4.0, 'seed': 20261103, 'name': 'R1_eps4_20261103'}, cells)

    def test_central_dp_reaches_the_engine_and_is_accounted(self):
        federated = load('run_effusion_federated')
        cfg = federated.config('x', 20261101, clients=80, features_path='p', report_sha256='0' * 64,
                               privacy={'mode': 'central', 'epsilon': 4.0, 'delta': 1e-5})
        self.assertEqual(cfg['privacy'], {'mode': 'central', 'epsilon': 4.0, 'delta': 1e-5})
        rng = np.random.default_rng(0)
        patients = np.repeat(np.arange(1, 41), 2).astype(str)
        y = np.repeat((rng.random(40) < 0.3).astype(np.int64), 2)
        x = rng.normal(size=(80, 6)).astype(np.float32); x[:, 0] += 2 * y
        record, margin = federated.run_one('dp', 20261101, x, y, patients, x, y, x.mean(0), x.std(0), clients=8,
                                           features_path='mem', report_sha256='0' * 64, rounds=3,
                                           privacy={'mode': 'central', 'epsilon': 4.0, 'delta': 1e-5})
        self.assertGreater(record['rounds'][0]['privacy']['noise_std'], 0)
        self.assertLessEqual(record['meta']['accounted_epsilon'], 4.0 + 1e-6)
        self.assertEqual(margin.shape, (80,))

    def test_width_contrast_is_cost_r1_minus_cost_r0(self):
        rng = np.random.default_rng(3)
        patients = np.repeat(np.arange(300), 2).astype(str)
        y = np.repeat((rng.random(300) < 0.3).astype(np.int64), 2)
        good = y + rng.normal(scale=0.5, size=600)
        noisy = rng.normal(size=600)
        margins = {}
        for seed in self.module.SEEDS:
            margins[f'R1_clean_{seed}'] = margins[f'R1_eps1_{seed}'] = good
            margins[f'R0_clean_{seed}'] = good
            margins[f'R0_eps1_{seed}'] = noisy
        result = self.module.contrasts(y, patients, margins, epsilons=(1.0,), reps=100, seed=5)
        self.assertAlmostEqual(result['cost']['R1']['1.0']['seedMean'], 0.0)
        self.assertGreater(result['cost']['R0']['1.0']['seedMean'], 0.2)
        width = result['widthContrast']['1.0']
        self.assertLess(width['point'], -0.2)
        self.assertLess(width['interval95'][1], 0)


if __name__ == '__main__':
    unittest.main()
