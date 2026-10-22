"""A11 local-DP study: declared grid and client-noise reporting through the engine."""
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


class LocalDpStudyTests(unittest.TestCase):
    def setUp(self):
        self.module = load('run_effusion_local_dp')

    def test_grid_is_the_a11_declaration(self):
        cells = self.module.cells()
        self.assertEqual(len(cells), 25)
        self.assertEqual(sorted({c['epsilon'] for c in cells}), [1.0, 4.0, 8.0, 32.0, 128.0])
        self.assertEqual(self.module.RECIPE, {'lr': 0.03, 'clip_norm': 0.05})
        self.assertIn({'epsilon': 32.0, 'seed': 20261104, 'name': 'local_eps32_20261104'}, cells)

    def test_a_local_run_reports_the_client_noise_it_added(self):
        federated = load('run_effusion_federated')
        rng = np.random.default_rng(0)
        patients = np.repeat(np.arange(1, 41), 2).astype(str)
        y = np.repeat((rng.random(40) < 0.3).astype(np.int64), 2)
        x = rng.normal(size=(80, 6)).astype(np.float32); x[:, 0] += 2 * y
        record, margin = federated.run_one('l', 20261101, x, y, patients, x, y, x.mean(0), x.std(0), clients=8,
                                           features_path='mem', report_sha256='0' * 64, rounds=2,
                                           privacy={'mode': 'local', 'epsilon': 8.0, 'delta': 1e-5},
                                           recipe=self.module.RECIPE)
        z = record['meta']['noise_multiplier']
        self.assertAlmostEqual(record['rounds'][-1]['privacy']['client_noise_std'], 2 * z * 0.05)
        self.assertEqual(record['rounds'][-1]['privacy']['server_noise_std'], 0.0)
        self.assertTrue(np.isfinite(margin).all())


if __name__ == '__main__':
    unittest.main()
