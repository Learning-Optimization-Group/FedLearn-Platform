"""A12 robust aggregation under local DP: declared grid and an attacked robust local-DP engine run."""
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


class LocalRobustTests(unittest.TestCase):
    def setUp(self):
        self.module = load('run_effusion_local_robust')

    def test_grid_is_the_a12_declaration(self):
        cells = self.module.cells()
        self.assertEqual(len(cells), 135)
        self.assertEqual(sorted({c['epsilon'] for c in cells}), [8.0, 32.0, 128.0])
        self.assertEqual({c['rule'] for c in cells}, {'mean', 'trimmed_mean', 'median'})
        self.assertEqual({c['condition'] for c in cells}, {'clean', 'sign_flip', 'label_flip'})
        self.assertIn({'epsilon': 8.0, 'rule': 'median', 'condition': 'sign_flip', 'seed': 20261103,
                       'name': 'local_eps8_median_sign_flip_20261103'}, cells)

    def test_an_attacked_median_local_dp_run_completes_with_client_noise_reported(self):
        federated = load('run_effusion_federated')
        rng = np.random.default_rng(0)
        patients = np.repeat(np.arange(1, 41), 2).astype(str)
        y = np.repeat((rng.random(40) < 0.3).astype(np.int64), 2)
        x = rng.normal(size=(80, 6)).astype(np.float32); x[:, 0] += 2 * y
        record, margin = federated.run_one('r', 20261101, x, y, patients, x, y, x.mean(0), x.std(0), clients=10,
                                           features_path='mem', report_sha256='0' * 64, rounds=2,
                                           privacy={'mode': 'local', 'epsilon': 32.0, 'delta': 1e-5},
                                           recipe=self.module.RECIPE, rule='median',
                                           behaviors=self.module.BEHAVIORS['sign_flip'])
        self.assertGreater(record['rounds'][-1]['privacy']['client_noise_std'], 0)
        self.assertTrue(np.isfinite(margin).all())


if __name__ == '__main__':
    unittest.main()
