"""A9 attack study: declared grid, attack parameters, rule/behaviour plumbing and DP compatibility."""
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


class AttackStudyTests(unittest.TestCase):
    def setUp(self):
        self.module = load('run_effusion_attacks')

    def test_grid_is_exactly_the_a9_declaration(self):
        cells = self.module.cells()
        self.assertEqual(len(cells), 100)
        no_dp = [c for c in cells if c['epsilon'] is None]
        dp = [c for c in cells if c['epsilon'] == 4.0]
        self.assertEqual((len(no_dp), len(dp)), (60, 40))
        self.assertEqual({c['rule'] for c in no_dp}, {'mean', 'trimmed_mean', 'median'})
        self.assertEqual({c['rule'] for c in dp}, {'mean', 'trimmed_mean'})
        self.assertEqual({c['condition'] for c in cells}, {'clean', 'sign_flip', 'label_flip', 'alie'})
        self.assertEqual(self.module.RECIPES, {False: {'lr': 0.1, 'clip_norm': 1.0}, True: {'lr': 0.03, 'clip_norm': 0.05}})

    def test_attack_parameters_are_the_declared_ones(self):
        self.assertEqual(self.module.BEHAVIORS['sign_flip'], [{'type': 'sign_flip_scale', 'fraction': 0.2, 'scale': -4.0}])
        self.assertEqual(self.module.BEHAVIORS['label_flip'], [{'type': 'label_flip', 'fraction': 0.2}])
        self.assertEqual(self.module.BEHAVIORS['alie'], [{'type': 'alie', 'fraction': 0.2, 'z': 0.4888}])
        self.assertEqual(self.module.BEHAVIORS['clean'], [])

    def test_config_carries_rule_and_behaviours_and_refuses_unknown_rules(self):
        federated = load('run_effusion_federated')
        cfg = federated.config('x', 20261101, clients=80, features_path='p', report_sha256='0' * 64,
                               rule='trimmed_mean', behaviors=[{'type': 'label_flip', 'fraction': 0.2}])
        self.assertEqual(cfg['server']['rule'], 'trimmed_mean')
        self.assertEqual(cfg['behaviors'], [{'type': 'label_flip', 'fraction': 0.2}])
        base = federated.config('y', 20261101, clients=80, features_path='p', report_sha256='0' * 64)
        self.assertEqual((base['server']['rule'], base['behaviors']), ('mean', []))
        with self.assertRaises(ValueError):
            federated.config('z', 20261101, clients=80, features_path='p', report_sha256='0' * 64, rule='krum')

    def test_an_attacked_trimmed_mean_dp_run_completes_through_the_engine(self):
        federated = load('run_effusion_federated')
        rng = np.random.default_rng(0)
        patients = np.repeat(np.arange(1, 41), 2).astype(str)
        y = np.repeat((rng.random(40) < 0.3).astype(np.int64), 2)
        x = rng.normal(size=(80, 6)).astype(np.float32); x[:, 0] += 2 * y
        record, margin = federated.run_one('a', 20261101, x, y, patients, x, y, x.mean(0), x.std(0), clients=10,
                                           features_path='mem', report_sha256='0' * 64, rounds=2,
                                           privacy={'mode': 'central', 'epsilon': 4.0, 'delta': 1e-5},
                                           recipe={'lr': 0.03, 'clip_norm': 0.05}, rule='trimmed_mean',
                                           behaviors=self.module.BEHAVIORS['sign_flip'])
        assignment = record['meta']['behavior_assignment']
        self.assertEqual(sum(kinds == ['sign_flip_scale'] for kinds in assignment.values()), 2)
        self.assertTrue(np.isfinite(margin).all())


if __name__ == '__main__':
    unittest.main()
