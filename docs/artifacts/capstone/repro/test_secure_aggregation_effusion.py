"""A5 secure aggregation: real LightSecAgg recovery, scoped engine swap, refusals and accounting."""
from collections import OrderedDict
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[3] / 'framework' / 'src'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def deltas(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [OrderedDict(w=torch.randn(2, 3, generator=g) * 1e-3, b=torch.randn(2, generator=g) * 1e-3) for _ in range(n)]


class SecureAggregationTests(unittest.TestCase):
    def setUp(self):
        self.module = load('secure_aggregation_effusion')

    def test_real_protocol_recovers_the_mean_within_the_fixed_point_bound(self):
        reducer = self.module.SecureMeanReducer()
        ds = deltas(6)
        out, kept = reducer.reduce(ds, [1] * 6, ['w', 'b'], 'mean', {})
        self.assertIsNone(kept)
        for key in ('w', 'b'):
            plain = torch.stack([d[key] for d in ds]).mean(0)
            self.assertEqual(out[key].shape, plain.shape)
            self.assertEqual(out[key].dtype, torch.float32)
            self.assertLessEqual(float((out[key] - plain).abs().max()), 0.5e-6 + 1e-7)
        run = reducer.runs[0]
        self.assertEqual((run['clients'], run['threshold'], run['scalars']), (6, 4, 8))
        self.assertGreater(run['bytes']['perClientUpload'], run['bytes']['maskedSubmission'])
        self.assertLessEqual(run['maxAbsErrorVsFloatMean'], run['fixedPointBoundOnMean'])

    def test_identical_inputs_reuse_one_protocol_run(self):
        reducer = self.module.SecureMeanReducer()
        ds = deltas(5)
        first, _ = reducer.reduce(ds, [1] * 5, ['w', 'b'], 'mean', {})
        second, _ = reducer.reduce(ds, [1] * 5, ['w', 'b'], 'mean', {})
        self.assertEqual(len(reducer.runs), 1)
        self.assertEqual(reducer.reuses, 1)
        self.assertTrue(all(torch.equal(first[k], second[k]) for k in first))

    def test_rules_needing_individual_updates_are_refused(self):
        reducer = self.module.SecureMeanReducer()
        for rule in ('trimmed_mean', 'median', 'krum', 'weighted_mean'):
            with self.assertRaisesRegex(ValueError, 'secure_aggregation_unavailable'):
                reducer.reduce(deltas(5), [1] * 5, ['w', 'b'], rule, {'trim_ratio': 0.2})

    def test_engine_swap_is_scoped_and_restored(self):
        import fedlearn.server.composed_strategy as composed
        original = composed.reduce_deltas
        reducer = self.module.SecureMeanReducer()
        with self.assertRaises(RuntimeError):
            with self.module.secure_mean(reducer):
                self.assertIsNot(composed.reduce_deltas, original)
                raise RuntimeError('boom')
        self.assertIs(composed.reduce_deltas, original)

    def test_secure_engine_run_tracks_the_plain_run(self):
        federated = load('run_effusion_federated')
        rng = np.random.default_rng(0)
        patients = np.repeat(np.arange(1, 41), 2).astype(str)
        y = np.repeat((rng.random(40) < 0.3).astype(np.int64), 2)
        x = rng.normal(size=(80, 6)).astype(np.float32); x[:, 0] += 2 * y
        kwargs = dict(clients=8, features_path='mem', report_sha256='0' * 64, rounds=2)
        _, plain = federated.run_one('p', 20261101, x, y, patients, x, y, x.mean(0), x.std(0), **kwargs)
        reducer = self.module.SecureMeanReducer()
        with self.module.secure_mean(reducer):
            _, secure = federated.run_one('s', 20261101, x, y, patients, x, y, x.mean(0), x.std(0), **kwargs)
        self.assertEqual(len(reducer.runs), 2)
        self.assertLess(float(np.abs(secure - plain).max()), 1e-4)


if __name__ == '__main__':
    unittest.main()
