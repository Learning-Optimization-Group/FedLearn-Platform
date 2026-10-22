"""Matched clean federated baseline: workload, configs, whole-patient clients, central control."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[3] / 'framework' / 'src'))


def load():
    spec = importlib.util.spec_from_file_location('run_effusion_federated', HERE / 'run_effusion_federated.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def data(seed=0):
    rng = np.random.default_rng(seed)
    patients = np.repeat(np.arange(1, 61), 3).astype(str)
    y_patient = (rng.random(60) < 0.3).astype(np.int64)
    y = y_patient.repeat(3)
    x = rng.normal(size=(180, 8)).astype(np.float32); x[:, 0] += 2.5 * y
    return x, y, patients


class FederatedHarnessTests(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def test_protocol_recipe_and_seeds_are_fixed(self):
        cfg = self.module.config('fed', 20261101, clients=80, features_path='x', report_sha256='0' * 64)
        self.assertEqual((cfg['federation']['num_clients'], cfg['federation']['rounds']), (80, 20))
        self.assertEqual(cfg['client'], {'algorithm': 'fedavg', 'local_epochs': 7, 'lr': 0.003, 'batch_size': 256})
        self.assertEqual((cfg['server']['clip_norm'], cfg['server']['rule'], cfg['privacy']['mode']), (0.05, 'mean', 'none'))
        self.assertEqual(cfg['seeds'], {'partition': 20261101, 'init': 20261101, 'client': 20261101,
                                        'attack': 20262101, 'noise': 20263101})
        self.assertEqual(self.module.SEEDS, (20261101, 20261102, 20261103, 20261104, 20261105))

    def test_workload_keeps_patients_whole_and_evaluates_only_given_selection(self):
        x, y, patients = data(); sx, sy, _ = data(1)
        mean, std = x.mean(0).astype(np.float64), x.std(0).astype(np.float64)
        workload = self.module.EffusionWorkload(x, y, patients, sx, sy, mean, std, clients=7, partition_seed=3, info={'k': 'v'})
        owners = {}
        for cid, part in enumerate(workload.parts):
            for i in part:
                owners.setdefault(patients[i], set()).add(cid)
        self.assertTrue(all(len(v) == 1 for v in owners.values()))
        self.assertEqual(sorted(np.concatenate(workload.parts).tolist()), list(range(len(y))))
        self.assertTrue(np.allclose(workload.test_x.numpy(), (sx - mean) / std, atol=1e-6))
        self.assertEqual(workload.test_y.tolist(), sy.tolist())
        self.assertEqual((workload.feat_dim, workload.n_classes), (8, 2))

    def test_federated_and_central_runs_use_the_production_engine(self):
        x, y, patients = data(); sx, sy, _ = data(1)
        mean, std = x.mean(0).astype(np.float64), x.std(0).astype(np.float64)
        out = {}
        for name, clients in (('fed', 6), ('central', 1)):
            out[name] = self.module.run_one(name, 20261101, x, y, patients, sx, sy, mean, std, clients=clients,
                                            features_path='mem', report_sha256='0' * 64, rounds=3)
            record, margin = out[name]
            self.assertEqual(len(record['rounds']), 3)
            self.assertEqual(record['meta']['workload']['parameters'], 18)
            self.assertEqual(margin.shape, (180,))
            self.assertAlmostEqual(record['rounds'][-1]['auc'], self.module.auroc(sy, margin), places=12)
        self.assertEqual(out['central'][0]['meta']['config']['federation']['num_clients'], 1)

    def test_failed_gate_needs_a_named_amendment_and_records_it(self):
        failed = {'gate': {'passed': False}}
        with self.assertRaisesRegex(ValueError, 'gate_not_passed'):
            self.module.check_gate(failed, None)
        with self.assertRaisesRegex(ValueError, 'unknown_amendment'):
            self.module.check_gate(failed, 'A9')
        self.assertEqual(self.module.check_gate(failed, 'A2')['gatePassed'], False)
        self.assertEqual(self.module.check_gate(failed, 'A2')['proceedingUnder'], 'A2')
        self.assertEqual(self.module.check_gate({'gate': {'passed': True}}, None)['proceedingUnder'], None)


    def test_script_imports_this_worktrees_engine_in_a_fresh_process(self):
        import subprocess
        result = subprocess.run([sys.executable, str(HERE / 'run_effusion_federated.py'), '--engine-path'],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-500:])
        engine = Path(result.stdout.strip().splitlines()[-1]).resolve()
        self.assertTrue(engine.is_relative_to((HERE.parents[3] / 'framework' / 'src').resolve()), engine)

if __name__ == '__main__':
    unittest.main()
