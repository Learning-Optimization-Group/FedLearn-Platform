"""A7 recipe study: grid, override plumbing, selection rule and confirmation cells."""
import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[3] / 'framework' / 'src'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class RecipeStudyTests(unittest.TestCase):
    def setUp(self):
        self.module = load('run_effusion_recipe')

    def test_grid_and_selection_seeds_are_the_a7_declaration(self):
        self.assertEqual(self.module.GRID, [(lr, clip) for lr in (0.003, 0.01, 0.03, 0.1) for clip in (0.05, 0.25, 1.0)])
        self.assertEqual(self.module.SELECTION_SEEDS, (20261101, 20261102))

    def test_recipe_override_reaches_the_engine_config_and_defaults_are_unchanged(self):
        federated = load('run_effusion_federated')
        base = federated.config('a', 20261101, clients=80, features_path='p', report_sha256='0' * 64)
        self.assertEqual((base['client']['lr'], base['server']['clip_norm']), (0.003, 0.05))
        tuned = federated.config('b', 20261101, clients=80, features_path='p', report_sha256='0' * 64,
                                 recipe={'lr': 0.03, 'clip_norm': 0.25})
        self.assertEqual((tuned['client']['lr'], tuned['server']['clip_norm']), (0.03, 0.25))
        self.assertEqual((tuned['client']['local_epochs'], tuned['client']['batch_size']), (7, 256))
        with self.assertRaises(ValueError):
            federated.config('c', 20261101, clients=80, features_path='p', report_sha256='0' * 64, recipe={'epochs': 3})

    def test_selection_uses_mean_auroc_then_smaller_clip_then_smaller_lr(self):
        results = [
            {'lr': 0.03, 'clip': 1.0, 'eligible': True, 'meanAuroc': 0.80},
            {'lr': 0.1, 'clip': 0.25, 'eligible': True, 'meanAuroc': 0.80},
            {'lr': 0.01, 'clip': 0.25, 'eligible': True, 'meanAuroc': 0.80},
            {'lr': 0.1, 'clip': 1.0, 'eligible': False, 'meanAuroc': 0.95},
        ]
        self.assertEqual(self.module.choose(results), 2)

    def test_confirmation_cells_cover_five_seeds_and_three_epsilons(self):
        cells = self.module.confirmation_cells()
        self.assertEqual(len(cells), 25)
        self.assertEqual(sorted({c['kind'] for c in cells}), ['central', 'dp', 'fed'])
        self.assertEqual(sorted({c['epsilon'] for c in cells if c['kind'] == 'dp'}), [1.0, 4.0, 8.0])

    def test_linear_contrasts_average_signed_auroc_terms_over_seeds(self):
        import numpy as np
        y = np.array([0, 1] * 50); patients = np.arange(100).astype(str)
        perfect, inverted = y.astype(float), -y.astype(float)
        scores = {'a': perfect, 'b': inverted, 'c': perfect}
        result = self.module.linear_contrasts(y, patients, scores, {'a-b': [{'a': 1, 'b': -1}, {'c': 1, 'b': -1}],
                                                                   'zero': [{'a': 1, 'c': -1}]}, reps=50, seed=1)
        self.assertAlmostEqual(result['a-b']['point'], 1.0)
        self.assertEqual(result['a-b']['interval95'], [1.0, 1.0])
        self.assertEqual(result['zero']['point'], 0.0)


if __name__ == '__main__':
    unittest.main()
