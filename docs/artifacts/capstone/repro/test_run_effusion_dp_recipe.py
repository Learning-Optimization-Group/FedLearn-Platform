"""A8 DP-specific recipe study: grid under DP, selection seeds and confirmation cells."""
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


class DpRecipeStudyTests(unittest.TestCase):
    def setUp(self):
        self.module = load('run_effusion_dp_recipe')

    def test_grid_and_selection_privacy_are_the_a8_declaration(self):
        self.assertEqual(self.module.GRID, [(lr, clip) for lr in (0.03, 0.1, 0.3) for clip in (0.01, 0.025, 0.05, 0.1)])
        self.assertEqual(self.module.SELECTION_SEEDS, (20261101, 20261102))
        self.assertEqual(self.module.SELECTION_PRIVACY, {'mode': 'central', 'epsilon': 4.0, 'delta': 1e-5})

    def test_confirmation_covers_clean_and_three_epsilons_on_five_seeds(self):
        cells = self.module.confirmation_cells()
        self.assertEqual(len(cells), 20)
        self.assertEqual(sorted({c['epsilon'] for c in cells}, key=lambda e: (e is not None, e)), [None, 1.0, 4.0, 8.0])
        self.assertEqual(sorted({c['seed'] for c in cells}), list(range(20261101, 20261106)))
        self.assertIn({'seed': 20261103, 'epsilon': 4.0, 'name': 'dp_eps4_20261103'}, cells)
        self.assertIn({'seed': 20261103, 'epsilon': None, 'name': 'clean_20261103'}, cells)


if __name__ == '__main__':
    unittest.main()
