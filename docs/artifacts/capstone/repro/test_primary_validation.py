"""Fast regressions for fixed-seed primary inference and missingness."""
import copy
import importlib.util
import math
from pathlib import Path
import statistics
import unittest

SPEC = importlib.util.spec_from_file_location(
    "study_verifier", Path(__file__).with_name("verify_phase2_study.py")
)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def fixture(missing_seed_indices=()):
    """Hand-set eight AUCs give D_i = i/1000, independently of the checker."""
    scores = {}
    rows = []
    missing = []
    effects = []
    for index in range(1, 11):
        seed = 20261100 + index
        components = {
            "clean/mean/none": .70,
            "clean/mean/eps4": .65,
            "clean/trimmed_mean/none": .71,
            "clean/trimmed_mean/eps4": .64,
            "sign_flip/mean/none": .60,
            "sign_flip/mean/eps4": .55,
            "sign_flip/trimmed_mean/none": .63,
            "sign_flip/trimmed_mean/eps4": .56 + index / 1000,
        }
        absent = []
        for key, value in list(components.items()):
            condition, rule, mode = key.split("/")
            name = f"primary_s{seed}_linear_n80_c005_{condition}_{rule}_{mode}"
            if index in missing_seed_indices and key == "clean/mean/none":
                absent.append(name)
                del components[key]
            else:
                scores[name] = value
        effect = None if absent else index / 1000
        if effect is not None:
            effects.append(effect)
        missing.extend(absent)
        rows.append(dict(seed=seed, components=components, missing=absent, effect=effect))
    mean = statistics.mean(effects) if effects else None
    sd = statistics.stdev(effects) if len(effects) > 1 else None
    half_width = 2.2621571628540993 * sd / math.sqrt(10) if len(effects) == 10 else None
    primary = dict(
        complete=len(effects) == 10,
        n_complete=len(effects),
        requested_seeds=10,
        per_seed=rows,
        missing=missing,
        mean=mean,
        sd=sd,
        range=[min(effects), max(effects)] if effects else None,
        ci95=[mean - half_width, mean + half_width] if half_width is not None else None,
        scope="conditional_simulation_seed_variation",
        interval_method="paired_t_df9_if_all10_complete",
    )
    return primary, scores


class PrimaryValidationTests(unittest.TestCase):
    def test_complete_and_partial_scores_reconcile_without_inventing_interval(self):
        for absent in [(), (10,), tuple(range(2, 11)), tuple(range(1, 11))]:
            with self.subTest(absent=absent):
                primary, scores = fixture(absent)
                original = copy.deepcopy(primary)
                self.assertEqual(VERIFIER.verify_primary(primary, scores), original)

    def test_extra_missing_reordered_and_duplicate_rows_are_rejected(self):
        for kind in ["extra", "nine", "reversed", "duplicate"]:
            with self.subTest(kind=kind):
                primary, scores = fixture()
                if kind == "extra":
                    primary["per_seed"].append(copy.deepcopy(primary["per_seed"][0]))
                elif kind == "nine":
                    primary["per_seed"].pop()
                    values = [i / 1000 for i in range(1, 10)]
                    primary.update(complete=False, n_complete=9, ci95=None,
                                   mean=statistics.mean(values), sd=statistics.stdev(values),
                                   range=[.001, .009])
                elif kind == "reversed":
                    primary["per_seed"].reverse()
                else:
                    primary["per_seed"][1] = copy.deepcopy(primary["per_seed"][0])
                with self.assertRaises(AssertionError):
                    VERIFIER.verify_primary(primary, scores)

    def test_primary_metadata_and_global_missing_cannot_be_forged(self):
        mutations = {"requested_seeds": 9, "scope": "clinical_population",
                     "interval_method": "unpaired_normal", "missing": ["invented"],
                     "complete": 1, "n_complete": 10.0, "extra": "unplanned"}
        for key, value in mutations.items():
            with self.subTest(key=key):
                primary, scores = fixture()
                primary[key] = value
                with self.assertRaises(AssertionError):
                    VERIFIER.verify_primary(primary, scores)

    def test_range_and_interval_require_exactly_two_finite_numbers(self):
        for key in ["range", "ci95"]:
            for value in [[], [0.0], [0.0, 0.0, 0.0], None,
                          [float("nan"), 0.0], [float("inf"), 0.0], [False, 0.0]]:
                with self.subTest(key=key, value=value):
                    primary, scores = fixture()
                    primary[key] = value
                    with self.assertRaises(AssertionError):
                        VERIFIER.verify_primary(primary, scores)

    def test_statistics_effect_and_components_must_match_verified_scores(self):
        for key in ["mean", "sd", "effect", "component"]:
            for value in [999.0, float("nan"), float("inf"), True, "0.1"]:
                with self.subTest(key=key, value=value):
                    primary, scores = fixture()
                    if key in ["mean", "sd"]:
                        primary[key] = value
                    elif key == "effect":
                        primary["per_seed"][0]["effect"] = value
                    else:
                        primary["per_seed"][0]["components"]["clean/mean/none"] = value
                    with self.assertRaises(AssertionError):
                        VERIFIER.verify_primary(primary, scores)

    def test_partial_results_require_exact_missing_identities_and_no_interval(self):
        for kind in ["hidden_global", "hidden_row", "duplicate_global", "duplicate_row",
                     "invented_component", "effect", "ci95"]:
            with self.subTest(kind=kind):
                primary, scores = fixture((10,))
                row = primary["per_seed"][-1]
                if kind == "hidden_global": primary["missing"] = []
                elif kind == "hidden_row": row["missing"] = []
                elif kind == "duplicate_global": primary["missing"] *= 2
                elif kind == "duplicate_row": row["missing"] *= 2
                elif kind == "invented_component": row["components"]["clean/mean/none"] = .7
                elif kind == "effect": row["effect"] = .010
                else: primary["ci95"] = [0.0, 1.0]
                with self.assertRaises(AssertionError):
                    VERIFIER.verify_primary(primary, scores)

    def test_zero_or_one_complete_seed_requires_null_sd(self):
        for absent in [tuple(range(1, 11)), tuple(range(2, 11))]:
            with self.subTest(absent=absent):
                primary, scores = fixture(absent)
                primary["sd"] = 0.0
                with self.assertRaises(AssertionError):
                    VERIFIER.verify_primary(primary, scores)


if __name__ == "__main__":
    unittest.main(verbosity=2)
