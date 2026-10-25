"""Behavioral checks for retrospective seed inference; all inputs are synthetic."""
import hashlib
import importlib
import json

import pytest


def module():
    return importlib.import_module("reanalyze_selection_seeds")


def test_seed_interval_matches_hand_checked_five_seed_fixture():
    result = module().seed_interval({1: .1, 2: .2, 3: .3, 4: .4, 5: .5})
    assert result["point"] == pytest.approx(.3)
    assert result["interval95"] == pytest.approx([.1036756839, .4963243161])
    assert result["n"] == 5


def test_paired_contrast_preserves_shared_seed_control():
    lookup = {"a1": {"seed": 1, "auroc": .4}, "b1": {"seed": 1, "auroc": .3},
              "a2": {"seed": 2, "auroc": .8}, "b2": {"seed": 2, "auroc": .7}}
    original = {"contrasts": {"difference": {"point": .1, "interval95": [.09, .11]}}}
    specs = {"difference": [{"a1": 1., "b1": -1.}, {"a2": 1., "b2": -1.}]}
    result = module().analyse_contrasts(original, specs, lookup, required_seeds=(1, 2))
    assert result["difference"]["seed"]["interval95"] == pytest.approx([.1, .1])
    assert result["difference"]["conditional_patient_interval95"] == [.09, .11]


@pytest.mark.parametrize("change", ["missing_metric", "wrong_seed", "missing_contrast", "wrong_point"])
def test_reanalysis_refuses_incomplete_or_inconsistent_evidence(change):
    lookup = {"a1": {"seed": 1, "auroc": .4}, "a2": {"seed": 2, "auroc": .8}}
    original = {"contrasts": {"mean": {"point": .6, "interval95": [.5, .7]}}}
    specs = {"mean": [{"a1": 1.}, {"a2": 1.}]}
    if change == "missing_metric":
        del lookup["a2"]
    elif change == "wrong_seed":
        lookup["a2"]["seed"] = 1
    elif change == "missing_contrast":
        specs = {}
    else:
        original["contrasts"]["mean"]["point"] = .7
    with pytest.raises(ValueError):
        module().analyse_contrasts(original, specs, lookup, required_seeds=(1, 2))


@pytest.mark.parametrize("values", [{1: .4}, {1: .4, 2: float('nan')}])
def test_seed_interval_refuses_insufficient_or_nonfinite_replicates(values):
    with pytest.raises(ValueError):
        module().seed_interval(values)


def test_reader_verifies_artifact_hash_before_using_metrics(tmp_path):
    path = tmp_path / "summary.json"
    payload = json.dumps({"runs": []}).encode()
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert module().read_verified_summary(path, expected) == {"runs": []}
    path.write_text('{"runs":[{"auroc":0.99}]}')
    with pytest.raises(ValueError, match="hash"):
        module().read_verified_summary(path, expected)


def test_lookup_refuses_duplicate_cells_and_unqualified_metrics():
    good = {"name": "cell_1", "seed": 1, "auroc": .6, "finite": True, "refusals": 0}
    with pytest.raises(ValueError, match="duplicate"):
        module().metric_lookup([good, dict(good)])
    for bad in ({**good, "finite": False}, {**good, "refusals": 1}, {**good, "auroc": None}):
        with pytest.raises(ValueError):
            module().metric_lookup([bad])
