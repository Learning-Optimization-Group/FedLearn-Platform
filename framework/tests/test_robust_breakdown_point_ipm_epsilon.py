"""The IPM attack scale can be held fixed across attacker fractions.

By default the breakdown harness calibrates IPM against FedAvg: epsilon = 2(1 - w)/w, where w is the
attackers' share of the training data, so epsilon changes with every fraction and the sweep mixes
"more attackers" with "a weaker attack per attacker". ``--ipm-epsilon`` fixes the scale in BOTH the
accuracy sweep and the round-0 estimator-deviation pass. A spy wrapped around the real attack
function records the epsilon every attacked round actually received.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
import robust_aggregation_attack as raa  # noqa: E402
import robust_breakdown_point as rbp  # noqa: E402

TINY = ["--clients", "6", "--rounds", "1", "--local-epochs", "1", "--train-per-class", "15",
        "--test-per-class", "5", "--hidden", "8", "--fractions", "0.0,0.34,0.5"]


def _run(tmp_path, monkeypatch, extra):
    seen = []
    real = raa._apply_attack_to_round

    def spy(**kw):
        if kw.get("attack") == "ipm":
            seen.append(kw["ipm_epsilon"])
        return real(**kw)

    monkeypatch.setattr(raa, "_apply_attack_to_round", spy)
    monkeypatch.setattr(sys, "argv", ["robust_breakdown_point.py", *TINY, "--out-dir", str(tmp_path), *extra])
    rbp.main()
    with open(tmp_path / "robust_breakdown_point.json") as fh:
        return json.load(fh), seen


def test_fixed_ipm_epsilon_reaches_every_attacked_round(tmp_path, monkeypatch):
    out, seen = _run(tmp_path, monkeypatch, ["--ipm-epsilon", "1.5"])
    assert seen, "no IPM round ran"
    assert set(seen) == {1.5}  # accuracy sweep AND deviation pass
    attacked = [r for r in out["records"] if r["attack"] == "ipm"]
    assert attacked and all(r["attack_params"]["ipm_epsilon"] == 1.5 for r in attacked)
    assert out["meta"]["ipm_epsilon"] == 1.5


def test_default_still_calibrates_against_fedavg_per_fraction(tmp_path, monkeypatch):
    out, seen = _run(tmp_path, monkeypatch, [])
    assert out["meta"]["ipm_epsilon"] is None
    sizes = out["meta"]["client_sizes"]
    # attackers are clients 0..a-1 with a = round(f * 6): 2 at f=0.34, 3 at f=0.5
    want = set()
    for a in (2, 3):
        w = sum(sizes[:a]) / sum(sizes)
        want.add(round(2 * (1 - w) / w, 4))
    assert {round(e, 4) for e in seen} == want


def test_report_header_states_which_ipm_scale_ran(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, ["--ipm-epsilon", "1.5"])
    assert "IPM scale: fixed at 1.5" in (tmp_path / "robust_breakdown_point.md").read_text()
    calibrated = tmp_path / "calibrated"
    calibrated.mkdir()
    _run(calibrated, monkeypatch, [])
    assert "IPM scale: calibrated per fraction, 2(1 − w)/w" in (calibrated / "robust_breakdown_point.md").read_text()
