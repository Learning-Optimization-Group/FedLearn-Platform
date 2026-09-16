"""The breakdown-point report writer must describe what the data shows, including "nothing broke".

Found by the five-seed pass: every same_dir_scale run exited 1 AFTER writing its JSON, because the
report assumed FedAvg always breaks and formatted its breakdown fraction with `:g`. The control
attack is the case where FedAvg does NOT break, so the harness crashed on exactly the run whose job
is to show nothing breaking -- and the n=40 control run had hit the same crash unnoticed.
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
import robust_breakdown_point as rbp  # noqa: E402


def _inputs(fractions, fedavg_bp=None, median_bp=None, trimmed_bp=None):
    meta = {
        "fractions": fractions, "task": "t", "model": "m", "clients": 40, "alpha": 0.5,
        "client_sizes": [1] * 40, "rounds": 25, "local_epochs": 2, "lr": 0.01, "trim_beta": 0.2,
        "attack": "same_dir_scale", "seed": 1234, "torch_version": "x", "total_seconds": 1.0,
        "broken_retention_threshold": 0.5,
    }
    sweep, breakdown = {}, {}
    firsts = {"fedavg": fedavg_bp, "median": median_bp, "trimmed_mean": trimmed_bp}
    for s in rbp.AGGREGATORS:
        bp = firsts.get(s)
        sweep[s] = [(f, (0.1 if bp is not None and f >= bp else 1.0)) for f in fractions]
        breakdown[s] = {
            "empirical_first_broken_fraction": bp,
            "theoretical_breakdown": "n/a",
            "estimate_deviation_ratio_by_fraction": [(f, 0.5) for f in fractions],
        }
    return meta, sweep, breakdown


def _bullet(md, name):
    return next(line for line in md.splitlines() if line.startswith(f"- **{name}**"))


def test_report_is_written_when_no_rule_ever_breaks(tmp_path):
    """The control case, on the seed-pass grid (which does not start at f=0)."""
    meta, sweep, breakdown = _inputs([0.35, 0.4, 0.45, 0.5])
    rbp._write_markdown(SimpleNamespace(out_dir=str(tmp_path)), meta, 1.0, sweep, breakdown)

    md = (tmp_path / "robust_breakdown_point.md").read_text()
    fedavg = _bullet(md, "FedAvg")
    assert "collapses" not in fedavg, f"reported a FedAvg collapse that did not happen: {fedavg}"
    assert "0.35" in fedavg and "0.5" in fedavg, "the no-breakdown statement must name the swept range"

    median = _bullet(md, "median")
    assert "~0.5" not in median, f"stated a guessed breakdown as if measured: {median}"


def test_fedavg_breakdown_is_reported_at_the_fraction_it_happened(tmp_path):
    """FedAvg broke at 0.3 under alie at n=40 -- not at the first non-zero fraction (0.05)."""
    meta, sweep, breakdown = _inputs([0.0, 0.05, 0.1, 0.2, 0.3, 0.4], fedavg_bp=0.3)
    rbp._write_markdown(SimpleNamespace(out_dir=str(tmp_path)), meta, 1.0, sweep, breakdown)

    fedavg = _bullet((tmp_path / "robust_breakdown_point.md").read_text(), "FedAvg")
    assert "f=0.3" in fedavg
    assert "first non-zero fraction" not in fedavg, f"false claim about where FedAvg broke: {fedavg}"


def test_median_breakdown_is_reported_when_measured(tmp_path):
    meta, sweep, breakdown = _inputs([0.0, 0.25, 0.5], median_bp=0.5)
    rbp._write_markdown(SimpleNamespace(out_dir=str(tmp_path)), meta, 1.0, sweep, breakdown)

    median = _bullet((tmp_path / "robust_breakdown_point.md").read_text(), "median")
    assert "f=0.5" in median
