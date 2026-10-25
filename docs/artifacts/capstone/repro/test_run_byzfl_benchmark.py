"""Synthetic benchmark execution and preservation contract."""
import importlib
import hashlib
import json

import pytest


def benchmark():
    return importlib.import_module("run_byzfl_benchmark")


def test_benchmark_runs_real_engine_and_reconciles_baselines(tmp_path):
    summary = benchmark().run_benchmark(tmp_path / "first", seeds=(71,), rounds=2)
    assert summary["scope"] == "synthetic_integration_only"
    assert len(summary["runs"]) == 28
    assert all(r["status"] == "completed" for r in summary["runs"])
    assert len(summary["baseline_comparisons"]) == 10
    assert all(c["max_parameter_difference"] <= 1e-6 for c in summary["baseline_comparisons"])
    saved = json.loads((tmp_path / "first" / "summary.json").read_text())
    assert saved == summary
    snapshot_path = tmp_path / "first" / "source-snapshot.json"
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == summary["source_snapshot_sha256"]
    snapshot = json.loads(snapshot_path.read_text())
    assert "framework/src/fedlearn/simulation/byzfl_strategy.py" in snapshot["files"]
    assert "byzfl/aggregators/aggregators.py" in snapshot["files"]
    assert all((tmp_path / "first" / r["record"]).is_file() for r in summary["runs"])
    assert all(hashlib.sha256((tmp_path / "first" / r["record"]).read_bytes()).hexdigest() == r["sha256"]
               for r in summary["runs"])
    byz = next(r for r in summary["runs"] if r["recipe"] == "byzfl_dp_center" and r["condition"] == "clean")
    record = json.loads((tmp_path / "first" / byz["record"]).read_text())
    assert record["meta"]["aggregation_backend"]["version"] == "0.0.11"


def test_benchmark_never_overwrites_existing_evidence(tmp_path):
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        benchmark().run_benchmark(output, seeds=(71,), rounds=1)
    assert (output / "keep.txt").read_text() == "preserve"


def test_failed_cell_is_retained_and_not_replaced(tmp_path, monkeypatch):
    module = benchmark()
    real = module.run
    calls = []

    def fail_once(cfg, **kwargs):
        calls.append(cfg["name"])
        if len(calls) == 1:
            raise RuntimeError("synthetic injected failure")
        return real(cfg, **kwargs)

    monkeypatch.setattr(module, "run", fail_once)
    summary = module.run_benchmark(tmp_path / "failure", seeds=(71,), rounds=1)
    assert len(calls) == 28
    failed = [r for r in summary["runs"] if r["status"] == "error"]
    assert len(failed) == 1
    receipt = json.loads((tmp_path / "failure" / failed[0]["record"]).read_text())
    assert receipt["error"] == "synthetic injected failure"
    assert any(c["status"] == "unavailable" for c in summary["baseline_comparisons"])
