"""True-engine qualification checks using generated bundles and prior-workload references."""
from collections import OrderedDict
import hashlib
import importlib
import json

import numpy as np
import pytest
import torch

from fedlearn.simulation.effusion_bundle import write_effusion_bundle
from fedlearn.simulation.engine import run
from run_effusion_federated import config
from run_effusion_gdp_recipe import calibration
from run_effusion_zero_head import ZeroHeadWorkload


@pytest.fixture
def inputs(tmp_path):
    torch.set_num_threads(1)
    x = np.zeros((8, 512), dtype=np.float32)
    x[:, 0] = np.tile([-1., 1.], 4)
    y = np.tile([0, 1], 4).astype(np.int64)
    train_patients = [str(1 + i // 2) for i in range(8)]
    selection_patients = [str(11 + i // 2) for i in range(8)]
    def split(patients, offset):
        return {"x": x.copy(), "y": y.copy(), "patient_ids": patients,
                "image_ids": [f"synthetic-{offset+i}" for i in range(8)]}
    bundle = tmp_path / "bundle"
    pin = write_effusion_bundle(bundle, split(train_patients, 0), split(selection_patients, 8),
                                provenance={"synthetic_fixture": True})
    settings = {"seed": 71, "clients": 2, "rounds": 1, "local_epochs": 1,
                "batch_size": 256, "lr": .03, "clip_norm": .05, "epsilon": 4., "delta": 1e-5}
    reference = tmp_path / "a17"
    reference.mkdir()
    rows = []
    for private, label in ((False, "clean"), (True, "eps4")):
        cal = calibration(4., rounds=1) if private else None
        privacy = {"mode": "central", "epsilon": cal["rdpEquivalentEpsilon"], "delta": 1e-5} if private else None
        cfg = config(f"prior-{label}", 71, clients=2, features_path=bundle,
                     report_sha256=pin, rounds=1, privacy=privacy,
                     recipe={"lr": .03, "clip_norm": .05})
        cfg["client"]["local_epochs"] = 1
        workload = ZeroHeadWorkload(x, y, train_patients, x, y, np.zeros(512), np.ones(512),
                                    clients=2, partition_seed=71, info={"synthetic_fixture": True})
        final = {}
        record = run(cfg, workload=workload, on_final_parameters=final.update)
        record["meta"]["finalHead"] = {k: v.tolist() for k, v in final.items()}
        name = f"zero_T1_{label}_71"
        (reference / f"{name}.json").write_text(json.dumps(record))
        rows.append({"name": name, "seed": 71, "rounds": 1, "epsilon": 4. if private else None,
                     "auroc": record["rounds"][-1]["auc"], "finite": True, "refusals": 0})
    summary = reference / "summary.json"
    summary.write_text(json.dumps({"runs": rows}))
    return {"bundle": bundle, "bundle_sha256": pin, "reference_dir": reference,
            "reference_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
            "settings": settings, "synthetic": True}


def runner():
    return importlib.import_module("run_effusion_byzfl_qualification")


def freeze(inputs, tmp_path):
    path = tmp_path / "plan"
    plan = runner().preflight(output=path, **inputs)
    return path, hashlib.sha256((path / "plan.json").read_bytes()).hexdigest(), plan


def test_preflight_performs_no_model_fits_and_freezes_hashes(inputs, tmp_path, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("preflight must not fit")
    monkeypatch.setattr(runner(), "run", forbid)
    path, _, plan = freeze(inputs, tmp_path)
    assert len(plan["cells"]) == 4
    assert plan["head_init"] == "zero"
    assert plan["synthetic"] is True
    assert (path / "public-manifest.json").is_file()
    assert plan["calibration"]["claimedEpsilon"] == 4.
    assert plan["calibration"]["rdpEquivalentEpsilon"] > 4.
    assert set(plan["resolved_configurations"]) == {c["name"] for c in plan["cells"]}


def test_true_engine_reconciles_old_workload_and_external_backend(inputs, tmp_path):
    path, pin, _ = freeze(inputs, tmp_path)
    output = tmp_path / "qualification"
    result = runner().execute(path, pin, output)
    assert result["status"] == "qualified"
    assert len(result["runs"]) == 4
    assert all(x["status"] == "completed" for x in result["runs"])
    assert all(x["max_parameter_difference"] <= 1e-6 for x in result["comparisons"])
    for row in result["runs"]:
        raw = (output / row["record"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        record = json.loads(raw)
        assert record["meta"]["head_init"] == "zero"
        if row["private"]:
            accounting = record["meta"]["qualification_accounting"]
            assert accounting["accountant"] == "gdp"
            assert accounting["claimed_epsilon"] == 4.
            assert accounting["exact_epsilon"] <= 4. + 1e-6
            assert accounting["engine_rdp_equivalent_epsilon"] > 4.


@pytest.mark.parametrize("changed", ["plan", "bundle", "reference"])
def test_changed_evidence_refuses_before_training(inputs, tmp_path, monkeypatch, changed):
    path, pin, plan = freeze(inputs, tmp_path)
    if changed == "plan":
        (path / "plan.json").write_text('{}')
    elif changed == "bundle":
        (inputs["bundle"] / "selection-y.npy").write_bytes(b"changed")
    else:
        (inputs["reference_dir"] / (plan["references"]["clean"]["name"] + ".json")).write_text('{}')
    def forbid(*args, **kwargs):
        raise AssertionError("modified evidence reached training")
    monkeypatch.setattr(runner(), "run", forbid)
    with pytest.raises(ValueError):
        runner().execute(path, pin, tmp_path / "refused")


def test_existing_output_is_preserved(inputs, tmp_path):
    path, pin, _ = freeze(inputs, tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        runner().execute(path, pin, output)


def test_failed_cell_is_retained_without_replacement(inputs, tmp_path, monkeypatch):
    path, pin, _ = freeze(inputs, tmp_path)
    real, calls = runner().run, []
    def fail_once(cfg, **kwargs):
        calls.append(cfg["name"])
        if len(calls) == 1:
            raise RuntimeError("injected qualification failure")
        return real(cfg, **kwargs)
    monkeypatch.setattr(runner(), "run", fail_once)
    result = runner().execute(path, pin, tmp_path / "failed")
    assert len(calls) == 4
    assert result["status"] == "not_qualified"
    assert sum(x["status"] == "error" for x in result["runs"]) == 1


def test_refused_engine_record_keeps_its_observed_rounds(inputs, tmp_path, monkeypatch):
    path, pin, _ = freeze(inputs, tmp_path)
    real, calls = runner().run, []
    def refuse_once(cfg, **kwargs):
        record = real(cfg, **kwargs)
        calls.append(cfg["name"])
        if len(calls) == 1:
            record["meta"]["refusals"] = [{"round": 1, "reason": "synthetic refusal"}]
        return record
    monkeypatch.setattr(runner(), "run", refuse_once)
    output = tmp_path / "refused-record"
    result = runner().execute(path, pin, output)
    saved = json.loads((output / result["runs"][0]["record"]).read_text())
    assert result["status"] == "not_qualified"
    assert len(saved["rounds"]) == 1
    assert saved["meta"]["finalHead"]
    assert saved["qualification_failure"]["error_type"] == "ValueError"


def test_crash_after_round_retains_round_journal(inputs, tmp_path, monkeypatch):
    path, pin, _ = freeze(inputs, tmp_path)
    real, calls = runner().run, []
    def crash_once(cfg, **kwargs):
        calls.append(cfg["name"])
        if len(calls) == 1:
            forward = kwargs.pop("on_round", None)
            def callback(line):
                if forward is not None:
                    forward(line)
                raise RuntimeError("synthetic crash after round")
            return real(cfg, on_round=callback, **kwargs)
        return real(cfg, **kwargs)
    monkeypatch.setattr(runner(), "run", crash_once)
    output = tmp_path / "crashed"
    result = runner().execute(path, pin, output)
    journal = output / "rounds" / "native_clean" / "round-0001.json"
    assert journal.is_file()
    assert json.loads(journal.read_text())["round"] == 1
    assert result["runs"][0]["status"] == "error"


def test_real_execution_requires_explicit_committed_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        runner()._checkpoint({"synthetic": False}, None, None)


def test_changed_source_snapshot_refuses_before_training(inputs, tmp_path, monkeypatch):
    path, pin, plan = freeze(inputs, tmp_path)
    name = next(iter(plan["source_sha256"]))
    (path / "sources" / name).write_text("changed snapshot")
    def forbid(*args, **kwargs):
        raise AssertionError("modified snapshot reached training")
    monkeypatch.setattr(runner(), "run", forbid)
    with pytest.raises(ValueError, match="snapshot"):
        runner().execute(path, pin, tmp_path / "changed-snapshot")


def test_forged_synthetic_plan_cannot_bypass_bundle_provenance(inputs, tmp_path, monkeypatch):
    path, _, plan = freeze(inputs, tmp_path)
    report_path = inputs["bundle"] / "report.json"
    report = json.loads(report_path.read_text())
    report["provenance"]["synthetic_fixture"] = False
    report_path.write_text(json.dumps(report))
    bundle_pin = hashlib.sha256(report_path.read_bytes()).hexdigest()
    plan["bundle_report_sha256"] = bundle_pin
    for cfg in plan["resolved_configurations"].values():
        cfg["workload"]["features_report_sha256"] = bundle_pin
    plan["configuration_sha256"] = {name: runner()._configuration_hash(cfg)
                                    for name, cfg in plan["resolved_configurations"].items()}
    public = json.loads((path / "public-manifest.json").read_text())
    public = {key: plan[key] for key in public}
    (path / "public-manifest.json").write_text(json.dumps(public))
    plan["public_manifest_sha256"] = hashlib.sha256((path / "public-manifest.json").read_bytes()).hexdigest()
    (path / "plan.json").write_text(json.dumps(plan))
    pin = hashlib.sha256((path / "plan.json").read_bytes()).hexdigest()
    def forbid(*args, **kwargs):
        raise AssertionError("unmarked bundle reached training")
    monkeypatch.setattr(runner(), "run", forbid)
    with pytest.raises(ValueError, match="synthetic"):
        runner().execute(path, pin, tmp_path / "forged-synthetic")
