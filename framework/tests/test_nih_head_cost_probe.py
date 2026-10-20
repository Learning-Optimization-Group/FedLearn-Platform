"""Fixed four-head costs, source-bound children and failure-preserving budgets."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from benchmarks import nih_head_cost_probe as probe
from fedlearn.simulation.config import config_hash
from fedlearn.simulation.workloads import LinearHead, MLPHead

PIN = "6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd"
PARAMETERS = {"linear": 1026, "mlp8": 4122, "mlp128": 65922, "mlp512": 263682}


def _record(cfg):
    qualification = {"version": 1, "scope": "simulation_execution_only", "accepted_clients": 80,
                     "attack_knowledge": "own_data_or_own_update", "privacy_mode": "none",
                     "evaluation_split": "validation", "feature_control": False,
                     "research_record_dp_safe": False}
    return {"meta": {
        "config": cfg, "config_hash": config_hash(cfg), "provenance": {"git_commit": "test-commit"},
        "participation_schema_version": 2, "stopped_early": False, "stop_reason": None, "refusals": [],
        "behavior_assignment": {}, "privacy_mode": "none", "adjacency": "replace_one",
        "server_clip_norm": .4, "noise_multiplier": 0., "accounted_epsilon": None,
        "core_qualification": qualification, "peak_rss_bytes": 700_000_000, "peak_rss_scope": "process",
        "wall_seconds": 2., "totals": {"down_bytes": 1000, "up_bytes": 1000,
                                      "client_flops": 100, "client_flops_analytic": 100},
        "workload": {"dataset": "nih", "features_report_sha256": PIN, "evaluation_split": "validation",
                     "auc_score": "margin_v2", "pretrained": True, "feat_dim": 512, "n_classes": 2,
                     "parameters": PARAMETERS[cfg["name"]], "training_device": "cpu", "torch_num_threads": 4}},
        "rounds": [{"round": r, "forced": False, "refusal": None,
                    "participants": {"sampled": 80, "dropped": 0, "straggled": 0, "submitted": 80,
                                     "admitted": 80, "accepted": 80, "aggregated": 80},
                    "privacy": {"sensitivity": .01, "noise_std": 0., "epsilon_spent": None},
                    "accuracy": .6, "auc": .4, "auc_probability": .4, "balanced_accuracy": .5,
                    "loss": 8., "ece": .5, "sensitivity": .5, "specificity": .5,
                    "compute": {"client_flops": 50, "client_flops_analytic": 50},
                    "time": {"wall_s": {"train": .5, "eval": .1}}} for r in (1, 2)]}


def test_fixed_recipes_and_actual_model_sizes_match_protocol():
    configs = probe.configs("cache", PIN)
    assert [c["name"] for c in configs] == ["linear", "mlp8", "mlp128", "mlp512"]
    assert [c["workload"]["head"] for c in configs] == [
        {"type": "linear", "hidden": 128}, {"type": "mlp", "hidden": 8},
        {"type": "mlp", "hidden": 128}, {"type": "mlp", "hidden": 512}]
    for cfg in configs:
        assert cfg["seed"] == 20261019 and cfg["federation"]["num_clients"] == 80
        assert cfg["federation"]["clients_per_round"] == 80 and cfg["federation"]["rounds"] == 2
        assert cfg["client"]["local_epochs"] == 7 and cfg["client"]["batch_size"] == 256
        assert cfg["client"]["lr"] == .003 and cfg["server"]["clip_norm"] == .4
        assert cfg["server"]["rule"] == "mean" and cfg["privacy"]["mode"] == "none" and not cfg["behaviors"]
        assert cfg["workload"]["evaluation_split"] == "validation" and not cfg["workload"]["balance"]
        assert cfg["workload"]["auc_score"] == "margin_v2" and not cfg["workload"]["allow_random_features"]
        head = cfg["workload"]["head"]
        model = LinearHead(512, 2) if head["type"] == "linear" else MLPHead(512, head["hidden"], 2)
        assert sum(p.numel() for p in model.parameters()) == PARAMETERS[cfg["name"]]


def test_summary_retains_poor_finite_utility_without_head_selection():
    cfg = probe.configs("cache", PIN)[0]
    summary = probe.summarize_record(_record(cfg), cfg, "test-commit")
    assert summary["final"]["auc"] == .4 and summary["final"]["loss"] == 8
    assert summary["parameters"] == 1026 and summary["stage_wall_seconds"] == {"train": 1., "eval": .2}
    assert summary["peak_rss_bytes"] == 700_000_000 and summary["flops_agree"]


@pytest.mark.parametrize("field", ["recipe", "source", "params", "rss", "threads", "device", "partial", "marker"])
def test_wrong_record_cannot_qualify_costs(field):
    cfg = probe.configs("cache", PIN)[0]
    rec = _record(cfg)
    if field == "recipe": rec["meta"]["config"] = {**cfg, "seed": 1}
    elif field == "source": rec["meta"]["provenance"]["git_commit"] = "wrong"
    elif field == "params": rec["meta"]["workload"]["parameters"] = 4122
    elif field == "rss": rec["meta"]["peak_rss_bytes"] = 2**31 + 1
    elif field == "threads": rec["meta"]["workload"]["torch_num_threads"] = 8
    elif field == "device": rec["meta"]["workload"]["training_device"] = "mps"
    elif field == "partial": rec["rounds"][0]["participants"]["accepted"] = 79
    elif field == "marker": rec["meta"]["core_qualification"]["research_record_dp_safe"] = True
    with pytest.raises(ValueError): probe.summarize_record(rec, cfg, "test-commit")


def test_wrong_pin_and_existing_output_refuse_before_cache_access(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "load_nih_cache", lambda *a, **k: pytest.fail("cache reached"))
    with pytest.raises(ValueError): probe.run_probe(tmp_path/"cache", "0"*64, tmp_path/"out")
    out = tmp_path/"out"; out.mkdir(); (out/"sentinel").write_text("preserve")
    with pytest.raises(FileExistsError): probe.run_probe(tmp_path/"cache", PIN, out)
    assert (out/"sentinel").read_text() == "preserve"


def test_protocol_document_is_bound_into_source_identity(tmp_path, monkeypatch):
    doc = tmp_path/probe.DOCUMENT
    doc.parent.mkdir(parents=True); doc.write_text("prospective protocol")
    monkeypatch.setattr(probe, "_source_identity", lambda _: {"git_commit": "test", "files": {}})
    before = probe.source_identity(tmp_path)
    assert before["files"][probe.DOCUMENT] == hashlib.sha256(doc.read_bytes()).hexdigest()
    doc.write_text("changed protocol")
    assert before != probe.source_identity(tmp_path)


def test_cli_bad_pin_fails_without_creating_output(tmp_path):
    root = Path(probe.__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-m", "benchmarks.nih_head_cost_probe", "--cache", str(tmp_path/"cache"),
                             "--expected-report-sha256", "0"*64, "--output", str(tmp_path/"out")],
                            cwd=root, env=dict(os.environ, PYTHONPATH="framework/src:framework"),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "fixed protocol cache pin" in result.stderr
    assert not (tmp_path/"out").exists()


def _orchestrate(tmp_path, monkeypatch, outcome, overhead=0):
    root = tmp_path/"repo"
    doc = root/probe.DOCUMENT; doc.parent.mkdir(parents=True); doc.write_text("fixed")
    monkeypatch.setattr(probe, "__file__", str(root/"framework/benchmarks/nih_head_cost_probe.py"))
    identity = {"git_commit": "test-commit", "files": {name: name + "-hash" for name in probe.PROOF_REQUIRED}}
    state = {"calls": 0, "changed": False, "now": 0.}
    def source(_):
        state["now"] += overhead
        return {**identity, "git_commit": "changed"} if state["changed"] else copy.deepcopy(identity)
    monkeypatch.setattr(probe, "source_identity", source)
    monkeypatch.setattr(probe, "verify_imports", lambda *a: {"files": identity["files"]})
    monkeypatch.setattr(probe, "load_nih_cache", lambda *a, **k: {"provenance": {"pretrained": True}})
    monkeypatch.setattr(probe.time, "monotonic", lambda: state["now"])
    if outcome == "budget": monkeypatch.setattr(probe, "TOTAL_SECONDS", 0)
    def child(command, **kwargs):
        state["calls"] += 1
        # Before any child runs, every planned cell is already retained.
        index = json.loads((tmp_path/"out/index.json").read_text())
        assert len(index["cases"]) == 4
        config = json.loads(Path(command[-3]).read_text())
        directory = Path(command[-2]); directory.mkdir()
        state["now"] += 300 if outcome == "exhaustion" else 3.
        for filename, phase in (("executed-source-start.json", "before_engine"),
                                ("executed-source.json", "after_engine_return")):
            if outcome == "missing_proof" and state["calls"] == 1 and phase == "after_engine_return": continue
            files = identity["files"] if outcome != "wrong_proof" else {k: "wrong" for k in identity["files"]}
            (directory/filename).write_text(json.dumps({"phase": phase, "files": files}))
        rec = _record(config)
        if state["calls"] == 1:
            if outcome == "partial": rec["rounds"][0]["participants"]["accepted"] = 79
            if outcome == "rss": rec["meta"]["peak_rss_bytes"] = 2**31 + 1
        (directory/"record.json").write_text(json.dumps(rec))
        (directory/"execution.json").write_text(json.dumps({"status": "ok", "record_encoding": "json", "core_execution_seconds": 2.5}))
        if state["calls"] == 1:
            if outcome == "error": raise subprocess.CalledProcessError(2, command)
            if outcome == "timeout": raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            if outcome == "mutation": state["changed"] = True
            if outcome == "child_source_error":
                (directory/"execution.json").write_text(json.dumps({"status": "failed", "error_kind": "SourceIntegrityError"}))
                raise subprocess.CalledProcessError(2, command)
        if outcome == "exhaustion":
            assert kwargs["timeout"] == 300
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(probe.subprocess, "run", child)
    result = probe.run_probe(tmp_path/"cache", PIN, tmp_path/"out")
    assert len(result["cases"]) == 4
    assert json.loads((tmp_path/"out/index.json").read_text())["cases"] == result["cases"]
    assert result["elapsed_seconds"] == pytest.approx(result["child_execution_seconds"] + result["parent_control_seconds"])
    if outcome == "success":
        assert result["complete"] and state["calls"] == 4
        assert all(c["status"] == "qualified" for c in result["cases"])
    elif outcome == "budget":
        assert state["calls"] == 0 and all(c["status"] == "not_run" for c in result["cases"])
    elif outcome == "exhaustion":
        assert state["calls"] == 2 and result["child_execution_seconds"] == 600
        assert [c["status"] for c in result["cases"]] == ["timeout", "timeout", "not_run", "not_run"]
    elif outcome in ("mutation", "wrong_proof", "missing_proof", "child_source_error"):
        assert state["calls"] == 1 and result["source_integrity_failed"]
        assert all(c["status"] == "not_run" for c in result["cases"][1:])
    else:
        assert state["calls"] == 4 and not result["complete"]
        assert result["cases"][0]["status"] == ("timeout" if outcome == "timeout" else "failed")
        assert all(c["status"] == "qualified" for c in result["cases"][1:])
        assert "record.json" in result["cases"][0]["artifacts"]
    return result


@pytest.mark.parametrize("outcome", ["success", "error", "timeout", "partial", "rss", "mutation",
                                    "budget", "exhaustion", "missing_proof", "wrong_proof", "child_source_error"])
def test_four_case_orchestration_preserves_failures_and_stops_untrusted_source(tmp_path, monkeypatch, outcome):
    _orchestrate(tmp_path, monkeypatch, outcome)


def test_parent_control_overhead_does_not_consume_child_budget(tmp_path, monkeypatch):
    rec = _orchestrate(tmp_path, monkeypatch, "success", overhead=1000)
    assert rec["child_execution_seconds"] == pytest.approx(12.)
    assert rec["parent_control_seconds"] >= 8000


def test_child_preserves_nonfinite_failed_postflight_record(tmp_path, monkeypatch):
    from fedlearn.simulation.core_qualification import CoreQualificationError
    cfg = probe.configs("cache", PIN)[0]
    record = _record(cfg); record["rounds"][0]["loss"] = float("nan")
    monkeypatch.setattr(probe, "verify_imports", lambda *a: {"files": {}})
    def refuse(*a, **k): raise CoreQualificationError("nonfinite utility", record=record)
    monkeypatch.setattr(probe, "run_core", refuse)
    protocol = tmp_path/"protocol.json"; protocol.write_text(json.dumps({"source_identity": {"files": {}}}))
    config = tmp_path/"config.json"; config.write_text(json.dumps(cfg))
    out = tmp_path/"child"
    assert probe.run_child(Path(probe.__file__).resolve().parents[2], config, out, protocol) == 2
    saved = json.loads((out/"record.json").read_text())
    assert saved["encoding"] == "nonfinite_tagged_v1"
    assert saved["record"]["rounds"][0]["loss"] == {"__nonfinite_float__": "nan"}
    assert json.loads((out/"execution.json").read_text())["status"] == "failed"
    assert json.loads((out/"executed-source.json").read_text())["phase"] == "after_engine_error"


def test_full_core_timer_includes_actual_engine_setup_outside_federation_loop(tmp_path, monkeypatch):
    from tests.test_simulation_engine_cli import _features
    from fedlearn.simulation import engine
    from fedlearn.simulation.config import resolve
    from fedlearn.simulation.workloads import FeatureHeadWorkload
    cfg = resolve({"workload": {"features_path": _features(tmp_path)},
                   "federation": {"num_clients": 2, "rounds": 1}, "client": {"local_epochs": 1}})
    original = FeatureHeadWorkload.__init__
    def delayed_setup(self, resolved):
        time.sleep(.05)
        original(self, resolved)
    monkeypatch.setattr(FeatureHeadWorkload, "__init__", delayed_setup)
    # Exercise real engine setup/loop; the fixed NIH recipe gate has its own tests.
    monkeypatch.setattr(probe, "configs", lambda *a: [cfg])
    monkeypatch.setattr(probe, "run_core", lambda cfg, **kw: engine.run(cfg))
    monkeypatch.setattr(probe, "verify_imports", lambda *a: {"files": {}})
    protocol = tmp_path/"protocol.json"; protocol.write_text(json.dumps({"source_identity": {"files": {}}}))
    config = tmp_path/"config.json"; config.write_text(json.dumps(cfg)); out = tmp_path/"child"
    assert probe.run_child(Path(probe.__file__).resolve().parents[2], config, out, protocol) == 0
    execution = json.loads((out/"execution.json").read_text())
    record = json.loads((out/"record.json").read_text())
    assert execution["core_execution_seconds"] >= record["meta"]["wall_seconds"] + .05


def test_config_hashes_exist_before_any_launch(tmp_path, monkeypatch):
    original = probe._checkpoint
    def inspect(path, value):
        if path.name == "protocol.json":
            assert value["config_hashes"] == {
                c["name"]: hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
                for c in value["configs"]}
        original(path, value)
    monkeypatch.setattr(probe, "_checkpoint", inspect)
    _orchestrate(tmp_path, monkeypatch, "success")


def test_actual_interrupted_child_keeps_start_proof_without_completion(tmp_path):
    root = Path(probe.__file__).resolve().parents[2]
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in ("framework/src", "framework/benchmarks") for p in (root/folder).rglob("*.py")}
    protocol = tmp_path/"protocol.json"; protocol.write_text(json.dumps({"source_identity": {"files": files}}))
    cfg = probe.configs("missing-cache", PIN)[0]
    config = tmp_path/"config.json"; config.write_text(json.dumps(cfg)); out = tmp_path/"child"
    # Only the slow engine boundary is replaced; actual run_child/proof/checkpoint code runs.
    bootstrap = """
import sys,time
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'framework/src'),str(root/'framework')]
from benchmarks import nih_head_cost_probe as probe
probe.run_core=lambda *a,**k: time.sleep(30)
raise SystemExit(probe.run_child(root,Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4])))
"""
    with (tmp_path/"child.log").open("w") as log:
        child = subprocess.Popen([sys.executable, "-I", "-c", bootstrap, str(root), str(config), str(out), str(protocol)],
                                 env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"), stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while not (out/"executed-source-start.json").exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert (out/"executed-source-start.json").exists(), (tmp_path/"child.log").read_text()
        finally:
            if child.poll() is None: child.kill()
            child.wait(timeout=5)
    assert not (out/"executed-source.json").exists() and not (out/"execution.json").exists()
