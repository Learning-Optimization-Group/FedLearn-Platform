"""Twelve prospective full-schedule NIH validation mechanism cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from benchmarks.nih_optimization_pilot import (
    FIXED_REPORT_SHA256, _config as _base_config, source_identity as _source_identity, verify_imports,
)
from benchmarks.nih_validation_pilot import _checkpoint
from benchmarks.nih_head_cost_probe import _save_record
from fedlearn.simulation.config import config_hash, resolve
from fedlearn.simulation.core_qualification import CoreQualificationError, qualify_core_record, run_core
from fedlearn.simulation.nih_cache import load_nih_cache

DOCUMENT = "docs/artifacts/capstone/nih-mechanism-qualification-v1-protocol.md"
PER_CELL_SECONDS, TOTAL_SECONDS, MAX_RSS_BYTES = 180, 1800, 2**31
PROOF_REQUIRED = ("framework/src/fedlearn/simulation/engine.py",
                  "framework/src/fedlearn/simulation/core_qualification.py",
                  "framework/benchmarks/nih_mechanism_qualification.py")
CHILD_BOOTSTRAP = """
import sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(root/'framework/src'),str(root/'framework')]
from benchmarks.nih_mechanism_qualification import run_child
raise SystemExit(run_child(root,Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4])))
"""


class SourceIntegrityError(ValueError):
    pass


def configs(cache, pin):
    """All declared matched cells; never condition the list on observed utility."""
    result = []
    for condition in ("clean", "sign_flip", "label_flip"):
        for rule in ("mean", "trimmed_mean"):
            for mode in ("none", "central"):
                name = f"{condition}_{rule}_{mode}"
                cfg = _base_config(cache, pin, 80, .003, name=name, seed=20261022, clip=.05, rule=rule)
                cfg["client"]["local_epochs"] = 7
                cfg["privacy"] = {"mode": mode, "epsilon": 4., "delta": 1e-5, "adjacency": "replace_one"}
                cfg["behaviors"] = ([] if condition == "clean" else
                    [{"type": "sign_flip_scale", "fraction": .2, "scale": -4.}] if condition == "sign_flip" else
                    [{"type": "label_flip", "fraction": .2}])
                result.append(resolve(cfg))
    return result


def source_identity(root):
    identity = _source_identity(root)
    identity["files"][DOCUMENT] = hashlib.sha256((Path(root)/DOCUMENT).read_bytes()).hexdigest()
    return identity


def child_command(root, config, output, protocol):
    return [sys.executable, "-I", "-c", CHILD_BOOTSTRAP, str(root), str(config), str(output), str(protocol)]


def _verify(root, identity):
    try:
        return verify_imports(root, identity["files"])
    except ValueError as error:
        raise SourceIntegrityError(str(error)) from error


def run_child(root, config_path, output, protocol_path):
    """Persist execution and failed postflight evidence; a killed child retains its start proof."""
    output.mkdir(parents=True)
    identity = json.loads(protocol_path.read_text())["source_identity"]
    outcome = {"status": "failed"}
    core_seconds = None
    try:
        proof = _verify(root, identity)
        _checkpoint(output/"executed-source-start.json", {**proof, "phase": "before_engine"})
        torch.set_num_threads(4)
        cfg = json.loads(config_path.read_text())
        if cfg not in configs(cfg["workload"]["features_path"], FIXED_REPORT_SHA256):
            raise ValueError("child recipe differs from fixed mechanism qualification protocol")
        core_started = time.monotonic()
        try:
            record = run_core(cfg, feature_report_sha256=FIXED_REPORT_SHA256, clients=cfg["federation"]["num_clients"], evaluation_split="validation")
        finally:
            core_seconds = time.monotonic()-core_started
        encoding = _save_record(output/"record.json", record)
        outcome = {"status": "ok", "record_encoding": encoding}
    except Exception as error:
        outcome = {"status": "failed", "error": str(error), "error_kind": type(error).__name__}
        if isinstance(error, CoreQualificationError) and error.record is not None:
            outcome["record_encoding"] = _save_record(output/"record.json", error.record)
        traceback.print_exc()
    finally:
        outcome["core_execution_seconds"] = core_seconds
        try:
            proof = _verify(root, identity)
            phase = "after_engine_return" if outcome["status"] == "ok" else "after_engine_error"
            _checkpoint(output/"executed-source.json", {**proof, "phase": phase})
        except Exception as error:
            outcome.update(status="failed", error=str(error), error_kind="SourceIntegrityError")
            traceback.print_exc()
        _checkpoint(output/"execution.json", outcome)
    return 0 if outcome["status"] == "ok" else 2


def summarize_record(record, expected_config, source_commit):
    """Requalify engine execution, preserve poor utility, then apply the declared resource gate."""
    try:
        meta, lines = record["meta"], record["rounds"]
        bare = {**record, "meta": {k: v for k, v in meta.items() if k != "core_qualification"}}
        qualified = qualify_core_record(bare, expected_config, feature_report_sha256=FIXED_REPORT_SHA256,
                                        clients=80, evaluation_split="validation")
        if meta["core_qualification"] != qualified or qualified["feature_control"]:
            raise ValueError("missing or inconsistent core qualification")
        wl = meta["workload"]
        parameters = 1026
        if (meta["provenance"]["git_commit"] != source_commit or wl["parameters"] != parameters
                or wl["training_device"] != "cpu" or wl["torch_num_threads"] != 4):
            raise ValueError("source, model size, device or thread recipe mismatch")
        rss = meta["peak_rss_bytes"]
        if type(rss) is not int or not 0 < rss <= MAX_RSS_BYTES or meta["peak_rss_scope"] not in ("run", "process"):
            raise ValueError("peak child RSS exceeds 2GiB or has invalid scope")
        wall = meta["wall_seconds"]
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
            raise ValueError("invalid engine timing")
        stages = {}
        for line in lines:
            for name, seconds in line["time"]["wall_s"].items():
                if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("invalid stage timing")
                stages[name] = stages.get(name, 0.) + seconds
        totals = meta["totals"]
        for key in ("down_bytes", "up_bytes", "client_flops", "client_flops_analytic"):
            if type(totals[key]) is not int or totals[key] < 0:
                raise ValueError("invalid cost total")
        return {"parameters": parameters, "federation_loop_seconds": wall, "stage_wall_seconds": stages,
                "peak_rss_bytes": rss, "peak_rss_scope": meta["peak_rss_scope"],
                "down_bytes": totals["down_bytes"], "up_bytes": totals["up_bytes"],
                "client_flops": totals["client_flops"], "client_flops_analytic": totals["client_flops_analytic"],
                "flops_agree": all(l["compute"]["client_flops"] == l["compute"]["client_flops_analytic"] for l in lines),
                "final": {key: lines[-1][key] for key in ("auc", "auc_probability", "loss", "accuracy", "ece",
                                                          "sensitivity", "specificity", "balanced_accuracy")}}
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"incomplete mechanism record: {error}") from error


def _artifacts(output, directory, log):
    paths = {name: directory/name for name in ("record.json", "execution.json", "executed-source-start.json", "executed-source.json")}
    paths["child.log"] = log
    return {name: {"file": path.relative_to(output).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "bytes": path.stat().st_size} for name, path in paths.items() if path.is_file()}


def _check_proofs(directory, identity):
    try:
        for name, phase in (("executed-source-start.json", "before_engine"), ("executed-source.json", "after_engine_return")):
            proof = json.loads((directory/name).read_text())
            files = proof["files"]
            if (proof["phase"] != phase or any(files.get(k) != identity["files"][k] for k in PROOF_REQUIRED)
                    or any(identity["files"].get(k) != v for k, v in files.items())):
                raise ValueError("loaded source fingerprint mismatch")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SourceIntegrityError(f"missing or inconsistent child source proof: {error}") from error


def run_qualification(cache, pin, output):
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    cache, output = Path(cache).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing existing mechanism output")
    if output.resolve().is_relative_to(cache):
        raise ValueError("output must be outside immutable cache")
    if pin != FIXED_REPORT_SHA256:
        raise ValueError("mechanism qualification requires fixed protocol cache pin")
    identity = source_identity(root)
    _verify(root, identity)
    if not load_nih_cache(cache, expected_report_sha256=pin)["provenance"]["pretrained"]:
        raise ValueError("mechanism qualification requires pretrained features")
    output = output.resolve(); output.mkdir(parents=True)
    for name in ("configs", "logs", "runs"): (output/name).mkdir()
    recipes = configs(cache, pin)
    protocol = {"probe_version": "nih-mechanism-qualification-v1", "source_identity": identity,
                "protocol_document_sha256": hashlib.sha256((root/DOCUMENT).read_bytes()).hexdigest(),
                "configs": recipes, "config_hashes": {c["name"]: config_hash(c) for c in recipes},
                "features_report_sha256": pin, "per_cell_seconds_cap": PER_CELL_SECONDS,
                "total_child_seconds_cap": TOTAL_SECONDS, "max_child_rss_bytes": MAX_RSS_BYTES,
                "official_test_evaluated": False}
    _checkpoint(output/"protocol.json", protocol)
    cases = [{"name": c["name"], "status": "not_run", "reason": "pending"} for c in recipes]
    for cfg in recipes: _checkpoint(output/"configs"/f"{cfg['name']}.json", cfg)
    _checkpoint(output/"index.json", {"cases": cases})
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    integrity_failed, spent = False, 0.
    for i, cfg in enumerate(recipes):
        name = cfg["name"]
        if integrity_failed or spent >= TOTAL_SECONDS:
            cases[i] = {"name": name, "status": "not_run", "reason": "source_integrity" if integrity_failed else "global_budget_exhausted"}
            _checkpoint(output/"index.json", {"cases": cases})
            continue
        before, child_seconds = time.monotonic(), 0.
        directory, log = output/"runs"/name, output/"logs"/f"{name}.log"
        try:
            if source_identity(root) != identity: raise SourceIntegrityError("source identity changed")
            with log.open("xb") as stream:
                child_started = time.monotonic()
                try:
                    subprocess.run(child_command(root, output/"configs"/f"{name}.json", directory, output/"protocol.json"),
                                   cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=min(PER_CELL_SECONDS, TOTAL_SECONDS-spent), check=True)
                finally:
                    child_seconds = time.monotonic()-child_started
                    spent += child_seconds
            if source_identity(root) != identity: raise SourceIntegrityError("source identity changed")
            _check_proofs(directory, identity)
            execution = json.loads((directory/"execution.json").read_text())
            if execution["status"] != "ok" or execution["record_encoding"] != "json":
                raise ValueError("child did not return a qualified JSON record")
            record = json.loads((directory/"record.json").read_text())
            summary = summarize_record(record, cfg, identity["git_commit"])
            core_seconds = execution["core_execution_seconds"]
            if (type(core_seconds) not in (int, float) or not math.isfinite(core_seconds)
                    or not summary["federation_loop_seconds"] <= core_seconds <= child_seconds):
                raise ValueError("invalid core execution timing")
            summary["core_execution_seconds"] = core_seconds
            case = {"name": name, "status": "qualified", "summary": summary}
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            integrity_failed = integrity_failed or isinstance(error, SourceIntegrityError)
            try:
                integrity_failed = integrity_failed or source_identity(root) != identity
                if (directory/"execution.json").exists():
                    execution = json.loads((directory/"execution.json").read_text())
                    integrity_failed = integrity_failed or execution.get("error_kind") == "SourceIntegrityError"
            except (OSError, ValueError, subprocess.SubprocessError):
                integrity_failed = True
            case = {"name": name, "status": "timeout" if isinstance(error, subprocess.TimeoutExpired) else "failed",
                    "error": str(error)}
        elapsed = time.monotonic()-before
        cases[i] = {**case, "elapsed_seconds": elapsed, "child_execution_seconds": child_seconds,
                    "parent_control_seconds": elapsed-child_seconds, "artifacts": _artifacts(output, directory, log)}
        _checkpoint(output/"index.json", {"cases": cases})
        print(json.dumps({"name": name, "status": case["status"], "child_seconds": child_seconds}), flush=True)
    elapsed = time.monotonic()-started
    result = {"probe_version": protocol["probe_version"], "source_identity": identity,
              "protocol_document_sha256": protocol["protocol_document_sha256"], "features_report_sha256": pin,
              "complete": not integrity_failed and all(c["status"] == "qualified" for c in cases),
              "source_integrity_failed": integrity_failed, "elapsed_seconds": elapsed,
              "child_execution_seconds": spent, "parent_control_seconds": elapsed-spent,
              "cases": cases, "official_test_evaluated": False}
    _checkpoint(output/"summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_qualification(args.cache, args.expected_report_sha256, args.output)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "source_identity")}))
    if not result["complete"]: parser.exit(2, "mechanism qualification incomplete; retain every failed/unrun cell\n")


if __name__ == "__main__":
    main()
