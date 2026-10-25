"""Bounded synthetic integration benchmark for native and pinned ByzFL reducers.

This program generates its own features. It accepts no patient-data input and
does not establish Effusion utility, clinical validity or production privacy.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "framework" / "src"))
from fedlearn.simulation.engine import run
from fedlearn.simulation.lab_contract import write_research_record


def recipes():
    result = []
    for rule, name in (("mean", "mean"), ("median", "median"), ("trimmed_mean", "trim"),
                       ("centered_clip", "center")):
        for external in (False, True):
            result.append({"name": f"{'byzfl' if external else 'native'}_{name}", "rule": rule,
                           "private": False, "external": external, "bucket_size": 1})
    result.append({"name": "byzfl_bucket_median", "rule": "median", "private": False,
                   "external": True, "bucket_size": 2})
    for rule, name in (("mean", "mean"), ("trimmed_mean", "trim")):
        for external in (False, True):
            result.append({"name": f"{'byzfl' if external else 'native'}_dp_{name}", "rule": rule,
                           "private": True, "external": external, "bucket_size": 1})
    result.append({"name": "byzfl_dp_center", "rule": "centered_clip", "private": True,
                   "external": True, "bucket_size": 1})
    return result


def run_benchmark(output, *, seeds=(20261201, 20261202), rounds=3, progress=None):
    output = Path(output)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    generator = torch.Generator().manual_seed(20261023)

    def split(n):
        x = torch.randn(n, 8, generator=generator)
        y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
        return x, y

    train_x, train_y = split(256)
    evaluation_x, evaluation_y = split(96)
    feature_path = output / "synthetic-features.pt"
    torch.save({"train_x": train_x, "train_y": train_y, "test_x": evaluation_x,
                "test_y": evaluation_y, "feat_dim": 8, "n_classes": 2}, feature_path)
    source_files = list((ROOT / "framework" / "src" / "fedlearn").rglob("*.py"))
    source_files += [Path(__file__).resolve(), Path(__file__).with_name("run_effusion_federated.py"),
                     ROOT / "framework" / "setup.py"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    dependency = importlib.util.find_spec("byzfl")
    if dependency is not None and dependency.origin is not None:
        dependency_root = Path(dependency.origin).parent
        hashes.update({"byzfl/" + p.relative_to(dependency_root).as_posix():
                       hashlib.sha256(p.read_bytes()).hexdigest() for p in dependency_root.rglob("*.py")})
    source_snapshot = output / "source-snapshot.json"
    write_research_record(source_snapshot, {"files": hashes,
                                           "meaning": "Exact source bytes; Git commit alone omits uncommitted changes"})
    configs = []
    for seed in seeds:
        for condition in ("clean", "sign_flip"):
            for recipe in recipes():
                name = f"{recipe['name']}_{condition}_{seed}"
                cfg = {"name": name, "seed": seed, "workload": {"features_path": str(feature_path)},
                       "federation": {"num_clients": 8, "rounds": rounds},
                       "client": {"local_epochs": 1, "lr": 0.1},
                       "server": {"rule": recipe["rule"], "clip_norm": 0.4,
                                  "rule_params": {"trim_ratio": 0.25, "tau": 0.2, "iterations": 1}},
                       "privacy": {"mode": "central" if recipe["private"] else "none",
                                   "epsilon": 4., "delta": 1e-5},
                       "behaviors": [] if condition == "clean" else [
                           {"type": "sign_flip_scale", "fraction": 0.25, "scale": -4.}]}
                if recipe["external"]:
                    cfg["server"]["external_aggregation"] = {
                        "library": "byzfl", "bucket_size": recipe["bucket_size"]}
                configs.append({"recipe": recipe["name"], "condition": condition, "seed": seed, "config": cfg})
    write_research_record(output / "planned-cells.json", {"scope": "synthetic_integration_only", "cells": configs})
    outcomes, heads = [], {}
    for cell in configs:
        cfg = cell["config"]
        relative = f"runs/{cfg['name']}.json"
        final = {}
        try:
            record = run(cfg, on_final_parameters=final.update)
            finite = bool(final) and all(torch.isfinite(t).all() for t in final.values())
            complete = len(record["rounds"]) == rounds and not record["meta"]["refusals"] and finite
            status = "completed" if complete else "refused_or_incomplete"
            record["meta"]["benchmark_scope"] = "synthetic_integration_only"
            record["meta"]["final_parameters"] = {k: v.tolist() for k, v in final.items()}
            auc = record["rounds"][-1]["auc"] if record["rounds"] else None
            if status == "completed":
                heads[cfg["name"]] = torch.cat([v.reshape(-1) for v in final.values()])
        except Exception as exc:
            status, auc = "error", None
            record = {"config": cfg, "status": status, "error_type": type(exc).__name__, "error": str(exc)}
        encoding = write_research_record(output / relative, record)
        digest = hashlib.sha256((output / relative).read_bytes()).hexdigest()
        outcome = {k: cell[k] for k in ("recipe", "condition", "seed")}
        outcome.update(name=cfg["name"], status=status, record=relative, sha256=digest, encoding=encoding,
                       final_auroc=float(auc) if auc is not None and math.isfinite(float(auc)) else None)
        outcomes.append(outcome)
        if progress is not None:
            progress(dict(outcome))
    comparisons = []
    for seed in seeds:
        for condition in ("clean", "sign_flip"):
            for suffix in ("mean", "median", "trim", "dp_mean", "dp_trim"):
                native, external = f"native_{suffix}_{condition}_{seed}", f"byzfl_{suffix}_{condition}_{seed}"
                comparison = {"native": native, "external": external, "status": "unavailable",
                              "max_parameter_difference": None}
                if native in heads and external in heads:
                    comparison.update(status="compared", max_parameter_difference=float(
                        (heads[native] - heads[external]).abs().max()))
                comparisons.append(comparison)
    summary = {"scope": "synthetic_integration_only", "seeds": list(seeds), "rounds": rounds,
               "runs": outcomes, "baseline_comparisons": comparisons,
               "features_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
               "source_snapshot_sha256": hashlib.sha256(source_snapshot.read_bytes()).hexdigest(),
               "limits": ["No patient inputs or holdout evaluation", "RDP accounting, full participation",
                          "History-centered rule differs from native zero-centered rule",
                          "Private bucketing and oracle attacks are refused",
                          "Research randomness and exports are not protected deployment releases"]}
    write_research_record(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261201, 20261202])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = run_benchmark(args.output, seeds=tuple(args.seeds), rounds=args.rounds,
                            progress=lambda r: print(json.dumps(r, sort_keys=True), flush=True))
    good = all(r["status"] == "completed" for r in summary["runs"])
    reconciled = all(c["status"] == "compared" and c["max_parameter_difference"] <= 1e-6
                     for c in summary["baseline_comparisons"])
    return 0 if good and reconciled else 1


if __name__ == "__main__":
    raise SystemExit(main())
