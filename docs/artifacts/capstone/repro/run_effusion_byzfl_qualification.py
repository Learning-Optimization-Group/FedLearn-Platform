"""Freeze and execute a four-cell train/selection-only workload reconciliation.

Preflight verifies existing inputs and creates no fitted model. Real-data execution
requires a local committed source/protocol checkpoint; synthetic tests are explicit.
"""
import argparse
from collections import OrderedDict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "framework" / "src"))
from fedlearn.estimators.params import trainable_state
from fedlearn.simulation.config import default_seeds, resolve
from fedlearn.simulation.effusion_bundle import load_effusion_bundle
from fedlearn.simulation.engine import run
from fedlearn.simulation.lab_contract import write_research_record
from fedlearn.simulation.workloads import NihEffusionBundleWorkload
from run_effusion_gdp_recipe import calibration, check_calibration

BUNDLE_PIN = "dec54dfb580232ed6bf8da16ea9466248db58de4925285c8fb0d07a381562ffa"
A17_PIN = "54d584fcc0dd92f7ba86d4edb530f433e02c08cb965f7e80208a068f8f3fc225"
PROTOCOL = ROOT / "docs/artifacts/capstone/effusion-byzfl-qualification-protocol-2026-10-24.md"
DEFAULTS = {"seed": 20261101, "clients": 80, "rounds": 20, "local_epochs": 7,
            "batch_size": 256, "lr": .03, "clip_norm": .05, "epsilon": 4., "delta": 1e-5}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _configuration_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read(path, expected):
    if Path(path).is_symlink() or sha(path) != expected:
        raise ValueError(f"input hash mismatch: {Path(path).name}")
    return json.loads(Path(path).read_text())


class ZeroBundleWorkload(NihEffusionBundleWorkload):
    def build_model(self, init_seed=None):
        model = super().build_model(init_seed)
        if init_seed is not None:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
        return model


def _runtime():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "scipy": importlib.metadata.version("scipy"),
            "byzfl": importlib.metadata.version("byzfl"),
            "torch_num_threads": torch.get_num_threads()}


def _source_paths():
    files = {p.relative_to(ROOT).as_posix(): p for p in (ROOT / "framework/src/fedlearn").rglob("*.py")}
    for path in (Path(__file__).resolve(), HERE / "run_effusion_gdp_recipe.py",
                 HERE / "run_effusion_federated.py", HERE / "run_effusion_recipe.py", PROTOCOL):
        files[path.relative_to(ROOT).as_posix()] = path
    dependency = importlib.util.find_spec("byzfl")
    if dependency is None or dependency.origin is None:
        raise ValueError("pinned ByzFL dependency is unavailable")
    directory = Path(dependency.origin).parent
    files.update({"byzfl/" + p.relative_to(directory).as_posix(): p for p in directory.rglob("*.py")})
    return files


def _configuration(cell, settings, bundle, bundle_sha256, cal):
    private = cell["private"]
    return {"name": cell["name"], "seeds": default_seeds(settings["seed"]),
            "workload": {"dataset": "nih", "features_path": str(bundle),
                         "features_report_sha256": bundle_sha256, "evaluation_split": "validation",
                         "allow_random_features": False, "balance": False, "auc_score": "margin_v2",
                         "head": {"type": "linear"}},
            "federation": {"num_clients": settings["clients"], "rounds": settings["rounds"],
                           "partition": {"type": "patient_iid"}},
            "client": {"algorithm": "fedavg", "local_epochs": settings["local_epochs"],
                       "lr": settings["lr"], "batch_size": settings["batch_size"]},
            "server": {"rule": "mean", "clip_norm": settings["clip_norm"],
                       **({"external_aggregation": {"library": "byzfl", "bucket_size": 1}}
                          if cell["backend"] == "byzfl" else {})},
            "privacy": {"mode": "central" if private else "none",
                        "epsilon": cal["rdpEquivalentEpsilon"] if private else settings["epsilon"],
                        "delta": settings["delta"]}, "behaviors": []}


def _reference_heads(record, shapes):
    values = record["meta"]["finalHead"]
    if set(values) != set(shapes):
        raise ValueError("reference parameter keys differ")
    result = OrderedDict()
    for key, shape in shapes.items():
        value = torch.tensor(values[key], dtype=torch.float32)
        if list(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError("reference parameter shape or finiteness differs")
        result[key] = value
    return result


def _validate_bundle_classification(bundle_data, bundle_sha256, synthetic):
    if type(synthetic) is not bool:
        raise ValueError("synthetic classification must be boolean")
    if synthetic and (bundle_sha256 == BUNDLE_PIN or bundle_data["report"]["provenance"].get("synthetic_fixture") is not True):
        raise ValueError("synthetic qualification requires an explicitly generated fixture bundle")


def preflight(bundle, bundle_sha256, reference_dir, reference_summary_sha256, output, *,
              synthetic=False, settings=None):
    bundle, reference_dir, output = Path(bundle).resolve(), Path(reference_dir).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("qualification plan directory already exists")
    if any("holdout" in part.lower() or part.lower() == "test" for p in (bundle, reference_dir) for part in p.parts):
        raise ValueError("holdout/test input paths are forbidden")
    settings = dict(DEFAULTS if settings is None else settings)
    if set(settings) != set(DEFAULTS):
        raise ValueError("qualification settings differ from the declared schema")
    for key in ("seed", "clients", "rounds", "local_epochs", "batch_size"):
        if type(settings[key]) is not int or settings[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid setting: {key}")
    for key in ("lr", "clip_norm", "epsilon", "delta"):
        if not math.isfinite(float(settings[key])) or float(settings[key]) <= 0:
            raise ValueError(f"invalid setting: {key}")
    if settings["delta"] >= 1 or torch.get_num_threads() != 1:
        raise ValueError("qualification requires delta in (0,1) and one Torch CPU thread")
    if not synthetic and (settings != DEFAULTS or bundle_sha256 != BUNDLE_PIN or reference_summary_sha256 != A17_PIN):
        raise ValueError("real qualification must use the declared workload/settings/pins")
    bundle_data = load_effusion_bundle(bundle, expected_report_sha256=bundle_sha256)
    _validate_bundle_classification(bundle_data, bundle_sha256, synthetic)
    summary = _read(reference_dir / "summary.json", reference_summary_sha256)
    cal = calibration(settings["epsilon"], rounds=settings["rounds"], delta=settings["delta"])
    cells = [{"name": f"{backend}_{'gdp4' if private else 'clean'}", "backend": backend, "private": private}
             for private in (False, True) for backend in ("native", "byzfl")]
    configurations = {cell["name"]: resolve(_configuration(cell, settings, bundle, bundle_sha256, cal))
                      for cell in cells}
    first = configurations[cells[0]["name"]]
    workload = ZeroBundleWorkload(first)
    initial = trainable_state(workload.build_model(settings["seed"]))
    if any(torch.count_nonzero(value) for value in initial.values()) or any(not len(p) for p in workload.parts):
        raise ValueError("zero initialization or nonempty cohort qualification failed")
    shapes = {key: list(value.shape) for key, value in initial.items()}
    refs = {}
    for private, label in ((False, "clean"), (True, f"eps{settings['epsilon']:g}")):
        name = f"zero_T{settings['rounds']}_{label}_{settings['seed']}"
        matching = [r for r in summary["runs"] if r["name"] == name]
        if len(matching) != 1 or matching[0].get("finite") is not True or matching[0].get("refusals") != 0:
            raise ValueError("missing or unqualified prior reference")
        row = matching[0]
        if row["seed"] != settings["seed"] or row["rounds"] != settings["rounds"] \
                or row["epsilon"] != (settings["epsilon"] if private else None):
            raise ValueError("reference configuration does not match")
        path = reference_dir / f"{name}.json"
        record = json.loads(path.read_text())
        _reference_heads(record, shapes)
        expected_z = cal["gdpNoiseMultiplier"] if private else 0.
        if abs(float(record["meta"]["noise_multiplier"]) - expected_z) > max(1e-12, expected_z * 1e-6):
            raise ValueError("prior reference noise multiplier differs")
        refs["private" if private else "clean"] = {"name": name, "record_sha256": sha(path),
                                                    "auroc": float(row["auroc"])}
    runtime = _runtime()
    if runtime["byzfl"] != "0.0.11":
        raise ValueError("qualification requires ByzFL0.0.11")
    sources = _source_paths()
    hashes = {name: sha(path) for name, path in sources.items()}
    public = {"schema": "capstone.effusion_byzfl_qualification.v1", "synthetic": bool(synthetic),
              "settings": settings, "cells": cells, "head_init": "zero", "calibration": cal,
              "configuration_sha256": {name: _configuration_hash(cfg) for name, cfg in configurations.items()},
              "bundle_report_sha256": bundle_sha256, "reference_summary_sha256": reference_summary_sha256,
              "references": refs, "parameter_shapes": shapes, "source_sha256": hashes, "runtime": runtime,
              "scope": "train and development selection only; implementation reconciliation, not new method ranking"}
    output.mkdir(parents=True, exist_ok=False)
    write_research_record(output / "public-manifest.json", public)
    plan = {**public, "bundle_path": str(bundle), "reference_dir": str(reference_dir),
            "resolved_configurations": configurations,
            "public_manifest_sha256": sha(output / "public-manifest.json")}
    write_research_record(output / "plan.json", plan)
    for name, source in sources.items():
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return plan


def _checkpoint(plan, commit, manifest_path):
    if plan["synthetic"]:
        return
    if commit is None or not re.fullmatch(r"[a-f0-9]{7,40}", commit) or manifest_path is None:
        raise ValueError("real execution requires the explicitly authorized local checkpoint commit and public manifest")
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_relative_to(ROOT) or sha(manifest_path) != plan["public_manifest_sha256"]:
        raise ValueError("checkpoint manifest differs from the frozen plan")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, capture_output=True)
    if ancestor.returncode != 0:
        raise ValueError("checkpoint is not in the current branch history")
    wanted = {**plan["source_sha256"], manifest_path.relative_to(ROOT).as_posix(): plan["public_manifest_sha256"]}
    for name, expected in wanted.items():
        if name.startswith("byzfl/"):
            continue
        blob = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=ROOT, capture_output=True)
        if blob.returncode != 0 or hashlib.sha256(blob.stdout).hexdigest() != expected:
            raise ValueError(f"source/protocol is not frozen in the checkpoint: {name}")


def execute(plan_dir, expected_plan_sha256, output, *, checkpoint_commit=None, checkpoint_manifest=None):
    plan_dir, output = Path(plan_dir), Path(output)
    if output.exists():
        raise FileExistsError("qualification output already exists")
    plan = _read(plan_dir / "plan.json", expected_plan_sha256)
    public = _read(plan_dir / "public-manifest.json", plan["public_manifest_sha256"])
    if {k: plan[k] for k in public} != public:
        raise ValueError("private plan differs from the prospective public manifest")
    current = {name: sha(path) for name, path in _source_paths().items()}
    if current != plan["source_sha256"] or _runtime() != plan["runtime"]:
        raise ValueError("source/dependency/runtime changed after preflight")
    snapshot = plan_dir / "sources"
    snapshot_files = {p.relative_to(snapshot).as_posix(): p for p in snapshot.rglob("*") if p.is_file()}
    if any(p.is_symlink() for p in snapshot.rglob("*")) or {
        name: sha(path) for name, path in snapshot_files.items()
    } != plan["source_sha256"]:
        raise ValueError("source snapshot differs from the frozen inventory/hashes")
    bundle, refs_dir = Path(plan["bundle_path"]), Path(plan["reference_dir"])
    bundle_data = load_effusion_bundle(bundle, expected_report_sha256=plan["bundle_report_sha256"])
    _validate_bundle_classification(bundle_data, plan["bundle_report_sha256"], plan["synthetic"])
    if not plan["synthetic"] and (plan["settings"] != DEFAULTS or plan["bundle_report_sha256"] != BUNDLE_PIN
                                  or plan["reference_summary_sha256"] != A17_PIN):
        raise ValueError("real qualification must use the declared workload/settings/pins")
    _checkpoint(plan, checkpoint_commit, checkpoint_manifest)
    _read(refs_dir / "summary.json", plan["reference_summary_sha256"])
    references = {key: _read(refs_dir / f"{ref['name']}.json", ref["record_sha256"])
                  for key, ref in plan["references"].items()}
    settings, cal = plan["settings"], plan["calibration"]
    configurations = {cell["name"]: resolve(_configuration(cell, settings, bundle, plan["bundle_report_sha256"], cal))
                      for cell in plan["cells"]}
    if configurations != plan["resolved_configurations"] or {
        name: _configuration_hash(cfg) for name, cfg in configurations.items()
    } != plan["configuration_sha256"]:
        raise ValueError("resolved configurations differ from the frozen plan")
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    shutil.copyfile(plan_dir / "plan.json", output / "frozen-plan.json")
    shutil.copyfile(plan_dir / "public-manifest.json", output / "public-manifest.json")
    shutil.copytree(plan_dir / "sources", output / "sources")
    outcomes, heads, metrics = [], {}, {}
    for cell in plan["cells"]:
        final, observed, record = {}, [], None
        cfg = configurations[cell["name"]]
        journal = output / "rounds" / cell["name"]
        journal.mkdir(parents=True, exist_ok=False)
        def save_round(line):
            write_research_record(journal / f"round-{line['round']:04d}.json", line)
            observed.append(line)
        try:
            workload = ZeroBundleWorkload(resolve(cfg))
            initial = trainable_state(workload.build_model(settings["seed"]))
            if any(torch.count_nonzero(value) for value in initial.values()):
                raise ValueError("head initialization is not zero")
            record = run(cfg, workload=workload, on_round=save_round, on_final_parameters=final.update)
            if len(record["rounds"]) != settings["rounds"] or record["meta"]["refusals"] \
                    or not final or not all(torch.isfinite(v).all() for v in final.values()):
                raise ValueError("incomplete, refused or nonfinite qualification cell")
            if any(line["participants"]["aggregated"] != settings["clients"] for line in record["rounds"]):
                raise ValueError("qualification cohort changed")
            exact = check_calibration(cal, record["meta"]["noise_multiplier"], settings["rounds"], settings["delta"]) if cell["private"] else None
            record["meta"].update(head_init="zero", finalHead={k: v.tolist() for k, v in final.items()},
                                   qualification_accounting={"accountant": "gdp" if cell["private"] else None,
                                   "claimed_epsilon": settings["epsilon"] if cell["private"] else None,
                                   "exact_epsilon": exact,
                                   "engine_rdp_equivalent_epsilon": cal["rdpEquivalentEpsilon"] if cell["private"] else None})
            status, auc = "completed", float(record["rounds"][-1]["auc"])
            if not math.isfinite(auc):
                raise ValueError("nonfinite qualification metric")
            heads[cell["name"]], metrics[cell["name"]] = final, auc
        except Exception as exc:
            status, auc = "error", None
            if record is None:
                record = {"config": cfg, "rounds": observed}
            record["qualification_failure"] = {"status": status, "error_type": type(exc).__name__, "error": str(exc)}
        if final:
            record.setdefault("meta", {})["finalHead"] = {key: value.tolist() for key, value in final.items()}
        relative = f"runs/{cell['name']}.json"
        write_research_record(output / relative, record)
        outcomes.append({**cell, "status": status, "record": relative, "sha256": sha(output / relative), "auroc": auc})
        write_research_record(output / f"cell-receipt-{cell['name']}.json", outcomes[-1])
    comparisons = []
    def compare(label, left, right, left_auc, right_auc):
        if left is None or right is None:
            comparisons.append({"comparison": label, "status": "unavailable", "max_parameter_difference": None})
            return
        same_keys = set(left) == set(right)
        difference = max(float((left[k] - right[k]).abs().max()) for k in left) if same_keys else math.inf
        metric_difference = abs(left_auc - right_auc)
        comparisons.append({"comparison": label, "status": "matched" if difference <= 1e-6 and metric_difference <= 1e-10 else "mismatch",
                            "max_parameter_difference": difference if math.isfinite(difference) else None,
                            "auroc_difference": metric_difference})
    for private, label in ((False, "clean"), (True, "gdp4")):
        native, external = f"native_{label}", f"byzfl_{label}"
        compare(f"backend-{label}", heads.get(native), heads.get(external), metrics.get(native, 0.), metrics.get(external, 0.))
        key = "private" if private else "clean"
        reference_head = _reference_heads(references[key], plan["parameter_shapes"])
        compare(f"prior-A17-{label}", heads.get(native), reference_head, metrics.get(native, 0.), plan["references"][key]["auroc"])
    qualified = all(x["status"] == "completed" for x in outcomes) and all(x["status"] == "matched" for x in comparisons)
    result = {"schema": "capstone.effusion_byzfl_qualification_result.v1", "status": "qualified" if qualified else "not_qualified",
              "synthetic": plan["synthetic"], "plan_sha256": expected_plan_sha256,
              "checkpoint_commit": checkpoint_commit, "runs": outcomes, "comparisons": comparisons,
              "scope": "fixed train/selection implementation reconciliation; no holdout, new method ranking or deployment privacy claim"}
    write_research_record(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("preflight")
    prepare.add_argument("--bundle", type=Path, required=True)
    prepare.add_argument("--bundle-sha256", default=BUNDLE_PIN)
    prepare.add_argument("--reference-dir", type=Path, required=True)
    prepare.add_argument("--reference-summary-sha256", default=A17_PIN)
    prepare.add_argument("--output", type=Path, required=True)
    launch = sub.add_parser("execute")
    launch.add_argument("--plan-dir", type=Path, required=True)
    launch.add_argument("--plan-sha256", required=True)
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--checkpoint-commit", required=True)
    launch.add_argument("--checkpoint-manifest", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "preflight":
        preflight(args.bundle, args.bundle_sha256, args.reference_dir, args.reference_summary_sha256, args.output)
        print(json.dumps({"status": "preflight_complete", "model_fits": 0,
                          "plan_sha256": sha(args.output / "plan.json"),
                          "public_manifest_sha256": sha(args.output / "public-manifest.json")}))
        return 0
    result = execute(args.plan_dir, args.plan_sha256, args.output,
                     checkpoint_commit=args.checkpoint_commit, checkpoint_manifest=args.checkpoint_manifest)
    print(json.dumps({"status": result["status"], "completed_cells": sum(x["status"] == "completed" for x in result["runs"])}))
    return 0 if result["status"] == "qualified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
