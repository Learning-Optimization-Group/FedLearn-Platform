"""Execute the fixed validation-only NIH exploratory pilot and retain failures."""
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

from fedlearn.simulation.config import resolve
from fedlearn.simulation.nih_cache import json_bytes, load_nih_cache


LEARNING_RATES = (0.001, 0.01, 0.1)
BASE_SEED = 20261019
PER_CELL_SECONDS = 300
TOTAL_SECONDS = 1800
METRICS = ("loss", "auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "ece")


def _config(name, cache, pin, clients, lr, *, seed=BASE_SEED, rounds=10, clip=None, rule="mean"):
    return resolve({"name": name, "seed": seed,
        "workload": {"dataset": "nih", "features_path": str(cache), "features_report_sha256": pin,
                     "evaluation_split": "validation", "balance": False, "allow_random_features": False,
                     "auc_score": "probability_v1",
                     "head": {"type": "linear"}},
        "federation": {"num_clients": clients, "rounds": rounds, "partition": {"type": "patient_iid"}},
        "client": {"local_epochs": 3, "lr": lr, "batch_size": 256},
        "server": {"rule": rule, "clip_norm": clip, "rule_params": {"trim_ratio": 0.2}},
        "privacy": {"mode": "none"}, "behaviors": [], "eval": {"per_client_every": 0}})


def initial_configs(cache, pin):
    return [_config(f"{group}_lr_{lr:g}", cache, pin, clients, lr)
            for group, clients in (("pooled", 1), ("federated", 10)) for lr in LEARNING_RATES]


def followup_configs(cache, pin, selected_lr):
    if selected_lr not in LEARNING_RATES:
        raise ValueError("selected LR is outside the fixed pilot budget")
    return [*[_config(f"federated_repeat_{seed}", cache, pin, 10, selected_lr, seed=seed)
              for seed in (20261020, 20261021)],
            _config("federated_clip_mean", cache, pin, 10, selected_lr, clip=0.4),
            _config("federated_clip_trimmed", cache, pin, 10, selected_lr, clip=0.4, rule="trimmed_mean"),
            _config("scale_1000_two_rounds", cache, pin, 1000, selected_lr, rounds=2)]


def summarize_record(record):
    """Qualify a completed validation record; reconstruct, rather than re-predict, CM."""
    try:
        meta, lines = record["meta"], record["rounds"]
        config, workload = meta["config"], meta["workload"]
        wl = config["workload"]
        if (wl["dataset"] != "nih" or workload["dataset"] != "nih" or wl["evaluation_split"] != "validation"
                or workload["evaluation_split"] != "validation" or wl["balance"] is not False
                or wl["allow_random_features"] is not False or workload["pretrained"] is not True
                or workload["features_report_sha256"] != wl["features_report_sha256"]
                or config["privacy"]["mode"] != "none" or config["behaviors"]
                or workload["parameters"] != 1026 or meta["stopped_early"] or meta["refusals"]
                or [line["round"] for line in lines] != list(range(1, config["federation"]["rounds"] + 1))):
            raise ValueError("pilot identity, model, evaluation or completion contract failed")
        counts = workload["split_counts"]["validation"]
        positive, negative, n = counts["positive_images"], counts["negative_images"], counts["images"]
        if (any(type(value) is not int or value <= 0 for value in (positive, negative, n))
                or positive + negative != n or n != workload["n_test"] or n >= 2**24):
            raise ValueError("invalid validation class counts for float32 confusion reconstruction")
        confusion = None
        for line in lines:
            if line["refusal"] is not None or any(not isinstance(line[key], (int, float))
                    or isinstance(line[key], bool) or not math.isfinite(line[key]) for key in METRICS):
                raise ValueError("refused or nonfinite pilot round")
            if line["loss"] < 0 or any(not 0 <= line[key] <= 1 for key in METRICS if key != "loss"):
                raise ValueError("invalid metric range")
            tp, tn = round(line["sensitivity"] * positive), round(line["specificity"] * negative)
            if (not math.isclose(line["sensitivity"], tp / positive, abs_tol=1e-6, rel_tol=0)
                    or not math.isclose(line["specificity"], tn / negative, abs_tol=1e-6, rel_tol=0)
                    or not math.isclose(line["accuracy"], (tp + tn) / n, abs_tol=1e-6, rel_tol=0)
                    or not math.isclose(line["balanced_accuracy"], (tp / positive + tn / negative) / 2,
                                        abs_tol=1e-6, rel_tol=0)):
                raise ValueError("confusion fractions/accuracy are inconsistent")
            confusion = {"tp": tp, "fn": positive - tp, "tn": tn, "fp": negative - tn}
        return {"name": config["name"], "seed": config["seed"], "lr": config["client"]["lr"],
                "clients": config["federation"]["num_clients"], "rounds": len(lines),
                "final": {key: lines[-1][key] for key in METRICS}, "confusion": confusion,
                "confusion_scope": "reconstructed from recorded fractions and known class counts",
                "majority_baseline": {"accuracy": max(positive, negative) / n, "balanced_accuracy": 0.5},
                "first_round_loss": lines[0]["loss"], "first_round_auc": lines[0]["auc"],
                "engine_wall_seconds": meta["wall_seconds"], "peak_rss_bytes": meta["peak_rss_bytes"],
                "peak_rss_scope": meta["peak_rss_scope"], "workload": workload}
    except (KeyError, IndexError, TypeError, OverflowError) as error:
        raise ValueError(f"incomplete or malformed pilot record: {error}") from error


def select_learning_rate(records):
    if len(records) != 6:
        raise ValueError("all six initial cells must qualify; do not select survivors")
    summaries = [summarize_record(record) for record in records]
    config = records[0]["meta"]["config"]
    expected = {c["name"]: c for c in initial_configs(config["workload"]["features_path"],
                                                    config["workload"]["features_report_sha256"])}
    actual = {record["meta"]["config"]["name"]: record["meta"]["config"] for record in records}
    if len(actual) != 6 or actual != expected:
        raise ValueError("initial cells differ from the prospectively fixed configs")
    federated = [s for s in summaries if s["clients"] == 10]
    return min(federated, key=lambda s: (-s["final"]["auc"], s["lr"]))["lr"]


def _checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        file.write(json_bytes(value))
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def run_pilot(cache: Path, expected_report_sha256: str, output: Path):
    started = time.monotonic()
    cache, output = Path(cache), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing existing pilot output")
    if output.resolve().is_relative_to(cache.resolve()):
        raise ValueError("pilot output must be outside the immutable cache")
    loaded = load_nih_cache(cache, expected_report_sha256=expected_report_sha256)
    if not loaded["provenance"]["pretrained"]:
        raise ValueError("real pilot requires pretrained features")
    cache, output = cache.resolve(), output.resolve()
    output.mkdir(parents=True)
    (output / "configs").mkdir()
    (output / "logs").mkdir()
    document = Path(__file__).resolve().parents[2] / "docs/artifacts/capstone/nih-validation-pilot-protocol.md"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=document.parents[3], text=True).strip()
    initial = initial_configs(cache, expected_report_sha256)
    protocol = {"pilot_version": "nih-validation-pilot-v1", "protocol_document_sha256": hashlib.sha256(document.read_bytes()).hexdigest(),
                "git_commit": commit, "initial_configs": initial,
                "conditional_followup_recipes": followup_configs("CACHE", expected_report_sha256, LEARNING_RATES[0]),
                "conditional_recipe_note": "LR and cache path in these recipes are placeholders; substitute only the selected fixed-candidate LR and verified cache",
                "selection": "highest final ten-client validation AUC; exact tie lower LR; all six initial cells must qualify",
                "total_seconds_cap": TOTAL_SECONDS, "per_cell_seconds_cap": PER_CELL_SECONDS,
                "torch_cpu_threads": 4, "official_test_evaluated": False}
    with (output / "protocol.json").open("xb") as file:
        file.write(json_bytes(protocol))
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    cases, records = [], []

    def execute(config):
        name = config["name"]
        config_path = output / "configs" / f"{name}.json"
        with config_path.open("xb") as file:
            file.write(json_bytes(config))
        remaining = TOTAL_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            return {"name": name, "status": "not_run", "reason": "global_budget_exhausted"}, None
        cell_start = time.monotonic()
        run_dir = output / "runs" / name
        try:
            with (output / "logs" / f"{name}.log").open("xb") as log:
                subprocess.run([sys.executable, "-m", "fedlearn.simulation.engine", str(config_path),
                                "--out", str(run_dir)], env=environment, stdout=log, stderr=subprocess.STDOUT,
                               cwd=document.parents[3], timeout=min(PER_CELL_SECONDS, remaining), check=True)
            entries = json.loads((run_dir / "index.json").read_text())["runs"]
            if len(entries) != 1 or entries[0]["status"] != "ok":
                raise ValueError(f"child cell failed: {entries}")
            filename = entries[0]["file"]
            if Path(filename).name != filename:
                raise ValueError("invalid child record filename")
            record_path = run_dir / filename
            raw = record_path.read_bytes()
            record = json.loads(raw)
            if record["meta"]["config"] != config or record["meta"]["provenance"]["git_commit"] != commit:
                raise ValueError("child config/code identity mismatch")
            summary = summarize_record(record)
            return {"name": name, "status": "ok", "elapsed_seconds": time.monotonic() - cell_start,
                    "record_file": record_path.relative_to(output).as_posix(),
                    "record_sha256": hashlib.sha256(raw).hexdigest(), "summary": summary}, record
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            return {"name": name, "status": "timeout" if isinstance(error, subprocess.TimeoutExpired) else "failed",
                    "elapsed_seconds": time.monotonic() - cell_start, "error": str(error)}, None

    for config in initial:
        case, record = execute(config)
        cases.append(case)
        if record is not None:
            records.append(record)
        _checkpoint(output / "index.json", {"cases": cases})
        print(json.dumps({"cell": case["name"], "status": case["status"]}), flush=True)
    selected = select_learning_rate(records) if len(records) == 6 else None
    follow = followup_configs(cache, expected_report_sha256, selected or LEARNING_RATES[0])
    for config in follow:
        case, _ = (execute(config) if selected is not None else
                   ({"name": config["name"], "status": "not_run", "reason": "initial_qualification_failed"}, None))
        cases.append(case)
        _checkpoint(output / "index.json", {"cases": cases})
        print(json.dumps({"cell": case["name"], "status": case["status"]}), flush=True)
    complete = all(case["status"] == "ok" for case in cases)
    best_auc = max((s["rounds"][-1]["auc"] for s in records if s["meta"]["config"]["federation"]["num_clients"] == 10), default=None)
    summary = {"pilot_version": protocol["pilot_version"], "git_commit": commit,
               "protocol_document_sha256": protocol["protocol_document_sha256"],
               "features_report_sha256": expected_report_sha256, "selected_lr": selected,
               "complete": complete, "elapsed_seconds": time.monotonic() - started, "cases": cases,
               "readiness": "incomplete_pilot" if not complete else "weak_baseline" if best_auc < 0.60 else "ready_to_draft_confirmatory_protocol",
               "official_test_evaluated": False, "record_scope": "public-dataset exploratory research; not a protected deployment export"}
    with (output / "summary.json").open("xb") as file:
        file.write(json_bytes(summary))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_pilot(args.cache, args.expected_report_sha256, args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, sort_keys=True))
    if not result["complete"]:
        parser.exit(2, "pilot incomplete; inspect retained failed/unrun cells\n")


if __name__ == "__main__":
    main()
