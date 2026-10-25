"""Retrospective paired-seed inference from pinned selection-only scalar summaries.

No training, patient manifest, per-image prediction, holdout or test file is read.
Original patient intervals are preserved as conditional on realized trained heads.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
from scipy.stats import t

SEEDS = tuple(range(20261101, 20261106))
INPUTS = {
    "A8": ("dp-recipe-r0-v1", "bd9868ae3300411e44371348aebf97856c4a7b178996ed1d7d32a42d210da42d"),
    "A11": ("local-dp-r0-v1", "6175fcbe5ee94c18896704417d2655e355ef5d5d9f147e38d17d5e9343b86fb5"),
    "A12": ("local-robust-r0-v1", "5de3c331fb35693a057320fa882c28ed75c568d44c3e08fe2701ff0683b1bc17"),
    "A14": ("strong-attacks-r0-v1", "57ff024b4ef9dc461606f9dd003c92e84495c4672c8dac31e7aa9e4f3de9d22f"),
    "A17": ("zero-head-r0-v1", "54d584fcc0dd92f7ba86d4edb530f433e02c08cb965f7e80208a068f8f3fc225"),
    "A19": ("noniid-r0-v1", "6c1e4010dff279db5a974d0fea9e219dd0850d2fb0113cf98dd6cb8f9dd8215e"),
    "A22": ("encoder-dp-r2-v1", "4196c213fb89e51face7a9ef1053ec9fc3fe2d74fe980827498377f887f61820"),
    "A23": ("distributed-dp-r0-v1", "cba952640c2dd8b9f71ecf77315765ab117574daaf5d7efc92bb680f43443465"),
}


def read_verified_summary(path, expected):
    path = Path(path)
    if path.is_symlink() or path.name != "summary.json":
        raise ValueError("only nonsymlink summary.json artifacts are accepted")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"summary hash mismatch: {path.parent.name}")
    return json.loads(raw)


def metric_lookup(runs, prefix=""):
    result = {}
    for row in runs:
        name, seed, value = prefix + row["name"], row["seed"], row.get("auroc")
        if name in result:
            raise ValueError(f"duplicate metric cell: {name}")
        if type(seed) is not int or value is None or not math.isfinite(float(value)) \
                or not 0 <= float(value) <= 1 or row.get("finite") is not True or row.get("refusals") != 0:
            raise ValueError(f"unqualified scalar metric: {name}")
        result[name] = {"seed": seed, "auroc": float(value)}
    return result


def seed_interval(values):
    if len(values) < 2 or not all(math.isfinite(float(x)) for x in values.values()):
        raise ValueError("at least two finite seed replicates are required")
    ordered = sorted(values)
    array = np.array([values[s] for s in ordered], dtype=np.float64)
    point, sd = float(array.mean()), float(array.std(ddof=1))
    halfwidth = float(t.ppf(.975, len(array) - 1) * sd / math.sqrt(len(array)))
    return {"point": point, "interval95": [point - halfwidth, point + halfwidth],
            "n": len(array), "sd": sd, "per_seed": {str(s): float(values[s]) for s in ordered},
            "method": "paired seed mean t interval; retrospective, unadjusted, fixed evaluation cohort"}


def analyse_contrasts(summary, specs, lookup, *, required_seeds=SEEDS):
    original = summary["contrasts"]
    if set(original) != set(specs):
        raise ValueError(f"contrast coverage mismatch: missing={sorted(set(original)-set(specs))}, "
                         f"unexpected={sorted(set(specs)-set(original))}")
    output = {}
    for name, terms in specs.items():
        values = {}
        for term in terms:
            if not term or any(key not in lookup for key in term):
                raise ValueError(f"missing component in {name}")
            seeds = {lookup[key]["seed"] for key in term}
            if len(seeds) != 1:
                raise ValueError(f"cross-seed components in {name}")
            seed = seeds.pop()
            if seed in values:
                raise ValueError(f"duplicate seed in {name}")
            values[seed] = sum(float(weight) * lookup[key]["auroc"] for key, weight in term.items())
        if set(values) != set(required_seeds):
            raise ValueError(f"incomplete seed coverage in {name}")
        estimate = seed_interval(values)
        if abs(estimate["point"] - float(original[name]["point"])) > 1e-10:
            raise ValueError(f"original point does not reconcile: {name}")
        patient = original[name]["interval95"]
        if len(patient) != 2 or not all(math.isfinite(float(x)) for x in patient) or patient[0] > patient[1]:
            raise ValueError(f"invalid conditional patient interval: {name}")
        output[name] = {"seed": estimate, "conditional_patient_interval95": patient,
                        "original_point_reproduced": True,
                        "conditional_interval_excludes_zero": patient[0] > 0 or patient[1] < 0,
                        "seed_interval_excludes_zero": estimate["interval95"][0] > 0 or estimate["interval95"][1] < 0}
    return output


def run_reanalysis(evidence_root, output):
    from selection_contrast_specs import build_specs, incremental_dp_cost_specs
    evidence_root, output = Path(evidence_root), Path(output)
    if output.exists():
        raise FileExistsError("reanalysis output already exists")
    summaries = {}
    for study, (directory, expected) in INPUTS.items():
        path = evidence_root / directory / "summary.json"
        if not path.resolve().is_relative_to(evidence_root.resolve()):
            raise ValueError("evidence path escapes the declared root")
        summaries[study] = read_verified_summary(path, expected)
    aliases = {
        "A11": metric_lookup(summaries["A8"]["confirmation"], "a8_"),
        "A12": {}, "A19": metric_lookup(summaries["A14"]["runs"], "iid_"),
        "A22": metric_lookup(summaries["A17"]["runs"], "a17_"),
        "A23": metric_lookup(summaries["A17"]["runs"], "a17_"),
    }
    results, lookups = {}, {}
    for study, aliases_for_study in aliases.items():
        lookups[study] = {**metric_lookup(summaries[study]["runs"]), **aliases_for_study}
        results[study] = analyse_contrasts(summaries[study], build_specs(study, summaries[study]), lookups[study])
    additional = {}
    for name, terms in incremental_dp_cost_specs(summaries["A19"]).items():
        values = {}
        for term in terms:
            seed_ids = {lookups["A19"][key]["seed"] for key in term}
            if len(seed_ids) != 1:
                raise ValueError("cross-seed incremental cost")
            seed = seed_ids.pop()
            if seed in values:
                raise ValueError("duplicate incremental-cost seed")
            values[seed] = sum(weight * lookups["A19"][key]["auroc"] for key, weight in term.items())
        if set(values) != set(SEEDS):
            raise ValueError("incomplete incremental-cost seed coverage")
        additional[name] = seed_interval(values)
    output.mkdir(parents=True, exist_ok=False)
    artifact = {
        "schema": "capstone.selection_seed_reanalysis.v1", "analysis_date": "2026-10-24",
        "scope": "selection-only retrospective inference; no new model fits",
        "inputs": {key: {"directory": value[0], "summary_sha256": value[1]} for key, value in INPUTS.items()},
        "studies": results, "additional_A19_incremental_cost": additional,
        "limitations": ["Five already-used development seeds; not independent confirmation",
                        "Unadjusted approximate t intervals; normality is not verified",
                        "Patient and seed intervals are separate, not a joint uncertainty interval",
                        "No equivalence test; selection/tuning optimism is not removed",
                        "No private release guarantee for research outputs or oracle attacks"],
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("reanalyze_selection_seeds.py", "selection_contrast_specs.py")},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (output / "summary.json").write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with (output / "contrasts.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["study", "contrast", "point", "n", "seed_low", "seed_high", "patient_low", "patient_high"])
        for study, contrasts in results.items():
            for name, result in contrasts.items():
                seed = result["seed"]
                writer.writerow([study, name, seed["point"], seed["n"], *seed["interval95"],
                                 *result["conditional_patient_interval95"]])
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_reanalysis(args.evidence_root, args.output)
    print(json.dumps({"status": "complete", "contrasts": {k: len(v) for k, v in result["studies"].items()}}))
