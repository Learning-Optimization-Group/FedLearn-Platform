"""Bounded dashboard recipes and diagnostic record encoding."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

from fedlearn.simulation.config import config_hash
from fedlearn.simulation.core_qualification import prepare_core_config


DEFAULT_CHOICES = {
    "workloadId": "nih-qualified", "head": "linear", "numClients": 10,
    "rounds": 3, "localEpochs": 1, "aggregation": "mean", "clipNorm": .05,
    "privacy": "none", "epsilon": 4, "behavior": "clean", "seed": 42, "recipe": "standard",
}
# Measured presets for the capstone Effusion R0 bundle (protocol amendments A7 clean, A8 DP): (client LR, clip).
RECIPES = {"effusion-clean": (.1, 1.0), "effusion-dp": (.03, .05)}
WORKLOAD_RECIPES = {"nih-qualified": ("standard",), "nih-effusion-r0": ("effusion-clean", "effusion-dp")}
EVALUATION = {"nih-qualified": "validation", "nih-effusion-r0": "development selection half"}
HEADS = {"linear": ("linear", 128, 1026), "mlp8": ("mlp", 8, 4122),
         "mlp128": ("mlp", 128, 65922), "mlp512": ("mlp", 512, 263682)}
ENUMS = {"workloadId": tuple(WORKLOAD_RECIPES), "recipe": ("standard", *RECIPES), "head": tuple(HEADS),
         "aggregation": ("mean", "trimmed_mean"), "privacy": ("none", "central"),
         "behavior": ("clean", "sign_flip", "label_flip")}


def resolve_choices(choices: dict, *, cache_path: Path, report_sha256: str) -> dict:
    """Resolve one server-bound validation recipe without loading/scoring the cache."""
    if type(choices) is not dict or set(choices) - set(DEFAULT_CHOICES):
        raise ValueError("choices must be an object containing only supported settings")
    values = {**DEFAULT_CHOICES, **choices}
    for key, allowed in ENUMS.items():
        if type(values[key]) is not str or values[key] not in allowed:
            raise ValueError(f"unsupported {key}")
    for key, lower, upper in (("rounds", 1, 20), ("localEpochs", 1, 7),
                              ("seed", 0, 1_000_000_000)):
        if type(values[key]) is not int or not lower <= values[key] <= upper:
            raise ValueError(f"{key} must be an integer in [{lower}, {upper}]")
    if type(values["numClients"]) is not int or values["numClients"] not in (10, 80, 1000):
        raise ValueError("numClients must be 10, 80 or 1000")
    if values["recipe"] not in WORKLOAD_RECIPES[values["workloadId"]]:
        raise ValueError("recipe does not apply to this workload")
    lr = .003
    if values["recipe"] in RECIPES:
        lr, clip = RECIPES[values["recipe"]]
        if "clipNorm" in choices and choices["clipNorm"] != clip:
            raise ValueError("clipNorm is fixed by the measured recipe")
        values["clipNorm"] = clip
    for key, allowed in (("clipNorm", (.05, .4) if values["recipe"] == "standard" else (values["clipNorm"],)),
                         ("epsilon", (1, 4, 8))):
        if (type(values[key]) not in (int, float) or not math.isfinite(values[key])
                or values[key] not in allowed):
            raise ValueError(f"unsupported {key}")
    kind, hidden, parameters = HEADS[values["head"]]
    behaviors = []
    if values["behavior"] == "sign_flip":
        behaviors = [{"type": "sign_flip_scale", "fraction": .2, "scale": -4.0}]
    elif values["behavior"] == "label_flip":
        behaviors = [{"type": "label_flip", "fraction": .2}]
    cfg = {
        "name": "simulation-lab", "seed": values["seed"],
        "workload": {"dataset": "nih", "features_path": str(cache_path),
                     "features_report_sha256": report_sha256, "evaluation_split": "validation",
                     "allow_random_features": False, "balance": False, "auc_score": "margin_v2",
                     "head": {"type": kind, "hidden": hidden}},
        "federation": {"num_clients": values["numClients"],
                       "clients_per_round": values["numClients"], "rounds": values["rounds"],
                       "partition": {"type": "patient_iid"}},
        "client": {"local_epochs": values["localEpochs"], "lr": lr, "batch_size": 256},
        "server": {"rule": values["aggregation"], "clip_norm": float(values["clipNorm"]),
                   "rule_params": {"trim_ratio": .2}},
        "privacy": {"mode": values["privacy"], "epsilon": float(values["epsilon"]),
                    "delta": 1e-5, "adjacency": "replace_one"},
        "behaviors": behaviors, "eval": {"per_client_every": 0},
    }
    cfg = prepare_core_config(cfg, feature_report_sha256=report_sha256,
                              clients=values["numClients"], evaluation_split="validation")
    display = {
        **values, "clipNorm": cfg["server"]["clip_norm"], "epsilon": cfg["privacy"]["epsilon"],
        "evaluationSplit": EVALUATION[values["workloadId"]], "learningRate": cfg["client"]["lr"],
        "batchSize": cfg["client"]["batch_size"], "parameters": parameters,
        "featureReportSha256": report_sha256, "delta": cfg["privacy"]["delta"],
        "trimRatio": .2 if cfg["server"]["rule"] == "trimmed_mean" else None,
        "maliciousFraction": .2 if behaviors else 0.0, "researchRecordDpSafe": False,
    }
    return {"config": cfg, "config_hash": config_hash(cfg), "display_recipe": display}


def _tag_nonfinite(value):
    if isinstance(value, dict):
        return {key: _tag_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tag_nonfinite(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        return {"__nonfinite_float__": str(value)}
    return value


def write_research_record(path: Path, record: dict) -> str:
    """Atomically create one finite or tagged failure record; never overwrite evidence."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("record already exists")
    encoding = "json"
    try:
        raw = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    except ValueError:
        encoding = "nonfinite_tagged_v1"
        raw = (json.dumps({"encoding": encoding, "record": _tag_nonfinite(record)},
                          sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as file:
            temporary = Path(file.name)
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        # Same-directory hard link publishes a complete file and refuses existing targets.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return encoding
