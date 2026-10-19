"""Run configuration for the simulation engine: defaults, validation, seeds, matrix expansion.

A config is a plain JSON object. ``resolve`` fills defaults, rejects unknown keys and invalid
combinations (naming the offending field), and returns the fully resolved config that every run
record stores. ``expand_matrix`` turns a ``matrix`` block into one resolved config per cell and
repeat; all cells of one repeat share the same seeds (the capstone harness's scheme).
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from typing import Any, Dict, List, Tuple

SCHEMA_VERSION = 1
RULES = ("mean", "weighted_mean", "trimmed_mean", "median", "krum", "multi_krum", "bulyan",
         "centered_clip")
CENTRAL_DP_RULES = ("mean", "trimmed_mean")
MALICIOUS = ("sign_flip_scale", "ipm", "alie", "label_flip", "backdoor", "free_rider")
FAULTY = ("dropout", "straggler", "label_noise")
SEED_KEYS = ("partition", "init", "client", "attack", "noise")

BEHAVIOR_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "sign_flip_scale": {"scale": -10.0},
    "ipm": {"scale": 4.0},
    "alie": {"z": 20.0},
    "label_flip": {},
    "backdoor": {"target_label": None, "poison_fraction": 0.5, "trigger": None},
    "free_rider": {"mode": "zero", "noise_std": 0.0},
    "dropout": {},
    "straggler": {"slowdown": 4.0},
    "label_noise": {"flip_prob": 0.3},
}

DEFAULTS: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "name": "run",
    "seed": 0,
    "seeds": None,
    "workload": {"dataset": "kermany", "features_path": None, "balance": True,
                 "head": {"type": "linear", "hidden": 128}},
    "federation": {"num_clients": 10, "clients_per_round": None, "rounds": 8,
                   "partition": {"type": "iid", "alpha": 0.5},
                   "network": {"uplink_mbps": 10.0, "downlink_mbps": 50.0, "latency_ms": 50.0,
                               "device_gflops": 10.0, "round_deadline_s": None}},
    "client": {"algorithm": "fedavg", "proximal_mu": 0.0, "local_epochs": 5, "lr": 0.5,
               "batch_size": None},
    "server": {"clip_norm": 0.4, "rule": "mean",
               "rule_params": {"trim_ratio": 0.2, "byzantine_fraction": None, "tau": 1.0,
                               "iterations": 1, "multi_krum_m": None},
               "optimizer": {"type": "none", "lr": 0.01}},
    "privacy": {"mode": "none", "epsilon": 1.0, "delta": 1e-5, "adjacency": "replace_one"},
    "behaviors": [],
    "eval": {"per_client_every": 0},
    "matrix": None,
}


class ConfigError(ValueError):
    """An invalid config; ``field`` is the dotted path of the offending key."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field


def default_seeds(base: int, rep: int = 0) -> Dict[str, int]:
    """The capstone harness's seed scheme: partition = init = client = base + rep,
    attack = base + 1000 + rep, noise = base + 2000 + rep."""
    return {"partition": base + rep, "init": base + rep, "client": base + rep,
            "attack": base + 1000 + rep, "noise": base + 2000 + rep}


def _merge(defaults: Dict[str, Any], given: Any, path: str) -> Dict[str, Any]:
    if not isinstance(given, dict):
        raise ConfigError(path or "<root>", "expected an object")
    out = copy.deepcopy(defaults)
    for key, value in given.items():
        where = f"{path}.{key}" if path else key
        if key not in defaults:
            raise ConfigError(where, "unknown key")
        if isinstance(defaults[key], dict) and value is not None:
            out[key] = _merge(defaults[key], value, where)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_behavior(b: Any, i: int) -> Dict[str, Any]:
    where = f"behaviors[{i}]"
    if not isinstance(b, dict):
        raise ConfigError(where, "expected an object")
    kind = b.get("type")
    if kind not in BEHAVIOR_DEFAULTS:
        raise ConfigError(f"{where}.type", f"must be one of {sorted(BEHAVIOR_DEFAULTS)}")
    out = {"type": kind, "fraction": b.get("fraction"), **copy.deepcopy(BEHAVIOR_DEFAULTS[kind])}
    for key, value in b.items():
        if key not in out:
            raise ConfigError(f"{where}.{key}", "unknown key")
        out[key] = value
    f = out["fraction"]
    if not isinstance(f, (int, float)) or not 0.0 <= float(f) <= 1.0:
        raise ConfigError(f"{where}.fraction", "must be a number in [0, 1]")
    if kind == "backdoor" and not isinstance(out["target_label"], int):
        raise ConfigError(f"{where}.target_label", "backdoor needs an integer target_label")
    if kind == "backdoor" and out["trigger"] is not None and (
            not isinstance(out["trigger"], dict) or out["trigger"].get("type", "patch") != "patch"):
        raise ConfigError(f"{where}.trigger", "only patch triggers are supported (prepare_features --trigger-patch)")
    if kind == "free_rider" and out["mode"] not in ("zero", "noise"):
        raise ConfigError(f"{where}.mode", "must be 'zero' or 'noise'")
    for key, ok, rule in _BEHAVIOR_RANGES.get(kind, ()):
        v = out[key]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not ok(float(v)):
            raise ConfigError(f"{where}.{key}", f"must be {rule}")
    return out


# Per-behaviour parameter ranges: (key, predicate, description). A value outside its range would
# otherwise fail mid-run or be silently meaningless (a straggler faster than everyone else).
_BEHAVIOR_RANGES = {
    "sign_flip_scale": (("scale", lambda v: v != 0.0, "a non-zero number (negative flips)"),),
    "ipm": (("scale", lambda v: v != 0.0, "a non-zero number"),),
    "alie": (("z", lambda v: v >= 0.0, ">= 0"),),
    "backdoor": (("poison_fraction", lambda v: 0.0 < v <= 1.0, "in (0, 1]"),),
    "free_rider": (("noise_std", lambda v: v >= 0.0, ">= 0"),),
    "straggler": (("slowdown", lambda v: v >= 1.0, ">= 1 (a straggler is slower than its peers)"),),
    "label_noise": (("flip_prob", lambda v: 0.0 <= v <= 1.0, "in [0, 1]"),),
}


def _require(cond: bool, field: str, message: str) -> None:
    if not cond:
        raise ConfigError(field, message)


def _validate(r: Dict[str, Any]) -> None:
    _require(r["schema_version"] == SCHEMA_VERSION, "schema_version", f"must be {SCHEMA_VERSION}")
    wl, fed, cl, sv, pv = r["workload"], r["federation"], r["client"], r["server"], r["privacy"]
    nih = wl["dataset"] == "nih"
    if nih:
        _require(wl["auc_score"] in ("probability_v1", "margin_v2"), "workload.auc_score",
                 "must be 'probability_v1' or 'margin_v2'")
        _require(wl["evaluation_split"] in ("validation", "test"), "workload.evaluation_split",
                 "NIH requires an explicit validation or test evaluation split")
        _require(isinstance(wl["features_report_sha256"], str)
                 and re.fullmatch(r"[0-9a-f]{64}", wl["features_report_sha256"]) is not None,
                 "workload.features_report_sha256", "NIH requires a trusted feature report SHA-256")
        _require(wl["balance"] is False, "workload.balance", "NIH requires explicit false; preserve prevalence")
        _require(type(wl["allow_random_features"]) is bool, "workload.allow_random_features", "must be boolean")
        _require(type(fed["num_clients"]) is int and fed["num_clients"] >= 1,
                 "federation.num_clients", "NIH client count must be a positive integer")
        _require(not any(b["type"] == "backdoor" for b in r["behaviors"]), "behaviors",
                 "NIH backdoor runs require separately prepared aligned triggered features")
    _require(isinstance(wl["features_path"], str) and wl["features_path"], "workload.features_path",
             "a prepared features file is required (benchmarks/prepare_features.py)")
    _require(wl["head"]["type"] in ("linear", "mlp"), "workload.head.type", "must be 'linear' or 'mlp'")
    _require(int(wl["head"]["hidden"]) >= 1, "workload.head.hidden", "must be >= 1")
    _require(int(fed["num_clients"]) >= 1, "federation.num_clients", "must be >= 1")
    _require(1 <= int(fed["clients_per_round"]) <= int(fed["num_clients"]),
             "federation.clients_per_round", "must be in [1, num_clients]")
    _require(int(fed["rounds"]) >= 1, "federation.rounds", "must be >= 1")
    _require(fed["partition"]["type"] in (("patient_iid",) if nih else ("iid", "dirichlet")),
             "federation.partition.type", "NIH requires 'patient_iid'" if nih else "must be 'iid' or 'dirichlet'")
    _require(float(fed["partition"]["alpha"]) > 0, "federation.partition.alpha", "must be > 0")
    net = fed["network"]
    for key in ("uplink_mbps", "downlink_mbps", "device_gflops"):
        _require(float(net[key]) > 0, f"federation.network.{key}", "must be > 0")
    _require(float(net["latency_ms"]) >= 0, "federation.network.latency_ms", "must be >= 0")
    _require(net["round_deadline_s"] is None or float(net["round_deadline_s"]) > 0,
             "federation.network.round_deadline_s", "must be null or > 0")
    _require(cl["algorithm"] in ("fedavg", "fedprox"), "client.algorithm", "must be 'fedavg' or 'fedprox'")
    if cl["algorithm"] == "fedprox":
        _require(float(cl["proximal_mu"]) > 0, "client.proximal_mu", "fedprox needs proximal_mu > 0")
    _require(int(cl["local_epochs"]) >= 1, "client.local_epochs", "must be >= 1")
    _require(float(cl["lr"]) > 0, "client.lr", "must be > 0")
    _require(cl["batch_size"] is None or int(cl["batch_size"]) >= 1, "client.batch_size",
             "must be null (full batch) or >= 1")
    _require(sv["rule"] in RULES, "server.rule", f"must be one of {list(RULES)}")
    _require(sv["clip_norm"] is None or float(sv["clip_norm"]) > 0, "server.clip_norm",
             "must be null or > 0")
    _require(0.0 <= float(sv["rule_params"]["trim_ratio"]) < 0.5, "server.rule_params.trim_ratio",
             "must be in [0, 0.5)")
    rp = sv["rule_params"]
    _require(rp["byzantine_fraction"] is None or 0.0 <= float(rp["byzantine_fraction"]) < 1.0,
             "server.rule_params.byzantine_fraction", "must be null or in [0, 1)")
    _require(float(rp["tau"]) > 0, "server.rule_params.tau", "must be > 0")
    _require(int(rp["iterations"]) >= 1, "server.rule_params.iterations", "must be >= 1")
    _require(rp["multi_krum_m"] is None or int(rp["multi_krum_m"]) >= 1, "server.rule_params.multi_krum_m",
             "must be null or >= 1")
    _require(sv["optimizer"]["type"] in ("none", "adam", "yogi"), "server.optimizer.type",
             "must be 'none', 'adam' or 'yogi'")
    _require(float(sv["optimizer"]["lr"]) > 0, "server.optimizer.lr", "must be > 0")
    _require(pv["mode"] in ("none", "central", "local"), "privacy.mode", "must be 'none', 'central' or 'local'")
    _require(pv["adjacency"] == "replace_one", "privacy.adjacency", "only 'replace_one' is supported")
    if pv["mode"] != "none":
        _require(float(pv["epsilon"]) > 0, "privacy.epsilon", "must be > 0")
        _require(0.0 < float(pv["delta"]) < 1.0, "privacy.delta", "must be in (0, 1)")
        _require(sv["clip_norm"] is not None, "server.clip_norm", "DP needs a clip norm")
    if pv["mode"] == "central":
        _require(sv["rule"] in CENTRAL_DP_RULES, "server.rule",
                 f"central DP needs a rule with a replace-one sensitivity bound: {list(CENTRAL_DP_RULES)}")
    malicious = sum(float(b["fraction"]) for b in r["behaviors"] if b["type"] in MALICIOUS)
    _require(malicious <= 1.0 + 1e-12, "behaviors", "malicious fractions must sum to at most 1")
    _require(sum(1 for b in r["behaviors"] if b["type"] == "dropout") <= 1, "behaviors",
             "at most one dropout behaviour")
    _require(int(r["eval"]["per_client_every"]) >= 0, "eval.per_client_every", "must be >= 0")


def resolve(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill defaults, reject unknown keys, validate; returns the resolved config (no ``matrix``)."""
    defaults = DEFAULTS
    workload = cfg.get("workload", {})
    if isinstance(workload, dict) and workload.get("dataset") == "nih":
        defaults = copy.deepcopy(DEFAULTS)
        defaults["workload"].update(features_report_sha256=None, evaluation_split=None,
                                     allow_random_features=False, auc_score="margin_v2")
    out = _merge(defaults, {k: v for k, v in cfg.items() if k != "matrix"}, "")
    fed = out["federation"]
    if fed["clients_per_round"] is None:
        fed["clients_per_round"] = fed["num_clients"]
    if out["seeds"] is None:
        out["seeds"] = default_seeds(int(out["seed"]))
    elif sorted(out["seeds"]) != sorted(SEED_KEYS):
        raise ConfigError("seeds", f"must have exactly the keys {list(SEED_KEYS)}")
    out["behaviors"] = [_resolve_behavior(b, i) for i, b in enumerate(out["behaviors"] or [])]
    out["matrix"] = None
    _validate(out)
    del out["matrix"]
    return out


def _set(cfg: Dict[str, Any], path: str, value: Any) -> None:
    node = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def expand_matrix(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """One ``(name, resolved config)`` per matrix cell and repeat; a config without ``matrix`` is one run."""
    base = resolve(cfg)
    matrix = cfg.get("matrix")
    if matrix and cfg.get("seeds") is not None:
        raise ConfigError("seeds", "cannot be combined with matrix: each repetition derives its seeds from 'seed'")
    if not matrix:
        return [(base["name"], base)]
    if not isinstance(matrix, dict):
        raise ConfigError("matrix", "expected an object")
    repeats = int(matrix.get("repeats", 1))
    _require(repeats >= 1, "matrix.repeats", "must be >= 1")
    axes = [(k, v) for k, v in matrix.items() if k != "repeats"]
    for k, v in axes:
        _require(isinstance(v, list) and len(v) > 0, f"matrix.{k}", "must be a non-empty list")
    runs = []
    for rep in range(repeats):
        for combo in itertools.product(*[vals for _, vals in axes]):
            raw = copy.deepcopy({k: v for k, v in cfg.items() if k != "matrix"})
            for (path, _), value in zip(axes, combo):
                _set(raw, path, value)
            raw["seeds"] = default_seeds(int(raw.get("seed", 0)), rep)
            label = "__".join(f"{p.split('.')[-1]}={v}" for (p, _), v in zip(axes, combo))
            raw["name"] = f"{base['name']}__{label}__r{rep}" if label else f"{base['name']}__r{rep}"
            runs.append((raw["name"], resolve(raw)))
    return runs


def config_hash(resolved: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()


# Asymptotic breakdown points, as in fedlearn.server.robust_aggregation._BREAKDOWN; the trimmed
# mean's is its trim ratio, and the plain means tolerate no attacker at all.
TOLERANCE = {"mean": 0.0, "weighted_mean": 0.0, "median": 0.5, "krum": 0.5, "multi_krum": 0.5,
             "bulyan": 0.25, "centered_clip": 0.5}


def tolerance_flags(resolved: Dict[str, Any]) -> List[str]:
    """Allowed but flagged: a malicious fraction above the rule's breakdown point is a legitimate
    experiment, so it is recorded in the run's ``meta.flags`` rather than rejected."""
    rule = resolved["server"]["rule"]
    tol = (float(resolved["server"]["rule_params"]["trim_ratio"]) if rule == "trimmed_mean"
           else TOLERANCE[rule])
    frac = sum(float(b["fraction"]) for b in resolved["behaviors"] if b["type"] in MALICIOUS)
    if frac > tol:
        return [f"malicious fraction {frac:g} exceeds the {rule} breakdown point {tol:g}"]
    return []
