"""Execution checks for a declared NIH core; neither a study protocol nor a DP export."""
from __future__ import annotations

import json
import math
import re

from fedlearn.privacy.dp_accountant import compute_rdp, get_epsilon, required_noise_multiplier
from fedlearn.simulation import engine
from fedlearn.simulation.behaviors import BehaviorPlan
from fedlearn.simulation.config import config_hash, resolve


class CoreQualificationError(ValueError):
    """A refused recipe or record; retain postflight evidence for the caller to save."""

    def __init__(self, message, *, record=None):
        super().__init__(message)
        self.record = record


def _require(condition, message):
    if not condition:
        raise CoreQualificationError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _close(value, expected):
    return _finite(value) and math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-12)


def prepare_core_config(cfg, *, feature_report_sha256, clients=80, evaluation_split="validation"):
    """Resolve one recipe and enforce the caller's independently declared policy before I/O."""
    try:
        _require(type(clients) is int and clients > 0, "expected cohort must be a positive integer")
        _require(isinstance(feature_report_sha256, str)
                 and re.fullmatch(r"[0-9a-f]{64}", feature_report_sha256), "invalid expected cache pin")
        _require(evaluation_split in ("validation", "test"), "invalid declared evaluation split")
        _require(isinstance(cfg, dict) and cfg.get("matrix") is None, "core requires one recipe, no matrix")
        resolved = resolve(cfg)
        # Generic historical config validation permits some wider numeric coercions.
        # A prospective recipe must be finite and have exact integer cohort/round counts.
        json.dumps(resolved, allow_nan=False)
        wl, fed, sv, pv = (resolved[key] for key in ("workload", "federation", "server", "privacy"))
        _require(all(b["type"] in ("sign_flip_scale", "label_flip") for b in resolved["behaviors"]),
                 "core behavior must use own data or own update; oracle/fault behaviors are excluded")
        cl, rp, net = resolved["client"], sv["rule_params"], fed["network"]
        integers = [resolved["schema_version"], resolved["seed"], *resolved["seeds"].values(),
                    wl["head"]["hidden"], fed["num_clients"], fed["clients_per_round"], fed["rounds"],
                    cl["local_epochs"], resolved["eval"]["per_client_every"], rp["iterations"]]
        integers += [v for v in (cl["batch_size"], rp["multi_krum_m"]) if v is not None]
        _require(all(type(value) is int for value in integers), "integer settings must use JSON integers")
        scalars = [cl["lr"], cl["proximal_mu"], sv["optimizer"]["lr"], pv["epsilon"], pv["delta"],
                   fed["partition"]["alpha"], rp["trim_ratio"], rp["tau"],
                   *[net[key] for key in ("uplink_mbps", "downlink_mbps", "latency_ms", "device_gflops")]]
        scalars += [v for v in (sv["clip_norm"], rp["byzantine_fraction"]) if v is not None]
        scalars += [v for b in resolved["behaviors"] for key, v in b.items() if key != "type"]
        _require(all(_finite(value) for value in scalars), "numeric settings must use finite JSON numbers")
        _require(wl["dataset"] == "nih", "core requires the NIH workload")
        _require(wl["features_report_sha256"] == feature_report_sha256, "cache identity differs from expected pin")
        _require(wl["evaluation_split"] == evaluation_split, "candidate evaluation differs from declared split")
        _require(wl["balance"] is False and wl["auc_score"] == "margin_v2", "core requires natural prevalence and margin_v2")
        _require(all(type(fed[key]) is int and fed[key] == clients
                     for key in ("num_clients", "clients_per_round")), "core requires the full declared cohort")
        _require(type(fed["rounds"]) is int and fed["rounds"] > 0, "rounds must be a positive integer")
        _require(fed["partition"]["type"] == "patient_iid", "core requires whole-patient IID clients")
        _require(fed["network"]["round_deadline_s"] is None, "core excludes deadline-based selection")
        _require(pv["mode"] in ("none", "central"), "core permits central or matched no-DP mode only")
        _require(sv["rule"] in ("mean", "trimmed_mean"), "core permits mean or trimmed_mean only")
        _require(_finite(sv["clip_norm"]) and sv["clip_norm"] > 0, "core requires finite positive clipping")
        plan = BehaviorPlan(resolved["behaviors"], clients, int(resolved["seeds"]["attack"]), 2)
        _require(len(plan.malicious_ids()) < clients, "core requires at least one protected honest client")
        return resolved
    except CoreQualificationError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise CoreQualificationError(f"invalid core recipe: {error}") from error


def qualify_core_record(record, cfg, *, feature_report_sha256, clients=80, evaluation_split="validation"):
    """Check complete fixed-cohort execution. The record remains diagnostic research output."""
    try:
        r = prepare_core_config(cfg, feature_report_sha256=feature_report_sha256,
                                clients=clients, evaluation_split=evaluation_split)
        meta, lines = record["meta"], record["rounds"]
        _require("core_qualification" not in meta, "record already has a qualification marker")
        _require(meta["config"] == r and meta["config_hash"] == config_hash(r), "record config identity mismatch")
        _require(type(meta["participation_schema_version"]) is int and meta["participation_schema_version"] == 2,
                 "core requires participation schema 2")
        _require(meta["stopped_early"] is False and meta["stop_reason"] is None and meta["refusals"] == [],
                 "core run stopped or refused")
        wl, sv, pv = meta["workload"], r["server"], r["privacy"]
        _require(wl["dataset"] == "nih" and wl["features_report_sha256"] == feature_report_sha256
                 and wl["evaluation_split"] == evaluation_split and wl["auc_score"] == "margin_v2"
                 and wl["feat_dim"] == 512 and wl["n_classes"] == 2, "record workload identity mismatch")
        _require(type(wl["pretrained"]) is bool, "missing pretrained/control provenance")
        plan = BehaviorPlan(r["behaviors"], clients, int(r["seeds"]["attack"]), 2)
        _require(meta["behavior_assignment"] == plan.assignment(), "behavior assignment mismatch")
        _require(meta["privacy_mode"] == pv["mode"] and meta["adjacency"] == "replace_one"
                 and _close(meta["server_clip_norm"], sv["clip_norm"]), "privacy mechanism metadata mismatch")
        z = meta["noise_multiplier"]
        central = pv["mode"] == "central"
        rounds, delta = r["federation"]["rounds"], float(pv["delta"])
        expected_z = (float(required_noise_multiplier(float(pv["epsilon"]), 1.0, rounds, delta))
                      if central else 0.0)
        _require(_close(z, expected_z), "noise multiplier differs from declared budget calibration")
        accounted = meta["accounted_epsilon"]
        expected_epsilon = (float(get_epsilon(compute_rdp(1.0, expected_z, rounds), delta)[0])
                            if central else None)
        _require((_close(accounted, expected_epsilon) and 0 < accounted <= pv["epsilon"] + 1e-6) if central
                 else accounted is None, "invalid accounted epsilon")
        _require(isinstance(lines, list) and len(lines) == r["federation"]["rounds"], "incomplete round sequence")
        trim = math.floor(float(sv["rule_params"]["trim_ratio"]) * clients) if sv["rule"] == "trimmed_mean" else 0
        sensitivity = 2 * float(sv["clip_norm"]) / (clients - 2 * trim)
        expected_counts = {"sampled": clients, "dropped": 0, "straggled": 0, "submitted": clients,
                           "admitted": clients, "accepted": clients, "aggregated": clients}
        previous_epsilon = 0.0
        for number, line in enumerate(lines, 1):
            _require(type(line["round"]) is int and line["round"] == number, "nonsequential round identity")
            _require(line["forced"] is False and line["refusal"] is None, "forced or refused round")
            counts = line["participants"]
            _require(set(counts) == set(expected_counts)
                     and all(type(counts[key]) is int and counts[key] == value
                             for key, value in expected_counts.items()), "incomplete or invalid accepted cohort")
            privacy = line["privacy"]
            _require(_close(privacy["sensitivity"], sensitivity)
                     and _close(privacy["noise_std"], z * sensitivity), "sensitivity/noise differs from fixed-cohort mechanism")
            spent = privacy["epsilon_spent"]
            if central:
                expected_spent = float(get_epsilon(compute_rdp(1.0, expected_z, number), delta)[0])
                _require(_close(spent, expected_spent) and previous_epsilon <= spent <= accounted + 1e-6,
                         "invalid cumulative epsilon")
                previous_epsilon = spent
            else:
                _require(spent is None, "no-DP control reports a privacy budget")
            for key in ("accuracy", "auc", "auc_probability", "loss", "sensitivity", "specificity",
                        "balanced_accuracy", "ece"):
                value = line[key]
                _require(_finite(value) and value >= 0 and (key == "loss" or value <= 1),
                         f"missing or invalid utility: {key}")
        if central:
            _require(_close(previous_epsilon, accounted), "last-round and total epsilon differ")
        return {"version": 1, "scope": "simulation_execution_only", "accepted_clients": clients,
                "attack_knowledge": "own_data_or_own_update", "privacy_mode": pv["mode"],
                "evaluation_split": evaluation_split, "feature_control": not wl["pretrained"],
                "research_record_dp_safe": False}
    except CoreQualificationError as error:
        error.record = record
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        raise CoreQualificationError(f"incomplete or invalid core record: {error}", record=record) from error


def run_core(cfg, *, feature_report_sha256, clients=80, evaluation_split="validation",
             on_round=None, should_stop=None, on_final_parameters=None):
    """Check policy before training, then label only a successfully qualified result.

    A caller must save ``CoreQualificationError.record`` on postflight failure.
    Engine exceptions propagate for its runner to retain alongside logs/tracebacks.
    Final-parameter callbacks receive owned CPU copies only after qualification;
    callback failure retains the qualified simulation record for the runner.
    """
    policy = {"feature_report_sha256": feature_report_sha256, "clients": clients,
              "evaluation_split": evaluation_split}
    resolved = prepare_core_config(cfg, **policy)
    captured = []
    capture = {"on_final_parameters": captured.append} if on_final_parameters is not None else {}
    record = engine.run(resolved, on_round=on_round, should_stop=should_stop, **capture)
    qualification = qualify_core_record(record, resolved, **policy)
    record["meta"]["core_qualification"] = qualification
    if on_final_parameters is not None:
        try:
            _require(len(captured) == 1, "missing final parameters")
            on_final_parameters(captured[0])
        except Exception as error:
            raise CoreQualificationError(f"final parameter callback failed: {error}", record=record) from error
    return record
