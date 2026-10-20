"""The NIH core refuses unsupported mechanisms and incomplete fixed cohorts."""
import copy
import hashlib

import pytest

from tests.test_capstone_nih_features import template as feature_template
from fedlearn.simulation import engine
from fedlearn.simulation import core_qualification as core


@pytest.fixture
def cfg(feature_template):
    cache = feature_template[1]
    return {
        "seed": 42,
        "workload": {"dataset": "nih", "features_path": str(cache), "balance": False,
                     "features_report_sha256": hashlib.sha256((cache / "report.json").read_bytes()).hexdigest(),
                     "evaluation_split": "validation", "allow_random_features": True},
        "federation": {"num_clients": 3, "rounds": 2, "partition": {"type": "patient_iid"}},
        "client": {"local_epochs": 1, "lr": .001}, "server": {"clip_norm": .4},
    }


def _policy(cfg):
    return {"feature_report_sha256": cfg["workload"]["features_report_sha256"], "clients": 3}


@pytest.fixture
def record(cfg):
    return core.run_core(cfg, **_policy(cfg))


@pytest.mark.parametrize("behavior", [
    {"type": "ipm", "fraction": .2}, {"type": "alie", "fraction": 0},
    {"type": "dropout", "fraction": 0}, {"type": "straggler", "fraction": .2},
    {"type": "label_noise", "fraction": .2}, {"type": "free_rider", "fraction": .2},
])
def test_unsupported_behavior_refuses_before_engine_access(cfg, monkeypatch, behavior):
    monkeypatch.setattr(engine, "run", lambda *a, **k: pytest.fail("engine reached"))
    cfg["behaviors"] = [behavior]
    with pytest.raises(core.CoreQualificationError, match="behavior") as error:
        core.run_core(cfg, **_policy(cfg))
    assert error.value.record is None


@pytest.mark.parametrize("path,value", [
    (("federation", "clients_per_round"), 2), (("federation", "rounds"), 1.5),
    (("federation", "network"), {"round_deadline_s": 5}),
    (("privacy", "mode"), "local"), (("server", "rule"), "weighted_mean"),
    (("server", "clip_norm"), None), (("client", "lr"), float("nan")),
    (("workload", "auc_score"), "probability_v1"),
    (("workload", "evaluation_split"), "test"),
    (("matrix",), {"repeats": 2}),
])
def test_unsupported_recipe_refuses_before_engine_access(cfg, monkeypatch, path, value):
    monkeypatch.setattr(engine, "run", lambda *a, **k: pytest.fail("engine reached"))
    target = cfg
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value
    with pytest.raises(core.CoreQualificationError):
        core.run_core(cfg, **_policy(cfg))


@pytest.mark.parametrize("path,value", [
    (("client", "lr"), "Infinity"), (("client", "proximal_mu"), "Infinity"),
    (("server", "optimizer", "lr"), "Infinity"),
    (("federation", "network", "device_gflops"), "Infinity"),
    (("privacy", "epsilon"), "Infinity"), (("seed",), "42"),
    (("client", "batch_size"), "256"),
])
def test_numeric_strings_refuse_before_engine_access(cfg, monkeypatch, path, value):
    monkeypatch.setattr(engine, "run", lambda *a, **k: pytest.fail("engine reached"))
    target = cfg
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value
    with pytest.raises(core.CoreQualificationError):
        core.run_core(cfg, **_policy(cfg))


def test_expected_pin_and_cohort_are_independent_of_candidate(cfg):
    with pytest.raises(core.CoreQualificationError, match="identity|pin"):
        core.prepare_core_config(cfg, clients=3, feature_report_sha256="0" * 64)
    with pytest.raises(core.CoreQualificationError, match="cohort"):
        core.prepare_core_config(cfg, feature_report_sha256=_policy(cfg)["feature_report_sha256"])
    with pytest.raises(core.CoreQualificationError):
        core.prepare_core_config(cfg, **{**_policy(cfg), "clients": True})


def test_rounded_attack_assignment_must_leave_an_honest_client(cfg):
    cfg["behaviors"] = [{"type": "label_flip", "fraction": .99}]
    with pytest.raises(core.CoreQualificationError, match="honest"):
        core.prepare_core_config(cfg, **_policy(cfg))


@pytest.mark.parametrize("privacy,rule,behaviors", [
    ("none", "mean", []),
    ("central", "mean", [{"type": "sign_flip_scale", "fraction": .34, "scale": -4}]),
    ("central", "trimmed_mean", [{"type": "label_flip", "fraction": .34}]),
])
def test_real_engine_qualifies_permitted_core_with_literal_sensitivity(cfg, privacy, rule, behaviors):
    cfg["privacy"] = {"mode": privacy, "epsilon": 4.0}
    cfg["server"].update(rule=rule, rule_params={"trim_ratio": .34})
    cfg["behaviors"] = behaviors
    rec = core.run_core(cfg, **_policy(cfg))
    summary = rec["meta"]["core_qualification"]
    assert summary["scope"] == "simulation_execution_only"
    assert summary["research_record_dp_safe"] is False and summary["accepted_clients"] == 3
    assert summary["feature_control"] is True
    expected_sensitivity = .8 if rule == "trimmed_mean" else .8 / 3
    for line in rec["rounds"]:
        assert line["participants"]["accepted"] == 3
        assert line["privacy"]["sensitivity"] == pytest.approx(expected_sensitivity)
        if privacy == "central":
            assert line["privacy"]["noise_std"] > 0
        else:
            assert line["privacy"]["noise_std"] == 0


@pytest.mark.parametrize("mutation", [
    "old_schema", "missing_count", "boolean_count", "fractional_count", "partial_count",
    "forced", "refusal", "stopped", "missing_round", "duplicate_round", "config_hash",
    "workload_pin", "evaluation", "assignment", "sensitivity", "noise", "utility",
])
def test_record_qualification_fails_closed(cfg, record, mutation):
    rec = copy.deepcopy(record)
    del rec["meta"]["core_qualification"]
    line, meta = rec["rounds"][0], rec["meta"]
    if mutation == "old_schema":
        del meta["participation_schema_version"]
    elif mutation == "missing_count":
        del line["participants"]["accepted"]
    elif mutation == "boolean_count":
        line["participants"]["dropped"] = False
    elif mutation == "fractional_count":
        line["participants"]["accepted"] = 3.0
    elif mutation == "partial_count":
        line["participants"]["admitted"] = 2
    elif mutation == "forced":
        line["forced"] = True
    elif mutation == "refusal":
        line["refusal"] = "refused"
    elif mutation == "stopped":
        meta["stopped_early"] = True
    elif mutation == "missing_round":
        rec["rounds"].pop()
    elif mutation == "duplicate_round":
        rec["rounds"][1]["round"] = 1
    elif mutation == "config_hash":
        meta["config_hash"] = "0" * 64
    elif mutation == "workload_pin":
        meta["workload"]["features_report_sha256"] = "0" * 64
    elif mutation == "evaluation":
        meta["workload"]["evaluation_split"] = "test"
    elif mutation == "assignment":
        meta["behavior_assignment"] = {"0": ["ipm"]}
    elif mutation == "sensitivity":
        line["privacy"]["sensitivity"] = .8 / 2
    elif mutation == "noise":
        line["privacy"]["noise_std"] = 1
    elif mutation == "utility":
        line["auc"] = float("nan")
    with pytest.raises(core.CoreQualificationError) as error:
        core.qualify_core_record(rec, cfg, **_policy(cfg))
    assert error.value.record is rec


@pytest.mark.parametrize("bad_update", ["zero_count", "extra_key"])
def test_actual_invalid_inputs_refuse_and_preserve_failed_record(cfg, monkeypatch, bad_update):
    fit = engine.HeadClient.fit

    def corrupt(self, params, config):
        update, count = fit(self, params, config)
        if self.cid == 0:
            if bad_update == "zero_count":
                count = 0
            else:
                update["unexpected"] = update[next(iter(update))].new_zeros(1)
        return update, count

    monkeypatch.setattr(engine.HeadClient, "fit", corrupt)
    with pytest.raises(core.CoreQualificationError) as error:
        core.run_core(cfg, **_policy(cfg))
    rec = error.value.record
    assert len(rec["rounds"]) == 2 and rec["rounds"][0]["participants"]["accepted"] == 2
    assert "core_qualification" not in rec["meta"]


def test_requested_stop_preserves_partial_record_without_qualification(cfg):
    seen = []
    with pytest.raises(core.CoreQualificationError) as error:
        core.run_core(cfg, **_policy(cfg), on_round=seen.append, should_stop=lambda: len(seen) == 1)
    assert len(seen) == len(error.value.record["rounds"]) == 1
    assert error.value.record["meta"]["stop_reason"] == "stopped by request"


def test_self_consistent_noise_fields_must_match_declared_budget(cfg):
    cfg["privacy"] = {"mode": "central", "epsilon": 4.0}
    rec = core.run_core(cfg, **_policy(cfg))
    del rec["meta"]["core_qualification"]
    rec["meta"]["noise_multiplier"] *= .5
    for line in rec["rounds"]:
        line["privacy"]["noise_std"] *= .5
    with pytest.raises(core.CoreQualificationError, match="multiplier|budget"):
        core.qualify_core_record(rec, cfg, **_policy(cfg))


@pytest.mark.parametrize("change", ["total", "prefix"])
def test_accounting_is_recomputed_for_every_release_prefix(cfg, change):
    cfg["privacy"] = {"mode": "central", "epsilon": 4.0}
    rec = core.run_core(cfg, **_policy(cfg))
    del rec["meta"]["core_qualification"]
    if change == "total":
        rec["meta"]["accounted_epsilon"] = 1.0
        for i, line in enumerate(rec["rounds"], 1):
            line["privacy"]["epsilon_spent"] = .5 * i
    else:
        rec["rounds"][0]["privacy"]["epsilon_spent"] = .001
    with pytest.raises(core.CoreQualificationError, match="epsilon|accounting"):
        core.qualify_core_record(rec, cfg, **_policy(cfg))


def test_configuration_cannot_be_mutated_by_round_callback(cfg):
    def change_candidate(line):
        cfg["server"]["clip_norm"] = 999

    rec = core.run_core(cfg, **_policy(cfg), on_round=change_candidate)
    assert rec["meta"]["config"]["server"]["clip_norm"] == .4
    assert rec["meta"]["core_qualification"]["accepted_clients"] == 3
