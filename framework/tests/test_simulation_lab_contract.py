"""Dashboard choices cannot broaden the qualified simulator's execution policy."""
import hashlib
import json

import pytest

from fedlearn.simulation.lab_contract import resolve_choices, write_research_record


PIN = "a" * 64


def resolved(tmp_path, choices=None):
    return resolve_choices({} if choices is None else choices,
                           cache_path=tmp_path / "prepared-cache", report_sha256=PIN)


def test_defaults_resolve_the_actual_core_recipe_and_safe_display(tmp_path):
    result = resolved(tmp_path)
    cfg = result["config"]
    assert cfg["name"] == "simulation-lab" and cfg["seed"] == 42
    assert cfg["seeds"] == dict(partition=42, init=42, client=42, attack=1042, noise=2042)
    assert cfg["workload"] == {
        "dataset": "nih", "features_path": str(tmp_path / "prepared-cache"),
        "features_report_sha256": PIN, "evaluation_split": "validation",
        "allow_random_features": False, "auc_score": "margin_v2", "balance": False,
        "head": {"type": "linear", "hidden": 128},
    }
    fed = cfg["federation"]
    assert (fed["num_clients"], fed["clients_per_round"], fed["rounds"]) == (10, 10, 3)
    assert fed["partition"]["type"] == "patient_iid"
    assert fed["network"]["round_deadline_s"] is None
    assert cfg["client"] == dict(algorithm="fedavg", proximal_mu=0.0,
                                 local_epochs=1, lr=.003, batch_size=256)
    assert cfg["server"]["rule"] == "mean" and cfg["server"]["clip_norm"] == .05
    assert cfg["server"]["rule_params"]["trim_ratio"] == .2
    assert cfg["server"]["optimizer"]["type"] == "none"
    assert cfg["privacy"] == dict(mode="none", epsilon=4.0, delta=1e-5, adjacency="replace_one")
    assert cfg["behaviors"] == []
    assert cfg["eval"]["per_client_every"] == 0
    assert result["config_hash"] == hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    display = result["display_recipe"]
    assert display["evaluationSplit"] == "validation" and display["workloadId"] == "nih-qualified"
    assert display["head"] == "linear" and display["numClients"] == 10
    assert display["researchRecordDpSafe"] is False
    assert str(tmp_path) not in json.dumps(display)


@pytest.mark.parametrize("head,kind,hidden,parameters", [
    ("linear", "linear", 128, 1026), ("mlp8", "mlp", 8, 4122),
    ("mlp128", "mlp", 128, 65922), ("mlp512", "mlp", 512, 263682),
])
def test_head_choice_changes_real_model_and_declared_size(tmp_path, head, kind, hidden, parameters):
    value = resolved(tmp_path, {"head": head})
    assert value["config"]["workload"]["head"] == dict(type=kind, hidden=hidden)
    assert value["display_recipe"]["parameters"] == parameters


@pytest.mark.parametrize("clients", [10, 80, 1000])
@pytest.mark.parametrize("behavior,expected", [
    ("clean", []), ("sign_flip", [dict(type="sign_flip_scale", fraction=.2, scale=-4.0)]),
    ("label_flip", [dict(type="label_flip", fraction=.2)]),
])
def test_supported_cohort_and_own_input_attack_reach_recipe(tmp_path, clients, behavior, expected):
    value = resolved(tmp_path, dict(numClients=clients, behavior=behavior, rounds=20,
                                    localEpochs=7, aggregation="trimmed_mean", clipNorm=.4,
                                    privacy="central", epsilon=8, seed=1_000_000_000))
    cfg = value["config"]
    assert cfg["federation"]["num_clients"] == cfg["federation"]["clients_per_round"] == clients
    assert cfg["federation"]["rounds"] == 20 and cfg["client"]["local_epochs"] == 7
    assert cfg["behaviors"] == expected and cfg["privacy"]["epsilon"] == 8.0
    assert cfg["server"]["rule"] == "trimmed_mean" and cfg["server"]["clip_norm"] == .4


@pytest.mark.parametrize("epsilon", [1, 4.0, 8])
def test_each_private_target_is_explicitly_resolved(tmp_path, epsilon):
    cfg = resolved(tmp_path, dict(privacy="central", epsilon=epsilon))["config"]
    assert cfg["privacy"] == dict(mode="central", epsilon=float(epsilon), delta=1e-5,
                                 adjacency="replace_one")


def test_identical_choices_retain_identity_and_changed_training_choice_changes_it(tmp_path):
    first = resolved(tmp_path)
    assert first == resolved(tmp_path, dict(seed=42, numClients=10, head="linear"))
    assert first["config_hash"] != resolved(tmp_path, dict(seed=43))["config_hash"]


@pytest.mark.parametrize("choice", [None, [], "mean", True, 42])
def test_choices_must_be_an_object(tmp_path, choice):
    with pytest.raises(ValueError):
        resolve_choices(choice, cache_path=tmp_path, report_sha256=PIN)


@pytest.mark.parametrize("key,value", [
    ("evaluationSplit", "test"), ("features_path", "/outside"), ("matrix", {}),
    ("command", "python"), ("featuresReportSha256", "b" * 64),
    ("allowRandomFeatures", True), ("workload", {"evaluation_split": "test"}),
    ("choices", {}), ("learningRate", 1), ("runId", "injected"),
])
def test_unknown_fields_cannot_inject_paths_study_or_engine_policy(tmp_path, key, value):
    with pytest.raises(ValueError):
        resolved(tmp_path, {key: value})


@pytest.mark.parametrize("key,value", [
    ("numClients", True), ("numClients", 10.0), ("numClients", 11),
    ("rounds", False), ("rounds", 1.0), ("rounds", 0), ("rounds", 21),
    ("localEpochs", "1"), ("localEpochs", 0), ("localEpochs", 8),
    ("seed", True), ("seed", -1), ("seed", 1_000_000_001),
    ("clipNorm", True), ("clipNorm", float("nan")), ("clipNorm", float("inf")),
    ("clipNorm", .1), ("epsilon", False), ("epsilon", "4"), ("epsilon", 2),
    ("epsilon", float("nan")), ("privacy", "local"), ("privacy", None),
    ("behavior", "ipm"), ("behavior", "alie"), ("head", "resnet50"),
    ("aggregation", "weighted_mean"), ("aggregation", {}), ("workloadId", "other"),
])
def test_invalid_types_values_and_broader_mechanisms_are_refused(tmp_path, key, value):
    with pytest.raises(ValueError):
        resolved(tmp_path, {key: value})


@pytest.mark.parametrize("pin", [None, "bad", "A" * 64, "0" * 63])
def test_server_cache_pin_must_be_explicit_valid_digest(tmp_path, pin):
    with pytest.raises(ValueError):
        resolve_choices({}, cache_path=tmp_path, report_sha256=pin)


def test_finite_record_preserves_canonical_json_encoding(tmp_path):
    path = tmp_path / "record.json"
    assert write_research_record(path, {"z": None, "a": [1, 2.5]}) == "json"
    assert path.read_bytes() == b'{\n  "a": [\n    1,\n    2.5\n  ],\n  "z": null\n}\n'


def test_nonfinite_failure_record_uses_existing_tagged_schema(tmp_path):
    path = tmp_path / "record.json"
    record = {"rounds": [{"loss": float("inf"), "delta": float("-inf"), "auc": float("nan")} ]}
    assert write_research_record(path, record) == "nonfinite_tagged_v1"
    assert json.loads(path.read_text()) == {"encoding": "nonfinite_tagged_v1", "record": {
        "rounds": [{"loss": {"__nonfinite_float__": "inf"},
                    "delta": {"__nonfinite_float__": "-inf"},
                    "auc": {"__nonfinite_float__": "nan"}}]}}


def test_record_writer_refuses_overwrite_and_symlink(tmp_path):
    original = tmp_path / "original.json"
    original.write_bytes(b"preserved\n")
    link = tmp_path / "record.json"
    link.symlink_to(original)
    for target in [original, link]:
        with pytest.raises(FileExistsError):
            write_research_record(target, {})
    assert original.read_bytes() == b"preserved\n"
