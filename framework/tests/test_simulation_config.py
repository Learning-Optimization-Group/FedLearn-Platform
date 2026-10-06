"""Run config: unknown keys and invalid combinations fail before any training, naming the field."""
import pytest

from fedlearn.simulation.config import (
    ConfigError, config_hash, default_seeds, expand_matrix, resolve, tolerance_flags,
)

BASE = {"workload": {"features_path": "/tmp/f.pt"}, "seed": 5000}


def test_defaults_are_filled_and_seeds_follow_the_harness_scheme():
    r = resolve(BASE)
    assert r["federation"]["clients_per_round"] == r["federation"]["num_clients"]
    assert r["seeds"] == {"partition": 5000, "init": 5000, "client": 5000, "attack": 6000, "noise": 7000}
    assert default_seeds(5000, 3) == {"partition": 5003, "init": 5003, "client": 5003,
                                      "attack": 6003, "noise": 7003}


def test_unknown_key_is_rejected_with_its_path():
    with pytest.raises(ConfigError) as e:
        resolve({**BASE, "server": {"clip_nrom": 0.4}})
    assert e.value.field == "server.clip_nrom"


@pytest.mark.parametrize("rule", ["median", "krum", "multi_krum", "bulyan", "centered_clip", "weighted_mean"])
def test_central_dp_rejects_rules_without_a_replace_one_bound(rule):
    with pytest.raises(ConfigError) as e:
        resolve({**BASE, "server": {"rule": rule}, "privacy": {"mode": "central"}})
    assert e.value.field == "server.rule"


def test_local_dp_accepts_any_rule():
    assert resolve({**BASE, "server": {"rule": "median"}, "privacy": {"mode": "local"}})["privacy"]["mode"] == "local"


def test_dp_requires_a_clip_norm():
    with pytest.raises(ConfigError) as e:
        resolve({**BASE, "server": {"clip_norm": None}, "privacy": {"mode": "central"}})
    assert e.value.field == "server.clip_norm"


def test_fedprox_requires_positive_mu():
    with pytest.raises(ConfigError) as e:
        resolve({**BASE, "client": {"algorithm": "fedprox"}})
    assert e.value.field == "client.proximal_mu"


def test_behaviors_get_defaults_and_validated_fractions():
    r = resolve({**BASE, "behaviors": [{"type": "ipm", "fraction": 0.2}, {"type": "dropout", "fraction": 0.1}]})
    assert r["behaviors"][0] == {"type": "ipm", "fraction": 0.2, "scale": 4.0}
    with pytest.raises(ConfigError) as e:
        resolve({**BASE, "behaviors": [{"type": "ipm", "fraction": 1.5}]})
    assert e.value.field == "behaviors[0].fraction"
    with pytest.raises(ConfigError) as e:
        resolve({**BASE, "behaviors": [{"type": "ipm", "fraction": 0.6}, {"type": "alie", "fraction": 0.6}]})
    assert e.value.field == "behaviors"
    with pytest.raises(ConfigError) as e:
        resolve({**BASE, "behaviors": [{"type": "backdoor", "fraction": 0.1}]})
    assert e.value.field == "behaviors[0].target_label"
    with pytest.raises(ConfigError) as e:
        resolve({**BASE, "behaviors": [{"type": "teleport", "fraction": 0.1}]})
    assert e.value.field == "behaviors[0].type"
    with pytest.raises(ConfigError) as e:
        resolve({**BASE, "behaviors": [{"type": "backdoor", "fraction": 0.1, "target_label": 1,
                                        "trigger": {"type": "blend"}}]})
    assert e.value.field == "behaviors[0].trigger"
    bd = resolve({**BASE, "behaviors": [{"type": "backdoor", "fraction": 0.1, "target_label": 1}]})["behaviors"][0]
    assert bd["trigger"] is None and bd["poison_fraction"] == 0.5


def test_fractions_above_a_rules_tolerance_are_flagged_not_rejected():
    attack = [{"type": "ipm", "fraction": 0.3}]
    r = resolve({**BASE, "server": {"rule": "trimmed_mean"}, "behaviors": attack})
    assert tolerance_flags(r) == ["malicious fraction 0.3 exceeds the trimmed_mean breakdown point 0.2"]
    assert tolerance_flags(resolve({**BASE, "server": {"rule": "median"}, "behaviors": attack})) == []
    assert tolerance_flags(resolve({**BASE, "server": {"rule": "bulyan"}, "behaviors": attack})) != []
    assert tolerance_flags(resolve(BASE)) == []


def test_matrix_expands_cells_times_repeats_with_shared_seeds_per_repeat():
    cfg = {**BASE, "name": "q", "matrix": {"privacy.mode": ["none", "central"],
                                          "server.rule": ["mean", "trimmed_mean"], "repeats": 2}}
    runs = expand_matrix(cfg)
    assert len(runs) == 8
    names = [n for n, _ in runs]
    assert names[0] == "q__mode=none__rule=mean__r0"
    rep1 = [r for n, r in runs if n.endswith("__r1")]
    assert {r["seeds"]["noise"] for r in rep1} == {7001}
    assert all("matrix" not in r for _, r in runs)


def test_matrix_with_invalid_combination_fails_before_running():
    cfg = {**BASE, "matrix": {"privacy.mode": ["central"], "server.rule": ["median"], "repeats": 1}}
    with pytest.raises(ConfigError):
        expand_matrix(cfg)


def test_config_hash_is_stable_and_sensitive():
    a, b = resolve(BASE), resolve(BASE)
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(resolve({**BASE, "seed": 1}))


def test_explicit_seeds_cannot_be_combined_with_a_matrix():
    with pytest.raises(ConfigError) as e:
        expand_matrix({**BASE, "seeds": default_seeds(1), "matrix": {"repeats": 2}})
    assert e.value.field == "seeds"


@pytest.mark.parametrize("over, field", [
    ({"federation": {"network": {"uplink_mbps": 0}}}, "federation.network.uplink_mbps"),
    ({"federation": {"network": {"downlink_mbps": -1}}}, "federation.network.downlink_mbps"),
    ({"federation": {"network": {"latency_ms": -5}}}, "federation.network.latency_ms"),
    ({"federation": {"network": {"device_gflops": 0}}}, "federation.network.device_gflops"),
    ({"federation": {"network": {"round_deadline_s": 0}}}, "federation.network.round_deadline_s"),
    ({"server": {"rule_params": {"byzantine_fraction": 1.0}}}, "server.rule_params.byzantine_fraction"),
    ({"server": {"rule_params": {"tau": 0}}}, "server.rule_params.tau"),
    ({"server": {"rule_params": {"iterations": 0}}}, "server.rule_params.iterations"),
    ({"server": {"rule_params": {"multi_krum_m": 0}}}, "server.rule_params.multi_krum_m"),
    ({"server": {"optimizer": {"type": "adam", "lr": 0}}}, "server.optimizer.lr"),
    ({"behaviors": [{"type": "backdoor", "fraction": 0.1, "target_label": 1, "poison_fraction": 1.5}]},
     "behaviors[0].poison_fraction"),
    ({"behaviors": [{"type": "label_noise", "fraction": 0.1, "flip_prob": -0.1}]}, "behaviors[0].flip_prob"),
    ({"behaviors": [{"type": "straggler", "fraction": 0.1, "slowdown": 0.5}]}, "behaviors[0].slowdown"),
    ({"behaviors": [{"type": "free_rider", "fraction": 0.1, "mode": "noise", "noise_std": -1}]},
     "behaviors[0].noise_std"),
    ({"behaviors": [{"type": "alie", "fraction": 0.1, "z": -1}]}, "behaviors[0].z"),
    ({"behaviors": [{"type": "sign_flip_scale", "fraction": 0.1, "scale": 0}]}, "behaviors[0].scale"),
    ({"behaviors": [{"type": "ipm", "fraction": 0.1, "scale": 0}]}, "behaviors[0].scale"),
])
def test_out_of_range_values_fail_before_training(over, field):
    with pytest.raises(ConfigError) as e:
        resolve({**BASE, **over})
    assert e.value.field == field
