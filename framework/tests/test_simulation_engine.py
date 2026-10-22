"""Engine: one config in, a complete, deterministic record out."""
import pytest
import torch

from fedlearn.simulation.engine import run

TRIGGER = {"type": "patch", "size": 16, "value": 1.0, "position": "bottom_right"}


def _features(tmp_path, trigger=False, d=16):
    g = torch.Generator().manual_seed(0)

    def split(n):
        y = (torch.rand(n, generator=g) < 0.5).long()
        return torch.randn(n, d, generator=g) + y[:, None].float() * 1.5, y

    tx, ty = split(400)
    vx, vy = split(200)
    rec = {"train_x": tx, "train_y": ty, "test_x": vx, "test_y": vy, "feat_dim": d, "n_classes": 2}
    if trigger:
        rec["trigger_train_x"], rec["trigger_test_x"] = tx + 4.0, vx + 4.0
        rec["trigger"] = dict(TRIGGER)
    p = tmp_path / ("ft.pt" if trigger else "f.pt")
    torch.save(rec, p)
    return str(p)


def _cfg(path, **over):
    base = {"seed": 5000, "workload": {"features_path": path},
            "federation": {"num_clients": 10, "rounds": 3},
            "client": {"local_epochs": 2, "lr": 0.5}, "server": {"clip_norm": 0.5}}
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return base


def _stable(rec):
    """Round lines without the measured-time fields (the only fields allowed to differ)."""
    out = []
    for line in rec["rounds"]:
        line = {k: v for k, v in line.items() if k != "time"}
        line["compute"] = {k: v for k, v in line["compute"].items() if k != "cpu_s"}
        out.append(line)
    return out


def test_a_run_produces_rounds_costs_and_provenance(tmp_path):
    rec = run(_cfg(_features(tmp_path), privacy={"mode": "central", "epsilon": 4.0},
                   server={"rule": "trimmed_mean"}, behaviors=[{"type": "ipm", "fraction": 0.2}]))
    assert len(rec["rounds"]) == 3 and [l["round"] for l in rec["rounds"]] == [1, 2, 3]
    r0 = rec["rounds"][0]
    assert {"accuracy", "auc", "ece", "fairness", "participants", "comm", "time", "compute", "privacy",
            "robustness", "updates", "refusal"} <= set(r0)
    assert r0["refusal"] is None and r0["accuracy"] is not None
    assert r0["comm"]["down_bytes"] == r0["comm"]["up_bytes"] > 0
    assert r0["compute"]["client_flops"] == r0["compute"]["client_flops_analytic"] > 0
    assert r0["privacy"]["noise_std"] > 0 and r0["privacy"]["noise_to_signal"] > 0
    assert r0["privacy"]["epsilon_spent"] < rec["rounds"][-1]["privacy"]["epsilon_spent"] <= 4.0 + 1e-6
    assert set(r0["robustness"]["exclusion"]) == {"honest", "attacker"}
    assignment = rec["meta"]["behavior_assignment"]
    assert len(assignment) == 2 and all(v == ["ipm"] for v in assignment.values())
    assert rec["meta"]["flags"] == []                     # 0.2 does not exceed the trim ratio 0.2
    totals = rec["meta"]["totals"]
    assert sum(totals["per_client_bytes"].values()) == totals["down_bytes"] + totals["up_bytes"]
    assert rec["meta"]["config_hash"] and rec["meta"]["peak_rss_bytes"] > 0
    assert rec["meta"]["round0_upload_digest"] and rec["meta"]["final_digest"]
    assert all(l["loss"] is not None for l in rec["rounds"])          # loss is a recorded utility metric
    assert rec["meta"]["server_clip_norm"] == pytest.approx(0.5)      # central: the server clips at C


def test_a_run_is_deterministic_except_measured_time(tmp_path):
    p = _features(tmp_path)
    cfg = _cfg(p, privacy={"mode": "local", "epsilon": 8.0}, behaviors=[{"type": "label_noise", "fraction": 0.3}])
    a, b = run(cfg), run(cfg)
    assert _stable(a) == _stable(b) and a["meta"]["final_digest"] == b["meta"]["final_digest"]
    assert run({**cfg, "seed": 6000})["meta"]["final_digest"] != a["meta"]["final_digest"]


def test_local_dp_privatizes_honest_clients_only(tmp_path):
    # Attackers do not privatize (the worst case). A training-time attacker has no upload override,
    # so its round-1 upload — trained from the same first global either way — must be identical with
    # and without local DP, while honest uploads change.
    p = _features(tmp_path)
    flip = [{"type": "label_flip", "fraction": 0.2}]
    plain = run(_cfg(p, behaviors=flip))["rounds"][0]["updates"]
    local = run(_cfg(p, behaviors=flip, privacy={"mode": "local", "epsilon": 8.0}))["rounds"][0]["updates"]
    assert local["attacker_norm_mean"] == pytest.approx(plain["attacker_norm_mean"], rel=1e-6)
    assert local["honest_norm_mean"] != pytest.approx(plain["honest_norm_mean"], rel=1e-3)


def test_partial_round_is_recorded_not_zeroed(tmp_path):
    # 1e-4 GFLOPS: an ordinary client needs about 0.2 s simulated, a 1000x straggler about 100 s.
    cfg = _cfg(_features(tmp_path), behaviors=[{"type": "straggler", "fraction": 0.3, "slowdown": 1000.0}],
               federation={"network": {"round_deadline_s": 1.0, "device_gflops": 1e-4}})
    r0 = run(cfg)["rounds"][0]
    assert r0["participants"] == {"sampled": 10, "dropped": 0, "straggled": 3, "submitted": 7,
                                  "admitted": 7, "accepted": 7, "aggregated": 7}
    assert r0["forced"] is True and r0["accuracy"] is not None and r0["refusal"] is None
    assert r0["comm"]["up_bytes"] == r0["comm"]["down_bytes"]      # late stragglers' bytes still count
    # a server cannot know a straggler is late before the deadline passes, so the round lasts until it
    assert r0["time"]["simulated_round_s"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad_update,admitted,accepted", [("zero_count", 9, 9), ("extra_key", 10, 9)])
def test_participation_counts_follow_coordinator_and_strategy_validation(
        tmp_path, monkeypatch, bad_update, admitted, accepted):
    from fedlearn.simulation.engine import HeadClient
    fit = HeadClient.fit

    def corrupt_one(self, params, config):
        update, count = fit(self, params, config)
        if self.cid == 0:
            if bad_update == "zero_count":
                count = 0
            else:
                update["unexpected"] = torch.zeros(1)
        return update, count

    monkeypatch.setattr(HeadClient, "fit", corrupt_one)
    rec = run(_cfg(_features(tmp_path), federation={"rounds": 2}))
    assert [line["round"] for line in rec["rounds"]] == [1, 2]
    for line in rec["rounds"]:
        assert line["participants"] == {
            "sampled": 10, "dropped": 0, "straggled": 0, "submitted": 10,
            "admitted": admitted, "accepted": accepted, "aggregated": 9,
        }
        assert line["refusal"] is None and line["accuracy"] is not None
        assert line["privacy"]["sensitivity"] == pytest.approx(1.0 / 9)
    assert rec["meta"]["participation_schema_version"] == 2


def test_all_rejected_updates_stop_without_claiming_validation_or_aggregation(tmp_path, monkeypatch):
    from fedlearn.simulation.engine import HeadClient
    fit = HeadClient.fit

    def zero_count(self, params, config):
        update, _ = fit(self, params, config)
        return update, 0

    monkeypatch.setattr(HeadClient, "fit", zero_count)
    rec = run(_cfg(_features(tmp_path)))
    assert len(rec["rounds"]) == 1 and rec["meta"]["stopped_early"]
    line = rec["rounds"][0]
    assert line["participants"] == {
        "sampled": 10, "dropped": 0, "straggled": 0, "submitted": 10,
        "admitted": 0, "accepted": None, "aggregated": 0,
    }
    assert line["forced"] and line["refusal"] and line["accuracy"] is None


def test_reducer_refusal_has_valid_inputs_but_no_aggregate(tmp_path):
    rec = run(_cfg(_features(tmp_path), federation={"num_clients": 3, "rounds": 1},
                   server={"rule": "krum", "rule_params": {"byzantine_fraction": 0.34}}))
    line = rec["rounds"][0]
    assert "krum" in line["refusal"] and line["accuracy"] is None
    assert line["participants"] == {
        "sampled": 3, "dropped": 0, "straggled": 0, "submitted": 3,
        "admitted": 3, "accepted": 3, "aggregated": 0,
    }


def test_backdoor_without_triggered_features_fails_fast(tmp_path):
    cfg = _cfg(_features(tmp_path), behaviors=[{"type": "backdoor", "fraction": 0.2, "target_label": 1}])
    with pytest.raises(ValueError, match="triggered"):
        run(cfg)


def test_backdoor_trigger_must_match_the_features_file(tmp_path):
    cfg = _cfg(_features(tmp_path, trigger=True),
               behaviors=[{"type": "backdoor", "fraction": 0.2, "target_label": 1, "trigger": {"size": 8}}])
    with pytest.raises(ValueError, match="trigger"):
        run(cfg)


def test_backdoor_reports_attack_success(tmp_path):
    cfg = _cfg(_features(tmp_path, trigger=True),
               behaviors=[{"type": "backdoor", "fraction": 0.3, "target_label": 1, "trigger": {"size": 16}}])
    assert run(cfg)["rounds"][-1]["backdoor_success"] is not None


def test_a_rule_refusal_is_recorded_not_zeroed(tmp_path):
    # Krum with f = int(0.5 * 4) = 2 needs n >= 2f + 3 = 7 clients; with 4 it refuses every round.
    rec = run(_cfg(_features(tmp_path), federation={"num_clients": 4},
                   server={"rule": "krum", "rule_params": {"byzantine_fraction": 0.5}}))
    assert len(rec["rounds"]) == 3
    for line in rec["rounds"]:
        assert line["accuracy"] is None and "krum" in line["refusal"]
    assert [r["round"] for r in rec["meta"]["refusals"]] == [1, 2, 3]


def test_a_round_where_every_client_drops_is_a_recorded_refusal(tmp_path):
    # The strategy never runs; the coordinator stops the run rather than aggregate nothing.
    rec = run(_cfg(_features(tmp_path), behaviors=[{"type": "dropout", "fraction": 1.0}]))
    line = rec["rounds"][0]
    assert line["accuracy"] is None and line["refusal"]
    assert line["participants"]["aggregated"] == 0
    assert rec["meta"]["stopped_early"] is True and rec["meta"]["refusals"][0]["round"] == 1


def test_per_client_fairness_is_opt_in_and_does_not_change_training(tmp_path):
    p = _features(tmp_path)
    off, on = run(_cfg(p)), run(_cfg(p, eval={"per_client_every": 2}))
    assert all(l["fairness"] is None for l in off["rounds"])
    got = [l["fairness"] for l in on["rounds"]]
    assert got[0] is None and got[2] is None and set(got[1]) == {"mean", "worst_10pct", "std"}
    assert on["meta"]["final_digest"] == off["meta"]["final_digest"]


def test_stop_request_writes_a_partial_record(tmp_path):
    seen = []
    rec = run(_cfg(_features(tmp_path)), on_round=seen.append, should_stop=lambda: len(seen) >= 1)
    assert len(rec["rounds"]) == 1 and rec["meta"]["stopped_early"] is True
    assert rec["meta"]["stop_reason"] == "stopped by request"


def test_a_round_with_missing_clients_waits_for_the_deadline(tmp_path):
    p = _features(tmp_path)
    drop = [{"type": "dropout", "fraction": 0.2}]
    lines = run(_cfg(p, behaviors=drop, federation={"network": {"round_deadline_s": 2.5}}))["rounds"]
    assert all(l["time"]["simulated_round_s"] == pytest.approx(2.5) for l in lines)
    assert all(l["time"]["simulated_round_s"] < 2.5 for l in run(_cfg(p, behaviors=drop))["rounds"])


def test_a_refused_round_reports_no_stale_utility(tmp_path):
    # Seed 5000: round 1 aggregates an on-time client; a later round has nobody on time and the
    # coordinator stops. That round must not inherit the previous round's accuracy.
    cfg = _cfg(_features(tmp_path), federation={"num_clients": 10, "clients_per_round": 2, "rounds": 6,
               "network": {"round_deadline_s": 1.0, "device_gflops": 1e-4}},
               behaviors=[{"type": "straggler", "fraction": 0.5, "slowdown": 1000.0}])
    lines = run(cfg)["rounds"]
    refused = [l for l in lines if l["refusal"] is not None]
    assert refused and lines[0]["accuracy"] is not None              # the scenario occurs
    for line in refused:
        assert all(line[k] is None for k in ("accuracy", "auc", "loss", "balanced_accuracy", "ece"))


def test_ipm_with_no_honest_client_in_the_cohort_is_recorded_not_raised(tmp_path):
    cfg = _cfg(_features(tmp_path), federation={"clients_per_round": 2, "rounds": 8},
               behaviors=[{"type": "ipm", "fraction": 0.5}])
    lines = run(cfg)["rounds"]
    notes = [l["attack_note"] for l in lines if l["attack_note"]]
    assert len(lines) == 8 and notes and "own" in notes[0]


def test_local_dp_server_clip_bounds_at_the_privatized_scale(tmp_path):
    import math
    rec = run(_cfg(_features(tmp_path), privacy={"mode": "local", "epsilon": 8.0}))
    z, c, d = rec["meta"]["noise_multiplier"], 0.5, rec["meta"]["workload"]["parameters"]
    assert rec["meta"]["server_clip_norm"] == pytest.approx(c + 2 * z * c * (math.sqrt(d) + 4))
    assert all(l["privacy"]["mean_shrink"] > 0.99 for l in rec["rounds"])   # honest uploads are not re-shrunk


def test_start_checks_fail_before_training_naming_the_field(tmp_path):
    from fedlearn.simulation.config import ConfigError
    p = _features(tmp_path)
    with pytest.raises(ConfigError) as e:
        run(_cfg(_features(tmp_path, trigger=True),
                 behaviors=[{"type": "backdoor", "fraction": 0.2, "target_label": 7}]))
    assert e.value.field == "behaviors[0].target_label"
    with pytest.raises(ConfigError) as e:
        run(_cfg(p, federation={"num_clients": 1000}))
    assert e.value.field == "federation.num_clients"
    tagged = tmp_path / "tagged.pt"
    data = torch.load(p, weights_only=True)
    data["dataset"] = "kermany"
    torch.save(data, tagged)
    with pytest.raises(ConfigError) as e:
        run(_cfg(str(tagged), workload={"dataset": "other"}))
    assert e.value.field == "workload.dataset"


def test_peak_memory_is_attributed_to_the_run_only_when_the_run_set_it(tmp_path, monkeypatch):
    from fedlearn.simulation import costs
    p = _features(tmp_path)
    readings = iter([100, 100])
    monkeypatch.setattr(costs, "peak_rss_bytes", lambda: next(readings))
    rec = run(_cfg(p))
    assert rec["meta"]["peak_rss_bytes"] == 100 and rec["meta"]["peak_rss_scope"] == "process"
    readings = iter([100, 250])
    rec = run(_cfg(p))
    assert rec["meta"]["peak_rss_bytes"] == 250 and rec["meta"]["peak_rss_scope"] == "run"


def test_fedprox_pulls_updates_toward_the_global(tmp_path):
    p = _features(tmp_path)
    avg = run(_cfg(p))["rounds"][0]["updates"]["honest_norm_mean"]
    prox = run(_cfg(p, client={"algorithm": "fedprox", "proximal_mu": 1.0}))["rounds"][0]["updates"]
    assert prox["honest_norm_mean"] < 0.9 * avg


def test_free_riders_do_no_training_and_upload_the_global(tmp_path):
    r0 = run(_cfg(_features(tmp_path), behaviors=[{"type": "free_rider", "fraction": 1.0}]))["rounds"][0]
    assert r0["compute"]["client_flops"] == 0 and r0["updates"]["attacker_norm_mean"] == 0.0


def test_an_injected_workload_is_trained_and_evaluated_instead_of_the_configured_one(tmp_path):
    from fedlearn.simulation.workloads import FeatureHeadWorkload
    from fedlearn.simulation.config import resolve
    cfg = _cfg(_features(tmp_path))
    default = run(cfg)
    injected = FeatureHeadWorkload(resolve(cfg))
    assert _stable(run(cfg, workload=injected)) == _stable(default)
    injected.test_y = 1 - injected.test_y  # an evaluation set only the injected workload has
    flipped = run(cfg, workload=injected)
    assert [l["accuracy"] for l in flipped["rounds"]] == pytest.approx([1 - l["accuracy"] for l in default["rounds"]])
