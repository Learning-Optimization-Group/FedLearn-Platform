# framework/tests/test_simulation_engine.py
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


def test_a_run_is_deterministic_except_measured_time(tmp_path):
    p = _features(tmp_path)
    cfg = _cfg(p, privacy={"mode": "local", "epsilon": 8.0}, behaviors=[{"type": "label_noise", "fraction": 0.3}])
    a, b = run(cfg), run(cfg)
    assert _stable(a) == _stable(b) and a["meta"]["final_digest"] == b["meta"]["final_digest"]
    assert run({**cfg, "seed": 6000})["meta"]["final_digest"] != a["meta"]["final_digest"]


def test_partial_round_is_recorded_not_zeroed(tmp_path):
    # 1e-4 GFLOPS: an ordinary client needs about 0.2 s simulated, a 1000x straggler about 100 s.
    cfg = _cfg(_features(tmp_path), behaviors=[{"type": "straggler", "fraction": 0.3, "slowdown": 1000.0}],
               federation={"network": {"round_deadline_s": 1.0, "device_gflops": 1e-4}})
    r0 = run(cfg)["rounds"][0]
    assert r0["participants"] == {"sampled": 10, "dropped": 0, "straggled": 3, "aggregated": 7}
    assert r0["forced"] is True and r0["accuracy"] is not None and r0["refusal"] is None
    assert r0["comm"]["up_bytes"] == r0["comm"]["down_bytes"]      # late stragglers' bytes still count
    assert r0["time"]["simulated_round_s"] < 1.0                    # the clock stops at the deadline's survivors


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
