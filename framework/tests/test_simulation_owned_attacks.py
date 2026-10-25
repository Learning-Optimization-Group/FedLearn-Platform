"""Synthetic checks for coalition-owned attack inputs and released-history wiring."""
from collections import OrderedDict
from copy import deepcopy
import inspect
import json

import pytest
import torch

from fedlearn.simulation.behaviors import BehaviorPlan
from fedlearn.simulation.config import ConfigError, resolve
from fedlearn.simulation.engine import run
from fedlearn.simulation.workloads import FeatureHeadWorkload


def config(kind="owned_pgd"):
    return {"workload": {"features_path": "unused", "balance": False},
            "federation": {"num_clients": 4, "rounds": 2},
            "client": {"local_epochs": 1, "lr": .1},
            "server": {"clip_norm": .5},
            "behaviors": [{"type": kind, "fraction": .5, **({"steps": 2, "restarts": 2} if kind == "owned_pgd" else {})}]}


def fixture(kind):
    resolved = resolve(config(kind))
    plan = BehaviorPlan(resolved["behaviors"], 4, 1000, 2)
    glob = OrderedDict([("head.weight", torch.zeros(2, 1)), ("head.bias", torch.zeros(2))])
    records = {c: {"honest_delta": OrderedDict([("head.weight", torch.tensor([[0.], [0.]])),
                                               ("head.bias", torch.tensor([3., 4.]))])}
               for c in range(4)}
    owned = {c: (torch.tensor([[-1.], [1.]]), torch.tensor([0, 1])) for c in plan.malicious_ids()}
    return plan, glob, records, owned


def invoke(plan, glob, records, owned, **kwargs):
    return plan.owned_upload_overrides(
        {c: records[c] for c in plan.malicious_ids()}, glob, 1, owned_data=owned,
        clip_norm=.5, total_submitted=4, released_delta=torch.zeros(4), **kwargs)


@pytest.mark.parametrize("kind", ["owned_pgd", "owned_sign_flip"])
def test_current_honest_records_cannot_affect_owned_uploads(kind):
    plan, glob, records, owned = fixture(kind)
    first = invoke(plan, glob, records, owned)
    for c in set(records) - plan.malicious_ids():
        records[c]["honest_delta"] = {"head.weight": torch.full((2, 1), float("nan"))}
    second = invoke(plan, glob, records, owned)
    assert set(first) == plan.malicious_ids()
    for c in first:
        assert all(torch.equal(first[c][k], second[c][k]) for k in glob)
        assert torch.equal(first[c]["head.weight"], first[min(first)]["head.weight"])
        assert torch.equal(first[c]["head.bias"], first[min(first)]["head.bias"])
        assert torch.cat([first[c][k].flatten() for k in glob]).norm() <= .5 + 1e-7
    params = inspect.signature(plan.owned_upload_overrides).parameters
    assert not any("selection" in key or "test" in key for key in params)


def test_sign_flip_normalizes_the_joint_own_mean_including_bias():
    plan, glob, records, owned = fixture("owned_sign_flip")
    output = invoke(plan, glob, records, owned)
    for upload in output.values():
        assert torch.equal(upload["head.weight"], torch.zeros(2, 1))
        assert torch.allclose(upload["head.bias"], torch.tensor([-.3, -.4]))
    for c in plan.malicious_ids():
        records[c]["honest_delta"]["head.bias"].zero_()
    assert all(torch.count_nonzero(v) == 0 for u in invoke(plan, glob, records, owned).values() for v in u.values())


def test_pgd_initialization_is_the_full_norm_baseline_and_budget_is_actual():
    plan, glob, records, owned = fixture("owned_pgd")
    invoke(plan, glob, records, owned)
    diag, = plan.last_owned_attack_diagnostics
    assert diag["initial_norm"] == pytest.approx(.5)
    assert diag["coalition_fraction"] == .5
    assert diag["gradient_evaluations"] == 4 and diag["loss_evaluations"] == 6
    assert diag["objective_after"] >= diag["objective_before"]
    assert diag["scope"] == "research-only; outside released-model DP claims"
    json.dumps(diag, allow_nan=False)
    assert not any("ids" in k or "delta" == k for k in diag)


@pytest.mark.parametrize("over", [
    {"workload": {"head": {"type": "mlp"}}},
    {"privacy": {"mode": "local"}},
    {"federation": {"clients_per_round": 3}},
    {"server": {"clip_norm": None}},
    {"federation": {"network": {"round_deadline_s": 1.}}},
    {"behaviors": [{"type": "owned_pgd", "fraction": .5}, {"type": "dropout", "fraction": 0.}]},
])
def test_owned_attacks_refuse_unsupported_configuration(over):
    cfg = config()
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    with pytest.raises(ConfigError):
        resolve(cfg)


@pytest.mark.parametrize("key,value", [("steps", True), ("steps", 1.5), ("restarts", 0), ("step_ratio", float("inf"))])
def test_search_budget_is_validated(key, value):
    cfg = config()
    cfg["behaviors"][0][key] = value
    with pytest.raises(ConfigError):
        resolve(cfg)


def workload(tmp_path, cfg):
    p = tmp_path / "synthetic.pt"
    x = torch.tensor([[-1.], [1.]] * 8)
    y = torch.tensor([0, 1] * 8)
    torch.save({"train_x": x, "train_y": y, "test_x": x[:4], "test_y": y[:4],
                "feat_dim": 1, "n_classes": 2}, p)
    cfg["workload"]["features_path"] = str(p)
    return FeatureHeadWorkload(resolve(cfg))


@pytest.mark.parametrize("kind", ["owned_pgd", "owned_sign_flip"])
@pytest.mark.parametrize("privacy", ["none", "central"])
def test_engine_metadata_is_opt_in_deterministic_and_history_uses_actual_release(tmp_path, kind, privacy, monkeypatch):
    from fedlearn.simulation import behaviors
    cfg = config(kind)
    cfg["privacy"] = {"mode": privacy, "epsilon": 8.}
    wl = workload(tmp_path, cfg)
    original = behaviors.BehaviorPlan.owned_upload_overrides
    calls = []

    def capture(self, records, glob, round_num, **kwargs):
        assert set(records) == self.malicious_ids()
        assert set(kwargs["owned_data"]) == self.malicious_ids()
        calls.append((deepcopy(glob), kwargs["released_delta"].clone()))
        return original(self, records, glob, round_num, **kwargs)

    monkeypatch.setattr(behaviors.BehaviorPlan, "owned_upload_overrides", capture)
    a = run(cfg, workload=wl)
    assert len(calls) == 2
    assert torch.count_nonzero(calls[0][1]) == 0
    actual = torch.cat([(calls[1][0][k] - calls[0][0][k]).flatten() for k in calls[0][0]])
    assert torch.equal(calls[1][1], actual)
    assert actual.norm() > 0
    assert all(l["participants"]["aggregated"] == 4 and l["refusal"] is None for l in a["rounds"])
    diag = a["rounds"][1]["owned_attack_diagnostics"]
    assert diag[0]["history_release_round"] == 1
    b = run(cfg, workload=wl)
    assert a["meta"]["final_digest"] == b["meta"]["final_digest"]
    assert a["rounds"][1]["owned_attack_diagnostics"] == b["rounds"][1]["owned_attack_diagnostics"]
    cfg["behaviors"] = []
    assert all("owned_attack_diagnostics" not in line for line in run(cfg, workload=wl)["rounds"])


@pytest.mark.parametrize("kind", ["owned_pgd", "owned_sign_flip"])
def test_engine_honest_training_and_selection_labels_cannot_change_first_owned_upload(tmp_path, kind, monkeypatch):
    cfg = config(kind)
    cfg["federation"]["rounds"] = 1
    wl = workload(tmp_path, cfg)
    plan = BehaviorPlan(resolve(cfg)["behaviors"], 4, 1000, 2)
    snapshots = []
    original = BehaviorPlan.owned_upload_overrides

    def capture(self, records, glob, round_num, **kwargs):
        for cid, (x, y) in kwargs["owned_data"].items():
            idx = torch.as_tensor(wl.parts[cid])
            assert torch.equal(x, wl.train_x[idx]) and torch.equal(y, wl.train_y[idx])
        output = original(self, records, glob, round_num, **kwargs)
        snapshots.append(deepcopy(output))
        return output

    monkeypatch.setattr(BehaviorPlan, "owned_upload_overrides", capture)
    run(cfg, workload=wl)
    for c in set(range(4)) - plan.malicious_ids():
        idx = torch.as_tensor(wl.parts[c])
        wl.train_x[idx] += 100.
        wl.train_y[idx] = 1 - wl.train_y[idx]
    wl.test_y = 1 - wl.test_y
    run(cfg, workload=wl)
    for cid in snapshots[0]:
        assert all(torch.equal(snapshots[0][cid][k], snapshots[1][cid][k]) for k in snapshots[0][cid])


def test_pgd_and_sign_flip_coalitions_do_not_share_private_updates_or_examples():
    cfg = config()
    cfg["behaviors"] = [{"type": "owned_pgd", "fraction": .25, "steps": 1, "restarts": 1},
                        {"type": "owned_sign_flip", "fraction": .25}]
    plan = BehaviorPlan(resolve(cfg)["behaviors"], 4, 1000, 2)
    _, glob, records, _ = fixture("owned_pgd")
    owned = {c: (torch.tensor([[-1.], [1.]]), torch.tensor([0, 1])) for c in plan.owned_ids()}
    first = invoke(plan, glob, records, owned)
    other = next(c for c, b in plan.malicious.items() if b["type"] == "owned_sign_flip")
    pgd = next(c for c, b in plan.malicious.items() if b["type"] == "owned_pgd")
    records[other]["honest_delta"]["head.bias"] *= -100.
    owned[other] = (torch.full((2, 1), 100.), torch.tensor([1, 0]))
    second = invoke(plan, glob, records, owned)
    assert all(torch.equal(first[pgd][k], second[pgd][k]) for k in glob)
    assert not torch.equal(first[other]["head.bias"], second[other]["head.bias"])
    assert [d["coalition_fraction"] for d in plan.last_owned_attack_diagnostics] == [.25, .25]


def test_owned_attack_api_refuses_honest_or_missing_views_and_incomplete_cohort():
    plan, glob, records, owned = fixture("owned_pgd")
    with pytest.raises(ValueError, match="malicious"):
        plan.owned_upload_overrides(records, glob, 1, owned_data=owned, clip_norm=.5,
                                    total_submitted=4, released_delta=torch.zeros(4))
    with pytest.raises(ValueError, match="complete"):
        plan.owned_upload_overrides({c: records[c] for c in plan.owned_ids()}, glob, 1,
                                    owned_data=owned, clip_norm=.5, total_submitted=3,
                                    released_delta=torch.zeros(4))


def test_refused_aggregate_and_previews_do_not_advance_public_history(tmp_path, monkeypatch):
    from fedlearn.server.composed_strategy import ComposedStrategy
    cfg = config()
    cfg["federation"]["rounds"] = 3
    cfg["privacy"] = {"mode": "central", "epsilon": 8.}
    wl = workload(tmp_path, cfg)
    aggregate = ComposedStrategy.aggregate_fit

    def refuse_second(self, server_round, results):
        if server_round == 2:
            self.last_round_stats = {"failed": True, "message": "synthetic reducer refusal", "aggregated": 4}
            return None
        return aggregate(self, server_round, results)

    monkeypatch.setattr(ComposedStrategy, "aggregate_fit", refuse_second)
    result = run(cfg, workload=wl)
    a, refused, b = result["rounds"]
    assert a["refusal"] is None and refused["refusal"] is not None and b["refusal"] is None
    assert a["owned_attack_diagnostics"][0]["history_release_round"] == 0
    assert refused["owned_attack_diagnostics"][0]["history_release_round"] == 1
    assert b["owned_attack_diagnostics"][0]["history_release_round"] == 1
    assert b["owned_attack_diagnostics"][0]["history_norm"] == refused["owned_attack_diagnostics"][0]["history_norm"]


def test_owned_training_keeps_the_baseline_global_warm_start(tmp_path, monkeypatch):
    from fedlearn.simulation.engine import HeadClient
    cfg = config()
    cfg["federation"]["rounds"] = 1
    wl = workload(tmp_path, cfg)
    original = HeadClient.fit
    starts = []

    def capture(self, global_params, client_cfg):
        starts.append(deepcopy(global_params))
        return original(self, global_params, client_cfg)

    monkeypatch.setattr(HeadClient, "fit", capture)
    run(cfg, workload=wl)
    cfg["behaviors"] = []
    run(cfg, workload=wl)
    assert len(starts) == 8
    assert all(torch.equal(starts[0][k], p[k]) for p in starts for k in starts[0])


def test_engine_refuses_actual_incomplete_collection_before_attack(tmp_path, monkeypatch):
    from fedlearn.simulation.federation import SimulatedFederation
    cfg = config()
    wl = workload(tmp_path, cfg)
    monkeypatch.setattr(SimulatedFederation, "_plan_round", lambda self, round_num: ([0, 1, 2, 3], [0], set()))
    with pytest.raises(ValueError, match="complete"):
        run(cfg, workload=wl)


def test_engine_refuses_nonbinary_workload_before_training(tmp_path):
    cfg = config()
    wl = workload(tmp_path, cfg)
    wl.n_classes = 3
    with pytest.raises(ConfigError, match="binary"):
        run(cfg, workload=wl)


@pytest.mark.parametrize("kind", ["owned_pgd", "owned_sign_flip"])
@pytest.mark.parametrize("flag", ["privatize", "match_norm"])
def test_owned_schema_refuses_local_privacy_flags_instead_of_ignoring_them(kind, flag):
    cfg = config(kind)
    cfg["behaviors"][0][flag] = True
    with pytest.raises(ConfigError, match="unknown key"):
        resolve(cfg)
