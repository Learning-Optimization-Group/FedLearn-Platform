"""Bad-client behaviours: seeded persistent assignment and effects equal to the attack harness."""
from collections import OrderedDict

import numpy as np
import pytest
import torch

from fedlearn.simulation.behaviors import BehaviorPlan
from fedlearn.simulation.config import resolve

from benchmarks import robust_aggregation_attack as raa


def _plan(behaviors, n=20, seed=11, classes=2):
    b = resolve({"workload": {"features_path": "x"}, "behaviors": behaviors})["behaviors"]
    return BehaviorPlan(b, n, seed, classes)


def test_first_malicious_set_equals_the_harness_draw():
    p = _plan([{"type": "ipm", "fraction": 0.2}], n=80, seed=6000)
    want = {int(i) for i in np.random.default_rng(6000).choice(80, 16, replace=False)}
    assert p.malicious_ids() == want


def test_malicious_sets_are_disjoint_and_faulty_may_overlap():
    p = _plan([{"type": "ipm", "fraction": 0.2}, {"type": "label_flip", "fraction": 0.2},
               {"type": "label_noise", "fraction": 0.5}], n=50)
    ipm = {c for c, b in p.malicious.items() if b["type"] == "ipm"}
    flip = {c for c, b in p.malicious.items() if b["type"] == "label_flip"}
    assert len(ipm) == 10 and len(flip) == 10 and not ipm & flip
    assert len([c for c, bs in p.faulty.items() if any(b["type"] == "label_noise" for b in bs)]) == 25
    # two halves must cover everyone exactly once (an overlap would leave a client unassigned)
    full = _plan([{"type": "ipm", "fraction": 0.5}, {"type": "alie", "fraction": 0.5}], n=20)
    assert len(full.malicious) == 20


def test_assignment_is_seeded():
    a = _plan([{"type": "alie", "fraction": 0.3}], seed=1).malicious_ids()
    assert a == _plan([{"type": "alie", "fraction": 0.3}], seed=1).malicious_ids()
    assert a != _plan([{"type": "alie", "fraction": 0.3}], seed=2).malicious_ids()


def _records(n, honest_scale=1.0):
    g = torch.Generator().manual_seed(0)
    glob = OrderedDict(w=torch.zeros(4), b=torch.zeros(1))
    recs = {}
    for c in range(n):
        d = OrderedDict((k, honest_scale * torch.randn(v.shape, generator=g)) for k, v in glob.items())
        recs[c] = {"client_state": OrderedDict((k, glob[k] + d[k]) for k in glob), "honest_delta": d, "n_examples": 5}
    return recs, glob


@pytest.mark.parametrize("kind, kw", [("ipm", {"ipm_epsilon": 4.0}), ("alie", {"alie_z": 20.0}),
                                      ("sign_flip_scale", {"attack_scale_signed": -10.0})])
def test_upload_attacks_equal_the_harness(kind, kw):
    behaviors = {"ipm": [{"type": "ipm", "fraction": 0.3, "scale": 4.0}],
                 "alie": [{"type": "alie", "fraction": 0.3, "z": 20.0}],
                 "sign_flip_scale": [{"type": "sign_flip_scale", "fraction": 0.3, "scale": -10.0}]}[kind]
    p = _plan(behaviors, n=10)
    recs, glob = _records(10)
    got = p.upload_overrides(recs, glob, round_num=0)
    want = raa._apply_attack_to_round(
        attack=kind, attacker_ids=p.malicious_ids(), client_records=recs, global_params=glob,
        attack_scale_signed=kw.get("attack_scale_signed"), ipm_epsilon=kw.get("ipm_epsilon"),
        alie_z=kw.get("alie_z"))
    assert set(got) == set(want)
    assert all(torch.equal(got[c][k], want[c][k]) for c in got for k in glob)


def test_label_flip_and_free_rider():
    p = _plan([{"type": "label_flip", "fraction": 0.5}, {"type": "free_rider", "fraction": 0.5}], n=2)
    flip = next(c for c, b in p.malicious.items() if b["type"] == "label_flip")
    rider = next(c for c, b in p.malicious.items() if b["type"] == "free_rider")
    x, y = torch.zeros(4, 3), torch.tensor([0, 1, 1, 0])
    assert torch.equal(p.training_data(flip, x, y, None, 1)[1], torch.tensor([1, 0, 0, 1]))
    assert p.training_data(rider, x, y, None, 1)[2] is False
    recs, glob = _records(2)
    assert torch.equal(p.upload_overrides(recs, glob, 0)[rider]["w"], glob["w"])


def test_label_noise_rate_and_backdoor_fraction():
    p = _plan([{"type": "label_noise", "fraction": 1.0, "flip_prob": 0.3}], n=1)
    y = torch.zeros(20_000, dtype=torch.long)
    _, noisy, _ = p.training_data(0, torch.zeros(20_000, 2), y, None, 5)
    assert noisy.float().mean().item() == pytest.approx(0.3, abs=0.02)
    q = _plan([{"type": "backdoor", "fraction": 1.0, "target_label": 1, "poison_fraction": 0.25}], n=1)
    x, trig = torch.zeros(100, 2), torch.ones(100, 2)
    bx, by, _ = q.training_data(0, x, torch.zeros(100, dtype=torch.long), trig, 5)
    assert int(bx[:, 0].sum()) == 25 and int(by.sum()) == 25
    assert q.needs_trigger() and q.backdoor_target() == 1
    with pytest.raises(ValueError):
        q.training_data(0, x, torch.zeros(100, dtype=torch.long), None, 5)


def test_stragglers_and_dropout():
    p = _plan([{"type": "straggler", "fraction": 0.5, "slowdown": 3.0}, {"type": "dropout", "fraction": 0.1}], n=10)
    slow = [c for c in range(10) if p.slowdown(c) == 3.0]
    assert len(slow) == 5 and p.dropout_fraction == pytest.approx(0.1)


def test_sign_flip_scale_uses_the_signed_scale():
    recs, glob = _records(10)
    flip = _plan([{"type": "sign_flip_scale", "fraction": 0.3}], n=10)                 # default -10 flips
    same = _plan([{"type": "sign_flip_scale", "fraction": 0.3, "scale": 10.0}], n=10)  # positive: same direction
    for plan, signed, kind in ((flip, -10.0, "sign_flip_scale"), (same, 10.0, "same_dir_scale")):
        got = plan.upload_overrides(recs, glob, 0)
        want = raa._apply_attack_to_round(attack=kind, attacker_ids=plan.malicious_ids(), client_records=recs,
                                          global_params=glob, attack_scale_signed=signed, ipm_epsilon=None,
                                          alie_z=None)
        assert set(got) == set(want) and all(torch.equal(got[c][k], want[c][k]) for c in got for k in glob)


def test_ipm_with_no_honest_client_uses_the_attackers_own_deltas():
    p = _plan([{"type": "ipm", "fraction": 1.0, "scale": 4.0}], n=4)
    recs, glob = _records(4)
    got = p.upload_overrides(recs, glob, 0)
    mean_ = {k: torch.stack([recs[c]["honest_delta"][k] for c in range(4)], dim=0).mean(dim=0) for k in glob}
    assert set(got) == {0, 1, 2, 3}
    assert all(torch.equal(got[c][k], glob[k].float() + (-4.0 * mean_[k])) for c in got for k in glob)
    assert "own" in p.last_attack_note


def _flat(state, glob):
    return torch.cat([(state[k] - glob[k]).reshape(-1) for k in glob])


@pytest.mark.parametrize("perturbation", ["std", "unit_vec", "sign"])
def test_min_max_sits_on_the_honest_diameter(perturbation):
    p = _plan([{"type": "min_max", "fraction": 0.25, "perturbation": perturbation}], n=20)
    recs, glob = _records(20)
    out = p.upload_overrides(recs, glob, 1)
    attackers = sorted(p.malicious_ids())
    honest = torch.stack([_flat(recs[c]["client_state"], glob) for c in sorted(recs) if c not in p.malicious])
    upload = _flat(out[attackers[0]], glob)
    diameter = float(torch.cdist(honest, honest).max())
    worst = float(torch.linalg.vector_norm(honest - upload, dim=1).max())
    assert worst <= diameter * (1 + 1e-6)
    assert worst == pytest.approx(diameter, rel=1e-4)  # gamma is as large as the constraint allows
    assert all(torch.equal(_flat(out[c], glob), upload) for c in attackers)
    mean = honest.mean(dim=0)
    assert float(torch.dot(upload - mean, mean)) < 0 or perturbation == "std"  # pushes against the mean


def test_fang_trim_uploads_lie_past_the_honest_extremes_against_the_honest_direction():
    p = _plan([{"type": "fang_trim", "fraction": 0.25}], n=20)
    recs, glob = _records(20)
    out = p.upload_overrides(recs, glob, 3)
    honest = torch.stack([_flat(recs[c]["client_state"], glob) for c in sorted(recs) if c not in p.malicious])
    mean, lo, hi = honest.mean(dim=0), honest.min(dim=0).values, honest.max(dim=0).values
    for c in p.malicious_ids():
        v = _flat(out[c], glob)
        assert torch.all(torch.where(mean > 0, v <= lo, v >= hi))
        assert torch.all(torch.where(mean > 0, v >= torch.where(lo > 0, lo / 2, lo * 2),
                                     v <= torch.where(hi > 0, hi * 2, hi / 2)))
    again = p.upload_overrides(recs, glob, 3)
    assert all(torch.equal(_flat(out[c], glob), _flat(again[c], glob)) for c in p.malicious_ids())


@pytest.mark.parametrize("behavior", [{"type": "min_max", "fraction": 0.1, "perturbation": "cos"},
                                      {"type": "fang_trim", "fraction": 0.1, "b": 1.0}])
def test_new_attack_parameters_are_validated(behavior):
    from fedlearn.simulation.config import ConfigError
    with pytest.raises(ConfigError):
        _plan([behavior])


def test_canaries_upload_their_own_fixed_unit_direction_scaled_to_norm():
    from fedlearn.simulation.behaviors import canary_direction
    p = _plan([{"type": "canary", "fraction": 0.25, "norm": 0.05}], n=20, seed=9)
    recs, glob = _records(20)
    first, later = p.upload_overrides(recs, glob, 1), p.upload_overrides(recs, glob, 7)
    canaries = sorted(p.malicious_ids())
    assert len(canaries) == 5 and set(first) == set(canaries)
    for c in canaries:
        u = _flat(canary_direction(9, c, glob), {k: torch.zeros_like(v) for k, v in glob.items()})
        assert float(torch.linalg.vector_norm(u)) == pytest.approx(1.0, rel=1e-6)
        assert torch.allclose(_flat(first[c], glob), 0.05 * u)
        assert torch.equal(_flat(first[c], glob), _flat(later[c], glob))  # same direction every round
    a, b = (_flat(first[c], glob) for c in canaries[:2])
    assert not torch.allclose(a, b)


def test_canaries_do_not_train():
    p = _plan([{"type": "canary", "fraction": 0.5}], n=4)
    c = next(iter(p.malicious_ids()))
    assert p.training_data(c, torch.zeros(3, 2), torch.zeros(3, dtype=torch.long), None, 0)[2] is False
