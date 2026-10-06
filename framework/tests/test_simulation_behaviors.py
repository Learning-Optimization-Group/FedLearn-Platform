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
