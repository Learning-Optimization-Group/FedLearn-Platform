# framework/tests/test_simulation_workloads.py
"""Workloads reproduce the capstone harness's data handling exactly and build the right heads."""
import numpy as np
import pytest
import torch

from fedlearn.estimators.params import trainable_state
from fedlearn.simulation.workloads import (
    FeatureHeadWorkload, LinearHead, MLPHead, balanced_indices, make_partition,
)


def _features(tmp_path, n_train=300, n_test=100, d=16, pos_frac=0.7, trigger=False):
    g = torch.Generator().manual_seed(0)

    def split(n):
        y = (torch.rand(n, generator=g) < pos_frac).long()
        return torch.randn(n, d, generator=g) + y[:, None].float() * 1.5, y

    tx, ty = split(n_train)
    vx, vy = split(n_test)
    rec = {"train_x": tx, "train_y": ty, "test_x": vx, "test_y": vy, "feat_dim": d, "n_classes": 2}
    if trigger:
        rec["trigger_train_x"], rec["trigger_test_x"] = tx + 5.0, vx + 5.0
    p = tmp_path / "f.pt"
    torch.save(rec, p)
    return p, rec


def _cfg(path, **over):
    from fedlearn.simulation.config import resolve
    return resolve({"seed": 5000, "workload": {"features_path": str(path), **over.pop("workload", {})},
                    "federation": {"num_clients": 10}, **over})


def test_balancing_matches_the_capstone_harness(tmp_path):
    from benchmarks.capstone_matched_study import balance_classes
    p, rec = _features(tmp_path)
    keep = balanced_indices(rec["train_y"], 5000)
    bx, by = balance_classes(rec["train_x"], rec["train_y"], 5000)
    assert torch.equal(rec["train_x"][keep], bx) and torch.equal(rec["train_y"][keep], by)


def test_partition_matches_the_capstone_harness():
    from benchmarks.capstone_matched_study import make_partition as harness
    labels = np.array([0, 1] * 100)
    for kind in ("iid", "dirichlet"):
        a, b = make_partition(labels, 10, 7, kind, 0.5), harness(labels, 10, 7, kind, 0.5)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_linear_head_init_matches_the_harness_head(tmp_path):
    from benchmarks.dp_on_head_xray import _Head
    from fedlearn.simulation.rng import torch_rng_scope
    p, _ = _features(tmp_path)
    wl = FeatureHeadWorkload(_cfg(p))
    with torch_rng_scope(5000):
        ref = _Head(16, 2)
    got = wl.build_model(5000)
    assert list(got.state_dict()) == ["head.weight", "head.bias"]
    assert all(torch.equal(a, b) for a, b in zip(got.state_dict().values(), ref.state_dict().values()))


def test_head_parameter_counts():
    assert sum(p.numel() for p in LinearHead(512, 2).parameters()) == 1026
    assert sum(p.numel() for p in MLPHead(512, 128, 2).parameters()) == 65922
    assert sum(p.numel() for p in MLPHead(512, 512, 2).parameters()) == 263682


def test_workload_balances_partitions_and_evaluates(tmp_path):
    p, _ = _features(tmp_path)
    wl = FeatureHeadWorkload(_cfg(p))
    sizes = [len(x) for x in wl.parts]
    assert sum(sizes) == len(wl.train_y) and int((wl.train_y == 0).sum()) == int((wl.train_y == 1).sum())
    model = wl.build_model(1)
    loss, metrics = wl.evaluate_fn()(1, trainable_state(model))
    assert set(metrics) >= {"accuracy", "auc", "sensitivity", "specificity", "balanced_accuracy", "ece"}
    assert "backdoor_success" not in metrics                # not applicable -> omitted, never None
    assert all(v is not None for v in metrics.values())


def test_per_client_accuracy_covers_every_client_and_averages_to_the_whole(tmp_path):
    p, _ = _features(tmp_path)
    wl = FeatureHeadWorkload(_cfg(p))
    accs = wl.per_client_accuracy(trainable_state(wl.build_model(3)))
    assert len(accs) == 10
    sizes = [len(x) for x in wl.parts]
    model = wl.build_model(3)
    with torch.no_grad():
        whole = (model(wl.train_x).argmax(1) == wl.train_y).float().mean().item()
    assert sum(a * s for a, s in zip(accs, sizes)) / sum(sizes) == pytest.approx(whole)


def test_triggered_rows_follow_the_same_balancing(tmp_path):
    p, rec = _features(tmp_path, trigger=True)
    wl = FeatureHeadWorkload(_cfg(p))
    assert wl.has_trigger
    x, y, idx = wl.client_data(0)
    assert torch.allclose(wl.trigger_rows(idx), x + 5.0)
