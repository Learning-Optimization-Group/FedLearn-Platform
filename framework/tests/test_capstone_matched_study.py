"""Capstone matched-study harness: the four cells must differ only in their two switches.

Q00 = no privacy noise, clipped mean; Q10 = noise, clipped mean; Q01 = no noise, trimmed mean;
Q11 = noise, trimmed mean. Clipping, partition, initialisation, local training, attack, and
weighting are shared, so a difference between cells is attributable to privacy and robustness.
Runs on small synthetic features so it is fast and never needs the X-ray data.
"""
import numpy as np
import pytest
import torch

from benchmarks.capstone_matched_study import balance_classes, make_partition, run_cell, run_quartet
from fedlearn.communication.safetensors_codec import save_safetensors

CELLS = ("Q00", "Q10", "Q01", "Q11")


def _features(n_train=400, n_test=200, d=16, shift=1.5, seed=0):
    g = torch.Generator().manual_seed(seed)

    def split(n):
        y = (torch.rand(n, generator=g) < 0.5).long()
        x = torch.randn(n, d, generator=g) + y[:, None].float() * shift
        return x, y

    tx, ty = split(n_train)
    vx, vy = split(n_test)
    return {"train_x": tx, "train_y": ty, "test_x": vx, "test_y": vy, "feat_dim": d, "n_classes": 2}


FEAT = _features()
BASE = dict(clients=10, rounds=3, local_epochs=2, lr=0.5, clip=0.5, trim_ratio=0.2,
            partition_seed=1, init_seed=1, attack_seed=1, noise_seed=1)


def _cell(cell, **over):
    return run_cell(FEAT, cell=cell, **{**BASE, "noise_multiplier": 2.0, **over})


# --------------------------------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------------------------------
def test_balance_classes_downsamples_the_majority_with_a_seed():
    x = torch.arange(100, dtype=torch.float32)[:, None]
    y = torch.tensor([0] * 30 + [1] * 70)
    bx, by = balance_classes(x, y, seed=3)
    assert int((by == 0).sum()) == 30 and int((by == 1).sum()) == 30
    assert set(bx[:, 0].long().tolist()) <= set(range(100))       # rows come from the input
    assert set(bx[by == 0, 0].long().tolist()) == set(range(30))  # the minority is kept whole
    again, _ = balance_classes(x, y, seed=3)
    other, _ = balance_classes(x, y, seed=4)
    assert torch.equal(bx, again)
    assert not torch.equal(bx, other)


@pytest.mark.parametrize("kind", ["iid", "dirichlet"])
def test_partition_is_disjoint_complete_and_seeded(kind):
    labels = np.array([0, 1] * 200)
    parts = make_partition(labels, clients=10, seed=5, kind=kind, alpha=0.5)
    flat = np.concatenate(parts)
    assert len(flat) == len(set(flat.tolist())) == 400            # disjoint: no sample twice
    assert sorted(flat.tolist()) == list(range(400))              # complete
    same = make_partition(labels, clients=10, seed=5, kind=kind, alpha=0.5)
    diff = make_partition(labels, clients=10, seed=6, kind=kind, alpha=0.5)
    assert all(np.array_equal(a, b) for a, b in zip(parts, same))
    assert not all(np.array_equal(a, b) for a, b in zip(parts, diff))


# --------------------------------------------------------------------------------------------------
# matched cells
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cell, want", [("Q00", 0.0), ("Q01", 0.0), ("Q10", 0.2), ("Q11", 1 / 3)])
def test_noise_std_is_z_times_the_cell_sensitivity(cell, want):
    # z=2, C=0.5, n=10: mean 2*2*0.5/10 = 0.2; trimmed (k=2) 2*2*0.5/6 = 1/3; no noise off-privacy.
    assert _cell(cell)["noise_std"] == pytest.approx(want, rel=1e-9)


def test_all_cells_see_identical_round0_client_uploads():
    attacked = dict(attack="ipm", attack_fraction=0.2, ipm_epsilon=4.0)
    digests = {c: _cell(c, **attacked)["round0_upload_digest"] for c in CELLS}
    assert len(set(digests.values())) == 1
    assert _cell("Q00", partition_seed=2, **attacked)["round0_upload_digest"] != digests["Q00"]


def test_noise_seed_moves_only_the_private_cells():
    for cell, private in (("Q00", False), ("Q01", False), ("Q10", True), ("Q11", True)):
        a = _cell(cell, noise_seed=1)["final_head_digest"]
        b = _cell(cell, noise_seed=2)["final_head_digest"]
        assert (a != b) == private, cell


def test_attackers_are_round_f_times_n_and_chosen_by_the_attack_seed():
    rec = _cell("Q00", attack="ipm", attack_fraction=0.3, ipm_epsilon=4.0)
    ids = rec["attacker_ids"]
    assert len(ids) == 3 and set(ids) <= set(range(10))
    assert _cell("Q00", attack="ipm", attack_fraction=0.3, ipm_epsilon=4.0)["attacker_ids"] == ids
    others = {tuple(_cell("Q00", attack="ipm", attack_fraction=0.3, ipm_epsilon=4.0,
                          attack_seed=s)["attacker_ids"]) for s in range(2, 7)}
    assert others != {tuple(ids)}


def test_the_attack_changes_the_trained_model():
    clean = _cell("Q00")["final_head_digest"]
    attacked = _cell("Q00", attack="ipm", attack_fraction=0.3, ipm_epsilon=8.0)
    assert attacked["final_head_digest"] != clean
    assert attacked["ipm_epsilon"] == 8.0


def test_the_clean_clipped_mean_learns_a_separable_task():
    rec = run_cell(_features(shift=3.0), cell="Q00", noise_multiplier=0.0, **BASE)
    assert rec["final_accuracy"] > 0.9


def test_a_cell_is_deterministic():
    a, b = _cell("Q11", attack="ipm", attack_fraction=0.2, ipm_epsilon=4.0), \
        _cell("Q11", attack="ipm", attack_fraction=0.2, ipm_epsilon=4.0)
    a.pop("wall_seconds"), b.pop("wall_seconds")
    assert a == b


def test_comm_bytes_count_one_download_and_one_upload_per_client_round():
    one = len(save_safetensors([("head.weight", np.zeros((2, 16), np.float32)),
                                ("head.bias", np.zeros(2, np.float32))]))
    assert _cell("Q00")["comm_bytes_total"] == 3 * 10 * 2 * one


def test_cli_writes_every_cell_condition_and_rep_with_provenance(tmp_path, monkeypatch):
    import json
    import sys

    from benchmarks import capstone_matched_study as cms

    feats = tmp_path / "features.pt"
    torch.save(FEAT, feats)
    out = tmp_path / "study.json"
    monkeypatch.setattr(sys, "argv", [
        "capstone_matched_study.py", "--features", str(feats), "--out", str(out),
        "--clients", "10", "--rounds", "2", "--local-epochs", "1", "--reps", "2",
        "--target-epsilon", "4", "--attack-fraction", "0.2", "--ipm-epsilon", "4",
    ])
    cms.main()
    data = json.loads(out.read_text())
    recs = data["records"]
    assert len(recs) == 4 * 2 * 2   # cells x {clean, attacked} x reps
    assert {(r["cell"], r["attack"] is not None, r["rep"]) for r in recs} == {
        (c, a, rep) for c in CELLS for a in (False, True) for rep in (0, 1)}
    meta = data["meta"]
    for key in ("torch_version", "platform", "features", "n_train", "n_test", "clients", "rounds",
                "target_epsilon", "delta", "clip", "trim_ratio", "ipm_epsilon", "attack_fraction",
                "partition", "seed_base", "reps", "total_seconds"):
        assert key in meta, key


def test_a_quartet_shares_one_accountant_noise_multiplier():
    recs = run_quartet(FEAT, target_epsilon=4.0, delta=1e-5, **BASE)
    assert sorted(r["cell"] for r in recs) == sorted(CELLS)
    zs = {r["noise_multiplier"] for r in recs}
    assert len(zs) == 1 and zs.pop() > 0
    for r in recs:
        assert 0 < r["accounted_epsilon"] <= 4.0 + 1e-9
        assert r["target_epsilon"] == 4.0
