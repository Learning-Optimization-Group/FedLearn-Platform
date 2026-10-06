"""The engine reproduces capstone_matched_study bit for bit: per-round accuracy, the round-0
upload digest and the final head digest, for all four cells, clean and attacked."""
import pytest
import torch

from benchmarks.capstone_matched_study import run_quartet
from fedlearn.simulation.engine import run

CELLS = {"Q00": ("none", "mean"), "Q10": ("central", "mean"), "Q01": ("none", "trimmed_mean"),
         "Q11": ("central", "trimmed_mean")}


def _features(tmp_path):
    g = torch.Generator().manual_seed(0)

    def split(n):
        y = (torch.rand(n, generator=g) < 0.5).long()
        return torch.randn(n, 16, generator=g) + y[:, None].float() * 1.5, y

    tx, ty = split(400)
    vx, vy = split(200)
    feats = {"train_x": tx, "train_y": ty, "test_x": vx, "test_y": vy, "feat_dim": 16, "n_classes": 2}
    p = tmp_path / "f.pt"
    torch.save(feats, p)
    return feats, str(p)


@pytest.mark.parametrize("attacked", [False, True])
def test_engine_equals_harness(tmp_path, attacked):
    feats, path = _features(tmp_path)
    seeds = dict(partition_seed=5000, init_seed=5000, attack_seed=6000, noise_seed=7000)
    attack = dict(attack="ipm", attack_fraction=0.2, ipm_epsilon=4.0) if attacked else {}
    harness = {r["cell"]: r for r in run_quartet(
        feats, target_epsilon=4.0, delta=1e-5, rounds=4, clients=10, local_epochs=2, lr=0.5,
        clip=0.5, trim_ratio=0.2, **seeds, **attack)}
    for cell, (mode, rule) in CELLS.items():
        cfg = {"seed": 5000,
               "seeds": {"partition": 5000, "init": 5000, "client": 5000, "attack": 6000, "noise": 7000},
               "workload": {"features_path": path, "balance": False},
               "federation": {"num_clients": 10, "rounds": 4},
               "client": {"local_epochs": 2, "lr": 0.5},
               "server": {"clip_norm": 0.5, "rule": rule, "rule_params": {"trim_ratio": 0.2}},
               "privacy": {"mode": mode, "epsilon": 4.0, "delta": 1e-5},
               "behaviors": [{"type": "ipm", "fraction": 0.2, "scale": 4.0}] if attacked else []}
        rec = run(cfg)
        h = harness[cell]
        assert [round(l["accuracy"], 6) for l in rec["rounds"]] == h["per_round_accuracy"], cell
        assert rec["meta"]["round0_upload_digest"] == h["round0_upload_digest"], cell
        assert rec["meta"]["final_digest"] == h["final_head_digest"], cell
