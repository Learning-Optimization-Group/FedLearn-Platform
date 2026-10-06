"""1,000 clients x 2 rounds completes with bounded memory (run with -m slow)."""
import pytest
import torch

from fedlearn.simulation.engine import run


@pytest.mark.slow
def test_thousand_clients_two_rounds(tmp_path):
    g = torch.Generator().manual_seed(0)
    y = (torch.rand(50_000, generator=g) < 0.5).long()
    x = torch.randn(50_000, 32, generator=g) + y[:, None].float()
    p = tmp_path / "f.pt"
    torch.save({"train_x": x, "train_y": y, "test_x": x[:2000], "test_y": y[:2000], "feat_dim": 32, "n_classes": 2}, p)
    rec = run({"seed": 1, "workload": {"features_path": str(p)},
               "federation": {"num_clients": 1000, "rounds": 2}, "client": {"local_epochs": 1},
               "privacy": {"mode": "central", "epsilon": 4.0},
               "behaviors": [{"type": "ipm", "fraction": 0.1}, {"type": "dropout", "fraction": 0.05}]})
    assert len(rec["rounds"]) == 2 and rec["meta"]["peak_rss_bytes"] < 4 * 1024 ** 3
