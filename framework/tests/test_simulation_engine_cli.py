# framework/tests/test_simulation_engine_cli.py
"""Matrix runner and CLI: every run written, failures recorded, streaming one line per round."""
import json
import os
import subprocess
import sys

import torch

from fedlearn.simulation import engine


def _features(tmp_path):
    g = torch.Generator().manual_seed(0)
    y = (torch.rand(200, generator=g) < 0.5).long()
    x = torch.randn(200, 8, generator=g) + y[:, None].float()
    p = tmp_path / "f.pt"
    torch.save({"train_x": x, "train_y": y, "test_x": x[:50], "test_y": y[:50], "feat_dim": 8, "n_classes": 2}, p)
    return str(p)


def _cfg(path):
    return {"name": "m", "seed": 1, "workload": {"features_path": path},
            "federation": {"num_clients": 5, "rounds": 2}, "client": {"local_epochs": 1},
            "matrix": {"server.rule": ["mean", "trimmed_mean"], "repeats": 2}}


def test_matrix_writes_every_run_and_an_index(tmp_path):
    out = tmp_path / "out"
    engine.run_matrix(_cfg(_features(tmp_path)), str(out))
    index = json.loads((out / "index.json").read_text())
    assert len(index["runs"]) == 4 and all(r["status"] == "ok" for r in index["runs"])
    assert all((out / f"{r['name']}.json").exists() for r in index["runs"])


def test_matrix_records_a_failed_cell_and_continues(tmp_path, monkeypatch):
    real = engine.run

    def flaky(cfg, **kw):
        if cfg["server"]["rule"] == "trimmed_mean":
            raise RuntimeError("boom")
        return real(cfg, **kw)

    monkeypatch.setattr(engine, "run", flaky)
    out = tmp_path / "out"
    engine.run_matrix(_cfg(_features(tmp_path)), str(out))
    index = json.loads((out / "index.json").read_text())
    failed = [r for r in index["runs"] if r["status"] == "failed"]
    assert len(failed) == 2 and len(index["runs"]) == 4
    rec = json.loads((out / f"{failed[0]['name']}.json").read_text())
    assert "boom" in rec["error"] and "Traceback" in rec["traceback"]


def test_cli_streams_one_line_per_round(tmp_path):
    cfg_path = tmp_path / "c.json"
    cfg = _cfg(_features(tmp_path))
    cfg["matrix"] = {"repeats": 1}
    cfg_path.write_text(json.dumps(cfg))
    fw = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run([sys.executable, "-m", "fedlearn.simulation.engine", str(cfg_path),
                           "--out", str(tmp_path / "o"), "--stream"], capture_output=True, text=True,
                          cwd=fw, env={**os.environ, "PYTHONPATH": os.path.join(fw, "src")}, check=True)
    lines = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith("{")]
    assert [l["round"] for l in lines] == [1, 2]                 # the coordinator numbers rounds from 1
    assert all(l["run"] == "m__r0" for l in lines)
