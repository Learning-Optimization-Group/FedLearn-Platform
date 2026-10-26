"""Generated-only selection packs and strict canonical server receipts."""

from collections import OrderedDict
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

SETTINGS = dict(
    lr=0.03, local_epochs=7, batch_size=256, clip_norm=0.05, optimizer="sgd", device="cpu", rounds=5
)


@pytest.fixture(autouse=True)
def runtime_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(tmp_path):
    x = np.zeros((4, 128), dtype=np.float32)
    x[:, 0] = [-2.0, -1.0, 1.0, 2.0]
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    np.save(tmp_path / "selection_x.npy", x)
    np.save(tmp_path / "selection_y.npy", y)
    ids = [f"generated:client-{i:05d}" for i in range(3)]
    plan = dict(
        schema="fedlearn.effusion_server.v1",
        synthetic=True,
        run_id="generated",
        method="projected128",
        width=128,
        num_clients=3,
        seed=14001,
        rounds=5,
        settings=dict(SETTINGS),
        client_ids=ids,
        client_hosts=dict(zip(ids, ("mac", "g14", "orin"))),
        bind_address="127.0.0.1:50051",
        selection=dict(
            x_path=str(tmp_path / "selection_x.npy"),
            y_path=str(tmp_path / "selection_y.npy"),
            x_sha256=sha(tmp_path / "selection_x.npy"),
            y_sha256=sha(tmp_path / "selection_y.npy"),
            split="selection",
            source_bundle_report_sha256="a" * 64,
        ),
    )
    path = tmp_path / "server.json"
    path.write_text(json.dumps(plan))
    return path, plan


def module():
    return importlib.import_module("effusion_server")


def load(path):
    return module().load_server_plan(path, sha(path))


@pytest.mark.parametrize(
    "damage",
    [
        "pin",
        "schema",
        "extra",
        "method",
        "width",
        "rounds",
        "settings",
        "optimizer",
        "synthetic_type",
        "medical_source",
        "duplicate",
        "order",
        "host",
        "missing_host",
        "public_bind",
        "bad_port",
        "split",
        "selection_extra",
        "payload_hash",
        "float64",
        "nonfinite",
        "labels",
        "label_dtype",
        "shape",
        "training_path",
        "symlink",
    ],
)
def test_server_plan_rejects_drift_and_nonselection_inputs(tmp_path, damage):
    path, p = pack(tmp_path)
    if damage == "pin":
        with pytest.raises(ValueError):
            module().load_server_plan(path, "0" * 64)
        return
    if damage == "schema":
        p["schema"] = "other"
    elif damage == "extra":
        p["attacks"] = []
    elif damage == "method":
        p["method"] = "mean512"
    elif damage == "width":
        p["width"] = True
    elif damage == "rounds":
        p["rounds"] = 20
    elif damage == "settings":
        p["settings"]["local_epochs"] = 1
    elif damage == "optimizer":
        p["settings"]["optimizer"] = "adam"
    elif damage == "synthetic_type":
        p["synthetic"] = 1
    elif damage == "medical_source":
        p["synthetic"] = False
    elif damage == "duplicate":
        p["client_ids"][1] = p["client_ids"][0]
    elif damage == "order":
        p["client_ids"].reverse()
    elif damage == "host":
        p["client_hosts"][p["client_ids"][0]] = "somewhere"
    elif damage == "missing_host":
        p["client_hosts"].pop(p["client_ids"][0])
    elif damage == "public_bind":
        p["bind_address"] = "0.0.0.0:50051"
    elif damage == "bad_port":
        p["bind_address"] = "localhost:99999"
    elif damage == "split":
        p["selection"]["split"] = "holdout"
    elif damage == "selection_extra":
        p["selection"]["test_path"] = "test.npy"
    elif damage == "payload_hash":
        p["selection"]["x_sha256"] = "0" * 64
    elif damage in ("float64", "nonfinite", "shape"):
        x = np.load(p["selection"]["x_path"])
        if damage == "float64":
            x = x.astype(np.float64)
        elif damage == "nonfinite":
            x[0, 0] = np.nan
        else:
            x = x[:, :64]
        np.save(p["selection"]["x_path"], x)
        p["selection"]["x_sha256"] = sha(Path(p["selection"]["x_path"]))
    elif damage in ("labels", "label_dtype"):
        y = (
            np.array([0, 0, 1, 2], dtype=np.int64)
            if damage == "labels"
            else np.zeros(4, np.float32)
        )
        np.save(p["selection"]["y_path"], y)
        p["selection"]["y_sha256"] = sha(Path(p["selection"]["y_path"]))
    elif damage == "training_path":
        old = Path(p["selection"]["x_path"])
        new = old.with_name("train_x.npy")
        old.rename(new)
        p["selection"]["x_path"] = str(new)
    elif damage == "symlink":
        old = Path(p["selection"]["x_path"])
        old.rename(old.with_name("other.npy"))
        old.symlink_to(old.with_name("other.npy"))
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        load(path)


def test_uniform_clipped_mean_records_actual_full_cohort_and_config(tmp_path):
    path, p = pack(tmp_path)
    strategy = module().EffusionStrategy(load(path))
    assert strategy.allowed_client_ids == tuple(p["client_ids"])
    assert strategy.min_fit_clients == strategy.clients_per_round == 3
    assert strategy.get_client_config() == dict(
        learning_rate="0.03",
        local_epochs="7",
        batch_size="256",
        optimizer="sgd",
        run_id="generated",
        num_clients="3",
        seed="14001",
        width="128",
        rounds="5",
    )
    assert all(torch.count_nonzero(v) == 0 for v in strategy.initial_parameters.values())
    results = []
    for i, n in enumerate((1, 10, 100)):
        params = OrderedDict((k, v.clone()) for k, v in strategy.initial_parameters.items())
        params["head.bias"][0] = (0.01, 0.02, 0.03)[i]
        results.append((p["client_ids"][i], params, n))
    output = strategy.aggregate_fit(1, list(reversed(results)))
    assert output["head.bias"][0].item() == pytest.approx(0.02)
    receipt = strategy.receipts[0]
    assert receipt["client_ids"] == p["client_ids"]
    assert receipt["num_examples"] == [1, 10, 100]
    assert receipt["client_count"] == 3
    assert len(receipt["head_sha256"]) == 64
    assert receipt["aggregation_elapsed_s"] >= 0
    assert receipt["stats"]["clip_rate"] == 0
    loss, metrics = strategy.evaluate(1, output)
    assert np.isfinite(loss) and metrics["auc"] == 0.5
    assert strategy.evaluations[0]["round"] == 1


@pytest.mark.parametrize(
    "damage",
    [
        "partial",
        "duplicate",
        "unknown",
        "stale",
        "ahead",
        "shape",
        "nonfinite",
        "dtype",
        "zero_examples",
    ],
)
def test_strategy_refuses_bad_cohorts_before_mutating_model(tmp_path, damage):
    path, p = pack(tmp_path)
    strategy = module().EffusionStrategy(load(path))
    original = {k: v.clone() for k, v in strategy.initial_parameters.items()}
    results = [
        (cid, OrderedDict((k, v.clone()) for k, v in original.items()), 1)
        for cid in p["client_ids"]
    ]
    if damage == "partial":
        results.pop()
    elif damage == "duplicate":
        results[1] = results[0]
    elif damage == "unknown":
        results[1] = ("stranger", results[1][1], 1)
    elif damage == "shape":
        results[0][1]["head.weight"] = torch.zeros((2, 512))
    elif damage == "nonfinite":
        results[0][1]["head.bias"][0] = float("inf")
    elif damage == "dtype":
        results[0][1]["head.bias"] = torch.zeros(2, dtype=torch.float64)
    elif damage == "zero_examples":
        results[0] = (results[0][0], results[0][1], 0)
    with pytest.raises(ValueError):
        strategy.aggregate_fit(0 if damage == "stale" else 2 if damage == "ahead" else 1, results)
    assert not strategy.receipts
    for key in original:
        assert torch.equal(strategy._global[key], original[key])


@pytest.mark.parametrize(
    "scores,labels,want",
    [
        ([0, 0, 1, 1], [0, 1, 0, 1], 0.5),
        ([1, 2, 3, 4], [0, 0, 1, 1], 1.0),
        ([4, 3, 2, 1], [0, 0, 1, 1], 0.0),
        ([0, 1, 1, 2], [0, 0, 1, 1], 0.875),
        ([1, 2], [0, 0], None),
    ],
)
def test_auc_uses_average_ranks_for_ties(scores, labels, want):
    assert module().rank_auc(np.asarray(labels), np.asarray(scores, dtype=float)) == want


def drive(strategy, rounds):
    history = []
    for r in range(1, rounds + 1):
        results = []
        for cid in strategy.allowed_client_ids:
            params = OrderedDict((k, v.clone()) for k, v in strategy._global.items())
            params["head.weight"][1, 0] += 0.01
            results.append((cid, params, 1))
        final = strategy.aggregate_fit(r, results)
        loss, metrics = strategy.evaluate(r, final)
        history.append((r, dict(loss=loss, **metrics)))
    return history, final


def test_execute_publishes_only_complete_run_and_preserves_auth_environment(tmp_path, monkeypatch):
    path, p = pack(tmp_path)
    server = module()
    from fedlearn.server import server as canonical

    observed = {}

    def start(address, config, strategy):
        import os

        observed.update(
            address=address,
            rounds=config.num_rounds,
            max_clients=os.environ["MAX_CLIENTS"],
            deadline=os.environ["FEDLEARN_ROUND_TIMEOUT_S"],
            auth=os.environ["FEDLEARN_REQUIRE_CLIENT_AUTH"],
            tls=os.environ["FEDLEARN_REQUIRE_TLS"],
        )
        return drive(strategy, 5)

    monkeypatch.setattr(canonical, "start_server", start)
    monkeypatch.setenv("FEDLEARN_REQUIRE_CLIENT_AUTH", "1")
    monkeypatch.setenv("FEDLEARN_REQUIRE_TLS", "1")
    monkeypatch.setenv("MAX_CLIENTS", "9")
    output = tmp_path / "out"
    summary = server.execute(path, sha(path), output)
    assert summary["status"] == "complete"
    assert summary["completed_rounds"] == 5
    assert observed == dict(
        address=p["bind_address"], rounds=5, max_clients="3", deadline="600", auth="1", tls="1"
    )
    import os

    assert os.environ["MAX_CLIENTS"] == "9"
    assert summary["runtime"]["device"] == "cpu"
    assert summary["runtime"]["peak_rss_bytes"] > 0
    assert (output / "head.npz").is_file()
    assert len(json.loads((output / "receipts.json").read_text())) == 5
    assert len(json.loads((output / "rounds.json").read_text())) == 5
    assert json.loads((output / "summary.json").read_text()) == summary
    with pytest.raises(FileExistsError):
        server.execute(path, sha(path), output)


@pytest.mark.parametrize("failure", ["partial", "throws", "changed_head", "missing_history"])
def test_failed_server_run_writes_error_summary_without_success_head(
    tmp_path, monkeypatch, failure
):
    path, _ = pack(tmp_path)
    from fedlearn.server import server as canonical

    def start(address, config, strategy):
        if failure == "throws":
            raise RuntimeError("controlled failure")
        history, final = drive(strategy, 2 if failure == "partial" else 5)
        if failure == "changed_head":
            final["head.bias"][0] += 1
        if failure == "missing_history":
            history.pop()
        return history, final

    monkeypatch.setattr(canonical, "start_server", start)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        module().execute(path, sha(path), output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["completed_rounds"] == (
        0 if failure == "throws" else 2 if failure == "partial" else 5
    )
    assert not (output / "head.npz").exists()
    assert not (output / "receipts.json").exists()


def test_clipping_applies_per_client_before_uniform_mean(tmp_path):
    path, p = pack(tmp_path)
    strategy = module().EffusionStrategy(load(path))
    results = []
    for cid, change, n in zip(p["client_ids"], (0.1, 0.01, -0.2), (1, 10, 100)):
        head = OrderedDict((k, v.clone()) for k, v in strategy.initial_parameters.items())
        head["head.bias"][0] = change
        results.append((cid, head, n))
    result = strategy.aggregate_fit(1, results)
    assert result["head.bias"][0].item() == pytest.approx(0.01 / 3, abs=1e-7)
    assert strategy.receipts[0]["stats"]["clip_rate"] == pytest.approx(2 / 3)


def test_wrong_method_type_is_a_validation_error(tmp_path):
    path, plan = pack(tmp_path)
    plan["method"] = []
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        load(path)


@pytest.mark.parametrize("failure", ["raises", "incomplete"])
def test_failed_run_retains_completed_partial_receipts_and_evaluations(
    tmp_path, monkeypatch, failure
):
    path, _ = pack(tmp_path)
    from fedlearn.server import server as canonical

    def start(address, config, strategy):
        history, final = drive(strategy, 1)
        if failure == "raises":
            raise RuntimeError("controlled exit after first completed round")
        return history, final

    monkeypatch.setattr(canonical, "start_server", start)
    output = tmp_path / "failed-after-one"
    with pytest.raises(RuntimeError):
        module().execute(path, sha(path), output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["completed_rounds"] == summary["aggregated_rounds"] == 1
    receipts = json.loads((output / "failed-receipts.json").read_text())
    rounds = json.loads((output / "failed-rounds.json").read_text())
    assert [r["round"] for r in receipts] == [1]
    assert receipts[0]["client_count"] == 3
    assert [r["round"] for r in rounds] == [1]
    assert rounds[0]["head_sha256"] == receipts[0]["head_sha256"]
    assert np.isfinite(rounds[0]["loss"])
    assert not (output / "head.npz").exists()
    assert not (output / "receipts.json").exists()
    assert not (output / "rounds.json").exists()
