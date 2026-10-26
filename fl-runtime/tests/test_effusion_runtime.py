"""Generated training packs preserve the frozen Effusion head recipe and isolation."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path
import sys
from threading import Lock

import numpy as np
import pytest
import torch

SETTINGS = dict(
    lr=0.03, local_epochs=7, batch_size=256, clip_norm=0.05, optimizer="sgd", device="cpu", rounds=5
)


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(directory, manifest):
    path = directory / "host.json"
    path.write_text(json.dumps(manifest))
    return path


def pack(tmp_path, *, width=128, counts=(259, 263)):
    rng = np.random.default_rng(911)
    x = rng.normal(size=(sum(counts), width)).astype(np.float32)
    y = rng.integers(0, 2, len(x), dtype=np.int64)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "y.npy", y)
    clients, start = [], 0
    for cid, count in enumerate(counts):
        clients.append(
            dict(
                client_id=cid,
                start=start,
                stop=start + count,
                identity=f"generated:client-{cid:05d}",
            )
        )
        start += count
    manifest = dict(
        schema="fedlearn.effusion_host_data.v1",
        dataset="NIH_ChestXray14",
        target="Effusion",
        training_only=True,
        selection_included=False,
        holdout_included=False,
        test_included=False,
        feat_dim=width,
        original_feat_dim=512,
        projection_seed=20261025,
        projection_sha256="a" * 64,
        width=width,
        run_id="generated",
        host_label="mac",
        num_clients=len(counts),
        seed=14001,
        clients=clients,
        payload_sha256={name: digest(tmp_path / name) for name in ("x.npy", "y.npy")},
        settings=dict(SETTINGS),
    )
    return write_manifest(tmp_path, manifest), manifest, x, y


def load(path):
    return importlib.import_module("effusion_runtime").load_host_data(path, digest(path))


@pytest.mark.parametrize("width", [128, 512])
def test_shared_data_stays_preprojected_and_heads_start_at_zero(tmp_path, width):
    path, _, x, y = pack(tmp_path, width=width)
    runtime = importlib.import_module("effusion_runtime")
    data = load(path)
    a, b = (runtime.FeatureState(data, cid, SETTINGS) for cid in (0, 1))
    assert data.x.device.type == data.y.device.type == "cpu"
    np.testing.assert_array_equal(data.x.numpy(), x)
    np.testing.assert_array_equal(data.y.numpy(), y)
    assert a.x.untyped_storage().data_ptr() == b.x.untyped_storage().data_ptr()
    assert list(a.get_parameters()) == ["head.weight", "head.bias"]
    assert a.get_parameters()["head.weight"].shape == (2, width)
    assert all(torch.count_nonzero(p) == 0 for p in a.get_parameters().values())
    with pytest.raises(TypeError):
        data.settings["lr"] = 1


@pytest.mark.parametrize(
    "damage",
    [
        "manifest_hash",
        "payload_hash",
        "selection",
        "holdout",
        "test",
        "dataset",
        "target",
        "extra_array",
        "extra_key",
        "duplicate",
        "gap",
        "overlap",
        "empty",
        "identity",
        "client_range",
        "negative_seed",
        "width",
        "projection",
        "float64",
        "nan",
        "label",
        "label_dtype",
        "symlink",
        "optimizer",
        "settings_extra",
        "settings_type",
        "duplicate_json",
    ],
)
def test_untrusted_or_incompatible_host_pack_is_rejected(tmp_path, damage):
    path, m, x, y = pack(tmp_path)
    if damage == "manifest_hash":
        with pytest.raises(ValueError):
            importlib.import_module("effusion_runtime").load_host_data(path, "0" * 64)
        return
    if damage == "payload_hash":
        with (tmp_path / "x.npy").open("ab") as f:
            f.write(b"changed")
    elif damage in ("selection", "holdout", "test"):
        m[damage + "_included"] = True
    elif damage in ("dataset", "target"):
        m[damage] = "Pneumonia"
    elif damage == "extra_array":
        np.save(tmp_path / "selection.npy", x)
    elif damage == "extra_key":
        m["selection_x"] = "elsewhere.npy"
    elif damage == "duplicate":
        m["clients"][1]["client_id"] = 0
    elif damage == "gap":
        m["clients"][1]["start"] += 1
    elif damage == "overlap":
        m["clients"][1]["start"] -= 1
    elif damage == "empty":
        m["clients"][0]["stop"] = 0
    elif damage == "identity":
        m["clients"][0]["identity"] = "another:client-00000"
    elif damage == "client_range":
        m["clients"][1]["client_id"] = 2
    elif damage == "negative_seed":
        m["seed"] = -1
    elif damage == "width":
        m["width"] = 512
    elif damage == "projection":
        m["projection_sha256"] = "invalid"
    elif damage in ("float64", "nan", "label", "label_dtype"):
        if damage == "float64":
            x = x.astype(np.float64)
        elif damage == "nan":
            x[0, 0] = np.nan
        elif damage == "label":
            y[0] = 2
        else:
            y = y.astype(np.float32)
        np.save(tmp_path / "x.npy", x)
        np.save(tmp_path / "y.npy", y)
        m["payload_sha256"] = {name: digest(tmp_path / name) for name in ("x.npy", "y.npy")}
    elif damage == "symlink":
        (tmp_path / "x.npy").rename(tmp_path / "elsewhere.npy")
        (tmp_path / "x.npy").symlink_to(tmp_path / "elsewhere.npy")
    elif damage == "optimizer":
        m["settings"]["optimizer"] = "adam"
    elif damage == "settings_extra":
        m["settings"]["momentum"] = 0.9
    elif damage == "settings_type":
        m["settings"]["rounds"] = 5.0
    write_manifest(tmp_path, m)
    if damage == "duplicate_json":
        path.write_text(path.read_text().replace('"seed": 14001', '"seed": 14001, "seed": 14001'))
    with pytest.raises(ValueError):
        load(path)


def reference(data):
    # Exercise the real recipe without loading any clinical bundle/evaluation rows.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/artifacts/capstone/repro"))
    from run_effusion_byzfl_qualification import ZeroBundleWorkload
    from fedlearn.simulation.behaviors import BehaviorPlan
    from fedlearn.simulation.costs import StageClock
    from fedlearn.simulation.engine import HeadClient

    workload = ZeroBundleWorkload.__new__(ZeroBundleWorkload)
    workload.head = {"type": "linear"}
    workload.feat_dim, workload.n_classes = data.x.shape[1], 2
    workload.train_x, workload.train_y = data.x, data.y
    workload.trigger_train_x = None
    workload.parts = [np.arange(c["start"], c["stop"]) for c in data.manifest["clients"]]
    cfg = {**SETTINGS, "algorithm": "fedavg", "proximal_mu": 0.0}
    return [
        HeadClient(
            workload,
            cid,
            BehaviorPlan([], len(workload.parts), 1, 2),
            cfg,
            1,
            {"measured": {cid: 0}, "analytic": {}},
            StageClock(),
        )
        for cid in range(len(workload.parts))
    ]


@pytest.mark.parametrize("width", [128, 512])
def test_concurrent_multiround_heads_and_batches_match_simulator(tmp_path, monkeypatch, width):
    path, _, _, _ = pack(tmp_path, width=width)
    runtime = importlib.import_module("effusion_runtime")
    from fedlearn.simulation.rng import ClientRng, torch_rng_scope

    data = load(path)
    states = [runtime.FeatureState(data, cid, SETTINGS) for cid in (0, 1)]
    clients, params = reference(data), states[0].get_parameters()
    original_randperm = torch.randperm
    observed, reference_batches, lock = [], [], Lock()

    def capture(n, *args, **kwargs):
        result = original_randperm(n, *args, **kwargs)
        with lock:
            (observed if "generator" in kwargs else reference_batches).append((n, result.clone()))
        return result

    monkeypatch.setattr(torch, "randperm", capture)
    for round_number in (1, 2, 5):
        observed.clear()
        reference_batches.clear()
        expected = []
        for cid, client in enumerate(clients):
            with torch_rng_scope(ClientRng(data.manifest["seed"], cid).torch_seed(round_number)):
                expected.append(client.fit(params, {})[0])
        rng_before = torch.get_rng_state().clone()
        with ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(lambda s: s.fit(params, round_number, lambda: False), states))
        assert torch.equal(torch.get_rng_state(), rng_before)
        for cid, (actual, n, diagnostics) in enumerate(outputs):
            assert n == len(states[cid].y)
            assert diagnostics["batch_count"] == 14
            assert diagnostics["fit_elapsed_s"] >= 0 and np.isfinite(diagnostics["loss"])
            assert set(diagnostics) <= {"loss", "batch_count", "fit_elapsed_s", "client_id"}
            for key in params:
                torch.testing.assert_close(actual[key], expected[cid][key], rtol=0, atol=1e-7)
        for n in (259, 263):
            batches = [b for b in observed if b[0] == n]
            refs = [b for b in reference_batches if b[0] == n]
            assert len(batches) == len(refs) == 7
            assert all(torch.equal(a[1], b[1]) for a, b in zip(batches, refs))
        params = OrderedDict((k, (outputs[0][0][k] + outputs[1][0][k]) / 2) for k in params)


def test_cancellation_never_publishes_a_partial_head(tmp_path):
    path, _, _, _ = pack(tmp_path)
    runtime = importlib.import_module("effusion_runtime")
    state = runtime.FeatureState(load(path), 0, SETTINGS)
    original, checks = state.get_parameters(), 0

    def stop():
        nonlocal checks
        checks += 1
        return checks > 2

    with pytest.raises(runtime.FitCancelled):
        state.fit(original, 1, stop)
    for key in original:
        assert torch.equal(state.get_parameters()[key], original[key])
    complete, _, _ = state.fit(original, 1, lambda: False)
    assert any(torch.count_nonzero(v) for v in complete.values())


@pytest.mark.parametrize(
    "change",
    [
        "optimizer",
        "lr",
        "missing",
        "unknown_client",
        "bool_client",
        "round",
        "shape",
        "nonfinite",
        "dtype",
        "missing_head",
    ],
)
def test_client_refuses_settings_identity_round_and_head_drift(tmp_path, change):
    path, _, _, _ = pack(tmp_path)
    runtime = importlib.import_module("effusion_runtime")
    data, settings = load(path), dict(SETTINGS)
    if change == "optimizer":
        settings["optimizer"] = "adam"
    elif change == "lr":
        settings["lr"] = 0.04
    elif change == "missing":
        settings.pop("clip_norm")
    if change in ("optimizer", "lr", "missing", "unknown_client", "bool_client"):
        with pytest.raises(ValueError):
            runtime.FeatureState(
                data,
                True if change == "bool_client" else 9 if change == "unknown_client" else 0,
                settings,
            )
        return
    state = runtime.FeatureState(data, 0, settings)
    params = state.get_parameters()
    if change == "shape":
        params["head.weight"] = torch.zeros((2, 512))
    elif change == "nonfinite":
        params["head.bias"][0] = float("inf")
    elif change == "dtype":
        params["head.bias"] = params["head.bias"].double()
    elif change == "missing_head":
        params.pop("head.bias")
    with pytest.raises(ValueError):
        state.fit(params, 6 if change == "round" else 1, lambda: False)


def test_sparse_global_ids_and_twenty_round_anchor_are_valid(tmp_path):
    path, manifest, _, _ = pack(tmp_path)
    manifest["num_clients"] = 80
    manifest["settings"]["rounds"] = 20
    for member, cid in zip(manifest["clients"], (27, 79)):
        member["client_id"] = cid
        member["identity"] = f"generated:client-{cid:05d}"
    write_manifest(tmp_path, manifest)
    runtime = importlib.import_module("effusion_runtime")
    data = load(path)
    before = torch.get_rng_state().clone()
    state = runtime.FeatureState(data, 79, manifest["settings"])
    assert torch.equal(torch.get_rng_state(), before)
    parameters = state.get_parameters()
    parameters["head.bias"].fill_(1.0)
    assert torch.count_nonzero(state.get_parameters()["head.bias"]) == 0
    trained, n, diagnostics = state.fit(parameters, 20, lambda: False)
    assert n == 263 and diagnostics["batch_count"] == 14
    assert not torch.equal(trained["head.weight"], parameters["head.weight"])
