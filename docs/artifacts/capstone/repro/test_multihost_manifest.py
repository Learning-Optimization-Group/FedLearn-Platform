"""Generated-only preparation contracts for the fixed three-host benchmark."""
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fedlearn.simulation.effusion_bundle import write_effusion_bundle
from fedlearn.simulation.partition import patient_iid_partition


def module():
    return importlib.import_module("multihost_manifest")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_exact_prospective_budget_and_historical_anchor_seeds():
    cells = module().cells()
    assert len(cells) == len({cell["name"] for cell in cells}) == 52
    assert sum(cell["rounds"] for cell in cells) == 320
    assert Counter(cell["stage"] for cell in cells) == {"systems": 48, "anchor": 4}
    assert cells[0] == dict(name="systems_n003_mean512_s14001_mac_only", stage="systems",
                           num_clients=3, method="mean512", width=512, seed=14001,
                           rounds=5, placement="mac_only")
    assert {cell["seed"] for cell in cells if cell["stage"] == "systems"} == {14001, 14002, 14003}
    assert {cell["seed"] for cell in cells if cell["stage"] == "anchor"} == {13001}
    assert all(cell["num_clients"] == 80 and cell["rounds"] == 20 for cell in cells[48:])
    cells[0]["rounds"] = 999
    assert module().cells()[0]["rounds"] == 5


def test_balanced_host_assignment_is_complete_without_duplicate_identities():
    mapping = module().assignments(80, "three_hosts")
    assert {host: len(ids) for host, ids in mapping.items()} == dict(mac=27, g14=27, orin=26)
    assert mapping["mac"] == list(range(0, 80, 3))
    assert sorted(cid for ids in mapping.values() for cid in ids) == list(range(80))
    assert module().assignments(6, "mac_only") == dict(mac=list(range(6)), g14=[], orin=[])
    for n, placement in [(0, "three_hosts"), (True, "mac_only"), (80, "unknown")]:
        with pytest.raises(ValueError):
            module().assignments(n, placement)


@pytest.fixture
def generated(tmp_path, monkeypatch):
    root = tmp_path / "source"
    for relative in ("framework/src/fedlearn/example.py", "fl-runtime/client.py", "proto/fedlearn.proto",
                     "docs/artifacts/capstone/repro/multihost_manifest.py",
                     "docs/artifacts/capstone/repro/test_multihost_manifest.py",
                     "docs/artifacts/capstone/repro/capstone_pilot_mechanisms.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source fixture\n")
    monkeypatch.setattr(module(), "ROOT", root, raising=False)
    rng = np.random.default_rng(119)
    x = rng.normal(size=(160, 512)).astype(np.float32)
    x[:, 0] = np.arange(160)
    train = dict(x=x, y=np.arange(160, dtype=np.int64) % 2,
                 patient_ids=[str(10001 + i // 2) for i in range(160)],
                 image_ids=[f"generated-row-{i}" for i in range(160)])
    selection = dict(x=rng.normal(size=(8, 512)).astype(np.float32), y=np.arange(8, dtype=np.int64) % 2,
                     patient_ids=[str(20001 + i) for i in range(8)], image_ids=[f"selection-row-{i}" for i in range(8)])
    bundle = tmp_path / "bundle"
    pin = write_effusion_bundle(bundle, train, selection, provenance={"synthetic": True})
    return tmp_path, root, bundle, pin, train, selection


def test_preparation_writes_pinned_complete_training_only_packs(generated, monkeypatch):
    temporary, root, bundle, pin, train, selection = generated
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "fl-runtime"))
    from effusion_runtime import load_host_data
    from effusion_server import load_server_plan
    output = temporary / "prepared"
    plan = module().prepare(bundle, pin, output, synthetic=True)
    assert set(plan) == {"schema", "synthetic", "bundle_path", "bundle_report_sha256", "port",
                         "sources", "public_manifest_sha256", "cells"}
    assert json.loads((output / "plan.json").read_text()) == plan
    assert len(plan["cells"]) == 52
    public = json.loads((output / "public-manifest.json").read_text())
    assert digest(output / "public-manifest.json") == plan["public_manifest_sha256"]
    assert set(public) == {"schema", "synthetic", "scope", "bundle_report_sha256", "source_sha256",
                           "cells", "settings_templates", "projection_seed", "projection_sha256",
                           "fit_concurrency_per_host", "round_timeout_s", "runtime", "input_commitments", "launch_runtimes", "execution_order"}
    assert public["scope"]["anchors"] == "historical-seed replay; not fresh confirmation"
    assert public["fit_concurrency_per_host"] == 2
    assert public["runtime"]["device"] == "cpu"
    assert str(temporary) not in json.dumps(public)
    assert "patient_ids" not in json.dumps(public) and "image_ids" not in json.dumps(public)
    assert set(plan["sources"]) == {"framework/src/fedlearn/example.py", "fl-runtime/client.py",
                                    "proto/fedlearn.proto",
                                    "docs/artifacts/capstone/repro/multihost_manifest.py",
                                    "docs/artifacts/capstone/repro/test_multihost_manifest.py",
                                    "docs/artifacts/capstone/repro/capstone_pilot_mechanisms.py"}
    for relative, expected in plan["sources"].items():
        assert digest(root / relative) == digest(output / "sources" / relative) == expected
    from capstone_pilot_mechanisms import public_projection
    projected_train = (torch.from_numpy(train["x"]).double() @ public_projection(512, 128, seed=20261025)).float().numpy()
    for cell in plan["cells"]:
        parts_for_cell = patient_iid_partition(train["patient_ids"], cell["num_clients"], cell["seed"])
        server = load_server_plan(output / cell["server_plan_path"], cell["server_plan_sha256"])
        assert server.manifest["synthetic"] is True
        assert server.manifest["settings"]["optimizer"] == "sgd"
        assert server.manifest["bind_address"] == "127.0.0.1:54051"
        members = []
        for host, reference in cell["hosts"].items():
            path = output / reference["manifest_path"]
            data = load_host_data(path, reference["manifest_sha256"])
            assert set(p.name for p in path.parent.iterdir()) == {"host.json", "x.npy", "y.npy"}
            assert data.manifest["selection_included"] is False
            assert data.manifest["holdout_included"] is False
            assert data.manifest["test_included"] is False
            assert list(data.clients) == module().assignments(cell["num_clients"], cell["placement"])[host]
            for cid, entry in data.clients.items():
                rows = parts_for_cell[cid]
                expected_x = train["x"] if cell["width"] == 512 else projected_train
                np.testing.assert_array_equal(data.x[entry["start"]:entry["stop"]].numpy(), expected_x[rows])
                np.testing.assert_array_equal(data.y[entry["start"]:entry["stop"]].numpy(), train["y"][rows])
            members.extend(data.clients)
            assert all(entry["stop"] > entry["start"] for entry in data.clients.values())
        assert sorted(members) == list(range(cell["num_clients"]))
    first = plan["cells"][0]
    data = load_host_data(output / first["hosts"]["mac"]["manifest_path"], first["hosts"]["mac"]["manifest_sha256"])
    parts = patient_iid_partition(train["patient_ids"], 3, 14001)
    expected_rows = np.concatenate(parts)
    np.testing.assert_array_equal(data.x.numpy(), train["x"][expected_rows])
    np.testing.assert_array_equal(data.y.numpy(), train["y"][expected_rows])
    projected = np.load(output / "selection/128/selection_x.npy")
    from capstone_pilot_mechanisms import public_projection
    expected = (torch.from_numpy(selection["x"]).double() @ public_projection(512, 128, seed=20261025)).float().numpy()
    np.testing.assert_array_equal(projected, expected)
    assert module().verify_prepared(output) == plan


@pytest.mark.parametrize("damage", ["source", "snapshot", "host", "selection", "public", "runtime"])
def test_frozen_preparation_rejects_changed_sources_payloads_and_runtime(generated, damage, monkeypatch):
    temporary, root, bundle, pin, *_ = generated
    output = temporary / "prepared"
    plan = module().prepare(bundle, pin, output, synthetic=True)
    if damage == "runtime":
        original = module().runtime
        monkeypatch.setattr(module(), "runtime", lambda: {**original(), "python": "changed"})
    else:
        reference = plan["cells"][0]["hosts"]["mac"]
        target = {
            "source": root / "fl-runtime/client.py",
            "snapshot": output / "sources/fl-runtime/client.py",
            "host": (output / reference["manifest_path"]).parent / "x.npy",
            "selection": output / "selection/128/selection_x.npy",
            "public": output / "public-manifest.json",
        }[damage]
        with target.open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        module().verify_prepared(output)


def test_medical_pin_and_unsafe_bundles_fail_before_preparation(generated):
    temporary, _, bundle, pin, *_ = generated
    with pytest.raises(ValueError, match="canonical"):
        module().prepare(temporary / "missing", pin, temporary / "medical")
    np.save(bundle / "holdout-x.npy", np.ones((1, 512), dtype=np.float32))
    with pytest.raises(ValueError, match="bundle"):
        module().prepare(bundle, pin, temporary / "unsafe", synthetic=True)
    assert not (temporary / "unsafe").exists()


def test_preparation_refuses_reused_output_and_changed_bundle(generated):
    temporary, _, bundle, pin, *_ = generated
    output = temporary / "existing"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        module().prepare(bundle, pin, output, synthetic=True)
    assert (output / "sentinel").read_text() == "keep"
    with (bundle / "train-x.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        module().prepare(bundle, pin, temporary / "changed", synthetic=True)


def test_public_commitment_rejects_resealed_private_training_payload(generated):
    temporary,_,bundle,pin,*_=generated
    output=temporary/'prepared'
    plan=module().prepare(bundle,pin,output,synthetic=True)
    ref=plan['cells'][0]['hosts']['mac'];path=output/ref['manifest_path']
    x_path=path.parent/'x.npy';x=np.load(x_path);x[0,0]+=1;np.save(x_path,x)
    manifest=json.loads(path.read_text());manifest['payload_sha256']['x.npy']=digest(x_path)
    path.write_text(json.dumps(manifest));ref['manifest_sha256']=digest(path)
    (output/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='commitment'):
        module().verify_prepared(output)
