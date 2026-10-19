"""Explicit NIH evaluation identity, immutable cache and whole-patient clients."""
import copy
import hashlib
import json

import numpy as np
import pytest

from tests.test_capstone_nih_features import template as feature_template
from fedlearn.simulation.config import ConfigError, config_hash, resolve
from fedlearn.simulation.engine import run
from fedlearn.simulation.nih_cache import load_nih_cache
from fedlearn.simulation.workloads import NihFeatureHeadWorkload


@pytest.fixture
def cfg(feature_template):
    output = feature_template[1]
    return {"seed": 42, "workload": {"dataset": "nih", "features_path": str(output),
            "features_report_sha256": hashlib.sha256((output / "report.json").read_bytes()).hexdigest(),
            "evaluation_split": "validation", "balance": False, "allow_random_features": True},
            "federation": {"num_clients": 3, "rounds": 2, "partition": {"type": "patient_iid"}},
            "client": {"local_epochs": 1, "lr": 0.001}, "server": {"clip_norm": None}}


@pytest.mark.parametrize("field,value", [("features_report_sha256", None), ("features_report_sha256", "bad"),
                                        ("evaluation_split", None), ("evaluation_split", "train"),
                                        ("balance", True), ("balance", 0), ("allow_random_features", 1)])
def test_nih_rejects_missing_or_unsafe_config_before_data_access(cfg, field, value):
    cfg["workload"][field] = value
    with pytest.raises(ConfigError):
        resolve(cfg)


def test_nih_requires_explicit_selection_digest_and_false_balancing(cfg):
    for field in ("features_report_sha256", "evaluation_split", "balance"):
        changed = copy.deepcopy(cfg)
        del changed["workload"][field]
        with pytest.raises(ConfigError):
            resolve(changed)


@pytest.mark.parametrize("partition", ["iid", "dirichlet"])
def test_nih_refuses_image_row_partitioning(cfg, partition):
    cfg["federation"]["partition"]["type"] = partition
    with pytest.raises(ConfigError, match="partition"):
        resolve(cfg)


@pytest.mark.parametrize("clients", [True, 1.5, 0])
def test_nih_client_count_is_a_positive_integer(cfg, clients):
    cfg["federation"]["num_clients"] = clients
    with pytest.raises(ConfigError):
        resolve(cfg)


def test_nih_random_cache_requires_explicit_control_flag(cfg):
    del cfg["workload"]["allow_random_features"]
    resolved = resolve(cfg)
    assert resolved["workload"]["allow_random_features"] is False
    with pytest.raises(ConfigError, match="random|pretrained"):
        NihFeatureHeadWorkload(resolved)


def test_nih_digest_is_verified_before_model_or_training(cfg, monkeypatch):
    cfg["workload"]["features_report_sha256"] = "0" * 64
    monkeypatch.setattr(NihFeatureHeadWorkload, "build_model", lambda *args: pytest.fail("model constructed"))
    with pytest.raises(ValueError, match="identity|digest"):
        run(cfg)


def test_nih_backdoor_refuses_before_model_construction(cfg, monkeypatch):
    cfg["behaviors"] = [{"type": "backdoor", "fraction": 0.2, "target_label": 1}]
    monkeypatch.setattr(NihFeatureHeadWorkload, "build_model", lambda *args: pytest.fail("model constructed"))
    with pytest.raises(ValueError, match="trigger|backdoor"):
        run(cfg)


def test_nih_preserves_whole_patients_exact_rows_counts_and_alignment(cfg):
    wl = NihFeatureHeadWorkload(resolve(cfg))
    cache = load_nih_cache(cfg["workload"]["features_path"])
    train, validation = cache["splits"]["train"], cache["splits"]["validation"]
    np.testing.assert_array_equal(wl.train_x.numpy(), train["x"])
    np.testing.assert_array_equal(wl.train_y.numpy(), train["y"])
    np.testing.assert_array_equal(wl.test_x.numpy(), validation["x"])
    np.testing.assert_array_equal(wl.test_y.numpy(), validation["y"])
    assert wl.train_patient_ids == train["patient_ids"]
    assert wl.train_image_ids == train["image_ids"]
    assert wl.evaluation_image_ids == validation["image_ids"]
    assert sorted(np.concatenate(wl.parts).tolist()) == list(range(len(train["y"])))
    patient_sets = []
    assignments = []
    for cid, part in enumerate(wl.parts):
        x, y, idx = wl.client_data(cid)
        np.testing.assert_array_equal(x.numpy(), train["x"][part])
        np.testing.assert_array_equal(y.numpy(), train["y"][part])
        assert idx.tolist() == part.tolist() and len(part) > 0
        patients = {train["patient_ids"][i] for i in part}
        patient_sets.append(patients)
        assert wl.info["clients"][str(cid)] == {
            "images": len(part), "patients": len(patients), "positive_images": int(y.sum()),
            "negative_images": len(y) - int(y.sum())}
        assignments.append({"client_id": cid, "image_ids": [train["image_ids"][i] for i in part]})
    assert all(not a & b for i, a in enumerate(patient_sets) for b in patient_sets[i + 1:])
    raw = (json.dumps({"algorithm": "nih-client-assignment-v1", "clients": assignments},
                      sort_keys=True, indent=2) + "\n").encode()
    assert wl.info["assignment_sha256"] == hashlib.sha256(raw).hexdigest()
    assert wl.info["partition_algorithm"] == "patient-iid-v1:sorted-patients-pcg64-array-split"
    assert wl.info["features_report_sha256"] == cache["report_sha256"]
    assert wl.info["pretrained"] is False
    assert wl.info["source_dataset"] == "NIH_ChestXray14" and wl.info["dataset"] == "nih"
    # The engine owns its tensors, never a writable alias to persisted cache arrays.
    before = np.array(train["x"], copy=True)
    wl.train_x.add_(1)
    np.testing.assert_array_equal(train["x"], before)


def test_patient_assignment_repeats_changes_with_seed_and_refuses_too_many(cfg):
    a, b = (NihFeatureHeadWorkload(resolve(cfg)) for _ in range(2))
    assert a.info["assignment_sha256"] == b.info["assignment_sha256"]
    changed = copy.deepcopy(cfg)
    changed["seed"] += 1
    assert NihFeatureHeadWorkload(resolve(changed)).info["assignment_sha256"] != a.info["assignment_sha256"]
    cfg["federation"]["num_clients"] = len(set(a.train_patient_ids)) + 1
    with pytest.raises(ValueError, match="patient|client"):
        NihFeatureHeadWorkload(resolve(cfg))


@pytest.mark.parametrize("evaluation", ["validation", "test"])
def test_tiny_engine_records_actual_evaluation_identity_and_cache(cfg, evaluation):
    cfg["workload"]["evaluation_split"] = evaluation
    rec = run(cfg)
    expected = load_nih_cache(cfg["workload"]["features_path"])["splits"][evaluation]
    info = rec["meta"]["workload"]
    assert info["evaluation_split"] == evaluation and info["n_test"] == len(expected["y"])
    assert info["features_report_sha256"] == cfg["workload"]["features_report_sha256"]
    assert info["evaluation_manifest_sha256"] == hashlib.sha256(
        (feature_path(cfg) / f"{evaluation}.csv").read_bytes()).hexdigest()
    assert len(rec["rounds"]) == 2 and all(r["loss"] is not None for r in rec["rounds"])
    assert info["assignment_sha256"] and info["clients"]
    assert "patient_ids" not in json.dumps(info) and "image_ids" not in json.dumps(info)


def feature_path(cfg):
    from pathlib import Path
    return Path(cfg["workload"]["features_path"])


@pytest.mark.parametrize("cfg,expected", [
    ({"workload": {"features_path": "fixture.pt"}},
     "78760775cf3a1b03e86e20c3ce3d14b069d7307beaab817521d476c897e1eee5"),
    ({"seed": 42, "workload": {"features_path": "fixture.pt", "head": {"type": "mlp", "hidden": 8}},
      "federation": {"num_clients": 100}, "privacy": {"mode": "central"}},
     "5dc8992c92c96360fa5e6a4214ddb7f4147a9318526bf7289cbc7717d3e2fafb")])
def test_legacy_resolved_configuration_hash_stays_exact(cfg, expected):
    resolved = resolve(cfg)
    assert set(resolved["workload"]) == {"dataset", "features_path", "balance", "head"}
    assert config_hash(resolved) == expected


@pytest.mark.parametrize("field,value", [("evaluation_split", "validation"), ("allow_random_features", True),
                                        ("features_report_sha256", "0" * 64)])
def test_legacy_workload_rejects_nih_only_options(field, value):
    with pytest.raises(ConfigError):
        resolve({"workload": {"features_path": "fixture.pt", field: value}})
