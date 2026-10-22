"""Effusion R0 bundle: holdout-free, hash-pinned train + selection features for the Simulation Lab."""
import hashlib
import json

import numpy as np
import pytest

from fedlearn.simulation.effusion_bundle import BUNDLE_VERSION, load_effusion_bundle, write_effusion_bundle


def _split(n, first_patient, seed):
    rng = np.random.default_rng(seed)
    patients = [str(first_patient + i // 2) for i in range(n)]
    return {"x": rng.normal(size=(n, 512)).astype(np.float32), "y": (rng.random(n) < 0.2).astype(np.int64),
            "patient_ids": patients, "image_ids": [f"{int(p):08d}_{i % 2:03d}.png" for i, p in enumerate(patients)]}


def _write(tmp_path, **overrides):
    train, selection = _split(40, 1, 0), _split(20, 1000, 1)
    out = tmp_path / "bundle"
    sha = write_effusion_bundle(out, train, selection, provenance={"lockSha256": "a" * 64, **overrides})
    return out, sha, train, selection


def test_round_trip_returns_exactly_the_written_arrays_and_identities(tmp_path):
    out, sha, train, selection = _write(tmp_path)
    assert sha == hashlib.sha256((out / "report.json").read_bytes()).hexdigest()
    bundle = load_effusion_bundle(out, expected_report_sha256=sha)
    assert bundle["report"]["bundle_version"] == BUNDLE_VERSION
    assert bundle["report"]["holdoutIncluded"] is False and bundle["report"]["testIncluded"] is False
    for name, split in (("train", train), ("evaluation", selection)):
        assert np.array_equal(bundle[name]["x"], split["x"]) and np.array_equal(bundle[name]["y"], split["y"])
        assert bundle[name]["patient_ids"] == split["patient_ids"]


def test_wrong_pin_and_changed_payload_are_refused(tmp_path):
    out, sha, _, _ = _write(tmp_path)
    with pytest.raises(ValueError, match="report"):
        load_effusion_bundle(out, expected_report_sha256="f" * 64)
    path = out / "train-x.npy"
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="payload"):
        load_effusion_bundle(out, expected_report_sha256=sha)


def test_patient_overlap_and_existing_output_are_refused(tmp_path):
    train, selection = _split(10, 1, 0), _split(10, 3, 1)
    with pytest.raises(ValueError, match="overlap"):
        write_effusion_bundle(tmp_path / "b", train, selection, provenance={})
    out, _, train, selection = _write(tmp_path)
    with pytest.raises(ValueError, match="exists"):
        write_effusion_bundle(out, train, selection, provenance={})


def test_non_binary_labels_or_wrong_width_are_refused(tmp_path):
    train, selection = _split(10, 1, 0), _split(10, 100, 1)
    bad = {**train, "y": np.full(10, 2, dtype=np.int64)}
    with pytest.raises(ValueError, match="label"):
        write_effusion_bundle(tmp_path / "a", bad, selection, provenance={})
    narrow = {**train, "x": train["x"][:, :256]}
    with pytest.raises(ValueError, match="512"):
        write_effusion_bundle(tmp_path / "c", narrow, selection, provenance={})


def _engine_config(bundle, sha, clients=4, privacy="none"):
    return {"name": "effusion-lab", "seed": 7,
            "workload": {"dataset": "nih", "features_path": str(bundle), "features_report_sha256": sha,
                         "evaluation_split": "validation", "allow_random_features": False, "balance": False,
                         "auc_score": "margin_v2", "head": {"type": "linear", "hidden": 128}},
            "federation": {"num_clients": clients, "clients_per_round": clients, "rounds": 2,
                           "partition": {"type": "patient_iid"}},
            "client": {"local_epochs": 1, "lr": 0.03, "batch_size": 256},
            "server": {"rule": "mean", "clip_norm": 0.05, "rule_params": {"trim_ratio": 0.2}},
            "privacy": {"mode": privacy, "epsilon": 4.0, "delta": 1e-5, "adjacency": "replace_one"},
            "behaviors": [], "eval": {"per_client_every": 0}}


def test_the_engine_dispatches_a_bundle_to_the_effusion_workload_and_scores_only_selection(tmp_path):
    from fedlearn.simulation.engine import run
    out, sha, train, selection = _write(tmp_path)
    record = run(_engine_config(out, sha))
    workload = record["meta"]["workload"]
    assert workload["bundle_version"] == BUNDLE_VERSION and workload["target"] == "Effusion"
    assert workload["n_train"] == 40 and workload["n_test"] == 20 and workload["feat_dim"] == 512
    assert workload["evaluation"] == "development selection half; no holdout or official test"
    assert all(0.0 <= line["auc"] <= 1.0 for line in record["rounds"])


def test_a_bundle_run_passes_core_qualification(tmp_path):
    from fedlearn.simulation.core_qualification import run_core
    out, sha, _, _ = _write(tmp_path)
    record = run_core(_engine_config(out, sha, privacy="central"), feature_report_sha256=sha, clients=4)
    assert record["meta"]["core_qualification"]["privacy_mode"] == "central"


def test_a_non_bundle_path_still_routes_to_the_historical_any_finding_workload(tmp_path, monkeypatch):
    from fedlearn.simulation import workloads

    class Sentinel:
        def __init__(self, resolved):
            self.resolved = resolved

    monkeypatch.setattr(workloads, "NihFeatureHeadWorkload", Sentinel)
    resolved = {"workload": {"features_path": str(tmp_path)}}
    assert isinstance(workloads.nih_workload(resolved), Sentinel)
