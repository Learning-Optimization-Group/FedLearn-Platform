import json
from dataclasses import replace

import numpy as np
import pytest
import xgboost as xgb

from benchmark import load_checkpoint, make_fixture, run_probe, save_checkpoint, train_probe
from benchmark import _metrics


def test_fixture_is_reproducible_with_disjoint_held_out_rows():
    first = make_fixture(42)
    second = make_fixture(42)
    ids = []
    for split, rows in (("train", 256), ("validation", 64), ("test", 128)):
        data = first[split]
        assert data.features.shape == (rows, 4)
        np.testing.assert_array_equal(data.features, second[split].features)
        np.testing.assert_array_equal(data.labels, second[split].labels)
        assert set(data.labels) == {0, 1}
        ids.extend(data.indices.tolist())
    assert len(ids) == len(set(ids)) == 448
    assert not np.array_equal(first["train"].features, make_fixture(43)["train"].features)


@pytest.mark.parametrize("format", ["json", "ubj"])
def test_checkpoint_preserves_predictions_and_continued_training(format, tmp_path):
    data = make_fixture()["train"]
    first = train_probe(data, rounds=8)
    path = tmp_path / f"checkpoint.{format}"
    save_checkpoint(first, path)
    loaded = load_checkpoint(path, data.feature_names)
    matrix = xgb.DMatrix(data.features, feature_names=list(data.feature_names))
    np.testing.assert_array_equal(loaded.predict(matrix), first.predict(matrix))

    resumed = train_probe(data, rounds=4, parent=loaded)
    uninterrupted = train_probe(data, rounds=12)
    assert first.num_boosted_rounds() == 8
    assert resumed.num_boosted_rounds() == 12
    np.testing.assert_array_equal(resumed.predict(matrix), uninterrupted.predict(matrix))
    assert np.max(np.abs(resumed.predict(matrix) - first.predict(matrix))) > 0.01


def test_checkpoint_refuses_permuted_features(tmp_path):
    data = make_fixture()["train"]
    path = tmp_path / "checkpoint.ubj"
    save_checkpoint(train_probe(data, rounds=2), path)
    with pytest.raises(ValueError, match="feature order"):
        load_checkpoint(path, tuple(reversed(data.feature_names)))


def test_checkpoint_refuses_corruption_before_native_loading(tmp_path):
    data = make_fixture()["train"]
    path = tmp_path / "checkpoint.ubj"
    save_checkpoint(train_probe(data, rounds=2), path)
    payload = path.read_bytes()
    path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(path, data.feature_names)


def test_probe_records_real_training_and_reload_evidence(tmp_path):
    report = run_probe(tmp_path / "probe", seed=42)
    saved = json.loads((tmp_path / "probe" / "report.json").read_text())
    assert saved == report
    assert report["scope"] == "single_process_cpu_probe"
    assert report["federation_implemented"] is False
    assert report["rounds"] == {"initial": 8, "continued": 12}
    assert report["checks"]["resume_max_abs_difference"] == 0.0
    assert report["checks"]["reload_max_abs_difference"] == 0.0
    assert report["metrics"]["continued"]["test"]["accuracy"] > 0.80
    assert report["metrics"]["continued"]["test"]["log_loss"] < np.log(2)
    assert report["metrics"]["initial"]["test"] != report["metrics"]["continued"]["test"]
    for name, artifact in report["artifacts"].items():
        assert (tmp_path / "probe" / name).stat().st_size == artifact["bytes"]
        assert len(artifact["sha256"]) == 64


def test_probe_does_not_overwrite_an_existing_experiment(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    original = directory / "report.json"
    original.write_text("previous experiment")
    with pytest.raises(FileExistsError):
        run_probe(directory)
    assert original.read_text() == "previous experiment"


def test_metrics_handle_tied_scores_with_a_known_confusion_matrix():
    metrics = _metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.8, 0.9]))
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["mcc"] == pytest.approx(1 / np.sqrt(3))
    assert metrics["roc_auc"] == 0.875
    assert metrics["log_loss"] == pytest.approx(0.5108256237659907)


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_metrics_do_not_report_auc_or_balanced_accuracy_for_a_single_class(labels):
    metrics = _metrics(np.array(labels), np.array([0.5, 0.5]))
    assert metrics["roc_auc"] is None
    assert metrics["balanced_accuracy"] is None
    assert metrics["mcc"] == 0.0


@pytest.mark.parametrize("change", ["infinite", "wrong_width", "duplicate_names", "wrong_labels"])
def test_training_refuses_incompatible_numeric_data(change):
    data = make_fixture()["train"]
    if change == "infinite":
        values = data.features.copy()
        values[0, 0] = np.inf
        data = replace(data, features=values)
    elif change == "wrong_width":
        data = replace(data, features=data.features[:, :3])
    elif change == "duplicate_names":
        data = replace(data, feature_names=("a", "a", "b", "c"))
    else:
        data = replace(data, labels=np.full(len(data.labels), 2))
    with pytest.raises(ValueError):
        train_probe(data, rounds=2)


def test_training_accepts_nan_as_missing_data():
    data = make_fixture()["train"]
    values = data.features.copy()
    values[:10, 0] = np.nan
    data = replace(data, features=values)
    model = train_probe(data, rounds=2)
    predictions = model.predict(xgb.DMatrix(data.features, feature_names=list(data.feature_names)))
    assert np.isfinite(predictions).all()


def test_repeated_probe_preserves_split_and_model_hashes(tmp_path):
    first = run_probe(tmp_path / "first", seed=42)
    second = run_probe(tmp_path / "second", seed=42)
    assert first["splits"] == second["splits"]
    assert first["artifacts"] == second["artifacts"]
