"""DA-14 Phase 1 (Target A) — MLP/ECG data loading collapses onto the recipe registry.

The client trainloader and server test loader were built by calling data_loaders.ecg_loader
directly from client.py / fl_server.py, with every hyperparameter hand-threaded from
config.get_dataset_config("ecg"). These pin that routing through
recipes.get_recipe("MLP").load_client_data()/.load_server_test_data() reproduces the exact same
partition — byte-for-byte identical dataset contents — provided the recipe sources its
hyperparameters from the ECG config (not recipe-style defaults).

Determinism notes:
- Comparison is on ECGDataset.X/.y (the partition CONTENTS), never batch order — the train
  DataLoader uses shuffle=True with no generator, so batch order is non-deterministic by design.
- get_or_create_split caches to ./data_splits/ keyed on (num_clients, alpha, frac, seed) — NOT on
  test_size/batch. The cwd is a tmp dir and the cache is cleared before each call so both sides
  recompute from scratch, making the test sensitive to test_size drift the cache key would hide.
- Fully offline: a synthetic CSV, no network, no real ecg.csv dependency.
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402
from data_loaders.ecg_loader import get_ecg_loaders, get_test_loader  # noqa: E402

# ECG config values (config.get_dataset_config("ecg")) the legacy call sites thread in by hand.
ECG_NUM_CLIENTS = 5
ECG_ALPHA = 1.0
ECG_TEST_SIZE = 0.2
ECG_SEED = 42
ECG_BATCH_TRAIN = 128
ECG_BATCH_TEST = 128
ECG_DATA_FRACTION = 1.0


def _write_synthetic_csv(path):
    """200 rows x (140 features + 1 label); both classes present so stratify + dirichlet work."""
    rng = np.random.RandomState(0)
    X = rng.randn(200, 140).astype(np.float32)
    y = np.array([0, 1] * 100, dtype=np.int64)
    pd.DataFrame(np.column_stack([X, y])).to_csv(path, header=False, index=False)
    return path


def _read_csv(path):
    df = pd.read_csv(path, header=None)
    return df.iloc[:, :-1].values.astype(np.float32), df.iloc[:, -1].values.astype(np.int64)


def _clear_cache():
    shutil.rmtree("data_splits", ignore_errors=True)


def test_recipe_mlp_client_partition_matches_legacy_ecg_loader(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = _write_synthetic_csv(os.path.join(tmp_path, "ecg.csv"))
    X, y = _read_csv(csv)

    _clear_cache()
    legacy_train, _, _ = get_ecg_loaders(
        X=X, y=y, client_id=0, num_clients=ECG_NUM_CLIENTS,
        batch_size_train=ECG_BATCH_TRAIN, batch_size_test=ECG_BATCH_TEST,
        data_fraction=ECG_DATA_FRACTION, alpha=ECG_ALPHA, test_size=ECG_TEST_SIZE,
        num_workers=0, seed=ECG_SEED)

    _clear_cache()
    recipe_train, _ = recipes.get_recipe("MLP").load_client_data(
        partition_id=0, num_clients=ECG_NUM_CLIENTS, dataset_path=csv)

    assert len(recipe_train.dataset) > 0  # not a degenerate empty-vs-empty pass
    assert torch.equal(legacy_train.dataset.X, recipe_train.dataset.X)
    assert torch.equal(legacy_train.dataset.y, recipe_train.dataset.y)


def test_recipe_mlp_server_testset_matches_legacy_ecg_loader(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = _write_synthetic_csv(os.path.join(tmp_path, "ecg.csv"))
    X, y = _read_csv(csv)

    _clear_cache()
    legacy_test, _ = get_test_loader(
        X=X, y=y, num_clients=ECG_NUM_CLIENTS, batch_size=ECG_BATCH_TEST,
        alpha=ECG_ALPHA, data_fraction=ECG_DATA_FRACTION, test_size=ECG_TEST_SIZE,
        num_workers=0, seed=ECG_SEED)

    _clear_cache()
    recipe_test = recipes.get_recipe("MLP").load_server_test_data(
        num_clients=ECG_NUM_CLIENTS, dataset_path=csv)

    assert len(recipe_test.dataset) > 0
    assert torch.equal(legacy_test.dataset.X, recipe_test.dataset.X)
    assert torch.equal(legacy_test.dataset.y, recipe_test.dataset.y)
