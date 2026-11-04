"""Durable round state must restore the same training trajectory without pickle."""
from collections import OrderedDict
import hashlib
import os
import random
import zipfile

import numpy as np
import pytest
import torch

from fedlearn.server.checkpoint import (
    CheckpointIdentity,
    RoundCheckpointStore,
    capture_round,
    restore_strategy,
)
from fedlearn.server.strategy import DPFedAvg, FedAvg, FedOpt, FedProx


def parameters(value=0.0):
    return OrderedDict(weight=torch.tensor([value, value + 1], dtype=torch.float32))


def identity(run_id="run-1", configuration="original"):
    return CheckpointIdentity(run_id, hashlib.sha256(configuration.encode()).hexdigest())


def make_strategy(kind="avg", **overrides):
    options = dict(initial_parameters=parameters(), min_fit_clients=2, clients_per_round=2)
    if kind == "prox":
        options.update(proximal_mu=.1, learning_rate=.01, local_epochs=2)
        constructor = FedProx
    elif kind in ("adam", "yogi"):
        options.update(variant=kind, server_learning_rate=.2, bias_correction=True)
        constructor = FedOpt
    else:
        constructor = FedAvg
    return constructor(**(options | overrides))


def snapshot(strategy=None, completed_round=1):
    strategy = strategy or make_strategy()
    for number in range(1, completed_round + 1):
        model = strategy.aggregate_fit(number, [(parameters(2), 2), (parameters(5), 3)])
    return capture_round(
        strategy, completed_round=completed_round, total_rounds=4, parameters=model,
        history=[(number, {"loss": .25}) for number in range(1, completed_round + 1)],
        partition_to_client={0: "a", 1: "b"},
    )


def test_checkpoint_round_trip_preserves_order_history_and_identity(tmp_path):
    store = RoundCheckpointStore(tmp_path / "private", identity())
    saved = snapshot()
    path = store.save(saved)
    loaded = store.load_latest()
    assert loaded.completed_round == 1
    assert loaded.total_rounds == 4
    assert loaded.history == [(1, {"loss": .25})]
    assert loaded.partition_to_client == {0: "a", 1: "b"}
    assert list(loaded.parameters) == ["weight"]
    assert torch.equal(loaded.parameters["weight"], saved.parameters["weight"])
    assert path.stat().st_mode & 0o077 == 0
    assert path.parent.stat().st_mode & 0o077 == 0
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {"manifest.json", "state.safetensors"}


@pytest.mark.parametrize("kind", ["avg", "prox", "adam", "yogi"])
def test_resumed_strategy_matches_uninterrupted_next_round(tmp_path, kind):
    uninterrupted = make_strategy(kind)
    store = RoundCheckpointStore(tmp_path / "private", identity())
    first = snapshot(uninterrupted)
    store.save(first)
    # The production aggregator consumes and clears upload dictionaries.
    expected = uninterrupted.aggregate_fit(2, [(parameters(-2), 2), (parameters(3), 5)])
    restored = make_strategy(kind)
    restore_strategy(restored, store.load_latest())
    actual = restored.aggregate_fit(2, [(parameters(-2), 2), (parameters(3), 5)])
    for name in expected:
        assert torch.equal(actual[name], expected[name])


def test_restore_recovers_python_numpy_and_torch_random_streams(tmp_path):
    store = RoundCheckpointStore(tmp_path / "private", identity())
    store.save(snapshot())
    expected = (random.random(), np.random.random(4), torch.rand(4))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_strategy(make_strategy(), store.load_latest())
    assert random.random() == expected[0]
    np.testing.assert_array_equal(np.random.random(4), expected[1])
    assert torch.equal(torch.rand(4), expected[2])


@pytest.mark.parametrize("other", [identity("run-2"), identity(configuration="changed")])
def test_wrong_run_or_configuration_is_refused(tmp_path, other):
    RoundCheckpointStore(tmp_path / "private", identity()).save(snapshot())
    with pytest.raises(ValueError, match="identity"):
        RoundCheckpointStore(tmp_path / "private", other).load_latest()


@pytest.mark.parametrize("kind,overrides", [
    ("prox", {}),
    ("avg", {"initial_parameters": OrderedDict(other=torch.zeros(2))}),
    ("avg", {"initial_parameters": OrderedDict(weight=torch.zeros(3))}),
    ("avg", {"clients_per_round": 3}),
])
def test_incompatible_restore_fails_before_mutating_target_or_randomness(kind, overrides):
    saved = snapshot()
    target = make_strategy(kind, **overrides)
    before = {name: value.clone() for name, value in target.initial_parameters.items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="configuration|strategy|layout"):
        restore_strategy(target, saved)
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in before.items():
        assert torch.equal(target.initial_parameters[name], value)


def test_committed_generation_cannot_be_overwritten(tmp_path):
    store = RoundCheckpointStore(tmp_path / "private", identity())
    saved = snapshot()
    path = store.save(saved)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        store.save(saved)
    assert path.read_bytes() == original


def test_failed_publication_retains_previous_committed_generation(tmp_path, monkeypatch):
    store = RoundCheckpointStore(tmp_path / "private", identity())
    store.save(snapshot())

    def disk_failure(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "link", disk_failure)
    with pytest.raises(OSError, match="disk full"):
        store.save(snapshot(completed_round=2))
    assert store.load_latest().completed_round == 1
    assert len(list(store.directory.iterdir())) == 1


def test_corrupt_latest_generation_is_refused_instead_of_rewinding(tmp_path):
    store = RoundCheckpointStore(tmp_path / "private", identity())
    store.save(snapshot())
    newest = store.save(snapshot(completed_round=2))
    newest.write_bytes(newest.read_bytes()[:40])
    with pytest.raises(ValueError, match="checkpoint"):
        store.load_latest()


def test_tensor_payload_checksum_is_verified(tmp_path):
    store = RoundCheckpointStore(tmp_path / "private", identity())
    path = store.save(snapshot())
    with zipfile.ZipFile(path) as archive:
        manifest = archive.read("manifest.json")
        state = bytearray(archive.read("state.safetensors"))
    state[-1] ^= 1
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", manifest)
        archive.writestr("state.safetensors", state)
    with pytest.raises(ValueError, match="checksum"):
        store.load_latest()


def test_generation_filename_must_match_recorded_round(tmp_path):
    store = RoundCheckpointStore(tmp_path / "private", identity())
    path = store.save(snapshot())
    path.rename(store.directory / "round-000000000002.ckpt")
    with pytest.raises(ValueError, match="round"):
        store.load_latest()


def test_nonfinite_model_is_never_persisted(tmp_path):
    store = RoundCheckpointStore(tmp_path / "private", identity())
    saved = snapshot()
    saved.parameters["weight"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        store.save(saved)
    assert store.load_latest() is None


def test_private_strategy_is_refused_until_accounting_adapter_is_qualified():
    strategy = DPFedAvg(parameters(), min_fit_clients=2, clients_per_round=2,
                        dp_clip_norm=1, dp_noise_multiplier=2, dp_seed=3)
    with pytest.raises(ValueError, match="unsupported checkpoint strategy"):
        snapshot(strategy)


def test_insecure_checkpoint_permissions_are_refused(tmp_path):
    store = RoundCheckpointStore(tmp_path / "private", identity())
    path = store.save(snapshot())
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        store.load_latest()


def test_invalid_identity_is_rejected():
    with pytest.raises(ValueError, match="identity"):
        CheckpointIdentity("", "not-a-digest")


def test_store_refuses_metadata_that_its_reader_cannot_load(tmp_path):
    store = RoundCheckpointStore(tmp_path / "private", identity())
    store.save(snapshot())
    saved = snapshot(completed_round=2)
    saved.history[0][1]["diagnostic"] = "x" * (4 * 1024 * 1024)
    with pytest.raises(ValueError, match="size"):
        store.save(saved)
    assert store.load_latest().completed_round == 1


def test_store_refuses_oversized_tensor_payload_before_publication(tmp_path, monkeypatch):
    from fedlearn.server import checkpoint
    store = RoundCheckpointStore(tmp_path / "private", identity())
    monkeypatch.setattr(checkpoint, "MAX_STATE_BYTES", 128, raising=False)
    with pytest.raises(ValueError, match="size"):
        store.save(snapshot())
    assert store.load_latest() is None


def test_changed_model_shape_cannot_poison_the_latest_generation(tmp_path):
    store = RoundCheckpointStore(tmp_path / "private", identity())
    store.save(snapshot())
    changed = snapshot(completed_round=2)
    changed.parameters["weight"] = torch.zeros(3)
    with pytest.raises(ValueError, match="layout"):
        store.save(changed)
    assert store.load_latest().completed_round == 1


@pytest.mark.parametrize("history", [[], [(1, None), (3, None)], [(1, None), (2, None)]])
def test_incomplete_history_cannot_be_committed_as_a_completed_round(tmp_path, history):
    store = RoundCheckpointStore(tmp_path / "private", identity())
    saved = snapshot(completed_round=3)
    saved.history = history
    with pytest.raises(ValueError, match="history"):
        store.save(saved)
    assert store.load_latest() is None


def test_completed_rounds_without_evaluation_retain_explicit_history_entries(tmp_path):
    store = RoundCheckpointStore(tmp_path / "private", identity())
    saved = snapshot(completed_round=3)
    saved.history = [(1, None), (2, None), (3, None)]
    store.save(saved)
    assert store.load_latest().history == saved.history
