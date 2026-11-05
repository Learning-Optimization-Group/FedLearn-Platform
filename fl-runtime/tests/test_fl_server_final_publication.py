"""Final reporting must use persisted model bytes and declared metric units."""
from collections import OrderedDict
from pathlib import Path
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import fl_server
import recipes


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    model = recipes.get_recipe('TINYNET_GOLDEN').build_model('cpu')
    path = tmp_path / 'model.npz'
    np.savez(path, **{k.replace('.', '__DOT__'): v.numpy() for k, v in model.state_dict().items()})
    original = path.read_bytes()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'argv', ['fl_server.py', '--model-path', str(path), '--project-id',
        '9b1e6d3a-2c47-4f85-a0d9-3e7c1b5a8f64', '--model-type', 'TINYNET_GOLDEN',
        '--model-name', 'tinynet_golden', '--strategy', 'FedAvg', '--num-rounds', '1', '--seed', '42'])
    monkeypatch.setattr(fl_server, 'INTERNAL_API_KEY', 'test-key')
    monkeypatch.setattr(fl_server, 'INTERNAL_RUN_TOKEN', 'test-run-token')
    posts = MagicMock()
    monkeypatch.setattr(fl_server.requests, 'post', posts)
    history = [(1, {'loss': .2, 'accuracy': 75.})]
    expected = OrderedDict()

    def server(**kwargs):
        expected.update((k, v.clone() + .25) for k, v in kwargs['strategy'].initial_parameters.items())
        return history, expected

    monkeypatch.setattr(fl_server.fl.server, 'start_server', server)
    return path, original, posts, history, expected


def partial_write(destination, **weights):
    if hasattr(destination, 'write'):
        destination.write(b'incomplete archive')
    else:
        Path(destination).write_bytes(b'incomplete archive')
    raise OSError('injected partial model write')


def test_failed_model_write_cannot_publish_stale_bytes_or_finish(completed_run, monkeypatch):
    path, original, posts, _, _ = completed_run
    monkeypatch.setattr(fl_server.np, 'savez', partial_write)
    with pytest.raises(RuntimeError, match='sav|persist'):
        fl_server.main()
    posts.assert_not_called()
    assert path.read_bytes() == original


def test_partial_model_write_preserves_the_previous_readable_model(completed_run, monkeypatch):
    path, original, _, _, _ = completed_run
    monkeypatch.setattr(fl_server.np, 'savez', partial_write)
    try:
        fl_server.main()
    except RuntimeError:
        pass
    assert path.read_bytes() == original
    with np.load(path, allow_pickle=False) as saved:
        assert len(saved.files) > 0
    assert sorted(p.name for p in path.parent.iterdir()) == ['model.npz']


@pytest.mark.parametrize('boundary', ['fsync', 'replace'])
def test_failed_model_commit_does_not_publish_or_finish(completed_run, monkeypatch, boundary):
    path, original, posts, _, _ = completed_run

    def fail(*args, **kwargs):
        raise OSError('injected model commit failure')

    monkeypatch.setattr(fl_server.os, boundary, fail)
    with pytest.raises(RuntimeError, match='sav|persist'):
        fl_server.main()
    posts.assert_not_called()
    assert path.read_bytes() == original
    assert sorted(p.name for p in path.parent.iterdir()) == ['model.npz']


@pytest.mark.parametrize('percentage', [0., .5, 1., 75., 100.])
def test_result_accuracy_converts_percentage_history_to_fraction(completed_run, percentage):
    path, _, posts, history, expected = completed_run
    history[0][1]['accuracy'] = percentage
    fl_server.main()
    results = [call.kwargs['json'] for call in posts.call_args_list
               if '/api/internal/results/' in call.args[0] and 'json' in call.kwargs]
    assert len(results) == 1
    assert results[0]['accuracy'] == percentage / 100.
    with np.load(path, allow_pickle=False) as saved:
        for name, value in expected.items():
            np.testing.assert_array_equal(saved[name.replace('.', '__DOT__')], value.numpy())
    assert posts.call_args_list[-1].args[0].endswith('/finished')


def test_failed_directory_sync_refuses_callbacks_after_complete_replacement(completed_run, monkeypatch):
    path, original, posts, _, expected = completed_run
    original_sync = fl_server.os.fsync
    calls = []

    def fail_directory_sync(descriptor):
        calls.append(descriptor)
        if len(calls) == 2:
            raise OSError('injected directory durability failure')
        original_sync(descriptor)

    monkeypatch.setattr(fl_server.os, 'fsync', fail_directory_sync)
    with pytest.raises(RuntimeError, match='sav|persist'):
        fl_server.main()
    posts.assert_not_called()
    assert len(calls) == 2
    assert path.read_bytes() != original
    with np.load(path, allow_pickle=False) as saved:
        for name, value in expected.items():
            np.testing.assert_array_equal(saved[name.replace('.', '__DOT__')], value.numpy())
    assert sorted(p.name for p in path.parent.iterdir()) == ['model.npz']


def test_readers_keep_the_previous_complete_model_until_replacement(completed_run, monkeypatch):
    path, original, _, _, _ = completed_run
    original_save = fl_server.np.savez
    observed = []

    def inspect_before_commit(output, **weights):
        original_save(output, **weights)
        observed.append(path.read_bytes())
        with np.load(path, allow_pickle=False) as saved:
            assert len(saved.files) > 0

    monkeypatch.setattr(fl_server.np, 'savez', inspect_before_commit)
    fl_server.main()
    assert observed == [original]
    assert path.read_bytes() != original


def test_absent_accuracy_is_not_reported_as_a_measured_zero(completed_run):
    _, _, posts, history, _ = completed_run
    del history[0][1]['accuracy']
    fl_server.main()
    results = [call.kwargs['json'] for call in posts.call_args_list
               if '/api/internal/results/' in call.args[0] and 'json' in call.kwargs]
    assert results[0]['accuracy'] is None
