"""A failed round must not register a model or call the finished callback."""
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

import fl_server
import recipes


def test_failed_server_result_keeps_checkpoint_and_never_reports_completion(tmp_path, monkeypatch):
    model = recipes.get_recipe('TINYNET_GOLDEN').build_model('cpu')
    checkpoint = tmp_path / 'initial.npz'
    np.savez(checkpoint, **{k.replace('.', '__DOT__'): v.numpy() for k, v in model.state_dict().items()})
    original = checkpoint.read_bytes()
    monkeypatch.setattr(sys, 'argv', ['fl_server.py', '--model-path', str(checkpoint), '--project-id',
        '9b1e6d3a-2c47-4f85-a0d9-3e7c1b5a8f64', '--model-type', 'TINYNET_GOLDEN',
        '--model-name', 'tinynet_golden', '--strategy', 'FedAvg', '--num-rounds', '1', '--seed', '42'])
    monkeypatch.setattr(fl_server.fl.server, 'start_server', lambda **kw: ([], {}))
    monkeypatch.setattr(fl_server, 'INTERNAL_API_KEY', 'test-key')
    post = MagicMock()
    monkeypatch.setattr(fl_server.requests, 'post', post)
    with pytest.raises(RuntimeError, match='completed.*model|model.*completed'):
        fl_server.main()
    assert checkpoint.read_bytes() == original
    post.assert_not_called()
