"""The golden TinyNet evaluator skips datasets, so startup must not download one."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

import fl_server


def test_tinynet_startup_does_not_load_an_unused_image_dataset(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('TinyNet must not load or download a dataset')

    monkeypatch.setattr(fl_server, 'load_server_test_data', unexpected)
    monkeypatch.setattr(fl_server.recipes, 'get_recipe', unexpected)
    args = SimpleNamespace(model_type='TINYNET_GOLDEN', dataset='cb')
    assert fl_server.load_evaluation_data(args, is_llm=False, num_clients=2, dataset_path=None) is None


def test_cnn_startup_keeps_its_existing_evaluation_loader(monkeypatch):
    expected = object()
    calls = []

    def load(is_llm, dataset):
        calls.append((is_llm, dataset))
        return expected

    monkeypatch.setattr(fl_server, 'load_server_test_data', load)
    args = SimpleNamespace(model_type='CNN', dataset='cb')
    assert fl_server.load_evaluation_data(args, is_llm=False, num_clients=2, dataset_path=None) is expected
    assert calls == [(False, None)]


def test_the_real_tinynet_server_entry_point_reaches_serving_without_a_dataset(monkeypatch, tmp_path):
    model = fl_server.get_model('TINYNET_GOLDEN', 'tinynet-golden', 'cpu')
    checkpoint = tmp_path / 'initial.npz'
    np.savez(checkpoint, **{k.replace('.', '__DOT__'): v.detach().numpy() for k, v in model.state_dict().items()})

    def unused_dataset(*args, **kwargs):
        raise AssertionError('A golden run must never request an image dataset')

    class StartedServing(Exception):
        pass

    def serve(*args, **kwargs):
        raise StartedServing

    monkeypatch.setattr(fl_server, 'load_server_test_data', unused_dataset)
    monkeypatch.setattr(fl_server.fl.server, 'start_server', serve)
    monkeypatch.setattr(sys, 'argv', ['fl_server.py', '--project-id', 'golden-test', '--model-path', str(checkpoint),
                                    '--model-type', 'TINYNET_GOLDEN', '--model-name', 'tinynet-golden',
                                    '--strategy', 'FedAvg', '--num-rounds', '1', '--min-clients', '1'])
    with pytest.raises(StartedServing):
        fl_server.main()
