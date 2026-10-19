"""Bounded research controls must refuse unsupported cohort budgets before fitting."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from google.protobuf.json_format import MessageToDict

from fedlearn.communication.generated import execution_contract_pb2 as pb


@pytest.mark.parametrize('clients', [1, 3])
def test_probe_refuses_a_different_cohort_before_opening_model_inputs(tmp_path, clients):
    contract = pb.ExecutionContract(recipe=pb.RECIPE_CIFAR_RESNET18, strategy=pb.STRATEGY_FEDPROX,
                                    num_rounds=2, clients_per_round=clients)
    contract.model_training.fedprox_mu = .1
    contract.model_training.local_training.batch_size = 32
    contract.model_training.data.source = pb.DATA_SOURCE_LOCAL_SNAPSHOT
    path = tmp_path / 'contract.json'
    path.write_text(json.dumps(MessageToDict(contract)))
    driver = Path(__file__).resolve().parents[2] / 'research/benchmarks/resnet_client_rates.py'
    result = subprocess.run([sys.executable, str(driver), '--contract', str(path),
                             '--initial-state', str(tmp_path / 'missing.npz'),
                             '--phone-package', str(tmp_path / 'missing-phone.zip'),
                             '--desktop-package', str(tmp_path / 'missing-desktop.zip'),
                             '--output-dir', str(tmp_path / 'output')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'requires exactly two clients' in result.stderr


@pytest.mark.parametrize('bad_logits', [float('nan'), float('inf'), -float('inf')])
def test_probe_refuses_nonfinite_diagnostic_logits(bad_logits):
    import importlib.util
    import torch
    driver = Path(__file__).resolve().parents[2] / 'research/benchmarks/resnet_client_rates.py'
    spec = importlib.util.spec_from_file_location('resnet_rate_probe', driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match='non-finite'):
        module.finite_cross_entropy(torch.tensor([[bad_logits, 0.]]), torch.tensor([0]))


def test_probe_refuses_loss_overflow_even_with_finite_logits():
    import importlib.util
    import torch
    driver = Path(__file__).resolve().parents[2] / 'research/benchmarks/resnet_client_rates.py'
    spec = importlib.util.spec_from_file_location('resnet_rate_probe', driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match='non-finite'):
        module.finite_cross_entropy(torch.tensor([[-3e38, 3e38]]), torch.tensor([0]))


def test_longer_controls_refuse_training_examples_in_validation():
    import importlib.util
    import torch
    driver = Path(__file__).resolve().parents[2] / 'research/benchmarks/resnet_client_rates.py'
    spec = importlib.util.spec_from_file_location('resnet_rate_probe', driver)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from torch.utils.data import TensorDataset
    first = TensorDataset(torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([0, 1]))
    heldout = TensorDataset(torch.tensor([[1., 2.]]), torch.tensor([1]))
    with pytest.raises(ValueError, match='overlap'):
        module.require_validation_disjoint([first], heldout)
    separated = TensorDataset(torch.tensor([[5., 6.]]), torch.tensor([1]))
    module.require_validation_disjoint([first], separated)


def test_server_rate_control_changes_real_fedopt_updates_without_changing_client_rate():
    import importlib.util
    from collections import OrderedDict
    import torch
    driver = Path(__file__).resolve().parents[2] / 'research/benchmarks/resnet_client_rates.py'
    spec = importlib.util.spec_from_file_location('resnet_rate_probe', driver)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    initial = OrderedDict(w=torch.tensor([0.]))
    # Aggregation consumes its input dictionaries; each independent control needs fresh updates.
    results = [module.make_aggregator('FedOpt', initial, rate).aggregate_fit(1,
        [('0', OrderedDict(w=torch.tensor([.1])), 64),
         ('1', OrderedDict(w=torch.tensor([.1])), 64)])['w'] for rate in (.001, .01)]
    assert torch.allclose(results[1], results[0] * 10)
    assert results[0].item() > 0
    import strategy_client_settings
    assert strategy_client_settings.client_learning_rate('FedOpt', 'CIFAR_RESNET18') == .001


def test_two_round_probe_refuses_fedavg_rate_labels_before_opening_inputs(tmp_path):
    driver = Path(__file__).resolve().parents[2] / 'research/benchmarks/resnet_client_rates.py'
    command = [sys.executable, str(driver)]
    for option in ('contract', 'initial-state', 'phone-package', 'desktop-package'):
        command += ['--'+option, str(tmp_path/'absent')]
    output = tmp_path/'results'
    result = subprocess.run(command + ['--output-dir', str(output), '--strategies', 'FedAvg'],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert 'FedAvg requires --longer' in result.stderr
    assert not output.exists()
