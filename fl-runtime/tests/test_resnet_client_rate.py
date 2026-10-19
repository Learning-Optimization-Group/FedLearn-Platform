"""New adaptive ResNet runs must publish the rate their server actually requests."""
from argparse import Namespace
from collections import OrderedDict

import pytest
import torch

import execution_plan
import fl_server


@pytest.mark.parametrize('strategy', ['FedOpt', 'FedProx'])
@pytest.mark.parametrize('recipe,rate', [('CIFAR_RESNET18', .001), ('CNN', .01),
                                       ('MLP', .01), ('PNEUMONIA_CNN', .01),
                                       ('TINYNET_GOLDEN', .01)])
def test_adaptive_server_and_plan_agree_on_recipe_rate(strategy, recipe, rate):
    args = Namespace(strategy=strategy, model_type=recipe, min_clients=2,
                     clients_per_round=2, fedopt_server_learning_rate=.01)
    server = fl_server.select_strategy(args, OrderedDict(weight=torch.ones(1)), None)
    plan = execution_plan.resolve_model_training(recipe, strategy, 'FULL', data_source='LOCAL_SNAPSHOT')
    optimizer = plan.local_training.WhichOneof('optimizer')
    assert float(server.get_client_config()['learning_rate']) == rate
    assert getattr(plan.local_training, optimizer).learning_rate == rate
    assert int(server.get_client_config()['local_epochs']) == plan.local_training.local_epochs == 1
    if strategy == 'FedOpt':
        assert server.eta == .01
    else:
        assert server.mu == plan.fedprox_mu == .1


@pytest.mark.parametrize('strategy', ['FedOpt', 'FedProx'])
def test_old_caller_without_recipe_keeps_legacy_client_rate(strategy):
    server = fl_server.select_strategy(Namespace(strategy=strategy, min_clients=1),
                                       OrderedDict(weight=torch.ones(1)), None)
    assert float(server.get_client_config()['learning_rate']) == .01


@pytest.mark.parametrize('strategy', ['FedOpt', 'FedProx'])
def test_server_normalizes_resnet_recipe_case(strategy):
    server = fl_server.select_strategy(Namespace(strategy=strategy, model_type='cifar_resnet18', min_clients=1),
                                       OrderedDict(weight=torch.ones(1)), None)
    assert float(server.get_client_config()['learning_rate']) == .001


@pytest.mark.parametrize('strategy,pb_strategy', [('FedOpt', execution_plan.pb.STRATEGY_FEDOPT),
                                                ('FedProx', execution_plan.pb.STRATEGY_FEDPROX)])
def test_old_contract_keeps_recorded_rate_and_refuses_changed_round(monkeypatch, strategy, pb_strategy):
    import client
    contract = execution_plan.pb.ExecutionContract(strategy=pb_strategy)
    contract.model_training.CopyFrom(execution_plan.resolve_model_training('CIFAR_RESNET18', strategy, 'FULL'))
    contract.model_training.local_training.adam.learning_rate = .01
    before = contract.SerializeToString()
    monkeypatch.setattr(client, 'EXECUTION_CONTRACT', contract)
    config = dict(learning_rate='.01', local_epochs='1', proximal_mu='.1' if strategy == 'FedProx' else '0')
    client._refuse_round_outside_contract(config)
    with pytest.raises(ValueError, match='execution contract'):
        client._refuse_round_outside_contract(dict(config, learning_rate='.001'))
    assert contract.SerializeToString() == before
