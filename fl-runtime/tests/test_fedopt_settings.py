import argparse
import pytest
import fl_server


def test_defaults_and_custom_fedopt_settings():
    parser = fl_server.build_arg_parser()
    args = parser.parse_args(['--project-id', 'p', '--strategy', 'fedopt', '--model-path', 'm.npz', '--model-type', 'CNN', '--model-name', 'net'])
    assert fl_server.resolve_fedopt_settings(args) == dict(variant='adam', server_learning_rate=1., beta1=.9, beta2=.99, tau=.001)
    args.fedopt_server_learning_rate = .01
    args.fedopt_variant = 'yogi'
    assert fl_server.resolve_fedopt_settings(args)['server_learning_rate'] == .01
    assert fl_server.resolve_fedopt_settings(args)['variant'] == 'yogi'


@pytest.mark.parametrize('field,value', [('server_learning_rate', 0), ('server_learning_rate', float('nan')), ('server_learning_rate', 11), ('beta1', 1), ('beta2', -.1), ('tau', float('inf')), ('variant', 'bogus')])
def test_invalid_fedopt_settings_are_refused(field, value):
    args = argparse.Namespace(strategy='fedopt', **{'fedopt_' + field: value})
    with pytest.raises(ValueError):
        fl_server.resolve_fedopt_settings(args)


def test_settings_on_other_strategy_are_refused():
    with pytest.raises(ValueError, match='only'):
        fl_server.resolve_fedopt_settings(argparse.Namespace(strategy='fedavg', fedopt_server_learning_rate=.01))


def test_production_strategy_uses_custom_values():
    from collections import OrderedDict
    import torch
    args = argparse.Namespace(strategy='fedopt', min_clients=1, fedopt_variant='yogi',
                              fedopt_server_learning_rate=.01, fedopt_beta1=.8,
                              fedopt_beta2=.95, fedopt_tau=.002)
    strategy = fl_server.select_strategy(args, OrderedDict(w=torch.zeros(1)), None)
    assert (strategy.variant, strategy.eta, strategy.beta1, strategy.beta2, strategy.tau) == ('yogi', .01, .8, .95, .002)
