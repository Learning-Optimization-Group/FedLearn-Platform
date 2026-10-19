"""Server optimizer moments must never adapt non-parameter running statistics."""
from collections import OrderedDict

import pytest
import torch

from fedlearn.server.strategy import FedOpt


@pytest.mark.parametrize('variant', ['adam', 'yogi'])
def test_declared_buffers_are_weighted_means_without_moment_state(variant):
    initial = OrderedDict(weight=torch.tensor([1.]), running_var=torch.tensor([.01]))
    strategy = FedOpt(initial, variant=variant, buffer_names=['running_var'])
    for round_ in (1, 2):
        result = strategy.aggregate_fit(round_, [
            (OrderedDict(weight=torch.tensor([1.01]), running_var=torch.tensor([.009])), 1),
            (OrderedDict(weight=torch.tensor([1.02]), running_var=torch.tensor([.008])), 3),
        ])
        expected = (.009 + 3 * .008) / 4
        assert result['running_var'].item() == pytest.approx(expected)
        assert set(strategy._m) == {'weight'}
        assert set(strategy._v) == {'weight'}
        assert torch.isfinite(result['weight']).all()


def test_unknown_buffer_name_is_refused_at_construction():
    with pytest.raises(ValueError, match='buffer names'):
        FedOpt(OrderedDict(weight=torch.tensor([1.])), buffer_names=['unknown'])
