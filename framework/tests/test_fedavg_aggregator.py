import pytest
import torch
from collections import OrderedDict
from fedlearn.server.strategy import FedAvgAggregator


def make_params(val: float) -> OrderedDict:
    """Helper: create a simple 2-parameter model state dict."""
    return OrderedDict([
        ("layer.weight", torch.tensor([[val, val]])),
        ("layer.bias", torch.tensor([val])),
    ])


class TestFedAvgAggregator:

    def setup_method(self):
        self.aggregator = FedAvgAggregator()

    def test_aggregate_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            self.aggregator.aggregate([])

    def test_aggregate_single_client_returns_same_values(self):
        updates = [(None, make_params(3.0), 100)]
        result = self.aggregator.aggregate(updates)
        assert torch.allclose(result["layer.bias"], torch.tensor([3.0]), atol=1e-5)

    def test_aggregate_two_clients_equal_weight(self):
        # Client A: value=2.0, Client B: value=4.0, equal samples -> avg=3.0
        updates = [
            (None, make_params(2.0), 100),
            (None, make_params(4.0), 100),
        ]
        result = self.aggregator.aggregate(updates)
        assert torch.allclose(result["layer.bias"], torch.tensor([3.0]), atol=1e-4)

    def test_aggregate_weighted_by_samples(self):
        # Client A: value=0.0, 100 samples. Client B: value=10.0, 900 samples.
        # Expected: (0*100 + 10*900) / 1000 = 9.0
        updates = [
            (None, make_params(0.0), 100),
            (None, make_params(10.0), 900),
        ]
        result = self.aggregator.aggregate(updates)
        assert torch.allclose(result["layer.bias"], torch.tensor([9.0]), atol=1e-4)

    def test_aggregate_filters_non_positive_num_examples(self):
        # Only the valid update (100 samples) should be used.
        # An entry with num_examples=0 must be silently dropped.
        updates = [
            (None, make_params(5.0), 0),   # invalid - should be dropped
            (None, make_params(5.0), 100), # valid
        ]
        result = self.aggregator.aggregate(updates)
        assert torch.allclose(result["layer.bias"], torch.tensor([5.0]), atol=1e-4)

    def test_aggregate_all_invalid_raises_value_error(self):
        updates = [(None, make_params(1.0), 0), (None, make_params(2.0), -1)]
        with pytest.raises(ValueError, match="No valid updates"):
            self.aggregator.aggregate(updates)

    def test_aggregate_caps_num_examples_at_max_samples(self):
        # Both clients claim 200_000 samples, but cap is 100_000.
        # After capping: both have equal weight -> average = 5.0
        updates = [
            (None, make_params(2.0), 200_000),
            (None, make_params(8.0), 200_000),
        ]
        result = self.aggregator.aggregate(updates)
        assert torch.allclose(result["layer.bias"], torch.tensor([5.0]), atol=1e-4)
