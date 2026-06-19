import pytest
import torch
import numpy as np
from collections import OrderedDict
from fedlearn.server.decomfl_strategy import DeComFL


def make_params(val: float = 0.0) -> OrderedDict:
    return OrderedDict([
        ("fc.weight", torch.tensor([[val, val, val]], dtype=torch.float32)),
        ("fc.bias",   torch.tensor([val], dtype=torch.float32)),
    ])


class TestDeComFLStrategy:

    def setup_method(self):
        self.strategy = DeComFL(
            initial_parameters=make_params(0.0),
            evaluate_fn=None,
            min_fit_clients=1,
            clients_per_round=2,
            num_local_steps=2,   # K=2
            num_perturbations=3, # P=3
            learning_rate=0.01,
            smoothing_param=0.001,
            seed=42,
        )

    def test_initialize_parameters_returns_initial(self):
        params = self.strategy.initialize_parameters()
        assert params is not None
        assert "fc.weight" in params

    def test_generate_seeds_returns_correct_shape(self):
        seeds = self.strategy.generate_seeds(round_idx=0)
        # Should be K x P = 2 x 3
        assert len(seeds) == 2       # K local steps
        assert len(seeds[0]) == 3   # P perturbations

    def test_generate_seeds_values_are_non_negative_integers(self):
        seeds = self.strategy.generate_seeds(round_idx=0)
        for k_seeds in seeds:
            for s in k_seeds:
                assert isinstance(s, int)
                assert s >= 0

    def test_flatten_unflatten_roundtrip(self):
        original = make_params(3.14)
        flat = self.strategy._flatten_params(original)
        recovered = self.strategy._unflatten_params(flat, original)
        for key in original:
            assert torch.allclose(original[key], recovered[key], atol=1e-6)

    def test_evaluate_returns_none_without_evaluate_fn(self):
        result = self.strategy.evaluate(1, make_params(0.0))
        assert result is None

    def test_aggregate_fit_empty_results_returns_none(self):
        result = self.strategy.aggregate_fit(server_round=1, results=[])
        assert result is None

    def test_aggregate_fit_updates_global_params(self):
        # Populate seed history so aggregate_fit can look up seeds
        seeds = self.strategy.generate_seeds(round_idx=0)
        self.strategy.seed_history[0] = seeds   # dict keyed by round (matches aggregate_fit(0))

        # Build fake gradient scalars: shape [K][P] = [2][3]
        grads = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        results = [
            ("c1", grads, 100),
            ("c2", grads, 100),
        ]
        result = self.strategy.aggregate_fit(server_round=0, results=results)
        assert result is not None
        assert isinstance(result, OrderedDict)
        assert "fc.weight" in result
