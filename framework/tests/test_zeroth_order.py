import pytest
import torch
import torch.nn as nn
from fedlearn.estimators.zeroth_order import ZerothOrderEstimator


class TinyModel(nn.Module):
    """Tiny 2-layer model for testing. Total params: 4*3 + 3 + 3*2 + 2 = 23"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestZerothOrderEstimator:

    def setup_method(self):
        self.estimator = ZerothOrderEstimator(smoothing_param=0.001, device='cpu')
        self.model = TinyModel()

    def test_generate_perturbation_same_seed_same_result(self):
        num_params = ZerothOrderEstimator.get_num_params(self.model)
        z1 = self.estimator.generate_perturbation(seed=42, num_params=num_params)
        z2 = self.estimator.generate_perturbation(seed=42, num_params=num_params)
        assert torch.allclose(z1, z2), "Same seed must produce identical perturbation"

    def test_generate_perturbation_different_seeds_different_results(self):
        num_params = ZerothOrderEstimator.get_num_params(self.model)
        z1 = self.estimator.generate_perturbation(seed=42, num_params=num_params)
        z2 = self.estimator.generate_perturbation(seed=99, num_params=num_params)
        assert not torch.allclose(z1, z2), "Different seeds should produce different perturbations"

    def test_generate_perturbation_correct_shape(self):
        num_params = ZerothOrderEstimator.get_num_params(self.model)
        z = self.estimator.generate_perturbation(seed=7, num_params=num_params)
        assert z.shape == (num_params,)

    def test_get_num_params_counts_all_trainable(self):
        # TinyModel: Linear(4,3) = 4*3 weights + 3 bias = 15
        #            Linear(3,2) = 3*2 weights + 2 bias =  8
        # Total = 23
        count = ZerothOrderEstimator.get_num_params(self.model)
        assert count == 23

    def test_get_num_params_ignores_frozen_params(self):
        # Freeze fc1
        for p in self.model.fc1.parameters():
            p.requires_grad = False
        count = ZerothOrderEstimator.get_num_params(self.model)
        assert count == 8  # Only fc2 params

    def test_flat_params_roundtrip_preserves_values(self):
        original_flat = ZerothOrderEstimator._get_flat_params(self.model)
        # Perturb the model
        modified = original_flat + 1.0
        ZerothOrderEstimator._set_flat_params(self.model, modified)
        recovered = ZerothOrderEstimator._get_flat_params(self.model)
        assert torch.allclose(recovered, modified, atol=1e-6)

    def test_flat_params_length_matches_num_params(self):
        flat = ZerothOrderEstimator._get_flat_params(self.model)
        count = ZerothOrderEstimator.get_num_params(self.model)
        assert len(flat) == count
