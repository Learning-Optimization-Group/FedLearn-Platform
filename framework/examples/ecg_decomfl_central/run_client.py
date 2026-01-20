# =============================================================================
# run_client.py - DeComFL Client Implementation (Algorithm 4)
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
from typing import List
from torch.utils.data import DataLoader

from config import Config
from model import create_model
from data import get_ecg_loaders


class DeComFLClient:
    """
    DeComFL Client implementing Algorithm 4.
    """

    def __init__(
            self,
            client_id: int,
            config: Config,
            model_params: torch.Tensor,
            train_loader: DataLoader,
            input_dim: int
    ):
        """
        Initialize DeComFL client

        Args:
            client_id: Unique client identifier
            config: Configuration object
            model_params: Initial model parameters from server
            train_loader: Training data loader
            input_dim: Input dimension for model
        """
        self.client_id = client_id
        self.config = config
        self.device = config.DEVICE
        self.train_loader = train_loader

        # Create local model
        self.model = create_model(
            input_dim=input_dim,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=config.NUM_CLASSES,
            device=self.device
        )

        # Set initial parameters from server
        self.model.set_flat_params(model_params)

        # Current model parameters
        self.x_current = self.model.get_flat_params()

        print(f"[CLIENT {client_id}] Initialized with {self.model.get_num_params():,} parameters")

    def _generate_perturbation(self, seed: int) -> torch.Tensor:
        """
        Generate perturbation vector z ~ N(0, I_d) from seed.
        Algorithm 4, Line 17
        """
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        z = torch.randn(
            self.model.get_num_params(),
            generator=generator,
            device=self.device
        )
        return z

    def _compute_gradient_scalar(
            self,
            x: torch.Tensor,
            z: torch.Tensor,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ) -> float:
        """
        Compute zeroth-order gradient scalar g^k_{i,r,p}.
        Algorithm 4, Line 18: g = (f(x + μz; ξ) - f(x; ξ)) / μ
        """
        mu = self.config.SMOOTHING_PARAM
        criterion = nn.CrossEntropyLoss()

        # Compute f(x; ξ)
        self.model.set_flat_params(x)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            loss_x = criterion(outputs, targets)

        # Compute f(x + μz; ξ)
        self.model.set_flat_params(x + mu * z)
        with torch.no_grad():
            outputs_perturbed = self.model(inputs)
            loss_x_perturbed = criterion(outputs_perturbed, targets)

        # Compute gradient scalar
        g = (loss_x_perturbed - loss_x) / mu

        return g.item()

    def local_update(
            self,
            seeds: List[List[int]],
            round_idx: int
    ) -> List[List[float]]:
        """
        Perform local update and compute gradient scalars.
        Algorithm 4, Procedure 2 (Lines 13-24)
        """
        K = self.config.NUM_LOCAL_STEPS
        P = self.config.NUM_PERTURBATIONS
        eta = self.config.LEARNING_RATE

        # Algorithm 4, Line 23: Store initial model for revert
        x_initial = self.x_current.clone()

        gradient_scalars = []
        data_iter = iter(self.train_loader)

        # Algorithm 4, Line 14: Loop over local steps k = 1, ..., K
        for k in range(K):
            delta = torch.zeros_like(self.x_current)
            k_gradient_scalars = []

            # Get data batch for this local step
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                inputs, targets = next(data_iter)

            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Algorithm 4, Line 16: Loop over perturbations p = 1, ..., P
            for p in range(P):
                # Algorithm 4, Line 17: Generate perturbation z^k_r,p
                z = self._generate_perturbation(seeds[k][p])

                # Algorithm 4, Line 18: Compute gradient scalar g^k_{i,r,p}
                g = self._compute_gradient_scalar(
                    self.x_current, z, inputs, targets
                )
                k_gradient_scalars.append(g)

                # Algorithm 4, Line 19: Accumulate update direction
                delta += g * z

            # Algorithm 4, Line 21: Update model
            self.x_current = self.x_current - (eta / P) * delta

            gradient_scalars.append(k_gradient_scalars)

        # Algorithm 4, Line 23: CRITICAL - Revert model back to initial state
        self.x_current = x_initial
        self.model.set_flat_params(self.x_current)

        # Algorithm 4, Line 24: Return gradient scalars
        return gradient_scalars


def run_client_round(
        client_id: int,
        seeds: List[List[int]],
        model_params: torch.Tensor,
        shared_data: dict,
        config: Config
) -> List[List[float]]:
    """
    Run one round of client training.
    """
    # Get client's training data
    train_loader, _, _ = get_ecg_loaders(
        X=shared_data['X'],
        y=shared_data['y'],
        client_id=client_id,
        num_clients=config.NUM_CLIENTS,
        batch_size_train=config.BATCH_SIZE_TRAIN,
        batch_size_test=config.BATCH_SIZE_TEST,
        data_fraction=config.DATA_FRACTION,
        alpha=config.ALPHA,
        test_size=config.TEST_SIZE,
        num_workers=0,  # Use 0 for main process
        seed=config.SEED
    )

    # Create client
    client = DeComFLClient(
        client_id=client_id,
        config=config,
        model_params=model_params,
        train_loader=train_loader,
        input_dim=shared_data['input_dim']
    )

    # Perform local update
    gradient_scalars = client.local_update(seeds, round_idx=0)

    return gradient_scalars