# src/fedlearn/client/decomfl_client.py
"""
DeComFL Client implementing Algorithm 4 from the paper.
"""

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from .client import Client
from fedlearn.estimators.zeroth_order import ZerothOrderEstimator

log = logging.getLogger(__name__)


class DeComFLClient(Client):
    """
    DeComFL client that computes gradient scalars using zeroth-order optimization.

    Key differences from standard FL client:
    - Uses zeroth-order gradient estimation instead of backpropagation
    - Returns gradient scalars instead of full model parameters
    - Implements model rebuilding for missed rounds
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader,
            smoothing_param: float = 0.001,
            device: str = 'cpu'
    ):
        """
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            smoothing_param: μ - smoothing parameter for ZO estimation
            device: Device to run on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device

        # Zeroth-order estimator
        self.zo_estimator = ZerothOrderEstimator(
            smoothing_param=smoothing_param,
            device=device
        )

        # Current model parameters (flattened)
        self.x_current = self.zo_estimator._get_flat_params(self.model)

        # For heartbeat integration
        self.grpc_client = None

        log.info(
            "DeComFLClient initialised with %d parameters",
            self.zo_estimator.get_num_params(self.model),
        )

    def set_grpc_client(self, grpc_client):
        """Set gRPC client for heartbeat updates."""
        self.grpc_client = grpc_client

    def load_global_model(self, parameters: OrderedDict[str, torch.Tensor]) -> None:
        """Adopt the server's global model (DeComFL requires every party to share x_0).

        Loads ``parameters`` into the local model and resets the flattened working copy
        ``x_current`` so the zeroth-order trajectory starts from the *shared* global model,
        not this client's constructor-time random init. Called once at startup; this is the
        O(d) initial download the paper assumes — per-round communication stays O(1).
        """
        self.model.load_state_dict(parameters)
        self.x_current = self.zo_estimator._get_flat_params(self.model).to(self.device)
        log.debug("Synced local model to server global (%d params)", len(self.x_current))

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """Return current model parameters."""
        return self.model.state_dict()

    def rebuild_model(
            self,
            rebuild_history: List[Dict],
            learning_rate: float
    ):
        """
        Rebuild model from missed rounds.
        Algorithm 2, Lines 2-9

        Args:
            rebuild_history: List of {round_number, seeds, gradients}
            learning_rate: Learning rate η
        """
        if not rebuild_history:
            return

        log.debug("Rebuilding model from %d missed rounds", len(rebuild_history))

        for round_data in rebuild_history:
            round_num = round_data['round_number']
            seeds = round_data['seeds']
            avg_gradients = round_data['gradients']

            K = len(seeds)
            P = len(seeds[0]) if K > 0 else 0

            # Replay each local step
            for k in range(K):
                delta = torch.zeros_like(self.x_current)

                for p in range(P):
                    # Regenerate perturbation from seed
                    z = self.zo_estimator.generate_perturbation(
                        seeds[k][p],
                        len(self.x_current)
                    )

                    # Get average gradient scalar
                    g = avg_gradients[k][p]

                    # Accumulate update
                    delta += g * z

                # Update model
                self.x_current = self.x_current - (learning_rate / P) * delta

        # Apply rebuilt parameters to model
        self.zo_estimator._set_flat_params(self.model, self.x_current)
        log.debug("Model rebuild complete")

    def fit(
            self,
            parameters: OrderedDict[str, torch.Tensor],
            config: dict
    ) -> Tuple[List[List[float]], int]:
        """
        Perform local training using zeroth-order optimization.
        Algorithm 4, Procedure 2 (Lines 13-24)

        Args:
            parameters: Initial model parameters (unused for DeComFL - uses seeds instead)
            config: Training configuration with seeds and hyperparameters

        Returns:
            gradient_scalars: Nested list [local_step][perturbation]
            num_examples: Number of training examples
        """
        # Extract config
        seeds = config.get('seeds', [])
        K = len(seeds)  # Number of local steps
        P = len(seeds[0]) if K > 0 else 0  # Number of perturbations
        eta = float(config.get('learning_rate', 0.001))
        # FR-10: μ is server-authoritative — apply the server's smoothing_param so the ZO estimate is
        # of the SAME smoothed function the server reconstructs (keep our default if the server omits
        # it). A mismatched μ makes the gradient scalars derivatives of a different function.
        mu = config.get('smoothing_param')
        if mu is not None:
            self.zo_estimator.mu = float(mu)

        log.debug("Starting local DeComFL training (K=%d, P=%d, mu=%s)", K, P, self.zo_estimator.mu)

        # Track total perturbation for in-place revert to avoid OOM
        total_perturbation = torch.zeros_like(self.x_current)

        gradient_scalars = []
        data_iter = iter(self.train_loader)
        total_steps = K

        # Algorithm 4, Line 14: Loop over local steps k = 1, ..., K
        for k in range(K):
            # FR-10: honour a server-driven stop between local steps. The heartbeat thread
            # latches the server's should_stop into the gRPC client's stop Event while this
            # loop blocks the training stub; checking it here bounds the abort latency to
            # ~one heartbeat interval + one local step instead of the full K-step round.
            # total_perturbation only accumulates APPLIED steps, so the revert below stays
            # exact for a partial run; the caller sees the stop via should_stop_training()
            # and must not submit the partial scalars (the server has stopped anyway).
            if self.grpc_client is not None and self.grpc_client.should_stop_training():
                log.info("Server requested stop; aborting local training at step %d/%d", k, K)
                break

            delta = torch.zeros_like(self.x_current)
            k_gradient_scalars = []

            # Get data batch for this local step
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Handle both tuple (standard) and unpacked batch formats
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch
            else:
                # Batch is already unpacked (shouldn't happen with wrapper, but safe)
                inputs, targets = batch, None

            # Handle both tensor inputs (CNN/MLP) and dict inputs (LLM)
            if isinstance(inputs, dict):
                # LLM format: move all dict values to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # Standard format: move tensor to device
                inputs = inputs.to(self.device)

            targets = targets.to(self.device)

            # Update heartbeat
            if self.grpc_client:
                self.grpc_client.update_status("training", k + 1, total_steps)

            # Algorithm 4, Line 16: Loop over perturbations p = 1, ..., P
            for p in range(P):
                # Algorithm 4, Line 17: Generate perturbation z^k_r,p
                z = self.zo_estimator.generate_perturbation(
                    seeds[k][p],
                    len(self.x_current)
                )

                # Algorithm 4, Line 18: Compute gradient scalar g^k_{i,r,p}
                g = self.zo_estimator.compute_gradient_scalar(
                    self.model,
                    self.x_current,
                    z,
                    inputs,
                    targets
                )
                k_gradient_scalars.append(g)

                # Algorithm 4, Line 19: Accumulate update direction
                delta += g * z

            # Algorithm 4, Line 21: Update model
            step_update = (eta / P) * delta
            self.x_current -= step_update
            total_perturbation -= step_update

            gradient_scalars.append(k_gradient_scalars)

            if (k + 1) % max(1, K // 5) == 0:
                log.debug("Completed local step %d/%d", k + 1, K)

        # SECURE: Revert by mathematically reversing the exact perturbation in-place
        self.x_current -= total_perturbation
        self.zo_estimator._set_flat_params(self.model, self.x_current)

        # Count training examples
        num_examples = len(self.train_loader.dataset)

        log.debug(
            "Local training complete: %d local steps × %d perturbations",
            len(gradient_scalars),
            len(gradient_scalars[0]) if gradient_scalars else 0,
        )

        # Algorithm 4, Line 24: Return gradient scalars
        return gradient_scalars, num_examples