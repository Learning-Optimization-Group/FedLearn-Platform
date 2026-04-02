# src/fedlearn/server/decomfl_strategy.py
from __future__ import annotations

"""
DeComFL Strategy implementing Algorithm 3 from the paper.
"""

from typing import Optional, Callable, Tuple, List, Dict
from collections import OrderedDict
import torch
import numpy as np
from .strategy import Strategy


class DeComFL(Strategy):
    """
    DeComFL strategy with dimension-free communication.

    Key features:
    - Communicates gradient scalars + seeds instead of full model parameters
    - Maintains seed history and gradient history for model rebuilding
    - Tracks client participation for proper synchronization
    """

    def __init__(
            self,
            initial_parameters: OrderedDict[str, torch.Tensor],
            evaluate_fn: Optional[Callable] = None,
            min_fit_clients: int = 1,
            clients_per_round: int = 2,
            num_local_steps: int = 1,
            num_perturbations: int = 10,
            learning_rate: float = 0.001,
            smoothing_param: float = 0.001,
            seed: int = 42
    ):
        """
        Args:
            initial_parameters: Initial model parameters
            evaluate_fn: Function to evaluate global model
            min_fit_clients: Minimum clients for aggregation
            clients_per_round: Number of clients per round
            num_local_steps: K - local SGD steps per round
            num_perturbations: P - number of perturbations
            learning_rate: η - learning rate
            smoothing_param: μ - smoothing parameter for ZO estimation
            seed: Random seed
        """
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round

        # DeComFL hyperparameters
        self.K = num_local_steps
        self.P = num_perturbations
        self.eta = learning_rate
        self.mu = smoothing_param

        # Algorithm 3, Line 2: Initialize history
        self.seed_history: List[List[List[int]]] = []  # [round][local_step][perturbation]
        self.gradient_history: List[List[List[float]]] = []  # [round][local_step][perturbation]

        # Track last participation round for each client
        self.client_last_round: Dict[str, int] = {}

        # Current global model parameters (flattened)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.global_params_flat = self._flatten_params(initial_parameters)


        # Random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"[DeComFL] Initialized with:")
        print(f"  - K (local steps): {self.K}")
        print(f"  - P (perturbations): {self.P}")
        print(f"  - η (learning rate): {self.eta}")
        print(f"  - μ (smoothing): {self.mu}")
        print(f"  - Model dimension: {len(self.global_params_flat):,}")

    def initialize_parameters(self) -> Optional[OrderedDict[str, torch.Tensor]]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def generate_seeds(self, round_idx: int) -> List[List[int]]:
        """
        Generate random seeds for perturbations.
        Algorithm 3, Line 5

        Returns:
            seeds[k][p] = seed for local step k, perturbation p
        """
        seeds = []
        for k in range(self.K):
            k_seeds = []
            for p in range(self.P):
                seed = np.random.randint(0, 2 ** 31 - 1)
                k_seeds.append(int(seed))
            seeds.append(k_seeds)

        return seeds

    def get_rebuild_history(self, client_id: str, current_round: int) -> List[Dict]:
        """Get history needed for client to rebuild model."""
        last_round = self.client_last_round.get(client_id, -1)

        if last_round >= current_round - 1:
            return []

        rebuild_history = []
        for r in range(last_round + 1, current_round):
            # Check if history exists for this round
            if r >= 0 and r < len(self.seed_history) and r < len(self.gradient_history):
                rebuild_history.append({
                    'round_number': r,
                    'seeds': self.seed_history[r],
                    'gradients': self.gradient_history[r]
                })

        return rebuild_history

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[str, List[List[float]], int]],  # (client_id, gradients, num_examples)
    ) -> Optional[OrderedDict[str, torch.Tensor]]:
        """
        Aggregate gradient scalars and update global model.
        Algorithm 3, Lines 10-12

        Args:
            results: List of (client_id, gradient_scalars, num_examples)
                    gradient_scalars[k][p] = gradient scalar for local step k, perturbation p

        Returns:
            Updated global model parameters
        """
        if not results:
            return None

        print(f"[DeComFL] Aggregating {len(results)} client updates for round {server_round}")

        # Extract gradient scalars from all clients
        client_gradients = {}
        for client_id, grad_scalars, num_examples in results:
            client_gradients[client_id] = grad_scalars
            # Update client's last participation round
            self.client_last_round[client_id] = server_round

        # Get current model parameters
        x_current = self.global_params_flat.clone()

        # For each local step
        for k in range(self.K):
            delta = torch.zeros_like(x_current)

            # Average gradients across clients
            num_clients = len(client_gradients)
            for client_id, grad_scalars in client_gradients.items():
                for p in range(self.P):
                    # Regenerate perturbation from seed
                    z = self._generate_perturbation(self.seed_history[server_round][k][p])

                    # Get gradient scalar for this client
                    g = grad_scalars[k][p]

                    # Accumulate gradient direction
                    delta += g * z

            # Average across clients and perturbations
            delta = delta / (num_clients * self.P)

            # Update model parameters
            x_current = x_current - self.eta * delta * self.P

        # Update global model
        self.global_params_flat = x_current

        # Convert back to OrderedDict format
        updated_params = self._unflatten_params(x_current, self.initial_parameters)

        return updated_params

    def _generate_perturbation(self, seed: int) -> torch.Tensor:
        """Generate perturbation vector from seed."""
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        z = torch.randn(
            len(self.global_params_flat),
            generator=generator,
            device=self.device
        )
        return z

    def _flatten_params(self, params: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten OrderedDict parameters to 1D tensor."""
        flat = []
        for name, tensor in params.items():
            flat.append(tensor.view(-1))
        return torch.cat(flat).to(self.device)

    def _unflatten_params(
            self,
            flat_params: torch.Tensor,
            template: OrderedDict[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        """Unflatten 1D tensor back to OrderedDict format."""
        params = OrderedDict()
        offset = 0
        for name, tensor in template.items():
            numel = tensor.numel()
            params[name] = flat_params[offset:offset + numel].view_as(tensor).cpu()
            offset += numel
        return params

    def evaluate(
            self,
            server_round: int,
            parameters: OrderedDict[str, torch.Tensor]
    ) -> Optional[Tuple[float, dict]]:
        """Evaluate the global model."""
        if self.evaluate_fn is None:
            return None

        loss, metrics = self.evaluate_fn(server_round, parameters)
        print(f"[DeComFL] Evaluation (Round {server_round}): Loss={loss:.4f}, Metrics={metrics}")
        return loss, metrics