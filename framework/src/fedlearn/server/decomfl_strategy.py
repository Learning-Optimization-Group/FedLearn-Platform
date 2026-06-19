# src/fedlearn/server/decomfl_strategy.py
"""
DeComFL server-side aggregation strategy.

Implements the server protocol of DeComFL — "Achieving Dimension-Free
Communication in Federated Learning via Zeroth-Order Optimization"
(Li, Ying, Liu, Dong, Yang; ICLR 2025; https://arxiv.org/abs/2405.15861).

Aligned with the authors' reference implementation
https://github.com/ZidongLiu/DeComFL (Apache-2.0); that attribution and license
are retained here per Apache-2.0 section 4.
"""

import logging
import threading
from typing import Optional, Callable, Tuple, List, Dict
from collections import OrderedDict
import torch
import numpy as np
from .strategy import Strategy
from fedlearn.estimators.perturbation import canonical_perturbation

log = logging.getLogger(__name__)


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
            clients_per_round: int = None,
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
        self.clients_per_round = clients_per_round if clients_per_round is not None else min_fit_clients

        # DeComFL hyperparameters
        self.K = num_local_steps
        self.P = num_perturbations
        self.eta = learning_rate
        self.mu = smoothing_param

        # Algorithm 3, Line 2: Initialize history. Keyed by ROUND NUMBER (1-based, matching
        # coordinator.current_round) so aggregate_fit / get_rebuild_history index by round
        # unambiguously. Fixes audit #28/#29: the old list+per-client-append produced N entries
        # per round (and off-by-one indexing), and handed each client a DIFFERENT perturbation
        # direction — breaking DeComFL's shared-seed invariant.
        self.seed_history: Dict[int, List[List[int]]] = {}        # round -> seeds[k][p]
        self.gradient_history: Dict[int, List[List[float]]] = {}  # round -> avg_grad[k][p]
        self._seed_lock = threading.Lock()  # guards get_or_create_seeds against concurrent client RPCs

        # Track last participation round for each client
        self.client_last_round: Dict[str, int] = {}

        # Current global model parameters (flattened)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.global_params_flat = self._flatten_params(initial_parameters)


        # Local RNG for seed generation. Does NOT mutate the process-global numpy/torch RNG
        # (B-2 fix): the old global np.random.seed/torch.manual_seed corrupted reproducibility for
        # anything else sharing the interpreter. Perturbations use canonical_perturbation's own
        # local CPU generator, so no global torch seeding is needed either.
        self._seed_rng = np.random.default_rng(seed)

        # One-shot startup banner — INFO so it's captured in normal logs.
        log.info(
            "DeComFL initialised: K=%d, P=%d, eta=%g, mu=%g, model_dim=%d",
            self.K, self.P, self.eta, self.mu, len(self.global_params_flat),
        )

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
                seed = int(self._seed_rng.integers(0, 2 ** 31 - 1))
                k_seeds.append(seed)
            seeds.append(k_seeds)

        return seeds

    def get_or_create_seeds(self, round_idx: int) -> List[List[int]]:
        """Return the seeds for ``round_idx``, generating them EXACTLY ONCE.

        DeComFL requires every client in a round to perturb along the same
        seed-derived direction z, so seeds must be generated once per round and
        shared by all clients — never regenerated per client RPC (audit #28).
        This is the single entry point grpc_servicer must call: it is idempotent
        and thread-safe, and records each round's seeds in ``seed_history`` once,
        keyed by the round number that ``aggregate_fit`` indexes with.
        """
        with self._seed_lock:
            seeds = self.seed_history.get(round_idx)
            if seeds is None:
                seeds = self.generate_seeds(round_idx)
                self.seed_history[round_idx] = seeds
            return seeds

    def get_rebuild_history(self, client_id: str, current_round: int) -> List[Dict]:
        """Get history needed for client to rebuild model."""
        last_round = self.client_last_round.get(client_id, -1)

        if last_round >= current_round - 1:
            return []

        rebuild_history = []
        for r in range(last_round + 1, current_round):
            # Check if history exists for this round (dicts keyed by round number)
            if r in self.seed_history and r in self.gradient_history:
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

        log.debug("Aggregating %d client updates for round %d", len(results), server_round)

        # Extract gradient scalars from all clients
        client_gradients = {}
        for client_id, grad_scalars, num_examples in results:
            client_gradients[client_id] = grad_scalars
            # Update client's last participation round
            self.client_last_round[client_id] = server_round

        # Get current model parameters
        x_current = self.global_params_flat.clone()
        num_clients = len(client_gradients)

        # For each local step
        for k in range(self.K):
            delta = torch.zeros_like(x_current)

            for p in range(self.P):
                # z depends only on (k, p), NOT on the client — regenerate it once and sum the
                # gradient scalars across clients (O(K*P) instead of the v1 O(K*P*N) loop, C-1).
                z = self._generate_perturbation(self.seed_history[server_round][k][p])
                g_sum = sum(grad_scalars[k][p] for grad_scalars in client_gradients.values())
                delta += g_sum * z

            # Average across clients and perturbations.
            delta = delta / (num_clients * self.P)

            # Update model parameters. The 1/P averaging above IS the paper's update; the v1 code
            # cancelled it with an extra * self.P, stepping the global model P x too far and off the
            # rebuild trajectory (Bug 1, B1-C1). No * self.P here.
            x_current = x_current - self.eta * delta

        # Update global model
        self.global_params_flat = x_current

        # Convert back to OrderedDict format
        updated_params = self._unflatten_params(x_current, self.initial_parameters)

        return updated_params

    def _generate_perturbation(self, seed: int) -> torch.Tensor:
        """Generate perturbation vector from seed.

        CPU-canonical and device-independent (Bug-2 fix): the server and every client must
        regenerate bit-identical z from the same seed, which a seeded torch.randn does NOT
        guarantee across CPU/CUDA/MPS. Delegates to the shared canonical helper, then moves the
        result to the compute device.
        """
        return canonical_perturbation(seed, len(self.global_params_flat)).to(self.device)

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
        log.info(
            "DeComFL eval round=%d loss=%.4f metrics=%s",
            server_round, loss, metrics,
        )
        return loss, metrics