import logging
from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple
from collections import OrderedDict
import torch
import json

from fedlearn.estimators.perturbation import canonical_perturbation

log = logging.getLogger(__name__)

class Strategy(ABC):
    """Abstract base class for learning strategies."""
    @abstractmethod
    def initialize_parameters(self)->Optional[OrderedDict[str, torch.Tensor]]:
        """Initialize the global"""
        pass


    @abstractmethod
    def aggregate_fit(
            self,
            server_round:int,
            results:list[Tuple[OrderedDict[str, torch.Tensor],int]],
    )-> Optional[OrderedDict[str, torch.Tensor]]:
        """Aggregate training results from clients"""
        pass

    @abstractmethod
    def evaluate(
            self, server_round: int, parameters: OrderedDict[str, torch.Tensor]
    ) -> Optional[Tuple[float, dict]]:
        """Evaluate the global model."""
        pass


class FedAvg(Strategy):
    """The default strategy for FedAvg."""

    def __init__(
            self,
            initial_parameters: OrderedDict[str, torch.Tensor],
            evaluate_fn:Optional[Callable]=None,
            min_fit_clients:int=1,
            clients_per_round:int=None
    ):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round if clients_per_round is not None else min_fit_clients
        self.aggregator = FedAvgAggregator()

    def initialize_parameters(self) -> Optional[OrderedDict[str, torch.Tensor]]:
        return self.initial_parameters

    def aggregate_fit(
            self,
            server_round: int,
            results: list[Tuple[OrderedDict[str, torch.Tensor], int]],
    ) -> Optional[OrderedDict[str, torch.Tensor]]:
        if not results:
            return None

        # Aggregate using the same logic as before
        return self.aggregator.aggregate(results)

    def evaluate(
            self, server_round: int, parameters: OrderedDict[str, torch.Tensor]
    ) -> Optional[Tuple[float, dict]]:
        if self.evaluate_fn is None:
            return None

        # Call the user-provided evaluation function
        loss, metrics = self.evaluate_fn(server_round, parameters)
        log.info(
            "FedAvg eval round=%d loss=%.4f metrics=%s",
            server_round, loss, metrics,
        )
        return loss, metrics


class FedAvgAggregator:
    MAX_SAMPLES = 100_000  # Cap to prevent model poisoning via inflated num_examples

    def aggregate(self, updates):
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        deserialized_updates = []
        for entry in updates:
            # If you have (client_id, params, num_examples)
            if len(entry) == 3:
                client_id, params, num_examples = entry
            else:
                params, num_examples = entry
                client_id = None

            # If params is a string (JSON), parse and convert to tensors
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                    params = OrderedDict({k: torch.tensor(v) for k, v in params.items()})
                except Exception as e:
                    raise ValueError(f"Failed to deserialize parameters from {client_id}: {e}")

            deserialized_updates.append((client_id, params, num_examples))

        updates = deserialized_updates

        if len(updates[0]) == 3:
            _, template_params, _ = updates[0]
        else:
            template_params, _ = updates[0]

        template_params = {k: v.to(device) for k, v in template_params.items()}

        aggregated_params = OrderedDict(
            [(key, torch.zeros_like(tensor, dtype=torch.float32)) for key, tensor in template_params.items()])

        # Sanitize num_examples: cap and reject invalid values
        sanitized_updates = [
            (cid, p, min(n, self.MAX_SAMPLES))
            for cid, p, n in updates if n > 0
        ]
        if not sanitized_updates:
            raise ValueError("No valid updates after sanitization.")

        total_examples = sum(num_examples for _, _, num_examples in sanitized_updates)

        for client_id, params, num_examples in sanitized_updates:
            weight = num_examples / total_examples
            for key in aggregated_params:
                if key in params:
                    torch.add(
                        aggregated_params[key], 
                        params[key].to(device).float(), 
                        alpha=weight, 
                        out=aggregated_params[key]
                    )
            
            # Aggressively free client memory buffer
            params.clear()

        return aggregated_params

    def aggregate_scalars(self, global_params, results, eta, num_perturbations):
        """FedAvg over ZO gradient scalars (DECISION D1).

        Reconstruct each client's update
            Δ_c = (eta/P)·Σ_{k,p} g_c[k][p]·canonical_perturbation(seed_c[k][p], d)
        and return the num_examples-weighted global
            global_new = global_old - Σ_c w_c·Δ_c
        as an OrderedDict matching global_params.

        Args:
            global_params: OrderedDict[str, Tensor] — the prior global model.
            results: list of (client_id, seeds, gradients, num_examples) where
                     seeds[k][p] and gradients[k][p] are the client's K×P
                     seed/g scalars.
            eta: learning rate η.
            num_perturbations: P (perturbations per local step).

        Returns:
            OrderedDict with the same keys/shapes as global_params.

        Raises:
            ValueError: if results is empty after sanitization, or if any
                        client's seeds/gradients have a shape mismatch.
        """
        if not results:
            raise ValueError("Cannot aggregate an empty list of updates.")

        # Sanitize num_examples: cap and reject invalid values (mirrors aggregate()).
        sanitized = [
            (cid, seeds, grads, min(n, self.MAX_SAMPLES))
            for cid, seeds, grads, n in results if n > 0
        ]
        if not sanitized:
            raise ValueError("No valid updates after sanitization.")

        # Flatten global params to a 1-D float32 CPU tensor.
        flat_global = torch.cat(
            [t.view(-1) for t in global_params.values()]
        ).float().cpu()
        d = flat_global.numel()

        total = sum(n for _, _, _, n in sanitized)
        agg_delta = torch.zeros(d, dtype=torch.float32)

        for cid, seeds, grads, n in sanitized:
            # Validate rectangular K×P layout — malformed payloads fail loudly.
            if len(seeds) != len(grads):
                raise ValueError(
                    f"Client {cid}: len(seeds)={len(seeds)} != len(gradients)={len(grads)}"
                )
            for k, (s_row, g_row) in enumerate(zip(seeds, grads)):
                if len(s_row) != num_perturbations or len(g_row) != num_perturbations:
                    raise ValueError(
                        f"Client {cid} step k={k}: expected P={num_perturbations} "
                        f"entries, got seeds={len(s_row)} gradients={len(g_row)}"
                    )

            # Δ_c = (eta/P) · Σ_{k,p} g[k][p] · canonical_perturbation(seed[k][p], d)
            delta_c = torch.zeros(d, dtype=torch.float32)
            for s_row, g_row in zip(seeds, grads):
                for seed, g in zip(s_row, g_row):
                    delta_c += g * canonical_perturbation(seed, d)
            delta_c *= eta / num_perturbations

            weight = n / total
            agg_delta += weight * delta_c

        new_flat = flat_global - agg_delta

        # Unflatten back to the same keys/shapes as global_params.
        out = OrderedDict()
        offset = 0
        for name, tensor in global_params.items():
            numel = tensor.numel()
            out[name] = new_flat[offset:offset + numel].view_as(tensor).clone()
            offset += numel
        return out


class FedLoRA(Strategy):
    """Federated LoRA. Clients communicate ONLY adapter params (FFA: B+head; FedIT: A+B+head).

    Reuses FedAvgAggregator (num-examples-weighted average over whatever keys are present).
    Under FFA_LORA, A is frozen+shared, so it is NOT aggregated — the strategy re-attaches the
    frozen A (captured from initial_parameters) to every aggregated global so it is redistributed
    unchanged and stays identical across clients (which makes avg(B)@A == avg(B@A) exact).
    """

    def __init__(self, initial_parameters, evaluate_fn=None, min_fit_clients=1,
                 clients_per_round=None, aggregation="FFA_LORA"):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round if clients_per_round is not None else min_fit_clients
        self.aggregation = aggregation
        self.aggregator = FedAvgAggregator()
        self._frozen_a = (
            OrderedDict((k, v.clone()) for k, v in initial_parameters.items() if "lora_A" in k)
            if aggregation == "FFA_LORA" else OrderedDict()
        )
        if aggregation == "FFA_LORA" and not self._frozen_a:
            raise ValueError(
                "FFA_LORA requires lora_A keys in initial_parameters (the global adapter must carry "
                "the shared frozen A); none found — ensure the initial adapter is the FULL adapter (A+B+head)."
            )

    def initialize_parameters(self):
        return self.initial_parameters

    def aggregate_fit(self, server_round, results):
        if not results:
            return None
        self._assert_homogeneous(results)
        aggregated = self.aggregator.aggregate(results)
        if self.aggregation == "FFA_LORA":
            for k, v in self._frozen_a.items():
                aggregated[k] = v.clone()
        return aggregated

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        loss, metrics = self.evaluate_fn(server_round, parameters)
        log.info("FedLoRA eval round=%d loss=%.4f metrics=%s", server_round, loss, metrics)
        return loss, metrics

    @staticmethod
    def _assert_homogeneous(results):
        """Raise ValueError if clients disagree on adapter key set or per-key shape (homogeneous rank)."""
        def params_of(entry):
            return entry[1] if len(entry) == 3 else entry[0]
        ref = params_of(results[0])
        ref_shapes = {k: tuple(v.shape) for k, v in ref.items()}
        for entry in results[1:]:
            shapes = {k: tuple(v.shape) for k, v in params_of(entry).items()}
            if shapes != ref_shapes:
                raise ValueError(
                    "Heterogeneous LoRA adapters across clients (key/shape mismatch). "
                    "FedLoRA requires homogeneous rank/config."
                )

