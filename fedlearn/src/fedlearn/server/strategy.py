from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple
from collections import OrderedDict
import torch
import json

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
            clients_per_round:int=2
    ):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round
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
        print(f"Strategy Evaluation (Round {server_round}): Loss={loss:.4f}, Metrics={metrics}")
        return loss, metrics


class FedAvgAggregator:
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

        total_examples = sum(num_examples for _, _, num_examples in updates)

        for client_id, params, num_examples in updates:
            weight = num_examples / total_examples
            for key in aggregated_params:
                if key in params:
                    aggregated_params[key] += params[key].to(device).float() * weight

        return aggregated_params


