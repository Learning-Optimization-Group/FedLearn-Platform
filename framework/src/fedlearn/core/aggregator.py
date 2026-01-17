# from abc import ABC, abstractmethod
# from collections import OrderedDict
# import torch
#
#
# class Aggregator(ABC):
#     @abstractmethod
#     def aggregate(self, updates: list[tuple[OrderedDict[str, torch.Tensor], int]]) -> OrderedDict[str, torch.Tensor]:
#         """Aggregate model updates from clients."""
#         pass
#
#
# class FedAvg(Aggregator):
#     def aggregate(self, updates: list[tuple[OrderedDict[str, torch.Tensor], int]]) -> OrderedDict[str, torch.Tensor]:
#         """
#         Federated Averaging (FedAvg) implementation.
#         """
#         if not updates:
#             raise ValueError("Cannot aggregate an empty list of updates.")
#
#         # Calculate the total number of examples from all client updates
#         total_examples = sum(num_examples for _, num_examples in updates)
#
#         if total_examples == 0:
#             # If no examples were trained on, return the first model's parameters as a default
#             return updates[0][0]
#
#         # Get the state_dict from the first client to use as a template for the new global model
#         template_params = updates[0][0]
#
#         # Create a new OrderedDict to store the aggregated parameters, initialized with zeros
#         aggregated_params = OrderedDict([
#             (key, torch.zeros_like(tensor, dtype=torch.float32)) for key, tensor in template_params.items()
#         ])
#
#         # Perform the weighted average
#         for params, num_examples in updates:
#             weight = num_examples / total_examples
#             for key in aggregated_params:
#                 if key in params:
#                     # Ensure tensors are float for accumulation
#                     aggregated_params[key] += params[key].float() * weight
#
#         return aggregated_params