# src/fedlearn/estimators/zeroth_order.py
from __future__ import annotations

"""
Zeroth-order gradient estimation for DeComFL.
Implements Algorithm 4 from the DeComFL paper.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Union, Dict
from collections import OrderedDict


class ZerothOrderEstimator:
    """
    Zeroth-order gradient estimator using forward difference method.

    Computes gradient scalars: g = (f(x + μz; ξ) - f(x; ξ)) / μ
    """

    def __init__(self, smoothing_param: float = 0.001, device: str = 'cpu'):
        """
        Args:
            smoothing_param: Smoothing parameter μ for ZO estimation
            device: Device to run computations on
        """
        self.mu = smoothing_param
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def generate_perturbation(
            self,
            seed: int,
            num_params: int
    ) -> torch.Tensor:
        """
        Generate perturbation vector z ~ N(0, I_d) from seed.

        Args:
            seed: Random seed for reproducibility
            num_params: Number of parameters (dimension d)

        Returns:
            Perturbation vector of shape (num_params,)
        """
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        z = torch.randn(num_params, generator=generator, device=self.device)
        return z

    def compute_gradient_scalar(
            self,
            model: nn.Module,
            flat_params: torch.Tensor,
            perturbation: torch.Tensor,
            inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
            targets: torch.Tensor
    ) -> float:
        """
        Compute zeroth-order gradient scalar g^k_{i,r,p}.

        Algorithm 4, Line 18: g = (f(x + μz; ξ) - f(x; ξ)) / μ

        Args:
            model: Neural network model
            flat_params: Current flattened model parameters
            perturbation: Perturbation vector z
            inputs: Input batch (tensor for CNN/MLP, dict for LLM)
            targets: Target labels

        Returns:
            Gradient scalar (float)
        """
        model.eval()

        # Determine if LLM or standard model
        is_llm = isinstance(inputs, dict)

        with torch.no_grad():
            # Compute f(x; ξ)
            self._set_flat_params(model, flat_params)

            if is_llm:
                # LLM: unpack dict as kwargs
                outputs = model(**inputs, labels=targets)
                loss_x = outputs.loss
            else:
                # Standard: direct forward pass
                outputs = model(inputs)
                loss_x = self.criterion(outputs, targets)

            # Compute f(x + μz; ξ)
            perturbed_params = flat_params + self.mu * perturbation
            self._set_flat_params(model, perturbed_params)

            if is_llm:
                # LLM: unpack dict as kwargs
                outputs_perturbed = model(**inputs, labels=targets)
                loss_x_perturbed = outputs_perturbed.loss
            else:
                # Standard: direct forward pass
                outputs_perturbed = model(inputs)
                loss_x_perturbed = self.criterion(outputs_perturbed, targets)

            # Compute gradient scalar
            g = (loss_x_perturbed - loss_x) / self.mu

        return g.item()

    @staticmethod
    def _get_flat_params(model: nn.Module) -> torch.Tensor:
        """Get model parameters as a flat vector."""
        params = []
        for p in model.parameters():
            if p.requires_grad:
                params.append(p.data.view(-1))
        return torch.cat(params)

    @staticmethod
    def _set_flat_params(model: nn.Module, flat_params: torch.Tensor):
        """Set model parameters from a flat vector."""
        offset = 0
        for p in model.parameters():
            if p.requires_grad:
                numel = p.numel()
                p.data.copy_(flat_params[offset:offset + numel].view_as(p.data))
                offset += numel

    @staticmethod
    def get_num_params(model: nn.Module) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)