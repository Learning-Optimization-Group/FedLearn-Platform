# src/fedlearn/estimators/zeroth_order.py
"""
Zeroth-order (forward finite-difference) gradient estimation for DeComFL.

Implements the client-side ZO gradient estimator of DeComFL — "Achieving
Dimension-Free Communication in Federated Learning via Zeroth-Order
Optimization" (Li, Ying, Liu, Dong, Yang; ICLR 2025;
https://arxiv.org/abs/2405.15861).

Aligned with the authors' reference implementation
https://github.com/ZidongLiu/DeComFL (Apache-2.0); that attribution and license
are retained here per Apache-2.0 section 4.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Union, Dict
from collections import OrderedDict

from fedlearn.estimators.perturbation import canonical_perturbation
from fedlearn.estimators import params


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
        # CPU-canonical (Bug-2 fix): identical to the server's perturbation for the same seed,
        # regardless of device. Generated on CPU, then moved to the compute device.
        return canonical_perturbation(seed, num_params).to(self.device)

    def _evaluate_loss(
            self,
            model: nn.Module,
            inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
            targets: torch.Tensor
    ) -> torch.Tensor:
        """One forward pass at the model's CURRENT parameters. Caller owns eval()/no_grad()."""
        if isinstance(inputs, dict):
            # LLM: unpack dict as kwargs
            return model(**inputs, labels=targets).loss
        # Standard: direct forward pass
        return self.criterion(model(inputs), targets)

    def compute_base_loss(
            self,
            model: nn.Module,
            flat_params: torch.Tensor,
            inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
            targets: torch.Tensor
    ) -> float:
        """Evaluate the UNPERTURBED loss f(x; ξ) once, for reuse across a local step's P
        perturbations.

        Within one DeComFL local step k both the base point ``flat_params`` and the batch ξ are
        fixed — only z varies — so f(x; ξ) is the same number for every perturbation. Hoisting it
        here turns a local step's cost from 2P forward passes into P+1, matching the authors'
        reference implementation, which computes ``pert_minus_loss`` once above the perturbation
        loop for the forward-difference method.

        The result is only valid for THIS (flat_params, inputs, targets) triple: x advances
        between local steps, so re-evaluate once per step. Determinism relies on ``model.eval()``
        (no dropout, batch-norm on running stats) — pass ``base_loss=None`` for any model whose
        forward is stochastic at inference.
        """
        model.eval()
        with torch.no_grad():
            self._set_flat_params(model, flat_params)
            return self._evaluate_loss(model, inputs, targets).item()

    def compute_gradient_scalar(
            self,
            model: nn.Module,
            flat_params: torch.Tensor,
            perturbation: torch.Tensor,
            inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
            targets: torch.Tensor,
            base_loss: "float | None" = None
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
            base_loss: Pre-computed f(x; ξ) from :meth:`compute_base_loss` for this same
                (flat_params, inputs, targets). Supplying it skips the redundant unperturbed
                forward pass — the scalar is bit-identical either way, since the base loss is
                deterministic under eval()/no_grad(). ``None`` recomputes it (back-compatible).

        Returns:
            Gradient scalar (float)
        """
        model.eval()

        with torch.no_grad():
            if base_loss is None:
                # Compute f(x; ξ)
                self._set_flat_params(model, flat_params)
                loss_x = self._evaluate_loss(model, inputs, targets).item()
            else:
                loss_x = base_loss

            # Compute f(x + μz; ξ)
            perturbed_params = flat_params + self.mu * perturbation
            self._set_flat_params(model, perturbed_params)
            loss_x_perturbed = self._evaluate_loss(model, inputs, targets).item()

            # Compute gradient scalar
            g = (loss_x_perturbed - loss_x) / self.mu

        return g

    # FR-14: the flat-param layout is owned by the canonical manifest (estimators/params.py) so the
    # client, the estimator, and the mobile export share ONE requires_grad-filtered named_parameters()
    # order. These stay as thin, back-compatible delegators (call sites + the golden generator keep
    # calling ZerothOrderEstimator._get_flat_params / _set_flat_params / get_num_params).
    @staticmethod
    def _get_flat_params(model: nn.Module) -> torch.Tensor:
        """Get the model's trainable parameters as a flat vector (canonical layout)."""
        return params.flat_params(model)

    @staticmethod
    def _set_flat_params(model: nn.Module, flat_params: torch.Tensor):
        """Set the model's trainable parameters from a flat vector (canonical layout)."""
        params.set_flat_params(model, flat_params)

    @staticmethod
    def get_num_params(model: nn.Module) -> int:
        """Number of trainable parameters (canonical count)."""
        return params.num_trainable(model)