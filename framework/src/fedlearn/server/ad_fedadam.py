"""FR-25 / Track C: Anchor-Decoupled FedAdam (AD-FedAdam).

Resolves the causal optimizer drag occurring when Byzantine-robust moving-center clipping
(Karimireddy et al., ICML 2021) is composed with server-side adaptive optimizers (Reddi et al., ICLR 2021).

Mathematical Formulation & Dynamical Mechanics:
----------------------------------------------
In standard modular federated learning pipelines, centered clipping maintains an anchor center:
    c_{t+1} = c_t + (1/N) * sum_{i=1}^N clip(Delta_{i,t} - c_t, tau)

Standard FedAdam consumes the accumulated return vector c_{t+1} and squares it in its second-moment
accumulator v_t. When clipping is tight (tau <= 0.5), obsolete momentum components in c_t induce
high-frequency residual oscillations. Adam's coordinate normalization (1 / (sqrt(v_t) + eps)) amplifies
these coordinates, collapsing accuracy (47.50% vs 80.30% under FedAvg).

AD-FedAdam decouples the anchor history from the adaptive variance estimation:
    1. Innovation extraction:
       r_t = (1/N) * sum_{i=1}^N clip(Delta_{i,t} - c_t, tau)
    2. Decoupled adaptive moment estimation over instantaneous innovations:
       m_t = beta1 * m_{t-1} + (1 - beta1) * r_t
       v_t = beta2 * v_{t-1} + (1 - beta2) * (r_t ** 2)
       m_hat = m_t / (1 - beta1 ** t)
       v_hat = v_t / (1 - beta2 ** t)
       step_t = eta_g * m_hat / (sqrt(v_hat) + eps)
    3. Anchor state update:
       c_{t+1} = c_t + r_t
    4. Model state update:
       w_{t+1} = w_t + step_t

Dynamical Scope & Convergence Characteristics:
---------------------------------------------
1. Empirical Scope: AD-FedAdam is an experimental unnoised optimization and Byzantine robustness
   heuristic (it does not claim Differential Privacy guarantees; DP noise must be accounted separately).
2. Trajectory Dynamics: On tasks like CIFAR-10 linear probe, the bounded-travel stall occurs early
   (the weight norm plateaus around 16 by round 5 and remains there through round 200), functioning
   similarly to early stopping and achieving 79.70% accuracy under tight clipping (tau=0.5) and
   77.60% under 20% Byzantine sign-flip attack.
3. Bounded-Travel Property & Stationary Stall on Constant Gradients:
   Because moments adapt solely to residual innovations r_t, if client updates remain strictly
   stationary and constant (Delta_t == g), the anchor center converges (c_t -> g), driving
   instantaneous innovations r_t -> 0. Consequently, under purely constant descent directions on
   convex quadratics without trajectory turns, stepping solely on innovations induces an asymptotic stall
   (distance to optimum plateaus around ~27). Stepping on the anchor (c_{t+1}) resolves the quadratic stall
   but collapses scale normalization and causes divergence on real tasks such as the CIFAR-10 linear probe
   (accuracy collapses to 45–58%, weight norm explodes >500). AD-FedAdam is therefore designated as an
   experimental bounded-travel strategy.
   Non-parameter buffers (e.g. BatchNorm running statistics) are linearly averaged rather than adapted
   through moment states.
"""

from collections import OrderedDict
import logging
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from fedlearn.server._update_normalize import normalize_updates
from fedlearn.server.strategy import Strategy

log = logging.getLogger(__name__)

_NORM_EPS = 1e-12


def ad_fedadam_step(
    stacked_deltas: torch.Tensor,
    centre: torch.Tensor,
    server_m: torch.Tensor,
    server_v: torch.Tensor,
    round_idx: int,
    tau: float = 0.5,
    eta_g: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.99,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute one round of Anchor-Decoupled FedAdam.

    Args:
        stacked_deltas: [N, D] client parameter updates.
        centre: [D] running anchor center c_t.
        server_m: [D] first-moment accumulator.
        server_v: [D] second-moment accumulator.
        round_idx: 1-based round index t.
        tau: clipping radius for centered clipping.
        eta_g: server learning rate.
        beta1: first-moment decay factor.
        beta2: second-moment decay factor.
        eps: numerical stabilizer for adaptive denominator.

    Returns:
        (param_step, next_centre, next_m, next_v)
    """
    n = stacked_deltas.shape[0]
    if n == 0:
        raise ValueError("ad_fedadam_step requires at least one client update.")
    if tau is None or not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"tau must be positive and finite, got {tau}")
    if isinstance(round_idx, bool) or not isinstance(round_idx, int) or not math.isfinite(round_idx) or round_idx < 1:
        raise ValueError(f"round_idx must be a finite positive integer >= 1, got {round_idx}")
    if not (0.0 <= beta1 < 1.0):
        raise ValueError(f"beta1 must be in [0.0, 1.0), got {beta1}")
    if not (0.0 <= beta2 < 1.0):
        raise ValueError(f"beta2 must be in [0.0, 1.0), got {beta2}")
    if eps is None or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be positive and finite, got {eps}")
    if eta_g is None or not math.isfinite(eta_g) or eta_g <= 0.0:
        raise ValueError(f"eta_g must be positive and finite, got {eta_g}")

    # Validate tensor finiteness
    if not torch.isfinite(stacked_deltas).all():
        raise ValueError("stacked_deltas contains NaN or Inf values.")
    if not torch.isfinite(centre).all():
        raise ValueError("centre contains NaN or Inf values.")
    if not torch.isfinite(server_m).all():
        raise ValueError("server_m contains NaN or Inf values.")
    if not torch.isfinite(server_v).all():
        raise ValueError("server_v contains NaN or Inf values.")

    # 1. Compute instantaneous clipped residual innovation
    orig_dtype = stacked_deltas.dtype
    comp_dtype = torch.float64 if stacked_deltas.device.type not in ("mps",) else torch.float32
    residuals_comp = stacked_deltas.to(comp_dtype) - centre.to(comp_dtype)
    res_norms = torch.linalg.vector_norm(residuals_comp, dim=1, keepdim=True)
    factors = torch.clamp(tau / torch.clamp(res_norms, min=_NORM_EPS), max=1.0)
    # After clipping, each row has norm <= tau, safe to downcast
    residual_mean = (residuals_comp * factors).mean(dim=0).to(orig_dtype)

    # 2. Update anchor center
    next_centre = centre + residual_mean

    # 3. Update decoupled moments on residual innovation r_t
    next_m = beta1 * server_m + (1.0 - beta1) * residual_mean
    next_v = torch.clamp(beta2 * server_v + (1.0 - beta2) * (residual_mean ** 2), min=0.0)

    # Bias correction
    m_hat = next_m / (1.0 - (beta1 ** round_idx))
    v_hat = next_v / (1.0 - (beta2 ** round_idx))

    # Adaptive step
    param_step = eta_g * m_hat / (torch.sqrt(v_hat) + eps)

    # Output finiteness guard — verify step, center, and adaptive moments
    if not (
        torch.isfinite(param_step).all()
        and torch.isfinite(next_centre).all()
        and torch.isfinite(next_m).all()
        and torch.isfinite(next_v).all()
    ):
        raise ValueError(
            "ad_fedadam_step produced non-finite outputs or moment states; check input magnitudes, "
            "eps, and tau for numerical feasibility."
        )

    return param_step, next_centre, next_m, next_v


class ADFedAdam(Strategy):
    """Anchor-Decoupled FedAdam Server Strategy."""

    def __init__(
        self,
        initial_parameters: OrderedDict[str, torch.Tensor],
        evaluate_fn: Optional[Callable] = None,
        min_fit_clients: int = 1,
        clients_per_round: Optional[int] = None,
        server_learning_rate: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 0.5,
        eps: float = 1e-4,
        buffer_names: Optional[Sequence[str]] = None,
    ):
        if not isinstance(initial_parameters, OrderedDict) or len(initial_parameters) == 0:
            raise ValueError("initial_parameters must be a non-empty OrderedDict")
        if min_fit_clients < 1:
            raise ValueError(f"min_fit_clients must be >= 1, got {min_fit_clients}")
        if clients_per_round is not None and clients_per_round < min_fit_clients:
            raise ValueError(f"clients_per_round ({clients_per_round}) cannot be less than min_fit_clients ({min_fit_clients})")
        if server_learning_rate is None or not math.isfinite(server_learning_rate) or server_learning_rate <= 0.0:
            raise ValueError(f"server_learning_rate must be positive and finite, got {server_learning_rate}")
        if tau is None or not math.isfinite(tau) or tau <= 0.0:
            raise ValueError(f"tau must be positive and finite, got {tau}")
        if beta1 is None or not math.isfinite(beta1) or not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if beta2 is None or not math.isfinite(beta2) or not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps is None or not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be positive and finite, got {eps}")

        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round if clients_per_round is not None else min_fit_clients

        self.eta_g = float(server_learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.tau = float(tau)
        self.eps = float(eps)
        self.buffer_names = frozenset(buffer_names) if buffer_names else frozenset()

        self._global = OrderedDict(
            (k, v.detach().clone().to(torch.float32)) for k, v in initial_parameters.items()
        )
        self._keys = list(initial_parameters.keys())
        self._param_keys = [
            k for k in self._keys
            if k not in self.buffer_names and not k.endswith((".running_mean", ".running_var", "num_batches_tracked"))
        ]
        self._buffer_keys = [k for k in self._keys if k not in self._param_keys]

        # Flattened state representations for trainable parameters only
        if self._param_keys:
            flat_initial = torch.cat([self._global[k].reshape(-1) for k in self._param_keys])
            self._param_dim = flat_initial.numel()
        else:
            self._param_dim = 0

        self._centre = torch.zeros(self._param_dim, dtype=torch.float32)
        self._server_m = torch.zeros(self._param_dim, dtype=torch.float32)
        self._server_v = torch.zeros(self._param_dim, dtype=torch.float32)

    def initialize_parameters(self) -> OrderedDict[str, torch.Tensor]:
        return self.initial_parameters

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[str, OrderedDict[str, torch.Tensor], int]]
    ) -> Optional[OrderedDict[str, torch.Tensor]]:
        if not results:
            return None

        # Validate and normalize updates to deltas from global
        expected_keys = set(self._keys)
        normalized_results = []
        for res in results:
            _cid, params, _n = normalize_updates([res])[0]
            if set(params.keys()) != expected_keys:
                raise ValueError(
                    f"Client update keys mismatch. Expected {sorted(expected_keys)}, got {sorted(params.keys())}"
                )
            for k in self._keys:
                if params[k].shape != self._global[k].shape:
                    raise ValueError(
                        f"Client update shape mismatch for key {k!r}: expected {self._global[k].shape}, got {params[k].shape}"
                    )
            normalized_results.append(params)

        new_global = OrderedDict()

        # 1. Update non-parameter buffers via linear client averaging (preventing sign-normalised blowup / negative variance)
        for k in self._buffer_keys:
            buf_tensors = [p[k].to(device=self._global[k].device, dtype=self._global[k].dtype) for p in normalized_results]
            mean_buf = torch.stack(buf_tensors, dim=0).mean(dim=0)
            if k.endswith(".running_var"):
                mean_buf = torch.clamp(mean_buf, min=0.0)
            new_global[k] = mean_buf

        # 2. Update parameters via ADFedAdam if param keys exist
        if self._param_keys:
            client_deltas: List[torch.Tensor] = []
            for params in normalized_results:
                # Co-locate to server parameter device before subtraction
                delta_flat = torch.cat([
                    (params[k].to(device=self._global[k].device, dtype=torch.float32) - self._global[k]).reshape(-1)
                    for k in self._param_keys
                ])
                client_deltas.append(delta_flat)

            stacked = torch.stack(client_deltas, dim=0)

            # Adopt device of incoming updates if necessary
            if stacked.device != self._centre.device:
                self._centre = self._centre.to(stacked.device)
                self._server_m = self._server_m.to(stacked.device)
                self._server_v = self._server_v.to(stacked.device)

            # Execute AD-FedAdam step
            step, self._centre, self._server_m, self._server_v = ad_fedadam_step(
                stacked_deltas=stacked,
                centre=self._centre,
                server_m=self._server_m,
                server_v=self._server_v,
                round_idx=server_round,
                tau=self.tau,
                eta_g=self.eta_g,
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.eps,
            )

            # Apply step to global parameters with fail-closed overflow validation
            offset = 0
            for k in self._param_keys:
                numel = self._global[k].numel()
                k_step = step[offset : offset + numel].reshape(self._global[k].shape)
                updated = self._global[k] + k_step.to(device=self._global[k].device, dtype=self._global[k].dtype)
                if not torch.isfinite(updated).all():
                    raise ValueError(
                        f"ADFedAdam produced non-finite global parameter {k!r} after applying step."
                    )
                new_global[k] = updated.clone()
                offset += numel

        self._global = new_global
        return OrderedDict((k, v.clone()) for k, v in self._global.items())

    def evaluate(self, server_round: int, parameters: OrderedDict[str, torch.Tensor]) -> Optional[Tuple[float, Dict]]:
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters)
