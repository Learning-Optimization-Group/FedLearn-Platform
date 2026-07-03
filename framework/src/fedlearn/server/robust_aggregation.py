"""FR-12 — Byzantine-robust aggregation for the FL server.

``RobustAggregator`` is a drop-in :class:`~fedlearn.server.strategy.Strategy` (selectable per
project via the strategy factory) that replaces FedAvg's num-examples-weighted mean with a
Byzantine-robust estimator, and hardens the ingress against poisoned tensors:

  1. **Coordinate-wise median** (Yin et al. 2018, "Byzantine-Robust Distributed Learning",
     https://arxiv.org/abs/1803.01498): for each parameter coordinate, take the statistical median
     across clients. Tolerates up to (but not including) half the clients being Byzantine.
  2. **beta-trimmed-mean** (same paper): for each coordinate, sort the n client values, drop
     ``k = floor(beta * n)`` from EACH end, and average the remaining ``n - 2k``. Tolerates up to
     the trim fraction ``beta`` being Byzantine; ``beta = 0`` recovers the plain mean.
  3. **Non-finite rejection**: any client whose update carries NaN/Inf is dropped BEFORE
     aggregation (reuses the canonical check in ``serializer._reject_non_finite`` — the same guard
     the FedAvg wire path enforces at deserialization). One malformed client cannot poison the
     round; the round completes on the honest survivors.
  4. **Server-side L2 norm clipping**: each client's *update* (delta from the current global) is
     clipped to a configurable global L2 norm ``S`` before aggregation, so no single client can
     contribute an unbounded pull. This is the model-space analogue of the DeComFL scalar clamp
     already applied in ``FLCoordinator.submit_decomfl_update``.
  5. **Byzantine-fraction guard**: if the operator's estimated malicious fraction exceeds the
     estimator's breakdown point (0.5 for median, ``beta`` for trimmed-mean), the round refuses to
     aggregate, leaves the global model untouched, and raises a ``last_round_failed`` signal
     (mirroring ``FLCoordinator.last_round_failed`` / ``last_round_message``). The coordinator
     already treats a ``None`` from ``aggregate_fit`` as a non-fatal round failure: it keeps the
     prior global model and continues the federated loop.

Aggregation is UNWEIGHTED by ``num_examples`` — this is deliberate and matches the robust-statistics
literature: an attacker controls its own reported ``num_examples``, so a weighted median/trimmed-mean
would hand the adversary back the very leverage these estimators exist to remove. The honest
``num_examples`` are still validated (>0) so a zero/negative-count client is dropped.

Robustness assumptions: median tolerates < 1/2 Byzantine clients; trimmed-mean tolerates <= beta.
Both are large-cohort defenses and degrade at the 1-3 client cohorts the platform often runs, which
is exactly why the estimator is opt-in per project rather than the default.
"""

import json
import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import torch

from fedlearn.communication.serializer import _reject_non_finite
from fedlearn.server.strategy import Strategy

log = logging.getLogger(__name__)

# Numerical floor so a zero-norm update never divides by zero during clipping.
_NORM_EPS = 1e-12


# --------------------------------------------------------------------------------------------------
# Pure estimators (module-level so they are unit-testable against their textbook definitions)
# --------------------------------------------------------------------------------------------------
def coordinate_wise_median(stacked: torch.Tensor) -> torch.Tensor:
    """Statistical median along dim 0 (the client axis).

    ``stacked`` has shape ``[num_clients, *param_shape]``; the return has shape ``[*param_shape]``.

    NOTE: this is NOT ``torch.median``. For an even client count ``torch.median`` returns the
    *lower* of the two central order statistics, which is a biased estimator; the true median (and
    the one Yin et al. analyse) is the mean of the two central order statistics. ``torch.quantile``
    with q=0.5 does exactly that (linear interpolation between the two middles), and reduces to the
    middle element for an odd count.
    """
    if stacked.shape[0] == 0:
        raise ValueError("coordinate_wise_median requires at least one client.")
    return torch.quantile(stacked.float(), 0.5, dim=0)


def trimmed_mean(stacked: torch.Tensor, trim_ratio: float) -> torch.Tensor:
    """beta-trimmed mean along dim 0 (the client axis).

    Sort each coordinate's ``n`` client values, drop ``k = floor(trim_ratio * n)`` from EACH end,
    and average the remaining ``n - 2k``. ``trim_ratio = 0`` is the plain mean.

    Raises:
        ValueError: if trimming would remove every value (``2k >= n``).
    """
    n = stacked.shape[0]
    if n == 0:
        raise ValueError("trimmed_mean requires at least one client.")
    k = int(trim_ratio * n)  # floor for non-negative inputs
    if 2 * k >= n:
        raise ValueError(
            f"trim_ratio={trim_ratio} removes every client (n={n}, k={k}); need 2*floor(beta*n) < n."
        )
    ordered, _ = torch.sort(stacked.float(), dim=0)
    kept = ordered[k: n - k]
    return kept.mean(dim=0)


def clip_l2_norm(
        update: "OrderedDict[str, torch.Tensor]", max_norm: float
) -> Tuple["OrderedDict[str, torch.Tensor]", float]:
    """Scale a whole update by ``min(1, max_norm / ||update||_2)`` and return ``(clipped, orig_norm)``.

    ``||update||_2`` is the L2 norm over ALL of the update's tensors concatenated (the same "global
    norm" convention as ``torch.nn.utils.clip_grad_norm_``), so a sprawl-across-many-layers attack
    is bounded jointly rather than per tensor. An update already within budget passes through
    unchanged (the scale is exactly 1.0).
    """
    total_norm = torch.sqrt(sum((t.float() * t.float()).sum() for t in update.values()))
    orig = float(total_norm.item())
    scale = min(1.0, max_norm / (orig + _NORM_EPS))
    clipped = OrderedDict((k, v.float() * scale) for k, v in update.items())
    return clipped, orig


# --------------------------------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------------------------------
_METHODS = ("median", "trimmed_mean")


class RobustAggregator(Strategy):
    """Byzantine-robust Strategy: coordinate-wise median or beta-trimmed-mean with norm clipping.

    Args:
        initial_parameters: the initial global model (also the clip reference for round 1).
        evaluate_fn: optional server-side evaluation callback (same contract as FedAvg).
        min_fit_clients / clients_per_round: cohort sizing (kept for parity with other strategies).
        method: ``"median"`` or ``"trimmed_mean"``.
        trim_ratio: beta in ``[0, 0.5)`` — the per-end trim fraction (trimmed-mean only); also the
            estimator's Byzantine tolerance for the guard.
        clip_norm: L2 bound ``S`` applied to each client's delta before aggregation; ``None``
            disables clipping.
        byzantine_fraction: the operator's ESTIMATE of the malicious client fraction. If it exceeds
            the estimator's breakdown point (0.5 for median, ``trim_ratio`` for trimmed-mean) the
            round refuses to aggregate.
    """

    def __init__(
            self,
            initial_parameters: "OrderedDict[str, torch.Tensor]",
            evaluate_fn: Optional[Callable] = None,
            min_fit_clients: int = 1,
            clients_per_round: int = None,
            method: str = "median",
            trim_ratio: float = 0.1,
            clip_norm: Optional[float] = None,
            byzantine_fraction: float = 0.0,
    ):
        method = str(method).lower()
        if method not in _METHODS:
            raise ValueError(f"RobustAggregator method must be one of {_METHODS}, got {method!r}")
        if not (0.0 <= trim_ratio < 0.5):
            raise ValueError(f"trim_ratio (beta) must be in [0, 0.5), got {trim_ratio}")
        if clip_norm is not None and clip_norm <= 0:
            raise ValueError(f"clip_norm (S) must be positive or None, got {clip_norm}")

        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round if clients_per_round is not None else min_fit_clients

        self.method = method
        self.trim_ratio = float(trim_ratio)
        self.clip_norm = None if clip_norm is None else float(clip_norm)
        self.byzantine_fraction = float(byzantine_fraction)

        # The clip reference / carry-across-rounds global (kept float32, mirroring the aggregator's
        # output dtype so a delta subtraction never silently upcasts).
        self._global: "OrderedDict[str, torch.Tensor]" = OrderedDict(
            (k, v.detach().clone().to(torch.float32)) for k, v in initial_parameters.items()
        )

        # last_round_failed-style signal (mirrors FLCoordinator's naming) surfaced when the
        # Byzantine guard refuses or every client is dropped as non-finite.
        self.last_round_failed = False
        self.last_round_message: Optional[str] = None

        log.info(
            "RobustAggregator initialised: method=%s trim_ratio=%g clip_norm=%s byz_frac=%g",
            self.method, self.trim_ratio, self.clip_norm, self.byzantine_fraction,
        )

    @property
    def tolerance(self) -> float:
        """The estimator's Byzantine breakdown point: 0.5 for median, beta for trimmed-mean."""
        return 0.5 if self.method == "median" else self.trim_ratio

    def initialize_parameters(self) -> Optional["OrderedDict[str, torch.Tensor]"]:
        return self.initial_parameters

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple["OrderedDict[str, torch.Tensor]", int]],
    ) -> Optional["OrderedDict[str, torch.Tensor]"]:
        self.last_round_failed = False
        self.last_round_message = None

        if not results:
            return None

        # Byzantine-fraction guard: refuse outright if the estimated malicious fraction is beyond
        # what this estimator can tolerate. The global model is left untouched.
        if self.byzantine_fraction > self.tolerance:
            self.last_round_failed = True
            self.last_round_message = (
                f"Estimated Byzantine fraction {self.byzantine_fraction:.3f} exceeds the "
                f"{self.method} tolerance {self.tolerance:.3f}; refusing to aggregate round "
                f"{server_round}."
            )
            log.error(self.last_round_message)
            return None

        # Normalise the wire formats FedAvgAggregator also accepts (2-/3-tuples, JSON-encoded
        # params) into a uniform list of (client_id, state_dict, num_examples).
        normalized = _normalize_updates(results)

        # Drop non-finite clients (reusing the canonical serializer check) and clip the survivors.
        survivors: List["OrderedDict[str, torch.Tensor]"] = []
        dropped = 0
        for client_id, params, num_examples in normalized:
            if num_examples <= 0:
                dropped += 1
                continue
            if not _is_finite(params):
                dropped += 1
                log.warning("RobustAggregator dropped non-finite update from client %s", client_id)
                continue
            survivors.append(self._clip_update(params))

        if not survivors:
            self.last_round_failed = True
            self.last_round_message = (
                f"Round {server_round}: every client update was dropped "
                f"(non-finite or invalid num_examples); nothing to aggregate."
            )
            log.error(self.last_round_message)
            return None

        aggregated = self._robust_reduce(survivors)

        if dropped:
            log.info(
                "RobustAggregator round %d: aggregated %d clients (%d dropped) via %s",
                server_round, len(survivors), dropped, self.method,
            )

        # Persist as the next round's clip reference / carry-across global.
        self._global = OrderedDict((k, v.clone()) for k, v in aggregated.items())
        return aggregated

    def evaluate(
            self, server_round: int, parameters: "OrderedDict[str, torch.Tensor]"
    ) -> Optional[Tuple[float, dict]]:
        if self.evaluate_fn is None:
            return None
        loss, metrics = self.evaluate_fn(server_round, parameters)
        log.info(
            "RobustAggregator eval round=%d loss=%.4f metrics=%s", server_round, loss, metrics
        )
        return loss, metrics

    # ---- internals -------------------------------------------------------------------------------
    def _clip_update(
            self, params: "OrderedDict[str, torch.Tensor]"
    ) -> "OrderedDict[str, torch.Tensor]":
        """Clip a client's DELTA (params - current global) to L2 norm ``clip_norm``, then
        reconstruct the clipped model as ``global + clipped_delta``.

        Clipping the delta (not the raw model) is what bounds each client's per-round *pull*: an
        honest client near the global has a small delta and passes unchanged; a hijacked client
        with a huge delta is scaled back so its contribution to the estimator is bounded by ``S``.
        No-op when ``clip_norm is None``. Both median and trimmed-mean are translation-equivariant,
        so when clipping is off this reduces to the robust estimator over the raw client models.
        """
        if self.clip_norm is None:
            return OrderedDict((k, v.float()) for k, v in params.items())

        delta = OrderedDict(
            (k, v.float() - self._global[k]) for k, v in params.items()
        )
        clipped_delta, _ = clip_l2_norm(delta, self.clip_norm)
        return OrderedDict(
            (k, self._global[k] + clipped_delta[k]) for k in params.keys()
        )

    def _robust_reduce(
            self, clients: List["OrderedDict[str, torch.Tensor]"]
    ) -> "OrderedDict[str, torch.Tensor]":
        """Apply the coordinate-wise estimator per parameter key over the stacked client tensors."""
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for key in clients[0].keys():
            stacked = torch.stack([c[key].float() for c in clients], dim=0)
            if self.method == "median":
                out[key] = coordinate_wise_median(stacked)
            else:
                out[key] = trimmed_mean(stacked, self.trim_ratio)
        return out


def _is_finite(params: "OrderedDict[str, torch.Tensor]") -> bool:
    """True iff every tensor is finite. Reuses the canonical serializer check (which RAISES on a
    non-finite tensor) so this second-layer drop shares one definition of "poisoned" with the wire
    path, rather than re-implementing an ``isfinite`` test that could drift out of sync."""
    for name, tensor in params.items():
        try:
            _reject_non_finite(name, tensor.detach().cpu().numpy())
        except ValueError:
            return False
    return True


def _normalize_updates(
        results: List[Tuple],
) -> List[Tuple[Optional[str], "OrderedDict[str, torch.Tensor]", int]]:
    """Coerce the accepted wire shapes into ``(client_id, state_dict, num_examples)``.

    Mirrors ``FedAvgAggregator.aggregate``'s front-matter: entries may be ``(client_id, params, n)``
    or ``(params, n)``, and ``params`` may be a JSON string that decodes to a plain dict of lists.
    """
    normalized = []
    for entry in results:
        if len(entry) == 3:
            client_id, params, num_examples = entry
        else:
            params, num_examples = entry
            client_id = None

        if isinstance(params, str):
            try:
                decoded = json.loads(params)
                params = OrderedDict({k: torch.tensor(v) for k, v in decoded.items()})
            except Exception as e:  # noqa: BLE001 — surface the offending client id
                raise ValueError(f"Failed to deserialize parameters from {client_id}: {e}")

        normalized.append((client_id, params, num_examples))
    return normalized
