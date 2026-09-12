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

import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import torch

from fedlearn.communication.serializer import _reject_non_finite
from fedlearn.server._update_normalize import normalize_updates
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


def _krum_scores(stacked: torch.Tensor, num_byzantine: int) -> torch.Tensor:
    """Per-client Krum scores: the sum of squared L2 distances to the ``n - f - 2`` closest others.

    Shared by :func:`krum_select` and :func:`multi_krum_select` so the two cannot drift apart —
    Multi-Krum is defined as "the m lowest Krum scores", and that only holds if it is literally the
    same score.

    Raises:
        ValueError: if ``num_byzantine`` is negative or ``n < 2f + 3``.
    """
    n = int(stacked.shape[0])
    f = int(num_byzantine)
    if f < 0:
        raise ValueError(f"num_byzantine must be non-negative, got {f}")
    if n < 2 * f + 3:
        raise ValueError(
            f"Krum requires n >= 2f + 3 for its guarantee; got n={n}, f={f} (needs {2 * f + 3})."
        )

    flat = stacked.float().reshape(n, -1)
    sq_dists = torch.cdist(flat, flat).pow(2)
    num_neighbours = n - f - 2

    scores = []
    for i in range(n):
        others = torch.cat([sq_dists[i, :i], sq_dists[i, i + 1:]])
        nearest, _ = torch.sort(others)
        scores.append(nearest[:num_neighbours].sum())
    return torch.stack(scores)


def krum_select(stacked: torch.Tensor, num_byzantine: int) -> int:
    """Krum (Blanchard et al., NIPS 2017, https://arxiv.org/abs/1703.02757): index of the selected client.

    ``stacked`` is ``[n_clients, D]`` — each row is a client's ENTIRE update flattened and
    concatenated across every parameter. This is not optional: Krum scores clients by L2 distance
    over the whole update, so applying it per parameter key would compute a different (and
    unjustified) rule. Median and trimmed-mean are coordinate-wise and do not have this constraint.

    For each client ``i``, the score is the sum of the squared L2 distances to its ``n - f - 2``
    CLOSEST other clients; the selected client is the argmin. The intuition is that an honest client
    sits inside the honest cluster and so has many close neighbours, while a Byzantine client that
    moved far enough to matter is far from everyone.

    Requires ``n >= 2f + 3`` — the paper's condition for the guarantee to hold. Below it the
    neighbour count is still arithmetically defined, which is exactly why this raises rather than
    silently returning a number with no robustness behind it.

    Raises:
        ValueError: if ``n < 2f + 3`` or ``num_byzantine`` is negative.
    """
    return int(_krum_scores(stacked, num_byzantine).argmin().item())


def multi_krum_select(stacked: torch.Tensor, num_byzantine: int, num_select: int) -> List[int]:
    """Multi-Krum (same paper): the ``m`` lowest-scoring client indices, in ASCENDING score order.

    ``m = 1`` reduces to :func:`krum_select` by construction — both read the same
    :func:`_krum_scores`. Returning score order rather than index order is deliberate: the caller
    averaging these clients may want to know which was most central, and a set loses that.

    Raises:
        ValueError: if ``num_select`` is not in ``[1, n]``, or via :func:`_krum_scores`.
    """
    n = int(stacked.shape[0])
    m = int(num_select)
    if not (1 <= m <= n):
        raise ValueError(f"num_select must be in [1, n={n}], got {m}")
    scores = _krum_scores(stacked, num_byzantine)
    return [int(i) for i in torch.argsort(scores)[:m].tolist()]


def bulyan_aggregate(stacked: torch.Tensor, num_byzantine: int) -> torch.Tensor:
    """Bulyan (Mhamdi et al., ICML 2018, https://arxiv.org/abs/1802.07927): robust aggregate vector.

    Krum picks ONE client, which leaves a gap the paper exploits: an attacker can sit close enough
    to the honest cluster to be selected while still being off in a few coordinates. Bulyan closes
    it in two stages:

      1. **Selection** — run Krum ``theta = n - 2f`` times, removing the winner from the pool each
         time, to get a selection set that is a majority-honest committee rather than one client.
      2. **Coordinate-wise trim** — over that set, for EACH coordinate independently, keep the
         ``beta = theta - 2f`` values closest to that coordinate's median and average them.

    Stage 2 is why Bulyan is not simply Multi-Krum: it bounds each coordinate separately, so a
    client selected for its overall proximity cannot still smuggle in one extreme coordinate.

    Unlike :func:`krum_select` this returns the aggregate itself (shape ``[D]``), because stage 2
    produces a value that is not any single client's update.

    Requires ``n >= 4f + 3`` — strictly stronger than Krum's ``2f + 3``, since stage 1 must run
    ``theta`` times and still leave ``beta >= 1``.

    Raises:
        ValueError: if ``n < 4f + 3`` or ``num_byzantine`` is negative.
    """
    n = int(stacked.shape[0])
    f = int(num_byzantine)
    if f < 0:
        raise ValueError(f"num_byzantine must be non-negative, got {f}")
    if n < 4 * f + 3:
        raise ValueError(
            f"Bulyan requires n >= 4f + 3; got n={n}, f={f} (needs {4 * f + 3}). Krum's weaker "
            f"n >= 2f + 3 is not sufficient — stage 1 runs theta = n - 2f selections."
        )

    flat = stacked.float().reshape(n, -1)
    theta = n - 2 * f

    # Stage 1: iterated Krum over a shrinking pool. Indices are tracked against the ORIGINAL rows so
    # the selection set refers to real clients after removals.
    remaining = list(range(n))
    selected: List[int] = []
    for _ in range(theta):
        pool = flat[remaining]
        # The pool shrinks, so f must shrink with it or the n >= 2f + 3 guard trips mid-loop; the
        # paper's accounting is that at most f of whatever remains is Byzantine.
        pool_f = min(f, max(0, (len(remaining) - 3) // 2))
        winner_local = krum_select(pool, num_byzantine=pool_f)
        selected.append(remaining.pop(winner_local))

    chosen = flat[selected]                      # [theta, D]
    beta = theta - 2 * f
    if beta < 1:
        raise ValueError(
            f"Bulyan's trim leaves beta = theta - 2f = {beta} < 1 (n={n}, f={f}); cohort too small."
        )

    # Stage 2: per coordinate, keep the beta values closest to that coordinate's median.
    median = torch.quantile(chosen, 0.5, dim=0, keepdim=True)     # [1, D]
    order = torch.argsort((chosen - median).abs(), dim=0)          # [theta, D]
    keep = torch.gather(chosen, 0, order[:beta])                   # [beta, D]
    return keep.mean(dim=0)


def centered_clip(
        stacked: torch.Tensor, centre: torch.Tensor, tau: float, iterations: int = 1
) -> torch.Tensor:
    """Centered Clipping (Karimireddy et al., ICML 2021, https://arxiv.org/abs/2012.10333).

    Iterate ``v <- v + (1/n) * sum_i clip(x_i - v, tau)``, where
    ``clip(z, tau) = z * min(1, tau / ||z||)``.

    The robustness is structural rather than statistical: every clipped term has norm at most
    ``tau``, so their mean does too, so ONE iteration moves the centre by at most ``tau`` no matter
    what an attacker sends. That is what gives it a 0.5 breakdown point without needing to sort,
    select or discard anyone — every client contributes, but none can contribute more than ``tau``.

    Unlike the selection rules this needs a ``centre`` to clip around. The natural choice on the
    server is the CURRENT GLOBAL MODEL: honest clients sit near it and pass through unclipped, while
    a client that has moved far is scaled back to the ``tau`` ball. A poor centre does not break
    correctness, only the rate — with ``tau`` large enough that nothing clips, one iteration is
    exactly the plain mean regardless of where the centre started.

    Args:
        stacked: ``[n_clients, D]`` — flattened client updates (see :func:`krum_select` on why this
            is flattened rather than per parameter key).
        centre: ``[D]`` — the point to clip around.
        tau: clipping radius; must be positive.
        iterations: how many times to re-centre. The paper finds small values suffice; 1 is the
            common choice and is what the strategy uses.

    Raises:
        ValueError: if ``tau <= 0``, ``iterations < 1``, or the cohort is empty.
    """
    n = int(stacked.shape[0])
    if n == 0:
        raise ValueError("centered_clip requires at least one client.")
    if tau <= 0:
        raise ValueError(f"tau (clipping radius) must be positive, got {tau}")
    if int(iterations) < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    flat = stacked.float().reshape(n, -1)
    v = centre.float().reshape(-1).clone()

    for _ in range(int(iterations)):
        deltas = flat - v                                             # [n, D]
        norms = torch.linalg.vector_norm(deltas, dim=1, keepdim=True)  # [n, 1]
        scale = torch.clamp(tau / (norms + _NORM_EPS), max=1.0)
        v = v + (deltas * scale).mean(dim=0)
    return v


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
# Coordinate-wise rules reduce each parameter coordinate independently, so they can be applied
# per parameter key. Vector rules score or clip a client's ENTIRE flattened update and therefore
# must see it concatenated across every key — applying them per key computes a different rule.
_COORDINATE_METHODS = ("median", "trimmed_mean")
_VECTOR_METHODS = ("krum", "multi_krum", "bulyan", "centered_clip")
_METHODS = _COORDINATE_METHODS + _VECTOR_METHODS

# Asymptotic Byzantine breakdown points. The EXACT per-round admissibility condition (n >= 2f + 3
# for Krum/Multi-Krum, n >= 4f + 3 for Bulyan) is enforced inside the estimators, where n is known;
# these are the cohort-size-independent values the guard compares a fraction against.
_BREAKDOWN = {
    "median": 0.5,
    "krum": 0.5,
    "multi_krum": 0.5,
    "bulyan": 0.25,
    "centered_clip": 0.5,
}


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
            centered_clip_tau: float = 1.0,
            centered_clip_iterations: int = 1,
            multi_krum_m: Optional[int] = None,
    ):
        method = str(method).lower()
        if method not in _METHODS:
            raise ValueError(f"RobustAggregator method must be one of {_METHODS}, got {method!r}")
        if not (0.0 <= trim_ratio < 0.5):
            raise ValueError(f"trim_ratio (beta) must be in [0, 0.5), got {trim_ratio}")
        if clip_norm is not None and clip_norm <= 0:
            raise ValueError(f"clip_norm (S) must be positive or None, got {clip_norm}")
        if centered_clip_tau <= 0:
            raise ValueError(f"centered_clip_tau must be positive, got {centered_clip_tau}")
        if multi_krum_m is not None and multi_krum_m < 1:
            raise ValueError(f"multi_krum_m must be >= 1 or None, got {multi_krum_m}")

        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round if clients_per_round is not None else min_fit_clients

        self.method = method
        self.trim_ratio = float(trim_ratio)
        self.clip_norm = None if clip_norm is None else float(clip_norm)
        self.byzantine_fraction = float(byzantine_fraction)
        # Centered Clipping's radius is scale-dependent: it must be comparable to a typical honest
        # update norm, so the default of 1.0 is a placeholder and should be tuned per task.
        self.centered_clip_tau = float(centered_clip_tau)
        self.centered_clip_iterations = int(centered_clip_iterations)
        # Multi-Krum's committee size; None -> n - 2f at aggregation time (the presumed-honest count).
        self.multi_krum_m = multi_krum_m

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
        """The estimator's Byzantine breakdown point.

        beta for trimmed-mean (operator-chosen), otherwise the rule's published value: 0.5 for
        median, Krum, Multi-Krum and Centered Clipping; 0.25 for Bulyan, whose n >= 4f + 3
        requirement is strictly stronger than Krum's n >= 2f + 3.
        """
        if self.method == "trimmed_mean":
            return self.trim_ratio
        return _BREAKDOWN[self.method]

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
        normalized = normalize_updates(results)

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
            # FR-19: drop a key/shape-mismatched update instead of crashing or wiping the global.
            if not self._conforms_to_global(params):
                dropped += 1
                log.warning(
                    "RobustAggregator dropped key/shape-mismatched update from client %s "
                    "(keys/shapes differ from the global model)", client_id,
                )
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
    def _conforms_to_global(self, params: "OrderedDict[str, torch.Tensor]") -> bool:
        """FR-19: a client update must carry exactly the global model's keys and per-key shapes.

        A missing/extra key or a wrong shape is a malformed (or Byzantine) update. Without this
        gate, ``torch.stack`` in :meth:`_robust_reduce` raises on a shape mismatch (crashing the
        aggregation thread after the client was already accepted), and an empty or mis-keyed
        ``clients[0]`` templates the reduction to a smaller key set — silently dropping those
        parameters from the aggregate and, once persisted, wiping them from the global model. Such
        clients are dropped exactly like non-finite ones rather than allowed to crash or corrupt the
        round.
        """
        if set(params.keys()) != set(self._global.keys()):
            return False
        return all(tuple(params[k].shape) == tuple(self._global[k].shape) for k in self._global)

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
        """Reduce the surviving client updates to one aggregate.

        Dispatches on the rule's KIND, not just its name: coordinate-wise rules go per parameter
        key; vector rules are handed the whole flattened update (see :func:`krum_select`).
        """
        if self.method in _VECTOR_METHODS:
            return self._vector_reduce(clients)

        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        # FR-19: template on the global model's keys, not clients[0] — every survivor has passed
        # _conforms_to_global, so all keys are present with matching shapes, and an empty/mis-keyed
        # first client can no longer silently shrink the aggregated key set.
        for key in self._global.keys():
            stacked = torch.stack([c[key].float() for c in clients], dim=0)
            if self.method == "median":
                out[key] = coordinate_wise_median(stacked)
            else:
                out[key] = trimmed_mean(stacked, self.trim_ratio)
        return out

    def _flatten(self, params: "OrderedDict[str, torch.Tensor]") -> torch.Tensor:
        """Concatenate an update into one 1-D vector, in the global model's key order."""
        return torch.cat([params[k].float().reshape(-1) for k in self._global.keys()])

    def _unflatten(self, flat: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        """Inverse of :meth:`_flatten`, templated on the global model's keys and shapes."""
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        offset = 0
        for key, ref in self._global.items():
            numel = ref.numel()
            out[key] = flat[offset: offset + numel].reshape(ref.shape).clone()
            offset += numel
        return out

    def _vector_reduce(
            self, clients: List["OrderedDict[str, torch.Tensor]"]
    ) -> "OrderedDict[str, torch.Tensor]":
        """Apply a whole-update rule (Krum family, Centered Clipping) and rebuild the key structure.

        ``f`` is derived from the operator's ``byzantine_fraction`` and the ACTUAL cohort size, so
        it tracks a round that came in short. The estimators enforce their own admissibility
        (n >= 2f + 3 / 4f + 3) and raise if the cohort cannot support the rule.
        """
        flat = torch.stack([self._flatten(c) for c in clients], dim=0)
        n = flat.shape[0]
        f = int(self.byzantine_fraction * n)

        if self.method == "krum":
            reduced = flat[krum_select(flat, f)]
        elif self.method == "multi_krum":
            m = self.multi_krum_m if self.multi_krum_m is not None else max(1, n - 2 * f)
            m = min(m, n)
            reduced = flat[multi_krum_select(flat, f, m)].mean(dim=0)
        elif self.method == "bulyan":
            reduced = bulyan_aggregate(flat, f)
        else:  # centered_clip -- clip around the CURRENT GLOBAL, which honest clients sit near
            reduced = centered_clip(
                flat, self._flatten(self._global),
                tau=self.centered_clip_tau, iterations=self.centered_clip_iterations,
            )
        return self._unflatten(reduced)


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
