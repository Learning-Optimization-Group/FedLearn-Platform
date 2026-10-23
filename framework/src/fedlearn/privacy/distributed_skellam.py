"""Distributed DP with the Skellam mechanism over secure aggregation.

Agarwal, Kairouz and Liu, "The Skellam Mechanism for Differentially Private Federated Learning",
NeurIPS 2021 (arXiv:2110.04995), Algorithm 1. Each client clips its update to ``c``, scales it by
``s``, applies a shared randomized Hadamard rotation, rounds it stochastically to the integer grid
(retrying until the norm bound of Proposition 4.2 holds), adds Skellam noise of variance ``mu / n``
and reduces modulo the secure-aggregation field. The server only ever sees the modular sum, which
carries Skellam noise of variance ``mu``; it lifts, unrotates and unscales it.

Privacy is Corollary 3.6 for integer orders, composed over rounds (no subsampling) and converted with
the same Mironov bound as ``dp_accountant.get_epsilon``. The modulo is post-processing.

Neighbouring relation: replace-one client by default (the capstone simulator), so the integer-domain
sensitivity is twice the rounded-norm bound; ``add_remove`` uses the bound itself.

The guarantee assumes every client follows the encoding. A client that skips the clip is not
detected: the server sees only the masked sum.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Sequence, Tuple

import numpy as np

from fedlearn.privacy.dp_accountant import get_epsilon

__all__ = [
    "INTEGER_ORDERS",
    "SkellamPlan",
    "largest_prime_below",
    "next_pow2",
    "fwht",
    "rotation_signs",
    "rounded_norm_bound_sq",
    "skellam_rdp",
    "skellam_epsilon",
    "calibrate_mu",
    "plan_distributed_skellam",
    "conditional_round",
    "encode",
    "decode",
    "wrapped_coordinates",
]

# Theorem 3.5 holds for integer alpha > 1 only.
INTEGER_ORDERS: Tuple[int, ...] = tuple(range(2, 64)) + (128, 256, 512)
_ADJACENCY_FACTOR = {"add_remove": 1.0, "replace_one": 2.0}


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):  # deterministic for n < 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def largest_prime_below(limit: int) -> int:
    """Largest prime strictly below ``limit`` (the field for ``bits``-bit aggregation is that below 2**bits)."""
    n = int(limit) - 1
    while n >= 2 and not _is_prime(n):
        n -= 1
    if n < 2:
        raise ValueError(f"no prime below {limit}")
    return n


def next_pow2(d: int) -> int:
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    return 1 << (int(d) - 1).bit_length()


def fwht(x: np.ndarray) -> np.ndarray:
    """Normalised Walsh-Hadamard transform along the last axis (length a power of two); its own inverse."""
    x = np.array(x, dtype=np.float64, copy=True)
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"length must be a power of two, got {n}")
    h = 1
    while h < n:
        y = x.reshape(x.shape[:-1] + (n // (2 * h), 2, h))
        a, b = y[..., 0, :].copy(), y[..., 1, :].copy()
        y[..., 0, :], y[..., 1, :] = a + b, a - b
        h *= 2
    return x / math.sqrt(n)


def rotation_signs(seed: int, d_pad: int) -> np.ndarray:
    """The shared random diagonal D of +/-1 (public randomness, e.g. derived from the round)."""
    return np.random.default_rng(int(seed)).choice(np.array([-1.0, 1.0]), size=int(d_pad))


def rounded_norm_bound_sq(scaled_norm: float, d_pad: int, beta: float) -> float:
    """Squared L2 bound a conditionally rounded vector satisfies (Prop. 4.2 and Algorithm 1)."""
    root = scaled_norm + math.sqrt(d_pad) / 2.0
    return min(root ** 2, scaled_norm ** 2 + d_pad / 4.0 + math.sqrt(2.0 * math.log(1.0 / beta)) * root)


def skellam_rdp(alpha: int, mu: float, delta1: float, delta2: float) -> float:
    """Corollary 3.6: RDP of the multidimensional Skellam mechanism at integer order ``alpha``."""
    if int(alpha) != alpha or alpha <= 1:
        raise ValueError(f"alpha must be an integer > 1, got {alpha}")
    if mu <= 0:
        raise ValueError(f"mu must be > 0, got {mu}")
    return (alpha * delta2 ** 2 / (2.0 * mu)
            + min(((2 * alpha - 1) * delta2 ** 2 + 6.0 * delta1) / (4.0 * mu ** 2), 3.0 * delta1 / (2.0 * mu)))


def skellam_epsilon(mu: float, delta1: float, delta2: float, rounds: int, delta: float,
                    orders: Sequence[int] = INTEGER_ORDERS) -> Tuple[float, int]:
    """(epsilon, order) after ``rounds`` full-participation releases at aggregate variance ``mu``."""
    rdp = [rounds * skellam_rdp(a, mu, delta1, delta2) for a in orders]
    eps, order = get_epsilon(rdp, delta, orders=[float(a) for a in orders])
    return eps, int(order)


def calibrate_mu(target_epsilon: float, delta1: float, delta2: float, rounds: int, delta: float,
                 orders: Sequence[int] = INTEGER_ORDERS) -> float:
    """Smallest aggregate variance ``mu`` (to 1e-9 relative) whose epsilon is at most the target."""
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be > 0")
    lo, hi = 1e-12, max(delta2 ** 2, 1.0)
    while skellam_epsilon(hi, delta1, delta2, rounds, delta, orders)[0] > target_epsilon:
        hi *= 2.0
        if hi > 1e300:
            raise ValueError("no mu reaches the target epsilon")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if skellam_epsilon(mid, delta1, delta2, rounds, delta, orders)[0] > target_epsilon:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + 1e-9:
            break
    return hi


@dataclass(frozen=True)
class SkellamPlan:
    """Everything the clients and the server share for one federation."""
    d: int
    d_pad: int
    clip_norm: float
    clients: int
    bits: int
    prime: int
    k: float
    beta: float
    scale: float
    mu: float
    norm_bound_sq: float
    delta1: float
    delta2: float
    epsilon: float
    order: int
    delta: float
    rounds: int
    adjacency: str

    @property
    def sum_noise_std(self) -> float:
        """Standard deviation of the aggregate noise per coordinate, in real (unscaled) units."""
        return math.sqrt(self.mu) / self.scale

    @property
    def signal_bound_std(self) -> float:
        """sigma-tilde of Algorithm 1 in integer units: the aggregate's per-coordinate scale."""
        return math.sqrt(self.scale ** 2 * self.clip_norm ** 2 * self.clients ** 2 / self.d_pad
                         + self.clients / 4.0 + self.mu)

    def as_dict(self) -> Dict:
        return {**asdict(self), "sum_noise_std": self.sum_noise_std, "signal_bound_std": self.signal_bound_std}


def _plan_at(scale, *, target_epsilon, delta, rounds, clip_norm, d, d_pad, clients, bits, prime, k, beta, adjacency):
    bound_sq = rounded_norm_bound_sq(scale * clip_norm, d_pad, beta)
    delta2 = _ADJACENCY_FACTOR[adjacency] * math.sqrt(bound_sq)
    delta1 = min(math.sqrt(d_pad) * delta2, delta2 ** 2)
    mu = calibrate_mu(target_epsilon, delta1, delta2, rounds, delta)
    eps, order = skellam_epsilon(mu, delta1, delta2, rounds, delta)
    return SkellamPlan(d=d, d_pad=d_pad, clip_norm=clip_norm, clients=clients, bits=bits, prime=prime, k=k, beta=beta,
                       scale=scale, mu=mu, norm_bound_sq=bound_sq, delta1=delta1, delta2=delta2, epsilon=eps,
                       order=order, delta=delta, rounds=rounds, adjacency=adjacency)


def plan_distributed_skellam(target_epsilon: float, delta: float, rounds: int, clip_norm: float, d: int,
                             clients: int, bits: int, k: float = 4.0, beta: float = math.exp(-0.5),
                             adjacency: str = "replace_one") -> SkellamPlan:
    """Largest scale ``s`` (to 1e-6 relative) with 2 k sigma-tilde within the ``bits``-bit prime field.

    A larger scale shrinks the rounding inflation, so the largest admissible one is the best; the
    field bound is the heuristic of Section 5 (Algorithm 1's shared-scale step).
    """
    if adjacency not in _ADJACENCY_FACTOR:
        raise ValueError(f"adjacency must be one of {sorted(_ADJACENCY_FACTOR)}, got {adjacency!r}")
    if not 2 <= bits <= 31:
        raise ValueError(f"bits must be in [2, 31] (int64 field products), got {bits}")
    if clip_norm <= 0 or clients < 1 or k <= 0 or not 0 < beta < 1:
        raise ValueError("clip_norm, clients and k must be positive and beta in (0, 1)")
    prime = largest_prime_below(2 ** bits)
    d_pad = next_pow2(d)
    common = dict(target_epsilon=target_epsilon, delta=delta, rounds=rounds, clip_norm=clip_norm, d=d, d_pad=d_pad,
                  clients=clients, bits=bits, prime=prime, k=k, beta=beta, adjacency=adjacency)

    def fits(plan: SkellamPlan) -> bool:
        return 2.0 * k * plan.signal_bound_std <= prime

    lo = 1e-6 / clip_norm
    if not fits(_plan_at(lo, **common)):
        raise ValueError(f"{bits} bits cannot hold the aggregate even at scale {lo:g}")
    hi = lo
    while fits(_plan_at(hi, **common)):
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if fits(_plan_at(mid, **common)):
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + 1e-6:
            break
    return _plan_at(lo, **common)


def conditional_round(values: np.ndarray, bound_sq: float, rng: np.random.Generator,
                      max_tries: int = 1000) -> Tuple[np.ndarray, int]:
    """Unbiased stochastic rounding, retried until the squared norm is within ``bound_sq``."""
    floor = np.floor(values)
    for tries in range(1, max_tries + 1):
        rounded = floor + (rng.random(values.shape) < (values - floor))
        if float(np.dot(rounded, rounded)) <= bound_sq:
            return rounded, tries
    raise RuntimeError(f"conditional rounding did not meet its bound in {max_tries} tries")


def encode(update: np.ndarray, plan: SkellamPlan, signs: np.ndarray, rng: np.random.Generator, *,
           clip: bool = True, max_tries: int = 1000) -> Tuple[np.ndarray, np.ndarray, int]:
    """One client's message: (field vector mod prime, the unreduced integers, rounding attempts).

    ``clip=False`` models a client that skips the protocol's clip and rounding check; nothing on the
    wire distinguishes it.
    """
    x = np.asarray(update, dtype=np.float64).reshape(-1)
    if x.size != plan.d:
        raise ValueError(f"update has {x.size} values, plan expects {plan.d}")
    if clip:
        norm = float(np.linalg.norm(x))
        if norm > plan.clip_norm:
            x = x * (plan.clip_norm / norm)
    padded = np.zeros(plan.d_pad)
    padded[:plan.d] = plan.scale * x
    rotated = fwht(signs * padded)
    rounded, tries = conditional_round(rotated, plan.norm_bound_sq if clip else math.inf, rng, max_tries)
    lam = plan.mu / (2.0 * plan.clients)
    noise = rng.poisson(lam, plan.d_pad).astype(np.int64) - rng.poisson(lam, plan.d_pad).astype(np.int64)
    integers = rounded.astype(np.int64) + noise
    return np.mod(integers, plan.prime), integers, tries


def decode(field_sum: np.ndarray, plan: SkellamPlan, signs: np.ndarray) -> np.ndarray:
    """The server's estimate of the sum of clipped updates (length ``d``, real units)."""
    v = np.mod(np.asarray(field_sum, dtype=np.int64), plan.prime)
    centred = np.where(v > plan.prime // 2, v - plan.prime, v).astype(np.float64)
    return (signs * fwht(centred))[:plan.d] / plan.scale


def wrapped_coordinates(integer_sum: np.ndarray, plan: SkellamPlan) -> int:
    """Coordinates whose true integer sum falls outside the field's centred range (decoded wrongly)."""
    half = plan.prime // 2
    s = np.asarray(integer_sum, dtype=np.int64)
    return int(np.count_nonzero((s > half) | (s < half - plan.prime + 1)))
