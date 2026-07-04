"""Rényi Differential Privacy (RDP) accountant for the Sampled Gaussian Mechanism.

Reference: Mironov, Talwar, Zhang, "Rényi Differential Privacy of the Sampled
Gaussian Mechanism" (2019), https://arxiv.org/abs/1908.10530 — the analysis Opacus
and TF-Privacy implement. This module is a self-contained re-implementation using
only the Python standard library and numpy (numpy is used only for array plumbing in
``compute_rdp`` / ``get_epsilon``; the scalar RDP math is pure ``math``). There is NO
opacus / tensorflow-privacy runtime dependency.

Correctness note — RDP → (ε, δ) conversion bound:
    ``get_epsilon`` uses the CLASSIC Mironov (2017) conversion
        ε = min_α [ rdp(α) + ln(1/δ)/(α − 1) ].
    Opacus' ``get_privacy_spent`` uses the tighter Balle et al. (2020) bound, which
    reports a *smaller* ε for the same RDP curve. The per-order RDP produced here
    matches Opacus to ~1e-9; the ε we report is therefore intentionally conservative
    (never under-reports privacy loss) relative to Opacus. The classic bound is the
    one specified for this accountant and the one against which the q=1 analytic
    oracle is defined.

Public API (stable — imported by the DP mechanism / strategy layer):
    DEFAULT_ORDERS
    compute_rdp(q, noise_multiplier, steps, orders=DEFAULT_ORDERS) -> list[float]
    get_epsilon(rdp, delta, orders=DEFAULT_ORDERS) -> (epsilon, best_order)
    required_noise_multiplier(target_epsilon, q, steps, delta, orders=DEFAULT_ORDERS) -> float
    RDPAccountant
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_ORDERS",
    "RDPAccountant",
    "compute_rdp",
    "get_epsilon",
    "required_noise_multiplier",
]

# Opacus' default order grid. Contains fractional orders (1.1 .. 10.9) and integer
# orders (12 .. 63, 128, 256, 512). Note it never contains α = 1 (the ε conversion
# divides by α − 1), and integer-valued floats such as 2.0 are treated as integers.
DEFAULT_ORDERS: List[float] = (
    [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]
)

# Terms in the fractional-α series below this log-magnitude are negligible; matches
# Opacus' stop condition for the do/until loop.
_LOG_TERM_CUTOFF = -30.0
# Hard safety cap on the fractional-α series length (Opacus relies solely on the
# cutoff; we add a bound so pathological inputs cannot spin forever).
_MAX_FRAC_TERMS = 10000


# ======================================================================================
# Log-space arithmetic (identical semantics to Opacus / TF-Privacy)
# ======================================================================================
def _log_add(log_x: float, log_y: float) -> float:
    """log(exp(log_x) + exp(log_y)), numerically stable."""
    a, b = min(log_x, log_y), max(log_x, log_y)
    if a == -math.inf:  # adding 0
        return b
    # exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b


def _log_sub(log_x: float, log_y: float) -> float:
    """log(exp(log_x) - exp(log_y)); requires log_x >= log_y."""
    if log_x < log_y:
        raise ValueError("_log_sub: result must be non-negative (log_x < log_y).")
    if log_y == -math.inf:  # subtracting 0
        return log_x
    if log_x == log_y:
        return -math.inf  # log(0)
    try:
        # exp(x) - exp(y) = (exp(x - y) - 1) * exp(y)
        return math.log(math.expm1(log_x - log_y)) + log_y
    except (OverflowError, ValueError):
        return log_x


# --- log(erfc(x)), stdlib-only, accurate in the far positive tail ---------------------
# For x where erfc(x) is comfortably representable we defer to math.erfc (full double
# precision). Beyond that (x >~ 26, erfc underflows to 0) we use the classical
# asymptotic expansion
#     erfc(x) = exp(-x²)/(x·√π) · Σ_{n≥0} (-1)^n (2n-1)!! / (2x²)^n ,
# whose log is stable. This mirrors scipy.special.log_ndtr's tail behaviour, since
# log(erfc(x)) = log(2) + log_ndtr(-x·√2).
_ERFC_TAIL_THRESHOLD = 25.0  # below this, math.erfc keeps full precision


def _log_erfc(x: float) -> float:
    """Natural log of the complementary error function, log(erfc(x))."""
    if x <= _ERFC_TAIL_THRESHOLD:
        val = math.erfc(x)
        if val > 0.0:
            return math.log(val)
    # Far positive tail: asymptotic series in 1/(2x²).
    inv_2x2 = 1.0 / (2.0 * x * x)
    term = 1.0
    series = 1.0
    for n in range(1, 64):
        term *= -(2 * n - 1) * inv_2x2
        series += term
        if abs(term) <= 1e-17 * abs(series):
            break
    # log(series) is safe: series -> 1 as x grows, and stays positive for large x.
    return -x * x - math.log(x) - 0.5 * math.log(math.pi) + math.log(series)


# ======================================================================================
# log(A_α) — the SGM moment, Section 3.3 of arXiv:1908.10530
# ======================================================================================
def _log_binom(n: float, k: int) -> float:
    """log of the (integer-α) binomial coefficient C(n, k) via lgamma, k <= n."""
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


def _compute_log_a_int(q: float, sigma: float, alpha: int) -> float:
    """log(A_α) for integer α — exact finite binomial sum in log space."""
    log_a = -math.inf
    for i in range(alpha + 1):
        log_coef_i = _log_binom(alpha, i) + i * math.log(q) + (alpha - i) * math.log1p(-q)
        s = log_coef_i + (i * i - i) / (2.0 * sigma * sigma)
        log_a = _log_add(log_a, s)
    return float(log_a)


def _compute_log_a_frac(q: float, sigma: float, alpha: float) -> float:
    """log(A_α) for fractional α — signed infinite series, truncated by magnitude.

    The fractional binomial C(α, i) alternates sign once i > α; it is carried as a
    running float value (updated multiplicatively, C(α, i+1) = C(α, i)·(α−i)/(i+1))
    so we never form gamma functions of non-positive arguments.
    """
    log_a0, log_a1 = -math.inf, -math.inf
    z0 = sigma * sigma * math.log(1.0 / q - 1.0) + 0.5
    sqrt2_sigma = math.sqrt(2.0) * sigma

    coef = 1.0  # C(α, 0)
    i = 0
    while i < _MAX_FRAC_TERMS:
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log1p(-q)
        log_t1 = log_coef + j * math.log(q) + i * math.log1p(-q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / sqrt2_sigma)
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / sqrt2_sigma)

        log_s0 = log_t0 + (i * i - i) / (2.0 * sigma * sigma) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2.0 * sigma * sigma) + log_e1

        if coef > 0.0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        if max(log_s0, log_s1) < _LOG_TERM_CUTOFF:
            break

        # Advance the running fractional binomial: C(α, i+1) = C(α, i)·(α−i)/(i+1).
        coef *= (alpha - i) / (i + 1)
        i += 1

    return _log_add(log_a0, log_a1)


def _compute_log_a(q: float, sigma: float, alpha: float) -> float:
    if float(alpha).is_integer():
        return _compute_log_a_int(q, sigma, int(alpha))
    return _compute_log_a_frac(q, sigma, alpha)


def _compute_rdp_single(q: float, sigma: float, alpha: float) -> float:
    """RDP of one application of the SGM at order α (before composition)."""
    if q == 0.0:
        return 0.0
    if q == 1.0:  # non-subsampled Gaussian: closed form
        return alpha / (2.0 * sigma * sigma)
    if math.isinf(alpha):
        return math.inf
    return _compute_log_a(q, sigma, alpha) / (alpha - 1.0)


# ======================================================================================
# Public API
# ======================================================================================
def compute_rdp(
    q: float,
    noise_multiplier: float,
    steps: int,
    orders: Sequence[float] = DEFAULT_ORDERS,
) -> List[float]:
    """RDP of ``steps`` compositions of the Sampled Gaussian Mechanism.

    Args:
        q: sampling rate in [0, 1].
        noise_multiplier: σ, the ratio of Gaussian noise std to L2 sensitivity (> 0).
        steps: number of (identical) SGM applications composed (>= 0).
        orders: RDP orders α (> 1). Defaults to ``DEFAULT_ORDERS``.

    Returns:
        list of RDP values, one per order, already multiplied by ``steps``.

    Raises:
        ValueError: on q outside [0, 1], noise_multiplier <= 0, or steps < 0.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"sample rate q must be in [0, 1], got {q!r}")
    if noise_multiplier <= 0.0:
        raise ValueError(f"noise_multiplier (σ) must be > 0, got {noise_multiplier!r}")
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps!r}")

    if steps == 0 or q == 0.0:
        return [0.0 for _ in orders]

    single = [_compute_rdp_single(q, noise_multiplier, float(a)) for a in orders]
    return [r * steps for r in single]


def get_epsilon(
    rdp: Sequence[float],
    delta: float,
    orders: Sequence[float] = DEFAULT_ORDERS,
) -> Tuple[float, float]:
    """Convert an RDP curve to (ε, δ)-DP via the classic Mironov bound.

        ε = min_α [ rdp(α) + ln(1/δ)/(α − 1) ]

    Note: the spec lists the parameters as ``(rdp, orders=DEFAULT_ORDERS, delta)``,
    which is not valid Python (a non-default parameter cannot follow a default one),
    and every concrete call site in the spec passes ``get_epsilon(rdp, delta)``.
    ``delta`` is therefore the second positional argument; ``orders`` keeps its
    default. Both names are preserved for keyword use.

    Args:
        rdp: RDP values aligned with ``orders``.
        delta: target δ in (0, 1).
        orders: the α grid ``rdp`` was computed on.

    Returns:
        (epsilon, best_order) — the minimising order.

    Raises:
        ValueError: if delta is not in (0, 1), or lengths mismatch.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")
    orders_list = list(orders)
    rdp_list = list(rdp)
    if len(orders_list) != len(rdp_list):
        raise ValueError(
            f"orders and rdp must have equal length: "
            f"{len(orders_list)} vs {len(rdp_list)}"
        )

    log_inv_delta = math.log(1.0 / delta)
    best_eps = math.inf
    best_order = float(orders_list[0])
    for a, r in zip(orders_list, rdp_list):
        a = float(a)
        if a <= 1.0:
            # α = 1 has no finite conversion; skip defensively.
            continue
        eps = r + log_inv_delta / (a - 1.0)
        if eps < best_eps:
            best_eps = eps
            best_order = a
    return best_eps, best_order


def required_noise_multiplier(
    target_epsilon: float,
    q: float,
    steps: int,
    delta: float,
    orders: Sequence[float] = DEFAULT_ORDERS,
    *,
    z_min: float = 1e-3,
    z_max: float = 1e6,
    tol: float = 1e-9,
) -> float:
    """Smallest noise multiplier z achieving ε(z) <= target_epsilon.

    ε is monotonically decreasing in z, so this is a monotone search. Uses geometric
    bisection because z ranges over several orders of magnitude.

    Raises:
        ValueError: if the target is infeasible for z in [z_min, z_max], or on bad args.
    """
    if target_epsilon <= 0.0:
        raise ValueError(f"target_epsilon must be > 0, got {target_epsilon!r}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")

    def eps_at(z: float) -> float:
        return get_epsilon(compute_rdp(q, z, steps, orders), delta, orders)[0]

    # Infeasible if even the largest noise can't reach the target.
    if eps_at(z_max) > target_epsilon:
        raise ValueError(
            f"target_epsilon={target_epsilon} is infeasible for q={q}, steps={steps}, "
            f"delta={delta} with z up to {z_max:g} (ε(z_max)={eps_at(z_max):.4g})."
        )
    # Already satisfied by the smallest noise: the target is very loose.
    if eps_at(z_min) <= target_epsilon:
        return z_min

    lo, hi = z_min, z_max  # eps(lo) > target >= eps(hi)
    for _ in range(200):
        mid = math.sqrt(lo * hi)  # geometric midpoint
        if eps_at(mid) <= target_epsilon:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.0 + tol:
            break
    return hi  # feasible side: ε(hi) <= target guaranteed


class RDPAccountant:
    """Accumulates RDP across FL rounds; reads out the running (ε, δ).

    A training loop calls ``step(...)`` once per round and reads ``get_privacy_spent``.
    Heterogeneous rounds (different q / σ each round) compose by summing their RDP
    vectors, which is exactly what ``step`` does.
    """

    def __init__(self, orders: Sequence[float] = DEFAULT_ORDERS) -> None:
        self.orders: List[float] = [float(a) for a in orders]
        self._rdp: np.ndarray = np.zeros(len(self.orders), dtype=float)

    def step(self, noise_multiplier: float, sample_rate: float, num_steps: int = 1) -> None:
        """Accumulate the RDP of ``num_steps`` SGM applications with these params."""
        rdp = compute_rdp(sample_rate, noise_multiplier, num_steps, self.orders)
        self._rdp = self._rdp + np.asarray(rdp, dtype=float)

    def get_privacy_spent(self, delta: float) -> Tuple[float, float]:
        """Current (ε, best_order) for the accumulated RDP at target ``delta``."""
        return get_epsilon(list(self._rdp), delta, self.orders)

    @property
    def rdp(self) -> List[float]:
        """A copy of the accumulated per-order RDP vector."""
        return list(self._rdp)

    def reset(self) -> None:
        """Clear all accumulated privacy loss."""
        self._rdp = np.zeros(len(self.orders), dtype=float)
