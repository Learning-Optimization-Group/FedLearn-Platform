"""One-run empirical privacy estimate from canary clients (Andrew et al., ICLR'24, arXiv:2302.03098).

Some clients are replaced by canaries that each upload a fixed random direction ``u_c``. After training,
each canary is scored by ``<final model, u_c>``. The same score is computed for random directions that
were never inserted. In ``d`` dimensions, independent random directions are nearly orthogonal to
everything, so the unseen scores show how the final model projects onto a direction it never saw.
The inserted scores show the same plus the canary's pull.

If both score sets are roughly Gaussian with a common spread, telling "inserted" from "not inserted"
is a Gaussian hypothesis test with

    mu_hat = (mean(inserted) - mean(unseen)) / sd(unseen),

which reads as an empirical mu-GDP and converts to an epsilon at the chosen delta.

This is an **estimate for one attack**: an adversary who knows the canary directions and sees the
final model. It is not a proof and not an upper bound. A value well **above** the accounted bound for
the same neighbour relation means the mechanism leaks more than claimed. A value below it is
consistent with the claim but does not establish it.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

from fedlearn.privacy.gdp_accountant import gdp_delta

__all__ = ["canary_audit", "gdp_epsilon_for_mu"]


def gdp_epsilon_for_mu(mu: float, delta: float, *, tol: float = 1e-10) -> float:
    """The smallest epsilon with delta(epsilon) <= ``delta`` for mu-GDP (0 when mu <= 0)."""
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")
    if mu <= 0.0 or gdp_delta(0.0, mu) <= delta:
        return 0.0
    lo, hi = 0.0, 1.0
    while gdp_delta(hi, mu) > delta:
        lo, hi = hi, hi * 2.0
        if hi > 1e6:
            return math.inf
    while hi - lo > tol * max(1.0, hi):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if gdp_delta(mid, mu) > delta else (lo, mid)
    return hi


def canary_audit(inserted: Sequence[float], unseen: Sequence[float], delta: float, *,
                 bootstrap: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Empirical mu and epsilon from inserted-canary and unseen-direction scores.

    Returns the point estimates plus a percentile-bootstrap 95% interval that resamples both
    score sets. Needs at least 2 scores in each set.
    """
    a = np.asarray(inserted, dtype=np.float64)
    b = np.asarray(unseen, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("canary_audit needs at least 2 inserted and 2 unseen scores")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("canary_audit scores must be finite")

    def mu_of(x: np.ndarray, y: np.ndarray) -> float:
        sd = float(y.std(ddof=1))
        return float((x.mean() - y.mean()) / sd) if sd > 0 else math.inf

    mu = mu_of(a, b)
    rng = np.random.default_rng(seed)
    draws = np.array([mu_of(rng.choice(a, a.size), rng.choice(b, b.size)) for _ in range(bootstrap)])
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    return {
        "mu": mu, "mu_interval95": [lo, hi],
        "epsilon": gdp_epsilon_for_mu(mu, delta),
        "epsilon_interval95": [gdp_epsilon_for_mu(lo, delta), gdp_epsilon_for_mu(hi, delta)],
        "inserted": int(a.size), "unseen": int(b.size), "delta": float(delta),
        "inserted_sd_over_unseen_sd": float(a.std(ddof=1) / b.std(ddof=1)) if b.std(ddof=1) > 0 else math.inf,
    }
