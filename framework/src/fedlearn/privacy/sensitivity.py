"""Sensitivity Registry and Falsification Engine for Differentially Private Aggregation.

Provides formal SensitivityClaim contracts, a registry of sound aggregation rules under
Central Differential Privacy, and an automated falsification engine that subjects
aggregation functions to structured and randomized adversarial witness search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

def clip_tensor_rows(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Clips each row of a 2D tensor to maximum L2 norm max_norm."""
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")
    norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    factors = torch.clamp(float(max_norm) / torch.clamp(norms, min=1e-12), max=1.0)
    return x * factors


@dataclass(frozen=True)
class SensitivityClaim:
    """Formal sensitivity claim for a differentially private aggregation rule."""

    rule_name: str
    adjacency: str  # 'replace_one' or 'add_remove'
    formula_latex: str
    centre_provenance: str  # 'none', 'constant', 'prior_release', 'private_state'
    cohort_semantics: str  # 'public_constant' or 'realised_cohort'
    calculator: Callable[..., float]

    def compute(self, num_clients: int, clip_norm: float, **kwargs) -> float:
        return self.calculator(num_clients=num_clients, clip_norm=clip_norm, **kwargs)


@dataclass
class FalsificationResult:
    """Outcome of attempting to falsify a sensitivity claim."""

    falsified: bool
    claimed_sensitivity: float
    max_observed_distance: float
    breach_factor: float
    iterations_run: int
    witness_notes: str


def _calc_mean_replace_one(num_clients: int, clip_norm: float, **kwargs) -> float:
    return 2.0 * float(clip_norm) / float(num_clients)


def _calc_mean_add_remove(num_clients: int, clip_norm: float, **kwargs) -> float:
    return float(clip_norm) / float(num_clients)


def _calc_trimmed_mean_replace_one(num_clients: int, clip_norm: float, trim_ratio: float = 0.0, **kwargs) -> float:
    k = int(trim_ratio * num_clients)
    kept = num_clients - 2 * k
    if kept <= 0:
        raise ValueError(f"Trim ratio {trim_ratio} removes all {num_clients} clients (2k={2*k})")
    return 2.0 * float(clip_norm) / float(kept)


def _calc_centered_clip_replace_one(
    num_clients: int, clip_norm: float, tau: Optional[float] = None, **kwargs
) -> float:
    tau_val = float(tau) if (tau is not None and tau > 0) else float(clip_norm)
    eff_tau = min(float(clip_norm), tau_val)
    return 2.0 * eff_tau / float(num_clients)


_BUILTIN_CLAIMS: Dict[Tuple[str, str], SensitivityClaim] = {
    ("mean", "replace_one"): SensitivityClaim(
        rule_name="mean",
        adjacency="replace_one",
        formula_latex=r"\frac{2C}{N}",
        centre_provenance="none",
        cohort_semantics="public_constant",
        calculator=_calc_mean_replace_one,
    ),
    ("mean", "add_remove"): SensitivityClaim(
        rule_name="mean",
        adjacency="add_remove",
        formula_latex=r"\frac{C}{N}",
        centre_provenance="none",
        cohort_semantics="public_constant",
        calculator=_calc_mean_add_remove,
    ),
    ("trimmed_mean", "replace_one"): SensitivityClaim(
        rule_name="trimmed_mean",
        adjacency="replace_one",
        formula_latex=r"\frac{2C}{N - 2\lfloor\beta N\rfloor}",
        centre_provenance="none",
        cohort_semantics="public_constant",
        calculator=_calc_trimmed_mean_replace_one,
    ),
    ("centered_clip", "replace_one"): SensitivityClaim(
        rule_name="centered_clip",
        adjacency="replace_one",
        formula_latex=r"\frac{2\min(C, \tau)}{N}",
        centre_provenance="prior_release",
        cohort_semantics="public_constant",
        calculator=_calc_centered_clip_replace_one,
    ),
}


class SensitivityRegistry:
    """Registry of verified DP sensitivity claims.
    
    DP runs fail-closed: an aggregation rule without an explicit registered claim
    cannot join an accounted DP pipeline.
    """

    def __init__(self):
        self._registry: Dict[Tuple[str, str], SensitivityClaim] = dict(_BUILTIN_CLAIMS)

    def register(self, claim: SensitivityClaim) -> None:
        key = (claim.rule_name, claim.adjacency)
        self._registry[key] = claim

    def get_claim(self, rule_name: str, adjacency: str) -> SensitivityClaim:
        key = (rule_name, adjacency)
        if key not in self._registry:
            raise ValueError(
                f"Rule {rule_name!r} with adjacency {adjacency!r} is not registered in the DP Sensitivity Registry. "
                f"Available rules: {sorted(list(self._registry.keys()))}. "
                f"To prevent silent under-noising, DP mechanisms fail closed on unregistered rules."
            )
        return self._registry[key]

    def is_registered(self, rule_name: str, adjacency: str) -> bool:
        return (rule_name, adjacency) in self._registry


GLOBAL_SENSITIVITY_REGISTRY = SensitivityRegistry()


def find_sensitivity_witness(
    aggregator_fn: Callable[[torch.Tensor], torch.Tensor],
    claimed_sensitivity: float,
    num_clients: int,
    dim: int,
    clip_norm: float,
    num_random_trials: int = 50,
    seed: int = 42,
    tolerance: float = 1e-4,
    adjacency: str = "replace_one",
) -> FalsificationResult:
    """Search for a witness pair of neighbouring cohorts that falsifies claimed sensitivity.

    Tests both structured extremal constructions (diametric vectors, coordinate extremes)
    and random restarts with gradient-free perturbation.
    Supports adjacency='replace_one' (n vs n clients) and adjacency='add_remove' (n vs n+1 clients).
    """
    if adjacency not in ("replace_one", "add_remove"):
        raise ValueError(f"adjacency must be 'replace_one' or 'add_remove', got {adjacency!r}")
    gen = torch.Generator().manual_seed(seed)
    max_dist = 0.0
    worst_note = "none"

    if adjacency == "add_remove":
        # --- Structured witness checks for add/remove (n vs n+1 clients) ---
        # 1. Opposite pole addition: n clients at -C*e_0, (n+1)-th client at +C*e_0
        d1 = torch.zeros(num_clients, dim, dtype=torch.float32)
        d1[:, 0] = -clip_norm
        d2 = torch.zeros(num_clients + 1, dim, dtype=torch.float32)
        d2[:num_clients, 0] = -clip_norm
        d2[num_clients, 0] = clip_norm
        out1 = aggregator_fn(d1)
        out2 = aggregator_fn(d2)
        dist = float(torch.linalg.vector_norm(out1 - out2))
        if dist > max_dist:
            max_dist = dist
            worst_note = "structured opposite pole addition"

        # 2. Positive pole addition: n clients at +C*e_0, (n+1)-th client at -C*e_0
        d1_pos = torch.zeros(num_clients, dim, dtype=torch.float32)
        d1_pos[:, 0] = clip_norm
        d2_pos = torch.zeros(num_clients + 1, dim, dtype=torch.float32)
        d2_pos[:num_clients, 0] = clip_norm
        d2_pos[num_clients, 0] = -clip_norm
        out1_pos = aggregator_fn(d1_pos)
        out2_pos = aggregator_fn(d2_pos)
        dist_pos = float(torch.linalg.vector_norm(out1_pos - out2_pos))
        if dist_pos > max_dist:
            max_dist = dist_pos
            worst_note = "structured positive pole addition"

        # 3. Addition to all-zeros base: n clients at 0, (n+1)-th client at +C*e_0
        d1_zero = torch.zeros(num_clients, dim, dtype=torch.float32)
        d2_zero = torch.zeros(num_clients + 1, dim, dtype=torch.float32)
        d2_zero[num_clients, 0] = clip_norm
        out1_zero = aggregator_fn(d1_zero)
        out2_zero = aggregator_fn(d2_zero)
        dist_zero = float(torch.linalg.vector_norm(out1_zero - out2_zero))
        if dist_zero > max_dist:
            max_dist = dist_zero
            worst_note = "structured zero base addition"

        # 4. Saturated coordinate spread: n clients spread, (n+1)-th client at +C*e_0
        d1_sat = torch.zeros(num_clients, dim, dtype=torch.float32)
        step = 2.0 * clip_norm / max(1, num_clients - 1)
        for i in range(num_clients):
            d1_sat[i, 0] = -clip_norm + i * step
        d2_sat = torch.zeros(num_clients + 1, dim, dtype=torch.float32)
        d2_sat[:num_clients] = d1_sat
        d2_sat[num_clients, 0] = clip_norm
        out1_sat = aggregator_fn(d1_sat)
        out2_sat = aggregator_fn(d2_sat)
        dist_sat = float(torch.linalg.vector_norm(out1_sat - out2_sat))
        if dist_sat > max_dist:
            max_dist = dist_sat
            worst_note = "structured coordinate spread addition"

        # 5. Median split pivot under add/remove:
        half = num_clients // 2
        d1_med = torch.zeros(num_clients, dim, dtype=torch.float32)
        d1_med[:half, 0] = -clip_norm
        d1_med[half:, 0] = clip_norm
        d2_med = torch.zeros(num_clients + 1, dim, dtype=torch.float32)
        d2_med[:num_clients] = d1_med
        d2_med[num_clients, 0] = clip_norm
        out1_med = aggregator_fn(d1_med)
        out2_med = aggregator_fn(d2_med)
        dist_med = float(torch.linalg.vector_norm(out1_med - out2_med))
        if dist_med > max_dist:
            max_dist = dist_med
            worst_note = "structured median split addition"

        # 6. Randomized witness search for add/remove (n vs n+1 clients)
        for trial in range(num_random_trials):
            base = torch.randn(num_clients, dim, generator=gen)
            base = clip_tensor_rows(base, max_norm=clip_norm)
            cand = torch.randn(1, dim, generator=gen)
            cand = clip_tensor_rows(cand, max_norm=clip_norm)

            c1 = base
            c2 = torch.cat([base, cand], dim=0)

            o1 = aggregator_fn(c1)
            o2 = aggregator_fn(c2)
            d = float(torch.linalg.vector_norm(o1 - o2))
            if d > max_dist:
                max_dist = d
                worst_note = f"random add/remove trial {trial}"

    else:
        # --- 1. Structured witness check: Diametric opposition on dimension 0 ---
        # Client 0 is at +C * e_0, replaced by -C * e_0. Other clients zero.
        d1 = torch.zeros(num_clients, dim, dtype=torch.float32)
        d2 = torch.zeros(num_clients, dim, dtype=torch.float32)
        d1[0, 0] = clip_norm
        d2[0, 0] = -clip_norm
        out1 = aggregator_fn(d1)
        out2 = aggregator_fn(d2)
        dist = float(torch.linalg.vector_norm(out1 - out2))
        if dist > max_dist:
            max_dist = dist
            worst_note = "structured diametric opposition"

        # --- 2. Structured witness check: Saturated coordinate spread ---
        d1_sat = torch.zeros(num_clients, dim, dtype=torch.float32)
        d2_sat = torch.zeros(num_clients, dim, dtype=torch.float32)
        step = 2.0 * clip_norm / max(1, num_clients - 1)
        for i in range(1, num_clients):
            val = -clip_norm + (i - 1) * step
            d1_sat[i, 0] = val
            d2_sat[i, 0] = val
        d1_sat[0, 0] = clip_norm
        d2_sat[0, 0] = -clip_norm
        out1_sat = aggregator_fn(d1_sat)
        out2_sat = aggregator_fn(d2_sat)
        dist_sat = float(torch.linalg.vector_norm(out1_sat - out2_sat))
        if dist_sat > max_dist:
            max_dist = dist_sat
            worst_note = "structured coordinate spread"

        # --- 3. Structured witness check: Trimming & Order-Statistic Boundaries ---
        max_k = max(1, (num_clients - 1) // 2)
        for k in range(1, max_k + 1):
            d1_trim = torch.zeros(num_clients, dim, dtype=torch.float32)
            d2_trim = torch.zeros(num_clients, dim, dtype=torch.float32)
            # k clients at +C, k clients at -C
            for i in range(1, min(num_clients, 1 + k)):
                d1_trim[i, 0] = clip_norm
                d2_trim[i, 0] = clip_norm
            for i in range(1 + k, min(num_clients, 1 + 2 * k)):
                d1_trim[i, 0] = -clip_norm
                d2_trim[i, 0] = -clip_norm
            d1_trim[0, 0] = clip_norm
            d2_trim[0, 0] = -clip_norm
            out1_trim = aggregator_fn(d1_trim)
            out2_trim = aggregator_fn(d2_trim)
            dist_trim = float(torch.linalg.vector_norm(out1_trim - out2_trim))
            if dist_trim > max_dist:
                max_dist = dist_trim
                worst_note = f"structured trimming boundary k={k}"

        # --- 4. Structured witness check: Median split pivot ---
        # Tests order-N sensitivity collapse on median-like rules by splitting cohort
        # evenly across opposite poles. Replacing client 0 pivots the median across the gap.
        half = num_clients // 2
        d1_med = torch.zeros(num_clients, dim, dtype=torch.float32)
        d2_med = torch.zeros(num_clients, dim, dtype=torch.float32)
        # half clients at -C, other clients at +C
        d1_med[:half, 0] = -clip_norm
        d1_med[half:, 0] = clip_norm
        d2_med = d1_med.clone()
        d2_med[0, 0] = clip_norm  # replace one -C with +C
        out1_med = aggregator_fn(d1_med)
        out2_med = aggregator_fn(d2_med)
        dist_med = float(torch.linalg.vector_norm(out1_med - out2_med))
        if dist_med > max_dist:
            max_dist = dist_med
            worst_note = "structured median split pivot"

        # --- 5. Structured witness check: Selection cluster pivot (Krum / Bulyan) ---
        # Two tight clusters at -C * e_0 and +C * e_0.
        # Replaced client 0 moves from cluster A to cluster B, shifting the selection minimum.
        if num_clients >= 4:
            d1_sel = torch.zeros(num_clients, dim, dtype=torch.float32)
            d1_sel[1:half, 0] = -clip_norm
            d1_sel[half:, 0] = clip_norm
            d1_sel[0, 0] = -clip_norm  # makes cluster A size half
            d2_sel = d1_sel.clone()
            d2_sel[0, 0] = clip_norm   # makes cluster B size (num_clients - half + 1)
            out1_sel = aggregator_fn(d1_sel)
            out2_sel = aggregator_fn(d2_sel)
            dist_sel = float(torch.linalg.vector_norm(out1_sel - out2_sel))
            if dist_sel > max_dist:
                max_dist = dist_sel
                worst_note = "structured selection cluster pivot"

        # --- 6. Randomized witness search ---
        for trial in range(num_random_trials):
            # Random honest cohort of size num_clients - 1
            honest = torch.randn(num_clients - 1, dim, generator=gen)
            honest = clip_tensor_rows(honest, max_norm=clip_norm)

            # Two candidate updates for replaced client 0
            cand1 = torch.randn(1, dim, generator=gen)
            cand1 = clip_tensor_rows(cand1, max_norm=clip_norm)
            cand2 = torch.randn(1, dim, generator=gen)
            cand2 = clip_tensor_rows(cand2, max_norm=clip_norm)

            c1 = torch.cat([cand1, honest], dim=0)
            c2 = torch.cat([cand2, honest], dim=0)

            o1 = aggregator_fn(c1)
            o2 = aggregator_fn(c2)
            d = float(torch.linalg.vector_norm(o1 - o2))
            if d > max_dist:
                max_dist = d
                worst_note = f"random restart trial {trial}"

    breach_factor = max_dist / claimed_sensitivity if claimed_sensitivity > 0 else float("inf")
    falsified = max_dist > claimed_sensitivity * (1.0 + tolerance)

    return FalsificationResult(
        falsified=falsified,
        claimed_sensitivity=claimed_sensitivity,
        max_observed_distance=max_dist,
        breach_factor=breach_factor,
        iterations_run=num_random_trials + 2,
        witness_notes=worst_note,
    )
