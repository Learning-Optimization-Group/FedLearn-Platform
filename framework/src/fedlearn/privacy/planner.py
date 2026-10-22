"""What a central-DP budget implies for noise, before a run starts.

Uses the platform RDP accountant (or, on request, the exact Gaussian-DP one) and the noise scale of :func:`fedlearn.privacy.dp_mechanism.dp_aggregate`
(the mechanism behind FedLoRA and DPFedAvg): per-coordinate Gaussian std ``c * z * S / N`` on the uniform mean
of jointly clipped client deltas, with ``q = 1`` (every enrolled client every round, no amplification). ``c`` is
1 under add/remove adjacency and 2 under replace-one.

``noise_norm_to_clip_ratio = c * z * sqrt(d) / N`` is the expected noise norm divided by the largest possible
update norm ``S``. Because the averaged clipped update can never exceed ``S``, it is a hard LOWER bound on
the per-round noise-to-signal ratio: real updates are shorter than ``S``, so the realised ratio is higher.
It does not depend on ``S``; width ``d`` raises it and cohort size ``N`` lowers it.
"""
from __future__ import annotations

import math

from fedlearn.privacy.dp_accountant import compute_rdp, get_epsilon, required_noise_multiplier
from fedlearn.privacy.dp_mechanism import ADJACENCY_SENSITIVITY_FACTOR
from fedlearn.privacy.gdp_accountant import gdp_epsilon, gdp_required_noise_multiplier

_ADJACENCY_LABEL = {"add_remove": "add/remove", "replace_one": "replace-one"}


def plan_central_dp(*, target_epsilon: float, delta: float, rounds: int, clients: int, clip_norm: float,
                    trainable_parameters: int, adjacency: str = "add_remove", accountant: str = "rdp") -> dict:
    """Return the noise multiplier and noise magnitudes an (epsilon, delta) budget implies.

    Args:
        target_epsilon: total epsilon over ``rounds`` releases (> 0).
        delta: in (0, 1).
        rounds: number of noised releases (>= 1).
        clients: fixed cohort size N per round (>= 1).
        clip_norm: joint L2 clip bound S (> 0).
        trainable_parameters: number of federated coordinates d (>= 1).
        adjacency: ``"add_remove"`` or ``"replace_one"``, the neighbour relation epsilon refers to.
        accountant: ``"rdp"`` (upper bound) or ``"gdp"`` (exact at q = 1, so less noise for the same epsilon).
    """
    if not target_epsilon > 0:
        raise ValueError("target_epsilon must be > 0")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if rounds < 1 or clients < 1 or trainable_parameters < 1:
        raise ValueError("rounds, clients and trainable_parameters must be >= 1")
    if not clip_norm > 0:
        raise ValueError("clip_norm must be > 0")
    if adjacency not in ADJACENCY_SENSITIVITY_FACTOR:
        raise ValueError(f"adjacency must be one of {sorted(ADJACENCY_SENSITIVITY_FACTOR)}")
    factor = ADJACENCY_SENSITIVITY_FACTOR[adjacency]
    if accountant == "gdp":
        z = float(gdp_required_noise_multiplier(target_epsilon, rounds, delta))
        accounted = float(gdp_epsilon(z, rounds, delta))
    elif accountant == "rdp":
        z = float(required_noise_multiplier(target_epsilon, 1.0, rounds, delta))
        accounted = float(get_epsilon(compute_rdp(1.0, z, rounds), delta)[0])
    else:
        raise ValueError("accountant must be 'rdp' or 'gdp'")
    std = factor * z * clip_norm / clients
    return {
        "noise_multiplier": z,
        "accounted_epsilon": accounted,
        "accountant": accountant,
        "per_coordinate_noise_std": std,
        "expected_noise_norm": std * math.sqrt(trainable_parameters),
        "noise_norm_to_clip_ratio": factor * z * math.sqrt(trainable_parameters) / clients,
        "adjacency": adjacency,
        "assumptions": f"q=1 fixed cohort, uniform mean of jointly clipped deltas, {_ADJACENCY_LABEL[adjacency]} "
                       "adjacency (dp_aggregate), client-level privacy unit",
    }
