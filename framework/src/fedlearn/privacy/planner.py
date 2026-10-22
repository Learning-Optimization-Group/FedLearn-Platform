"""What a central-DP budget implies for noise, before a run starts.

Uses the platform RDP accountant and the noise scale of :func:`fedlearn.privacy.dp_mechanism.dp_aggregate`
(the mechanism behind FedLoRA and DPFedAvg): per-coordinate Gaussian std ``z * S / N`` on the uniform mean
of jointly clipped client deltas, with ``q = 1`` (every enrolled client every round, no amplification).

``noise_norm_to_clip_ratio = z * sqrt(d) / N`` is the expected noise norm divided by the largest possible
update norm ``S``. Because the averaged clipped update can never exceed ``S``, it is a hard LOWER bound on
the per-round noise-to-signal ratio: real updates are shorter than ``S``, so the realised ratio is higher.
It does not depend on ``S``; width ``d`` raises it and cohort size ``N`` lowers it.
"""
from __future__ import annotations

import math

from fedlearn.privacy.dp_accountant import compute_rdp, get_epsilon, required_noise_multiplier


def plan_central_dp(*, target_epsilon: float, delta: float, rounds: int, clients: int, clip_norm: float,
                    trainable_parameters: int) -> dict:
    """Return the noise multiplier and noise magnitudes an (epsilon, delta) budget implies.

    Args:
        target_epsilon: total epsilon over ``rounds`` releases (> 0).
        delta: in (0, 1).
        rounds: number of noised releases (>= 1).
        clients: fixed cohort size N per round (>= 1).
        clip_norm: joint L2 clip bound S (> 0).
        trainable_parameters: number of federated coordinates d (>= 1).
    """
    if not target_epsilon > 0:
        raise ValueError("target_epsilon must be > 0")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if rounds < 1 or clients < 1 or trainable_parameters < 1:
        raise ValueError("rounds, clients and trainable_parameters must be >= 1")
    if not clip_norm > 0:
        raise ValueError("clip_norm must be > 0")
    z = float(required_noise_multiplier(target_epsilon, 1.0, rounds, delta))
    std = z * clip_norm / clients
    return {
        "noise_multiplier": z,
        "accounted_epsilon": float(get_epsilon(compute_rdp(1.0, z, rounds), delta)[0]),
        "per_coordinate_noise_std": std,
        "expected_noise_norm": std * math.sqrt(trainable_parameters),
        "noise_norm_to_clip_ratio": z * math.sqrt(trainable_parameters) / clients,
        "assumptions": "q=1 fixed cohort, uniform mean of jointly clipped deltas, add/remove adjacency "
                       "(dp_aggregate), client-level privacy unit",
    }
