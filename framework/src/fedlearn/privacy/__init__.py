"""Differential-privacy primitives for the FedLearn framework.

Currently exposes the RDP accountant for the Sampled Gaussian Mechanism
(Mironov-Talwar-Zhang 2019). Pure Python (stdlib + numpy); no opacus /
tensorflow-privacy runtime dependency.
"""

from fedlearn.privacy.dp_accountant import (
    DEFAULT_ORDERS,
    RDPAccountant,
    compute_rdp,
    get_epsilon,
    required_noise_multiplier,
)

__all__ = [
    "DEFAULT_ORDERS",
    "RDPAccountant",
    "compute_rdp",
    "get_epsilon",
    "required_noise_multiplier",
]
