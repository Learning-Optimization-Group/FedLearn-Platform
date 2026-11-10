"""Differential-privacy primitives for the FedLearn framework.

Currently exposes the RDP accountant for the Sampled Gaussian Mechanism
(Mironov-Talwar-Zhang 2019). Pure Python (stdlib + numpy); no opacus /
tensorflow-privacy runtime dependency.
"""

from fedlearn.privacy.adaptive_clipping import DPAdaptiveClipping
from fedlearn.privacy.dp_accountant import (
    DEFAULT_ORDERS,
    RDPAccountant,
    compute_rdp,
    get_epsilon,
    required_noise_multiplier,
)
from fedlearn.privacy.gdp_accountant import (
    gdp_delta,
    gdp_epsilon,
    gdp_required_noise_multiplier,
)
from fedlearn.privacy.subspace import SubspaceProjector, build_orthonormal_basis
from fedlearn.privacy.subspace_dp import subspace_dp_aggregate

__all__ = [
    "DEFAULT_ORDERS",
    "DPAdaptiveClipping",
    "RDPAccountant",
    "SubspaceProjector",
    "build_orthonormal_basis",
    "compute_rdp",
    "get_epsilon",
    "gdp_delta",
    "gdp_epsilon",
    "gdp_required_noise_multiplier",
    "required_noise_multiplier",
    "subspace_dp_aggregate",
]
