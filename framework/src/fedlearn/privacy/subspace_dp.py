"""Differentially Private Federated Learning via Intrinsic Subspace Projections.

Executes DP clipping, uniform averaging, and calibrated Gaussian noise addition strictly
within an orthonormal intrinsic subspace R^k (k << d), then reconstructs the model update
in R^d through the public round basis.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

from fedlearn.privacy.dp_mechanism import ADJACENCY_SENSITIVITY_FACTOR, dp_aggregate
from fedlearn.privacy.subspace import SubspaceProjector
from fedlearn.server._update_normalize import normalize_updates


def subspace_dp_aggregate(
    results: Sequence[tuple],
    global_params: "OrderedDict[str, torch.Tensor]",
    aggregatable_keys: Iterable[str],
    clip_norm: float,
    noise_multiplier: float,
    subspace_dim: int = 128,
    server_round: int = 0,
    subspace_seed: int = 42,
    rotating_subspace: bool = True,
    adjacency: str = "add_remove",
    generator: Optional[torch.Generator] = None,
    public_cohort: Optional[int] = None,
) -> "OrderedDict[str, torch.Tensor]":
    """Central Differential Privacy in an intrinsic subspace R^k (Yu et al. 2021).

    Args:
        results: Sequence of client updates (cid, params, num_examples) or similar.
        global_params: Current global model parameters.
        aggregatable_keys: Keys to include in the subspace projection and aggregation.
        clip_norm: Maximum L2 norm C applied to projected coordinate vectors in R^k.
        noise_multiplier: Noise multiplier z.
        subspace_dim: Dimension k of the projection subspace (k << d).
        server_round: Current federated round index (rotates the seed when rotating_subspace is True).
        subspace_seed: Base PRG seed for generating the orthonormal basis.
        rotating_subspace: Whether to rotate the subspace basis across rounds.
        adjacency: "add_remove" (default, c=1) or "replace_one" (c=2).
        generator: Dedicated torch.Generator for Gaussian noise. Fresh OS entropy used if None.
        public_cohort: Optional public cohort size N; raises if len(results) differs.

    Returns:
        OrderedDict of updated parameters in R^d.
    """
    if adjacency not in ADJACENCY_SENSITIVITY_FACTOR:
        raise ValueError(f"adjacency must be one of {sorted(ADJACENCY_SENSITIVITY_FACTOR)}, got {adjacency!r}")
    if not results:
        raise ValueError("subspace_dp_aggregate requires at least one client update.")
    if clip_norm is None or clip_norm <= 0:
        raise ValueError(f"clip_norm (C) must be positive, got {clip_norm}")
    if noise_multiplier is None or noise_multiplier < 0:
        raise ValueError(f"noise_multiplier (z) must be >= 0, got {noise_multiplier}")
    if subspace_dim is None or subspace_dim < 1:
        raise ValueError(f"subspace_dim (k) must be >= 1, got {subspace_dim}")

    keys = sorted(aggregatable_keys)
    if not keys:
        raise ValueError("subspace_dp_aggregate requires at least one aggregatable key.")

    normalized = normalize_updates(results)
    n = len(normalized)
    if public_cohort is not None and n != public_cohort:
        raise ValueError(
            f"received {n} updates but public_cohort is {public_cohort}; cohort size mismatch alters sensitivity"
        )

    # Ambient parameter dimension d
    d = sum(global_params[k].numel() for k in keys)

    # Fall back to standard DP if subspace_dim >= ambient dim (no dimension reduction possible)
    if subspace_dim >= d:
        return dp_aggregate(
            results=results,
            global_params=global_params,
            aggregatable_keys=keys,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            generator=generator,
            adjacency=adjacency,
            public_cohort=public_cohort,
        )

    projector = SubspaceProjector(
        in_dim=d,
        subspace_dim=subspace_dim,
        base_seed=subspace_seed,
        rotating=rotating_subspace,
    )

    # Reference parameters in CPU float32
    ref: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys:
        if k not in global_params:
            raise KeyError(f"aggregatable key {k!r} missing from global_params.")
        ref[k] = global_params[k].detach().float().cpu()

    # Project and clip each client's delta in R^k
    clipped_z = []
    for _cid, params, _num_examples in normalized:
        delta = OrderedDict((k, params[k].detach().float().cpu() - ref[k]) for k in keys)
        # Verify finite
        for k in keys:
            if not torch.isfinite(delta[k]).all():
                raise ValueError("Non-finite client update delta encountered in subspace_dp_aggregate.")
        z_i = projector.project_dict(delta, keys, server_round=server_round)
        if not torch.isfinite(z_i).all():
            raise ValueError("Non-finite projected coordinates encountered in subspace_dp_aggregate.")
        z_norm = float(torch.linalg.vector_norm(z_i.double()).item())
        if math.isnan(z_norm) or math.isinf(z_norm):
            raise ValueError("Non-finite projected norm encountered in subspace_dp_aggregate.")
        scale = min(1.0, float(clip_norm) / (z_norm + 1e-12))
        clipped_z.append(z_i * scale)

    # Uniform average in R^k
    sum_z = torch.zeros(subspace_dim, dtype=torch.float32)
    for z in clipped_z:
        sum_z = sum_z + z
    mean_z = sum_z / float(n)

    # Add calibrated Gaussian DP noise in R^k: sensitivity is c * C / n
    std = ADJACENCY_SENSITIVITY_FACTOR[adjacency] * float(noise_multiplier) * float(clip_norm) / float(n)
    if std > 0.0:
        if generator is None:
            generator = torch.Generator()
            generator.seed()
        noise = torch.empty(subspace_dim, dtype=torch.float32)
        noise.normal_(mean=0.0, std=std, generator=generator)
        noisy_z = mean_z + noise
    else:
        noisy_z = mean_z

    # Reconstruct update delta in R^d: Delta_global = P_t * noisy_z
    recon_delta = projector.reconstruct_dict(noisy_z, ref, keys, server_round=server_round)

    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys:
        out[k] = ref[k] + recon_delta[k]
        if "running_var" in k or k.endswith(".var"):
            out[k] = torch.clamp(out[k], min=1e-6)

    return out
