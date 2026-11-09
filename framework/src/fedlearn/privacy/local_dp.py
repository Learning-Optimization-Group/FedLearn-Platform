"""Local differential privacy: each honest client privatises its own update before upload.

The client clips its update (delta from the global) jointly to L2 norm C and adds Gaussian noise
of per-coordinate std ``z * 2C``: replacing the client's data can move its clipped update by at
most 2C, so this is the Gaussian mechanism at sensitivity 2C. ``z`` comes from the RDP accountant
for the target (epsilon, delta) over the rounds the client can participate in.
"""
from collections import OrderedDict
from typing import Optional

import torch

from fedlearn.server.robust_aggregation import clip_l2_norm


def privatize(delta: "OrderedDict[str, torch.Tensor]", clip_norm: float, noise_multiplier: float,
              generator: Optional[torch.Generator] = None) -> "OrderedDict[str, torch.Tensor]":
    if clip_norm is None or clip_norm <= 0:
        raise ValueError(f"clip_norm must be positive, got {clip_norm}")
    if noise_multiplier is None or noise_multiplier < 0:
        raise ValueError(f"noise_multiplier must be >= 0, got {noise_multiplier}")
    clipped, _ = clip_l2_norm(OrderedDict((k, v.detach().float().cpu()) for k, v in delta.items()), clip_norm)
    std = float(noise_multiplier) * 2.0 * float(clip_norm)
    if generator is None:
        generator = torch.Generator()
        generator.seed()
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in clipped.items():
        if std > 0.0:
            noise = torch.empty_like(v)
            noise.normal_(mean=0.0, std=std, generator=generator)
            v = v + noise
        out[k] = v
    return out
