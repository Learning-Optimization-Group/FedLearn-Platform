"""Differentially Private Adaptive Clipping (Andrew et al., NeurIPS 2021).

Implements private quantile estimation for dynamic calibration of the clipping threshold C:
- Mitigates update distortion without manual threshold tuning.
- Estimates the empirical fraction of clipped client updates using a Gaussian indicator mechanism
  with sensitivity Delta = 1.
- Multiplicatively updates C via geometric gradient descent on the quantile loss.

Reference:
    Andrew, G., Thakkar, O., McMahan, H. B., & Ramaswamy, S. (2021).
    Differentially Private Learning with Adaptive Clipping.
    NeurIPS 2021. https://arxiv.org/abs/1905.03871
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from fedlearn.server.robust_aggregation import clip_l2_norm


class DPAdaptiveClipping:
    """Stateful differentially private adaptive clipping coordinator.

    Args:
        initial_clip_norm: Initial clipping bound C_0 > 0.
        target_quantile: Fraction of updates that should fall below the clipping bound (e.g. 0.5 or 0.8).
        clip_learning_rate: Multiplicative step size eta > 0.
        noise_multiplier: Noise multiplier z_b added to the indicator sum (std = z_b * 1.0).
        min_clip_norm: Lower bound clamp on C to prevent collapse to zero.
        max_clip_norm: Optional upper bound clamp on C.
        seed: Optional RNG seed for deterministic testing. Omit in production for fresh OS entropy.
    """

    def __init__(
        self,
        initial_clip_norm: float = 1.0,
        target_quantile: float = 0.5,
        clip_learning_rate: float = 0.2,
        noise_multiplier: float = 0.5,
        min_clip_norm: float = 1e-4,
        max_clip_norm: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        if initial_clip_norm <= 0:
            raise ValueError(f"initial_clip_norm must be positive, got {initial_clip_norm}")
        if not (0.0 < target_quantile < 1.0):
            raise ValueError(f"target_quantile must be in (0, 1), got {target_quantile}")
        if clip_learning_rate <= 0:
            raise ValueError(f"clip_learning_rate must be positive, got {clip_learning_rate}")
        if noise_multiplier < 0:
            raise ValueError(f"noise_multiplier must be >= 0, got {noise_multiplier}")
        if min_clip_norm <= 0:
            raise ValueError(f"min_clip_norm must be positive, got {min_clip_norm}")
        if max_clip_norm is not None and max_clip_norm < min_clip_norm:
            raise ValueError(f"max_clip_norm ({max_clip_norm}) cannot be less than min_clip_norm ({min_clip_norm})")

        self.clip_norm = float(initial_clip_norm)
        self.target_quantile = float(target_quantile)
        self.lr = float(clip_learning_rate)
        self.z_b = float(noise_multiplier)
        self.min_clip_norm = float(min_clip_norm)
        self.max_clip_norm = float(max_clip_norm) if max_clip_norm is not None else None

        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(int(seed))
        else:
            self._generator.seed()

        self.last_stats: Dict[str, float] = {}

    def step(self, client_norms: Sequence[float], cohort_size: Optional[int] = None) -> float:
        """Update the clipping norm C using the noisy indicator mechanism.

        Args:
            client_norms: Sequence of unclipped L2 norms ||Delta_i||_2 for participants in this round.
            cohort_size: Optional total expected cohort size N. If None, len(client_norms) is used.

        Returns:
            The updated clipping bound C_{t+1}.
        """
        n = len(client_norms)
        if n == 0:
            raise ValueError("client_norms must contain at least one value.")
        denom = float(cohort_size if cohort_size is not None and cohort_size > 0 else n)

        # Count clients whose norm is strictly within budget: b_i = 1[||Delta_i|| <= C]
        unclipped_count = sum(1.0 for x in client_norms if x <= self.clip_norm)
        raw_fraction = unclipped_count / n

        # Add zero-mean Gaussian noise to count: Delta = 1.0 sensitivity
        noise_std = self.z_b * 1.0
        if noise_std > 0.0:
            noise = float(torch.empty(1).normal_(mean=0.0, std=noise_std, generator=self._generator).item())
        else:
            noise = 0.0

        noisy_fraction = (unclipped_count + noise) / denom

        # Geometric gradient update: C_{t+1} = C_t * exp(-lr * (noisy_fraction - target_quantile))
        # If noisy_fraction < target (too many clipped), factor > 1 -> increases C.
        # If noisy_fraction > target (too few clipped), factor < 1 -> decreases C.
        factor = math.exp(-self.lr * (noisy_fraction - self.target_quantile))
        new_clip = self.clip_norm * factor

        # Clamp bounds
        new_clip = max(new_clip, self.min_clip_norm)
        if self.max_clip_norm is not None:
            new_clip = min(new_clip, self.max_clip_norm)

        self.last_stats = {
            "applied_clip_norm": self.clip_norm,
            "next_clip_norm": new_clip,
            "raw_unclipped_fraction": raw_fraction,
            "noisy_unclipped_fraction": noisy_fraction,
            "indicator_noise_std": noise_std,
        }
        self.clip_norm = new_clip
        return self.clip_norm

    def clip_deltas(
        self, deltas: Sequence["OrderedDict[str, torch.Tensor]"]
    ) -> Tuple[List["OrderedDict[str, torch.Tensor]"], List[float]]:
        """Clip all updates to the current clip_norm C and return (clipped_deltas, orig_norms)."""
        clipped = []
        norms = []
        for d in deltas:
            c, orig = clip_l2_norm(d, self.clip_norm)
            clipped.append(c)
            norms.append(orig)
        return clipped, norms
