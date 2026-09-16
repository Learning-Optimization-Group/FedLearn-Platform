"""FR-13 — central differential privacy on the FFA-LoRA adapter delta.

An honest-but-curious server adds Gaussian noise at aggregation time to the released global
adapter, giving client-level (user-level) (epsilon, delta)-DP: one client = one privacy unit.

The mechanism operates over the *aggregatable* adapter keys ONLY — the adapter ``B`` and the
classification head, i.e. every client key that is NOT a frozen ``lora_A`` key. The frozen ``A``
is never touched here (the caller re-attaches it bit-identically), which is precisely what keeps
the FFA invariant ``avg(B) @ A == avg(B @ A)`` exact under DP: noise and clipping change only the
released ``B`` + head dimensions, and the shared ``A`` factors out unchanged.

Steps (mirrors ``RobustAggregator``'s delta pattern, ``robust_aggregation.py:261-282``):

1. ``delta_i = client[k] - global[k]`` over the aggregatable keys.
2. Clip ``delta_i`` JOINTLY to L2 norm ``S`` (global-norm convention, :func:`clip_l2_norm`). One
   client's sensitivity to the summed delta is then exactly ``S``.
3. **Uniform** average (weight ``1/N``), NOT num_examples-weighted — weighting by an
   attacker-reported example count would inflate that client's sensitivity above ``S`` and void
   the epsilon claim. The DP path deliberately drops the weighting (DP-path-only).
4. Add Gaussian noise ``N(0, (z*S/N)^2)`` per coordinate on the aggregatable keys. ``z`` is the
   noise multiplier; ``z=0`` is the noiseless clip+uniform-average sanity path.
5. Return ``{k: global[k] + mean_delta[k]}`` for the aggregatable keys; the caller re-attaches
   the frozen keys.

The RNG is a caller-supplied :class:`torch.Generator` so empirical-noise / reproducibility tests
are deterministic; ``None`` draws fresh entropy (the production default).

All arithmetic is done in float32 on CPU: this is a small, server-side reduction over adapter
deltas, and a CPU generator keeps the noise reproducible regardless of the aggregator's device.
"""

from collections import OrderedDict
from typing import Iterable, Optional, Sequence

import torch

# Reuse the exact same global-L2-norm clip the robust-aggregation path (FR-12) uses, so DP clipping
# and Byzantine clipping share one definition of "an update's norm". Imported at module top; the
# strategy imports THIS module lazily (inside the DP branch) to avoid an import cycle.
from fedlearn.server._update_normalize import normalize_updates
from fedlearn.server.robust_aggregation import clip_l2_norm


def dp_aggregate(
    results: Sequence[tuple],
    global_params: "OrderedDict[str, torch.Tensor]",
    aggregatable_keys: Iterable[str],
    clip_norm: float,
    noise_multiplier: float,
    generator: Optional[torch.Generator] = None,
) -> "OrderedDict[str, torch.Tensor]":
    """Central-DP aggregation of adapter deltas over the aggregatable keys only.

    Args:
        results: client updates, each ``(client_id, params, num_examples)`` or ``(params,
            num_examples)``; ``params`` is an ``OrderedDict[str, Tensor]`` or a JSON string
            decoding to a dict of lists (mirrors ``FedAvgAggregator``'s accepted wire shapes).
            ``num_examples`` is intentionally ignored — the DP average is uniform.
        global_params: the current running global (delta reference). Must contain every key in
            ``aggregatable_keys``.
        aggregatable_keys: the client keys to privatise (adapter ``B`` + head) — every client key
            that is NOT a frozen ``lora_A`` key. Clipping and noise touch ONLY these keys.
        clip_norm: the L2 clip bound ``S`` (> 0), applied jointly to each client's delta.
        noise_multiplier: ``z`` (>= 0). Per-coordinate Gaussian std is ``z*S/N``; ``z=0`` => no
            noise (deterministic clip + uniform average).
        generator: optional :class:`torch.Generator` for reproducible noise; ``None`` => fresh
            entropy.

    Returns:
        ``OrderedDict`` ``{k: global_params[k] + mean_delta[k]}`` for ``k`` in ``aggregatable_keys``
        (in that order), as float32 CPU tensors. Frozen keys are the caller's to re-attach.

    Raises:
        ValueError: on empty ``results``, non-positive ``clip_norm``, negative ``noise_multiplier``,
            or empty ``aggregatable_keys``.
        KeyError: if a key is missing from ``global_params`` or from a client's params.
    """
    if not results:
        raise ValueError("dp_aggregate requires at least one client update.")
    if clip_norm is None or clip_norm <= 0:
        raise ValueError(f"clip_norm (S) must be positive, got {clip_norm}")
    if noise_multiplier is None or noise_multiplier < 0:
        raise ValueError(f"noise_multiplier (z) must be >= 0, got {noise_multiplier}")

    keys = list(aggregatable_keys)
    if not keys:
        raise ValueError("dp_aggregate requires at least one aggregatable key.")

    # Delta reference restricted to the aggregatable keys, as float32 CPU.
    ref: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys:
        if k not in global_params:
            raise KeyError(f"aggregatable key {k!r} missing from global_params.")
        ref[k] = global_params[k].detach().float().cpu()

    # The DP average is uniform, so num_examples is intentionally ignored (the third element).
    normalized = normalize_updates(results)
    n = len(normalized)

    # Sum of per-client CLIPPED deltas over the aggregatable keys.
    sum_delta: "OrderedDict[str, torch.Tensor]" = OrderedDict(
        (k, torch.zeros_like(ref[k])) for k in keys
    )
    for _client_id, params, _num_examples in normalized:
        delta: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for k in keys:
            if k not in params:
                raise KeyError(f"client update missing aggregatable key {k!r}.")
            delta[k] = params[k].detach().float().cpu() - ref[k]
            # A non-finite (NaN/Inf) coordinate makes the L2 norm non-finite, so the clip becomes a
            # no-op (scale = min(1.0, S/NaN) = 1.0) or produces NaN (Inf*0), silently defeating the
            # sensitivity bound the (eps, delta) guarantee rests on and corrupting the whole
            # aggregated coordinate for every client. Reject it loudly, like the other contract checks.
            if not torch.isfinite(delta[k]).all():
                raise ValueError(
                    f"client {_client_id!r} sent a non-finite value in aggregatable key {k!r} "
                    "(NaN/Inf); a non-finite update defeats the DP L2 clip and would corrupt the "
                    "aggregate — rejecting the round."
                )
        # Joint L2 clip of the whole delta to S (sensitivity of one client to the sum is then S).
        clipped, _orig_norm = clip_l2_norm(delta, clip_norm)
        for k in keys:
            sum_delta[k] = sum_delta[k] + clipped[k]

    # Uniform average (1/N) + Gaussian noise N(0, (z*S/N)^2) per coordinate on the aggregatable keys.
    std = float(noise_multiplier) * float(clip_norm) / n
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys:
        mean_delta = sum_delta[k] / n
        if std > 0.0:
            noise = torch.empty_like(mean_delta)
            noise.normal_(mean=0.0, std=std, generator=generator)
            mean_delta = mean_delta + noise
        out[k] = ref[k] + mean_delta
    return out
