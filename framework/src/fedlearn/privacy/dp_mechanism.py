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
4. Add Gaussian noise ``N(0, (c*z*S/N)^2)`` per coordinate on the aggregatable keys. ``z`` is the
   noise multiplier; ``z=0`` is the noiseless clip+uniform-average sanity path. ``c`` is the
   adjacency factor: 1 for add/remove (one client's delta vs. none, over a fixed public
   denominator ``N``), 2 for replace-one (one client's delta vs. any other clipped delta). The
   accountant reads ``z`` as noise std / sensitivity under the chosen relation, so the same ``z``
   carries the same epsilon, for a different neighbour definition.
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
from fedlearn.server.robust_aggregation import clip_l2_norm, trimmed_mean


# Rules whose replace-one L2 sensitivity shrinks with the cohort. The coordinate-wise median is
# deliberately absent: one client can move it by as much as S at any n, so it has no 1/n bound.
_REPLACE_ONE_RULES = ("mean", "trimmed_mean")

# Neighbour relations dp_aggregate calibrates for, and the sensitivity of its clipped mean under each,
# in units of S/N. Add/remove holds only over a fixed public denominator: dividing by a cohort size that
# depends on who took part makes one removal move the mean by up to 2S/N (tests/test_dp_adjacency.py).
ADJACENCY_SENSITIVITY_FACTOR = {"add_remove": 1.0, "replace_one": 2.0}


def replace_one_sensitivity(
    rule: str, num_clients: int, clip_norm: float, trim_ratio: float = 0.0
) -> float:
    """L2 sensitivity of a clipped aggregate when ONE client's update is replaced by another.

    Each client's delta is jointly clipped to L2 norm ``S``, so a replaced delta moves by at most
    ``2S``. The uniform mean therefore moves by at most ``2S/n``. A coordinate-wise trimmed mean
    that drops ``k = floor(trim_ratio * n)`` values from each end (the same ``k`` as
    :func:`fedlearn.server.robust_aggregation.trimmed_mean`) moves by at most ``2S/(n - 2k)``:
    every coordinate of the trimmed SUM moves by at most the change in the replaced client's value
    at that coordinate, so the joint L2 change is at most ``||x - x'||_2 / (n - 2k)``.

    This is replace-one adjacency, which is NOT the default add/remove convention of :func:`dp_aggregate`
    (sensitivity ``S/n``); the two must not be mixed within one comparison.

    Raises:
        ValueError: on an unsupported rule, ``num_clients < 1``, non-positive ``clip_norm``,
            negative ``trim_ratio``, or a trim that removes every client.
    """
    if rule not in _REPLACE_ONE_RULES:
        raise ValueError(f"rule must be one of {_REPLACE_ONE_RULES}, got {rule!r}")
    if num_clients is None or num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if clip_norm is None or clip_norm <= 0:
        raise ValueError(f"clip_norm (S) must be positive, got {clip_norm}")
    if rule == "mean":
        return 2.0 * float(clip_norm) / num_clients
    if trim_ratio is None or trim_ratio < 0:
        raise ValueError(f"trim_ratio must be >= 0, got {trim_ratio}")
    k = int(trim_ratio * num_clients)  # floor, matching robust_aggregation.trimmed_mean
    kept = num_clients - 2 * k
    if kept <= 0:
        raise ValueError(
            f"trim_ratio={trim_ratio} removes every client (n={num_clients}, k={k})."
        )
    return 2.0 * float(clip_norm) / kept


def _clipped_deltas(results, global_params, keys, clip_norm):
    """Per-client deltas from ``global_params`` over ``keys``, each JOINTLY clipped to L2 norm ``S``.

    Returns ``(ref, clipped)``: the float32 CPU delta reference and one clipped ``OrderedDict`` per
    client, in arrival order. ``num_examples`` is ignored — every DP rule here is unweighted.
    """
    # Delta reference restricted to the aggregatable keys, as float32 CPU.
    ref: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys:
        if k not in global_params:
            raise KeyError(f"aggregatable key {k!r} missing from global_params.")
        ref[k] = global_params[k].detach().float().cpu()

    clipped_deltas = []
    for _client_id, params, _num_examples in normalize_updates(results):
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
        clipped_deltas.append(clipped)
    return ref, clipped_deltas


# Public name for the shared clip step (ComposedStrategy reuses it so its clipping is the same code
# as dp_robust_aggregate's). An alias rather than a rename: dp_aggregate and dp_robust_aggregate bind
# a local named ``clipped_deltas``, which would shadow a function of that name.
clipped_deltas = _clipped_deltas


def dp_aggregate(
    results: Sequence[tuple],
    global_params: "OrderedDict[str, torch.Tensor]",
    aggregatable_keys: Iterable[str],
    clip_norm: float,
    noise_multiplier: float,
    generator: Optional[torch.Generator] = None,
    adjacency: str = "add_remove",
    public_cohort: Optional[int] = None,
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
        noise_multiplier: ``z`` (>= 0). Per-coordinate Gaussian std is ``c*z*S/N`` with ``c`` from
            :data:`ADJACENCY_SENSITIVITY_FACTOR`; ``z=0`` => no noise (deterministic clip + uniform
            average).
        generator: optional :class:`torch.Generator` for reproducible noise; ``None`` => fresh
            entropy.
        adjacency: ``"add_remove"`` (default) or ``"replace_one"``: the neighbour relation the
            accounted epsilon refers to.
        public_cohort: the fixed, public cohort size the denominator must equal. When given, a call
            with any other number of updates is refused, since the mean's sensitivity would no
            longer be the calibrated one. ``None`` skips the check (offline callers that fix ``N``
            themselves).

    Returns:
        ``OrderedDict`` ``{k: global_params[k] + mean_delta[k]}`` for ``k`` in ``aggregatable_keys``
        (in that order), as float32 CPU tensors. Frozen keys are the caller's to re-attach.

    Raises:
        ValueError: on empty ``results``, non-positive ``clip_norm``, negative ``noise_multiplier``,
            empty ``aggregatable_keys``, an unknown ``adjacency``, or a cohort that differs from
            ``public_cohort``.
        KeyError: if a key is missing from ``global_params`` or from a client's params.
    """
    if adjacency not in ADJACENCY_SENSITIVITY_FACTOR:
        raise ValueError(f"adjacency must be one of {sorted(ADJACENCY_SENSITIVITY_FACTOR)}, got {adjacency!r}")
    if not results:
        raise ValueError("dp_aggregate requires at least one client update.")
    if clip_norm is None or clip_norm <= 0:
        raise ValueError(f"clip_norm (S) must be positive, got {clip_norm}")
    if noise_multiplier is None or noise_multiplier < 0:
        raise ValueError(f"noise_multiplier (z) must be >= 0, got {noise_multiplier}")

    keys = list(aggregatable_keys)
    if not keys:
        raise ValueError("dp_aggregate requires at least one aggregatable key.")

    ref, clipped_deltas = _clipped_deltas(results, global_params, keys, clip_norm)
    n = len(clipped_deltas)
    if public_cohort is not None and n != public_cohort:
        raise ValueError(
            f"dp_aggregate received {n} client updates but the public cohort is {public_cohort}: the "
            f"noise is calibrated for a fixed denominator, so releasing on a different cohort size "
            f"would understate epsilon."
        )

    # Sum of per-client CLIPPED deltas over the aggregatable keys.
    sum_delta: "OrderedDict[str, torch.Tensor]" = OrderedDict(
        (k, torch.zeros_like(ref[k])) for k in keys
    )
    for clipped in clipped_deltas:
        for k in keys:
            sum_delta[k] = sum_delta[k] + clipped[k]

    # Uniform average (1/N) + Gaussian noise N(0, (c*z*S/N)^2) per coordinate on the aggregatable keys.
    std = ADJACENCY_SENSITIVITY_FACTOR[adjacency] * float(noise_multiplier) * float(clip_norm) / n
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys:
        mean_delta = sum_delta[k] / n
        if std > 0.0:
            noise = torch.empty_like(mean_delta)
            noise.normal_(mean=0.0, std=std, generator=generator)
            mean_delta = mean_delta + noise
        out[k] = ref[k] + mean_delta
        if "running_var" in k or k.endswith(".var"):
            out[k] = torch.clamp(out[k], min=1e-6)
    return out


def dp_robust_aggregate(
    results: Sequence[tuple],
    global_params: "OrderedDict[str, torch.Tensor]",
    aggregatable_keys: Iterable[str],
    clip_norm: float,
    noise_multiplier: float,
    rule: str = "mean",
    trim_ratio: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> "OrderedDict[str, torch.Tensor]":
    """Central DP over a clipped mean or coordinate-wise trimmed mean, under replace-one adjacency.

    Clips each client's delta jointly to L2 norm ``S``, aggregates with ``rule`` (``"mean"`` or
    ``"trimmed_mean"``, unweighted), then adds Gaussian noise with per-coordinate std
    ``z * replace_one_sensitivity(rule, n, S, trim_ratio)``. Using one adjacency for both rules keeps
    a mean-vs-trimmed-mean comparison matched at the same ``z``; the accountant's ``z`` is
    sensitivity-normalized, so it carries over unchanged.

    Args mirror :func:`dp_aggregate`, plus ``rule`` and ``trim_ratio`` (the trimmed mean drops
    ``floor(trim_ratio * n)`` values from each end of every coordinate).

    Returns:
        ``OrderedDict`` ``{k: global_params[k] + aggregate[k]}`` for ``k`` in ``aggregatable_keys``
        (in that order), as float32 CPU tensors. Frozen keys are the caller's to re-attach.

    Raises:
        ValueError: on empty ``results``, negative ``noise_multiplier``, empty ``aggregatable_keys``,
            a non-finite client delta, or any configuration :func:`replace_one_sensitivity` rejects.
        KeyError: if a key is missing from ``global_params`` or from a client's params.
    """
    if not results:
        raise ValueError("dp_robust_aggregate requires at least one client update.")
    if noise_multiplier is None or noise_multiplier < 0:
        raise ValueError(f"noise_multiplier (z) must be >= 0, got {noise_multiplier}")
    keys = list(aggregatable_keys)
    if not keys:
        raise ValueError("dp_robust_aggregate requires at least one aggregatable key.")

    sensitivity = replace_one_sensitivity(rule, len(results), clip_norm, trim_ratio)
    ref, clipped_deltas = _clipped_deltas(results, global_params, keys, clip_norm)

    std = float(noise_multiplier) * sensitivity
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k in keys:
        stacked = torch.stack([c[k] for c in clipped_deltas], dim=0)
        agg = stacked.mean(dim=0) if rule == "mean" else trimmed_mean(stacked, trim_ratio)
        if std > 0.0:
            noise = torch.empty_like(agg)
            noise.normal_(mean=0.0, std=std, generator=generator)
            agg = agg + noise
        out[k] = ref[k] + agg
        if "running_var" in k or k.endswith(".var"):
            out[k] = torch.clamp(out[k], min=1e-6)
    return out
