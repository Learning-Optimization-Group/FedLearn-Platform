"""Composable server pipeline: validate -> clip -> rule -> central noise -> server optimizer.

One Strategy instead of a separate class per combination. Updates are aggregated as DELTAS from
the current global, which is how ``dp_robust_aggregate`` works; for the mean and trimmed mean the
staged code below performs the same per-key operations in the same order and draws noise from one
generator persisting across rounds, so it is bit-identical to ``dp_robust_aggregate``. The other
rules are translation-equivariant, so delta aggregation equals ``RobustAggregator``'s absolute
aggregation in exact arithmetic (float32-close, not bit-identical).

The server optimizer reuses ``FedOpt`` unchanged by handing it the pipeline's output as a single
client of weight 1, which its weighted mean returns exactly.
"""
import logging
import math
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch

from fedlearn.privacy.dp_mechanism import clipped_deltas, replace_one_sensitivity
from fedlearn.server._update_normalize import normalize_updates
from fedlearn.server.robust_aggregation import (
    bulyan_aggregate, centered_clip, coordinate_wise_median, krum_select, multi_krum_select, trimmed_mean,
)
from fedlearn.server.strategy import FedOpt, Strategy

log = logging.getLogger(__name__)

RULES = ("mean", "weighted_mean", "trimmed_mean", "median", "krum", "multi_krum", "bulyan",
         "centered_clip")
CENTRAL_DP_RULES = ("mean", "trimmed_mean")
_MAX_SAMPLES = 100_000  # same cap as FedAvgAggregator
_RULE_DEFAULTS = {"trim_ratio": 0.2, "byzantine_fraction": 0.0, "tau": 1.0, "iterations": 1,
                  "multi_krum_m": None}


def reduce_deltas(deltas: List["OrderedDict[str, torch.Tensor]"], weights: List[int], keys: List[str],
                  rule: str, rule_params: Dict) -> Tuple["OrderedDict[str, torch.Tensor]", Optional[List[float]]]:
    """Reduce per-client deltas with ``rule``; returns the reduced delta and, for the trimmed mean,
    Krum and Multi-Krum, each client's share of coordinates kept (``None`` for other rules)."""
    n = len(deltas)
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    kept: Optional[List[float]] = None
    if rule in ("mean", "trimmed_mean", "median", "weighted_mean"):
        k = int(rule_params["trim_ratio"] * n) if rule == "trimmed_mean" else 0
        counts = [0] * n
        total = 0
        w = [min(int(x), _MAX_SAMPLES) for x in weights]
        for key in keys:
            stacked = torch.stack([d[key] for d in deltas], dim=0)
            if rule == "mean":
                out[key] = stacked.mean(dim=0)
            elif rule == "median":
                out[key] = coordinate_wise_median(stacked)
            elif rule == "weighted_mean":
                acc = torch.zeros_like(stacked[0])
                for i in range(n):
                    torch.add(acc, stacked[i], alpha=w[i] / sum(w), out=acc)
                out[key] = acc
            else:
                out[key] = trimmed_mean(stacked, rule_params["trim_ratio"])
                order = torch.argsort(stacked.reshape(n, -1), dim=0)[k:n - k]
                for i, c in enumerate(torch.bincount(order.reshape(-1), minlength=n).tolist()):
                    counts[i] += c
                total += stacked[0].numel()
        if rule == "trimmed_mean":
            kept = [c / total for c in counts]
        return out, kept
    flat = torch.stack([torch.cat([d[key].reshape(-1) for key in keys]) for d in deltas], dim=0)
    f = int(float(rule_params["byzantine_fraction"] or 0.0) * n)
    if rule == "krum":
        sel = krum_select(flat, f)
        reduced, kept = flat[sel], [1.0 if i == sel else 0.0 for i in range(n)]
    elif rule == "multi_krum":
        m = rule_params["multi_krum_m"] or max(1, n - 2 * f)
        chosen = multi_krum_select(flat, f, min(m, n))
        reduced, kept = flat[chosen].mean(dim=0), [1.0 if i in chosen else 0.0 for i in range(n)]
    elif rule == "bulyan":
        reduced = bulyan_aggregate(flat, f)
    else:  # centered_clip around the current global, i.e. the zero delta
        reduced = centered_clip(flat, torch.zeros_like(flat[0]), tau=rule_params["tau"],
                                iterations=int(rule_params["iterations"]))
    offset = 0
    for key in keys:
        numel = deltas[0][key].numel()
        out[key] = reduced[offset:offset + numel].reshape(deltas[0][key].shape).clone()
        offset += numel
    return out, kept


def ks_statistic(values: "torch.Tensor", std: float) -> float:
    """Kolmogorov-Smirnov distance between the empirical distribution of ``values`` and N(0, std^2)."""
    x, _ = torch.sort(values.double().reshape(-1))
    n = x.numel()
    cdf = 0.5 * torch.erfc(-x / (std * math.sqrt(2.0)))
    i = torch.arange(1, n + 1, dtype=torch.float64)
    return float(torch.maximum(i / n - cdf, cdf - (i - 1) / n).max())


def ks_critical(n: int, alpha: float) -> float:
    """Asymptotic critical KS distance at level ``alpha`` for ``n`` values: sqrt(-ln(alpha/2) / 2) / sqrt(n)."""
    return math.sqrt(-math.log(alpha / 2.0) / 2.0) / math.sqrt(n)


class ComposedStrategy(Strategy):
    def __init__(self, initial_parameters, rule: str = "mean", clip_norm: Optional[float] = None,
                 rule_params: Optional[Dict] = None, central_noise_multiplier: float = 0.0,
                 noise_seed: int = 0, optimizer: str = "none", optimizer_lr: float = 0.01,
                 evaluate_fn: Optional[Callable] = None, min_fit_clients: int = 1,
                 clients_per_round: Optional[int] = None,
                 norm_band: Optional[Tuple[float, float]] = None,
                 ks_reference_std: Optional[float] = None, ks_alpha: float = 1e-4):
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
        if clip_norm is not None and clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive or None, got {clip_norm}")
        if central_noise_multiplier < 0:
            raise ValueError("central_noise_multiplier must be >= 0")
        if central_noise_multiplier > 0:
            if rule not in CENTRAL_DP_RULES:
                raise ValueError(f"central DP needs a rule with a replace-one bound {CENTRAL_DP_RULES}, got {rule!r}")
            if clip_norm is None:
                raise ValueError("central DP needs a clip_norm")
        if norm_band is not None and not (0.0 <= float(norm_band[0]) < float(norm_band[1])):
            raise ValueError(f"norm_band must be (low, high) with 0 <= low < high, got {norm_band}")
        if optimizer not in ("none", "adam", "yogi"):
            raise ValueError(f"optimizer must be 'none', 'adam' or 'yogi', got {optimizer!r}")
        self.initial_parameters = initial_parameters
        self.rule = rule
        self.clip_norm = None if clip_norm is None else float(clip_norm)
        self.rule_params = {**_RULE_DEFAULTS, **(rule_params or {})}
        if self.rule_params["byzantine_fraction"] is None:
            self.rule_params["byzantine_fraction"] = 0.0
        self.z = float(central_noise_multiplier)
        # Noise-aware filtering (Xiang et al., SIGMOD'23): an upload whose delta norm falls outside the band
        # an honest privatized update occupies is dropped before clipping. It reads only privatized uploads,
        # so it is post-processing and costs no privacy.
        self.norm_band = None if norm_band is None else (float(norm_band[0]), float(norm_band[1]))
        # With ks_reference_std, an in-band upload is also dropped when its delta coordinates are not
        # plausibly N(0, s^2): a noise-free update rescaled to the band's norm passes the norm test but not this.
        if ks_reference_std is not None and not (float(ks_reference_std) > 0 and 0 < float(ks_alpha) < 1):
            raise ValueError("ks_reference_std must be > 0 and ks_alpha in (0, 1)")
        self.ks_reference_std = None if ks_reference_std is None else float(ks_reference_std)
        self.ks_alpha = float(ks_alpha)
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round if clients_per_round is not None else min_fit_clients
        self._generator = torch.Generator().manual_seed(int(noise_seed))
        self._global = OrderedDict((k, v.detach().float().cpu().clone()) for k, v in initial_parameters.items())
        self._fedopt = (FedOpt(initial_parameters, server_learning_rate=optimizer_lr, variant=optimizer)
                        if optimizer != "none" else None)
        self.last_round_stats: Dict = {}

    def initialize_parameters(self):
        return self.initial_parameters

    def _valid(self, params) -> bool:
        if set(params.keys()) != set(self._global.keys()):
            return False
        for k, ref in self._global.items():
            v = params[k]
            if tuple(v.shape) != tuple(ref.shape) or not torch.isfinite(v.float()).all():
                return False
        return True

    def _pipeline(self, server_round, results, *, apply_noise: bool):
        keys = list(self._global.keys())
        normalized = normalize_updates(results)
        ok, dropped = [], []
        for cid, params, n in normalized:
            (ok if n > 0 and self._valid(params) else dropped).append((cid, params, n))
        filtered = []
        if self.norm_band is not None:
            lo, hi = self.norm_band
            keep = []
            for cid, params, n in ok:
                norm = math.sqrt(sum(float(((params[k].detach().float().cpu() - self._global[k]) ** 2).sum())
                                     for k in self._global))
                in_band = lo <= norm <= hi
                if in_band and self.ks_reference_std is not None:
                    delta = torch.cat([(params[k].detach().float().cpu() - self._global[k]).reshape(-1)
                                       for k in self._global])
                    in_band = ks_statistic(delta, self.ks_reference_std) <= ks_critical(delta.numel(), self.ks_alpha)
                (keep if in_band else filtered).append((cid, params, n))
            ok = keep
        stats = {"aggregated": len(ok), "dropped": [str(c) for c, _, _ in dropped], "failed": False,
                 "filtered": [str(c) for c, _, _ in filtered],
                 "message": None, "timings": {"clip": 0.0, "rule": 0.0, "noise": 0.0, "optimizer": 0.0}}
        if not ok:
            stats.update(failed=True, message=f"round {server_round}: every client update was invalid or filtered")
            return None, stats
        t = time.perf_counter()
        raw = [OrderedDict((k, p[k].detach().float().cpu() - self._global[k]) for k in keys) for _, p, _ in ok]
        norms = [math.sqrt(sum(float((d[k] * d[k]).sum()) for k in keys)) for d in raw]
        if self.clip_norm is not None:
            _, deltas = clipped_deltas(ok, self._global, keys, self.clip_norm)
            shrink = [min(1.0, self.clip_norm / (x + 1e-12)) for x in norms]
        else:
            deltas, shrink = raw, [1.0] * len(raw)
        stats["timings"]["clip"] = time.perf_counter() - t
        t = time.perf_counter()
        try:
            reduced, kept = reduce_deltas(deltas, [n for _, _, n in ok], keys, self.rule, self.rule_params)
        except ValueError as exc:  # the rule's own precondition, e.g. Krum's n >= 2f + 3
            stats.update(failed=True, message=f"round {server_round}: rule {self.rule!r} refused: {exc}")
            return None, stats
        stats["timings"]["rule"] = time.perf_counter() - t
        stats["aggregate_delta_norm"] = math.sqrt(sum(float((reduced[k] ** 2).sum()) for k in keys))
        sens = (replace_one_sensitivity(self.rule, len(ok), self.clip_norm, self.rule_params["trim_ratio"])
                if self.rule in CENTRAL_DP_RULES and self.clip_norm is not None else None)
        std = self.z * sens if (sens is not None and self.z > 0) else 0.0
        t = time.perf_counter()
        if apply_noise and std > 0.0:
            for k in keys:
                noise = torch.empty_like(reduced[k])
                noise.normal_(mean=0.0, std=std, generator=self._generator)
                reduced[k] = reduced[k] + noise
        stats["timings"]["noise"] = time.perf_counter() - t
        out = OrderedDict((k, self._global[k] + reduced[k]) for k in keys)
        stats.update(
            orig_norms={str(c): x for (c, _, _), x in zip(ok, norms)},
            clip_rate=sum(1 for s in shrink if s < 1.0) / len(shrink),
            mean_shrink=sum(shrink) / len(shrink),
            sensitivity=sens, noise_std=std,
            kept_share=({str(c): s for (c, _, _), s in zip(ok, kept)} if kept is not None else None),
        )
        return out, stats

    def aggregate_fit(self, server_round, results):
        if not results:
            self.last_round_stats = {"aggregated": 0, "failed": True, "message": "no results"}
            return None
        out, stats = self._pipeline(server_round, results, apply_noise=True)
        if out is None:
            self.last_round_stats = stats
            log.error(stats["message"])
            return None
        if self._fedopt is not None:
            t = time.perf_counter()
            out = self._fedopt.aggregate_fit(server_round, [(OrderedDict((k, v.clone()) for k, v in out.items()), 1)])
            stats["timings"]["optimizer"] = time.perf_counter() - t
        self._global = OrderedDict((k, v.detach().float().cpu().clone()) for k, v in out.items())
        self.last_round_stats = stats
        return out

    def preview(self, results):
        """Noise-free aggregate of ``results`` with this strategy's rule and clip, without changing
        any state (used to measure an attack's deviation from the honest-only aggregate)."""
        out, _ = self._pipeline(-1, results, apply_noise=False)
        return out

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters)
