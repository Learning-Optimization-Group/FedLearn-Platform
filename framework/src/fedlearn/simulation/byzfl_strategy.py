"""Optional pinned ByzFL reducers for controlled simulator benchmarks.

Private centered clipping retains only the previous released delta. Upstream's
unnoised persistent center is deliberately discarded after each reduction.
"""
from collections import OrderedDict
import importlib
from importlib.metadata import version
import math

import torch

from fedlearn.server._update_normalize import normalize_updates
from fedlearn.server.composed_strategy import ComposedStrategy

BYZFL_VERSION = "0.0.11"
SUPPORTED_RULES = ("mean", "median", "trimmed_mean", "centered_clip")


class ByzFLStrategy(ComposedStrategy):
    def __init__(self, initial_parameters, rule="mean", bucket_size=1, bucket_seed=0, **kwargs):
        if rule not in SUPPORTED_RULES:
            raise ValueError(f"ByzFL benchmark supports only {SUPPORTED_RULES}")
        if not isinstance(bucket_size, int) or isinstance(bucket_size, bool) or bucket_size < 1:
            raise ValueError("bucket_size must be a positive integer")
        params = dict(kwargs.get("rule_params") or {})
        tau, iterations = params.get("tau", 1.), params.get("iterations", 1)
        if not math.isfinite(float(tau)) or float(tau) <= 0:
            raise ValueError("tau must be finite and positive")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        z = float(kwargs.get("central_noise_multiplier", 0.))
        if not math.isfinite(z) or z < 0:
            raise ValueError("noise multiplier must be finite and nonnegative")
        clip = kwargs.get("clip_norm")
        if clip is not None and (not math.isfinite(float(clip)) or float(clip) <= 0):
            raise ValueError("clip_norm must be finite and positive")
        if z > 0:
            if bucket_size != 1 or rule == "median" or (rule == "centered_clip" and iterations != 1):
                raise ValueError("private ByzFL needs mean, trimmed mean, or one-step unbucketed centered clipping")
            if not kwargs.get("clients_per_round"):
                raise ValueError("private ByzFL needs a fixed clients_per_round")
            if kwargs.get("norm_band") is not None:
                raise ValueError("private ByzFL does not support data-dependent filtering")
        # The base class validates standard noise/clipping settings; the centered
        # rule's separately checked bound is supplied by _sensitivity below.
        super().__init__(initial_parameters, rule="mean" if z > 0 and rule == "centered_clip" else rule, **kwargs)
        self.rule = rule
        try:
            self._byzfl = importlib.import_module("byzfl")
            installed = version("byzfl")
        except ImportError as exc:
            raise RuntimeError("ByzFL benchmark dependencies are missing; install fedlearn[byzfl] in an isolated environment") from exc
        if installed != BYZFL_VERSION:
            raise RuntimeError(f"ByzFL benchmark requires {BYZFL_VERSION}, found {installed}")
        self.bucket_size, self.bucket_seed = bucket_size, int(bucket_seed)
        self._released_center = None
        self._releases = 0
        self.provenance = {"library": "byzfl", "version": installed, "bucket_size": bucket_size,
                           "bucket_seed": self.bucket_seed,
                           "bucket_rng": "local_torch_generator", "bucket_weighting": "equal_bucket_means",
                           "center": "previous_released_delta" if rule == "centered_clip" else None,
                           "centered_sensitivity": "2*min(C,tau)/n for one iteration" if rule == "centered_clip" else None,
                           "diagnostics": "research_only"}

    def _pipeline(self, server_round, results, *, apply_noise):
        if self.z > 0:
            normalized = normalize_updates(results)
            valid = [(c, p, n) for c, p, n in normalized if n > 0 and self._valid(p)]
            if len(valid) != self.clients_per_round or len({c for c, _, _ in valid}) != len(valid):
                return None, {"aggregated": len(valid), "failed": True, "dropped": [],
                              "message": "private ByzFL requires the complete fixed cohort", "timings": {}}
        return super()._pipeline(server_round, results, apply_noise=apply_noise)

    def _prepared_vectors(self, deltas, keys):
        vectors = torch.stack([torch.cat([d[k].reshape(-1) for k in keys]) for d in deltas]).float()
        groups = [[i] for i in range(len(vectors))]
        if self.bucket_size > 1:
            # Upstream Bucketing consumes global RNG. Use its permutation/grouping
            # semantics with a local generator so previews and other runs cannot
            # alter the partition. A partial final bucket receives equal weight.
            generator = torch.Generator().manual_seed(self.bucket_seed + self._releases)
            order = torch.randperm(len(vectors), generator=generator)
            groups = [part.tolist() for part in order.split(self.bucket_size)]
            vectors = vectors[order]
            vectors = torch.stack([part.mean(0) for part in vectors.split(self.bucket_size)])
        return vectors, groups

    def _reduce_deltas(self, deltas, weights, keys):
        vectors, _ = self._prepared_vectors(deltas, keys)
        if self.rule == "mean":
            reduced = self._byzfl.Average()(vectors)
        elif self.rule == "median":
            reduced = self._byzfl.Median()(vectors)
        elif self.rule == "trimmed_mean":
            reduced = self._byzfl.TrMean(f=math.floor(self.rule_params["trim_ratio"] * len(vectors)))(vectors)
        else:
            reducer = self._byzfl.CenteredClipping(L=self.rule_params["iterations"], tau=float(self.rule_params["tau"]))
            # Version 0.0.11 rejects every explicit constructor m argument. The
            # documented persistent state is assignable after construction.
            reducer.m = (torch.zeros_like(vectors[0]) if self._released_center is None
                         else self._released_center.clone())
            reduced = reducer(vectors)
        if reduced.shape != vectors[0].shape or not torch.isfinite(reduced).all():
            raise ValueError("ByzFL produced an invalid aggregate")
        out, offset = OrderedDict(), 0
        for key in keys:
            count = deltas[0][key].numel()
            out[key] = reduced[offset:offset + count].reshape(deltas[0][key].shape).clone()
            offset += count
        return out, None

    def _sensitivity(self, n):
        if self.rule == "centered_clip":
            if self.rule_params["iterations"] != 1:
                return None
            radius = float(self.rule_params["tau"])
            if self.clip_norm is not None:
                radius = min(radius, self.clip_norm)
            return 2. * radius / n
        return super()._sensitivity(n)

    def _reduction_diagnostics(self, deltas, keys):
        vectors, groups = self._prepared_vectors(deltas, keys)
        diagnostics = {"research_only": True, "bucket_size": self.bucket_size,
                       "bucket_count": len(groups), "last_bucket_size": len(groups[-1]),
                       "bucket_members_by_input_index": groups}
        if self.rule == "centered_clip":
            center = torch.zeros_like(vectors[0]) if self._released_center is None else self._released_center
            norms = torch.linalg.vector_norm(vectors - center, dim=1)
            tau = float(self.rule_params["tau"])
            mean = vectors.mean(0)
            center_norm, mean_norm = float(torch.linalg.vector_norm(center)), float(torch.linalg.vector_norm(mean))
            alignment = (float(torch.clamp(torch.dot(center, mean) / (center_norm * mean_norm), -1., 1.))
                         if center_norm > 0 and mean_norm > 0 else None)
            diagnostics.update(center_source="previous_released_delta", center_norm=center_norm,
                               tau=tau, inner_clip_diagnostic_iteration=1,
                               iterations=self.rule_params["iterations"],
                               inner_clip_rate=float((norms > tau).float().mean()),
                               inner_mean_shrink=float(torch.clamp(tau / norms, max=1.).mean()),
                               mean_input_norm=mean_norm,
                               mean_centered_deviation_norm=float(torch.linalg.vector_norm(mean - center)),
                               center_alignment_with_input_mean=alignment)
        return {"aggregation_diagnostics": diagnostics}

    def aggregate_fit(self, server_round, results):
        before = self._global
        out = super().aggregate_fit(server_round, results)
        if out is not None:
            self._released_center = torch.cat([(out[k] - before[k]).reshape(-1) for k in before]).detach().clone()
            self._releases += 1
        return out
