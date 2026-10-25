"""Bad clients: who misbehaves, and how.

Malicious sets are drawn in config order from one ``numpy.random.default_rng(attack_seed)``; the
first equals the capstone harness's ``choice(n, k, replace=False)``, later ones draw from the
clients not yet taken. Faulty sets (straggler, label noise) are drawn independently and may overlap
attackers. Identities persist across rounds. Upload-time attack arithmetic is the same as
``benchmarks/robust_aggregation_attack._apply_attack_to_round`` (tests compare them directly).
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from fedlearn.simulation.owned_data_attack import ascent_upload

OWNED = ("owned_pgd", "owned_sign_flip")

MALICIOUS = ("sign_flip_scale", "ipm", "alie", "label_flip", "backdoor", "free_rider", "min_max", "fang_trim", "canary", *OWNED)
_OMNISCIENT = ("ipm", "alie", "min_max", "fang_trim")  # built from the honest clients' deltas


class BehaviorPlan:
    def __init__(self, behaviors: List[dict], num_clients: int, attack_seed: int, n_classes: int):
        self.behaviors, self.n, self.n_classes = behaviors, int(num_clients), int(n_classes)
        self.attack_seed = int(attack_seed)
        self.malicious: Dict[int, dict] = {}
        self.faulty: Dict[int, List[dict]] = {}
        rng = np.random.default_rng(self.attack_seed)
        taken: Set[int] = set()
        first = True
        for b in behaviors:
            if b["type"] not in MALICIOUS:
                continue
            k = int(round(b["fraction"] * self.n))
            if first:
                chosen = [int(i) for i in rng.choice(self.n, k, replace=False)]
                first = False
            else:
                pool = np.array(sorted(set(range(self.n)) - taken), dtype=np.int64)
                if k > len(pool):
                    raise ValueError(f"not enough clients left for {b['type']} ({k} > {len(pool)})")
                chosen = [int(i) for i in rng.choice(pool, k, replace=False)] if k else []
            for c in chosen:
                self.malicious[c] = b
            taken |= set(chosen)
        for i, b in enumerate(behaviors):
            if b["type"] in ("straggler", "label_noise"):
                r = np.random.default_rng((self.attack_seed, 7919, i))
                for c in r.choice(self.n, int(round(b["fraction"] * self.n)), replace=False):
                    self.faulty.setdefault(int(c), []).append(b)
        self.dropout_fraction = sum(float(b["fraction"]) for b in behaviors if b["type"] == "dropout")
        self.malicious_fraction = len(self.malicious) / self.n
        self.last_attack_note: Optional[str] = None
        self.last_owned_attack_diagnostics: List[dict] = []

    def malicious_ids(self) -> Set[int]:
        return set(self.malicious)

    def assignment(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for c in range(self.n):
            kinds = ([self.malicious[c]["type"]] if c in self.malicious else []) + \
                    [b["type"] for b in self.faulty.get(c, [])]
            if kinds:
                out[str(c)] = kinds
        return out

    def backdoor_target(self) -> Optional[int]:
        for b in self.behaviors:
            if b["type"] == "backdoor":
                return int(b["target_label"])
        return None

    def needs_trigger(self) -> bool:
        return self.backdoor_target() is not None

    def slowdown(self, cid: int) -> float:
        s = 1.0
        for b in self.faulty.get(cid, []):
            if b["type"] == "straggler":
                s *= float(b["slowdown"])
        return s

    def training_data(self, cid: int, x: torch.Tensor, y: torch.Tensor, trigger_x: Optional[torch.Tensor],
                      data_seed: int):
        """Apply training-time behaviours; returns ``(x, y, train)``; ``train`` is False for a free-rider."""
        b = self.malicious.get(cid)
        if b is not None and b["type"] in ("free_rider", "canary"):
            return x, y, False
        if b is not None and b["type"] == "label_flip":
            y = (self.n_classes - 1) - y
        if b is not None and b["type"] == "backdoor":
            if trigger_x is None:
                raise ValueError("backdoor needs triggered features; prepare the features file with --trigger-patch")
            g = torch.Generator().manual_seed(int(data_seed))
            rows = torch.randperm(len(y), generator=g)[: int(round(b["poison_fraction"] * len(y)))]
            x, y = x.clone(), y.clone()
            x[rows] = trigger_x[rows]
            y[rows] = int(b["target_label"])
        for fb in self.faulty.get(cid, []):
            if fb["type"] == "label_noise":
                g = torch.Generator().manual_seed(int(data_seed) + 1)
                flip = torch.rand(len(y), generator=g) < float(fb["flip_prob"])
                shift = torch.randint(1, self.n_classes, (len(y),), generator=g)
                y = torch.where(flip, (y + shift) % self.n_classes, y)
        return x, y, True

    def owned_ids(self) -> Set[int]:
        return {c for c, b in self.malicious.items() if b["type"] in OWNED}

    def owned_upload_overrides(self, owned_records: Dict[int, dict], global_params, round_num: int, *,
                               owned_data, clip_norm, total_submitted, released_delta,
                               history_release_round=0) -> Dict[int, "OrderedDict"]:
        """Share one bounded upload per coalition, accepting malicious-only input views.

        Current honest records and evaluation labels have no place in this interface.
        The engine enforces a complete cohort before constructing these views.
        Diagnostics are research artifacts, outside released-model DP claims.
        """
        self.last_owned_attack_diagnostics = []
        expected = self.owned_ids()
        if set(owned_records) != expected or set(owned_data) != expected:
            raise ValueError("owned attack requires exactly the malicious owned-client views")
        if total_submitted != self.n:
            raise ValueError("owned attack requires a fixed complete submitted cohort")
        if not math.isfinite(float(clip_norm)) or float(clip_norm) <= 0:
            raise ValueError("owned attack needs a finite positive clip norm")
        keys = ["head.weight", "head.bias"]
        if set(global_params) != set(keys) or global_params[keys[0]].ndim != 2 \
                or global_params[keys[0]].shape[0] != 2 or global_params[keys[1]].shape != (2,):
            raise ValueError("owned attack requires a binary linear head")
        out = {}
        for kind in OWNED:
            group = sorted(c for c in expected if self.malicious[c]["type"] == kind)
            if not group:
                continue
            b = self.malicious[group[0]]
            own_mean = torch.stack([torch.cat([owned_records[c]["honest_delta"][k].detach().double().reshape(-1)
                                              for k in keys]) for c in group]).mean(0)
            if not torch.isfinite(own_mean).all():
                raise ValueError("nonfinite owned mean update")
            norm = torch.linalg.vector_norm(own_mean)
            initial = -float(clip_norm) * own_mean / norm if norm > 0 else torch.zeros_like(own_mean)
            rho = len(group) / total_submitted
            diag = {"type": kind, "scope": "research-only; outside released-model DP claims",
                    "knowledge": "same-type coalition-owned training examples/updates and public model/history only",
                    "coalition_size": len(group), "total_submitted": total_submitted, "coalition_fraction": rho,
                    "clip_norm": float(clip_norm), "initialization": "negative full-norm coalition mean; zero when mean is zero",
                    "initial_norm": float(initial.norm()), "history_release_round": history_release_round,
                    "history_norm": float(released_delta.double().norm()),
                    "gradient_evaluations": 0, "loss_evaluations": 0}
            delta = initial
            if kind == "owned_pgd":
                x = torch.cat([owned_data[c][0] for c in group])
                y = torch.cat([owned_data[c][1] for c in group])
                seed = (self.attack_seed * 1_000_003 + round_num * 10_007) % (2 ** 63)
                result = ascent_upload(weight=global_params[keys[0]], bias=global_params[keys[1]],
                                       owned_x=x, owned_y=y, initial_delta=initial, clip_norm=clip_norm,
                                       coalition_fraction=rho, seed=seed, released_delta=released_delta,
                                       steps=b["steps"], restarts=b["restarts"], step_ratio=b["step_ratio"])
                delta = result["delta"]
                diag.update({k: v for k, v in result.items() if k != "delta"})
                diag.update(steps=b["steps"], restarts=b["restarts"], step_ratio=b["step_ratio"], search_seed=seed)
            upload, at = OrderedDict(), 0
            for k in keys:
                size = global_params[k].numel()
                upload[k] = global_params[k].float() + delta[at:at + size].reshape(global_params[k].shape).float()
                at += size
            actual = torch.cat([(upload[k].double() - global_params[k].float().double()).reshape(-1) for k in keys])
            diag["upload_norm"] = float(actual.norm())
            self.last_owned_attack_diagnostics.append(diag)
            for c in group:
                out[c] = OrderedDict((k, v.clone()) for k, v in upload.items())
        return out

    def upload_overrides(self, records: Dict[int, dict], global_params, round_num: int) -> Dict[int, "OrderedDict"]:
        """Final uploads for this round's attackers (honest clients are untouched by the caller).

        IPM, ALIE, min-max and Fang's trimmed-mean attack are built from the honest clients' deltas
        (full knowledge). When a cohort holds no honest client, the attackers fall back to their own
        honestly-trained deltas (ALIE's own threat model) and ``last_attack_note`` says so; the attack
        is never silently skipped."""
        keys = list(global_params.keys())
        self.last_attack_note = None
        honest = [c for c in sorted(records) if c not in self.malicious]
        need_stats = any(self.malicious.get(c, {}).get("type") in _OMNISCIENT for c in records)
        if need_stats:
            basis = honest
            if not basis:
                basis = sorted(records)
                self.last_attack_note = (f"round {round_num}: no honest client in the cohort; IPM/ALIE "
                                         "statistics from the attackers' own honestly-trained deltas")
            stacked = {k: torch.stack([records[c]["honest_delta"][k] for c in basis], dim=0) for k in keys}
            mean_ = {k: stacked[k].mean(dim=0) for k in keys}
            std_ = {k: stacked[k].std(dim=0, unbiased=False) for k in keys}
        min_max_delta = None
        out: Dict[int, "OrderedDict"] = {}
        for c in sorted(records):
            b = self.malicious.get(c)
            if b is None:
                continue
            t = b["type"]
            if t == "sign_flip_scale":
                s = float(b["scale"])  # signed: negative flips, positive scales in the same direction
                out[c] = OrderedDict((k, global_params[k].float() + s * records[c]["honest_delta"][k]) for k in keys)
            elif t == "ipm":
                out[c] = OrderedDict((k, global_params[k].float() + (-float(b["scale"]) * mean_[k])) for k in keys)
            elif t == "alie":
                out[c] = OrderedDict((k, global_params[k].float() + (mean_[k] - float(b["z"]) * std_[k])) for k in keys)
            elif t == "min_max":
                if min_max_delta is None:
                    min_max_delta = _min_max_delta(stacked, mean_, std_, keys, b["perturbation"])
                out[c] = OrderedDict((k, global_params[k].float() + min_max_delta[k]) for k in keys)
            elif t == "fang_trim":
                g = torch.Generator().manual_seed((self.attack_seed * 1_000_003 + round_num * 10_007 + c) % (2 ** 63))
                out[c] = OrderedDict((k, global_params[k].float() + _fang_trim_values(stacked[k], mean_[k], float(b["b"]), g))
                                     for k in keys)
            elif t == "canary":
                u = canary_direction(self.attack_seed, c, global_params)
                out[c] = OrderedDict((k, global_params[k].float() + float(b["norm"]) * u[k]) for k in keys)
            elif t == "free_rider":
                if b["mode"] == "zero":
                    out[c] = OrderedDict((k, global_params[k].float().clone()) for k in keys)
                else:
                    g = torch.Generator().manual_seed((self.attack_seed * 1_000_003 + round_num * 10_007 + c) % (2 ** 63))
                    out[c] = OrderedDict(
                        (k, global_params[k].float() + float(b["noise_std"]) *
                         torch.randn(global_params[k].shape, generator=g)) for k in keys)
        return out


def _min_max_delta(stacked: Dict[str, torch.Tensor], mean_: Dict[str, torch.Tensor], std_: Dict[str, torch.Tensor],
                   keys: List[str], perturbation: str) -> "OrderedDict[str, torch.Tensor]":
    """Shejwalkar and Houmansadr's min-max attack (NDSS'21): ``mean + gamma * p`` with the largest
    gamma that keeps the upload no farther from any honest delta than two honest deltas can be.

    ``p`` is the negated per-coordinate std (``"std"``), the negated unit mean (``"unit_vec"``) or the
    negated sign of the mean (``"sign"``). Every min-max attacker uploads the same vector.
    """
    honest = torch.cat([stacked[k].reshape(stacked[k].shape[0], -1) for k in keys], dim=1).double()
    mean = torch.cat([mean_[k].reshape(-1) for k in keys]).double()
    if perturbation == "std":
        p = -torch.cat([std_[k].reshape(-1) for k in keys]).double()
    elif perturbation == "unit_vec":
        p = -mean / (torch.linalg.vector_norm(mean) + 1e-12)
    else:
        p = -torch.sign(mean)
    threshold = float(torch.cdist(honest, honest).max())

    def worst(gamma: float) -> float:
        return float(torch.linalg.vector_norm(honest - (mean + gamma * p), dim=1).max())

    lo, hi = 0.0, 1.0
    if float(torch.linalg.vector_norm(p)) > 0.0 and worst(0.0) <= threshold:
        while worst(hi) <= threshold and hi < 1e12:
            lo, hi = hi, hi * 2.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            lo, hi = (mid, hi) if worst(mid) <= threshold else (lo, mid)
    flat = (mean + lo * p).float()
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    at = 0
    for k in keys:
        size = mean_[k].numel()
        out[k] = flat[at:at + size].reshape(mean_[k].shape)
        at += size
    return out


def _fang_trim_values(stacked: torch.Tensor, mean: torch.Tensor, b: float, generator: torch.Generator) -> torch.Tensor:
    """Fang et al.'s full-knowledge attack on the trimmed mean and median (USENIX Security'20).

    On each coordinate the attacker pushes against the honest direction ``sign(mean)``: where honest
    clients move up it uploads a value below their minimum (in ``[min/b, min]`` if ``min > 0``, else
    ``[b*min, min]``); otherwise a value above their maximum (``[max, b*max]`` if ``max > 0``, else
    ``[max/b, max]``). Values are uniform in that interval, drawn per attacker.
    """
    lo_h, hi_h = stacked.min(dim=0).values, stacked.max(dim=0).values
    u = torch.rand(mean.shape, generator=generator)
    below_lo = torch.where(lo_h > 0, lo_h / b, lo_h * b)
    above_hi = torch.where(hi_h > 0, hi_h * b, hi_h / b)
    down = below_lo + u * (lo_h - below_lo)
    up = hi_h + u * (above_hi - hi_h)
    return torch.where(mean > 0, down, up)


def canary_direction(attack_seed: int, direction_id: int, like: Dict[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """A seeded random unit direction over all of ``like``'s keys (jointly unit L2 norm).

    A canary client uploads ``global + norm * canary_direction(attack_seed, its id, global)`` every
    round (Andrew et al., ICLR'24). An audit draws never-inserted directions from the same function with
    ids no client has, so inserted and unseen directions share one distribution.
    """
    g = torch.Generator().manual_seed((int(attack_seed) * 1_000_003 + 7_777_777 + int(direction_id)) % (2 ** 63))
    raw = OrderedDict((k, torch.randn(tuple(v.shape), generator=g)) for k, v in like.items())
    norm = torch.sqrt(sum((v.double() ** 2).sum() for v in raw.values())).item()
    return OrderedDict((k, (v / norm).float()) for k, v in raw.items())
