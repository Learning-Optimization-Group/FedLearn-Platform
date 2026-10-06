# framework/src/fedlearn/simulation/behaviors.py
"""Bad clients: who misbehaves, and how.

Malicious sets are drawn in config order from one ``numpy.random.default_rng(attack_seed)``; the
first equals the capstone harness's ``choice(n, k, replace=False)``, later ones draw from the
clients not yet taken. Faulty sets (straggler, label noise) are drawn independently and may overlap
attackers. Identities persist across rounds. Upload-time attack arithmetic is the same as
``benchmarks/robust_aggregation_attack._apply_attack_to_round`` (tests compare them directly).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Set

import numpy as np
import torch

MALICIOUS = ("sign_flip_scale", "ipm", "alie", "label_flip", "backdoor", "free_rider")


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
        if b is not None and b["type"] == "free_rider":
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

    def upload_overrides(self, records: Dict[int, dict], global_params, round_num: int) -> Dict[int, "OrderedDict"]:
        keys = list(global_params.keys())
        honest = [c for c in sorted(records) if c not in self.malicious]
        need_stats = any(self.malicious.get(c, {}).get("type") in ("ipm", "alie") for c in records)
        if need_stats:
            if not honest:
                raise ValueError("IPM and ALIE need at least one honest client in the round")
            stacked = {k: torch.stack([records[c]["honest_delta"][k] for c in honest], dim=0) for k in keys}
            mean_ = {k: stacked[k].mean(dim=0) for k in keys}
            std_ = {k: stacked[k].std(dim=0, unbiased=False) for k in keys}
        out: Dict[int, "OrderedDict"] = {}
        for c in sorted(records):
            b = self.malicious.get(c)
            if b is None:
                continue
            t = b["type"]
            if t == "sign_flip_scale":
                s = -abs(float(b["scale"]))
                out[c] = OrderedDict((k, global_params[k].float() + s * records[c]["honest_delta"][k]) for k in keys)
            elif t == "ipm":
                out[c] = OrderedDict((k, global_params[k].float() + (-float(b["scale"]) * mean_[k])) for k in keys)
            elif t == "alie":
                out[c] = OrderedDict((k, global_params[k].float() + (mean_[k] - float(b["z"]) * std_[k])) for k in keys)
            elif t == "free_rider":
                if b["mode"] == "zero":
                    out[c] = OrderedDict((k, global_params[k].float().clone()) for k in keys)
                else:
                    g = torch.Generator().manual_seed((self.attack_seed * 1_000_003 + round_num * 10_007 + c) % (2 ** 63))
                    out[c] = OrderedDict(
                        (k, global_params[k].float() + float(b["noise_std"]) *
                         torch.randn(global_params[k].shape, generator=g)) for k in keys)
        return out
