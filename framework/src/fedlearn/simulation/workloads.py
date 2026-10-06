"""What the simulation trains: a linear or MLP head over a prepared features file.

Data handling reproduces ``benchmarks/capstone_matched_study.py`` exactly (balancing seeds,
partitioning), which is what lets the engine reproduce that harness bit for bit.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from fedlearn.server.subset_federation import apply_trainable_subset
from fedlearn.simulation.config import ConfigError
from fedlearn.simulation.metrics import backdoor_success, classification_metrics
from fedlearn.simulation.partition import dirichlet_partition, iid_partition
from fedlearn.simulation.rng import torch_rng_scope


class LinearHead(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        return self.head(x)


class MLPHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.hidden = nn.Linear(feat_dim, hidden)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.head(torch.relu(self.hidden(x)))


def balanced_indices(y: torch.Tensor, seed: int) -> torch.Tensor:
    """Indices that down-sample every class to the minority count (seeded); the minority is kept
    whole. Same draws as ``capstone_matched_study.balance_classes``."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.bincount(y)
    m = int(counts[counts > 0].min())
    keep = []
    for c in range(len(counts)):
        idx = (y == c).nonzero(as_tuple=True)[0]
        if len(idx) > m:
            idx = idx[torch.randperm(len(idx), generator=g)[:m]]
        keep.append(idx)
    return torch.sort(torch.cat(keep)).values


def make_partition(labels, clients: int, seed: int, kind: str = "iid", alpha: float = 0.5) -> List[np.ndarray]:
    if kind == "iid":
        return iid_partition(len(labels), clients, seed)
    return dirichlet_partition(labels, clients, alpha, seed, min_partition_size=1)


class FeatureHeadWorkload:
    """A head over a features file written by ``benchmarks/prepare_features.py``."""

    def __init__(self, resolved: dict):
        wl, fed = resolved["workload"], resolved["federation"]
        data = torch.load(wl["features_path"], weights_only=True)
        tx, ty = data["train_x"].float(), data["train_y"].long()
        vx, vy = data["test_x"].float(), data["test_y"].long()
        trig_t, trig_v = data.get("trigger_train_x"), data.get("trigger_test_x")
        if wl["balance"]:
            ti = balanced_indices(ty, int(resolved["seed"]))
            vi = balanced_indices(vy, int(resolved["seed"]) + 1)
            tx, ty, vx, vy = tx[ti], ty[ti], vx[vi], vy[vi]
            trig_t = trig_t[ti] if trig_t is not None else None
            trig_v = trig_v[vi] if trig_v is not None else None
        self.train_x, self.train_y, self.test_x, self.test_y = tx, ty, vx, vy
        self.trigger_train_x, self.trigger_test_x = trig_t, trig_v
        self.has_trigger = trig_t is not None and trig_v is not None
        self.feat_dim, self.n_classes = int(data["feat_dim"]), int(data["n_classes"])
        self.head = dict(wl["head"])
        self.info = {k: data[k] for k in ("dataset", "backbone", "pretrained", "trigger") if k in data}
        if int(fed["num_clients"]) > len(ty):
            raise ConfigError("federation.num_clients",
                              f"{fed['num_clients']} clients but only {len(ty)} training samples; every client "
                              "needs at least one")
        self.parts = make_partition(ty.numpy(), int(fed["num_clients"]), int(resolved["seeds"]["partition"]),
                                    fed["partition"]["type"], float(fed["partition"]["alpha"]))

    def build_model(self, init_seed: Optional[int] = None) -> nn.Module:
        def make():
            if self.head["type"] == "linear":
                return LinearHead(self.feat_dim, self.n_classes)
            return MLPHead(self.feat_dim, int(self.head["hidden"]), self.n_classes)
        if init_seed is None:
            return make()
        with torch_rng_scope(int(init_seed)):
            return make()

    def layer_dims(self) -> List[Tuple[int, int]]:
        if self.head["type"] == "linear":
            return [(self.feat_dim, self.n_classes)]
        h = int(self.head["hidden"])
        return [(self.feat_dim, h), (h, self.n_classes)]

    def client_data(self, cid: int):
        idx = torch.as_tensor(self.parts[cid], dtype=torch.long)
        return self.train_x[idx], self.train_y[idx], idx

    def trigger_rows(self, idx: torch.Tensor) -> Optional[torch.Tensor]:
        return self.trigger_train_x[idx] if self.trigger_train_x is not None else None

    def evaluate_fn(self, backdoor_target: Optional[int] = None) -> Callable:
        model = self.build_model()

        def evaluate(server_round, params):
            apply_trainable_subset(model, params)
            model.eval()
            with torch.no_grad():
                logits = model(self.test_x)
                trig = model(self.trigger_test_x) if (backdoor_target is not None and self.has_trigger) else None
            out = classification_metrics(logits, self.test_y)
            out["backdoor_success"] = (backdoor_success(trig, self.test_y, backdoor_target)
                                       if trig is not None else None)
            # SimulatedFederation's RoundRecord applies float() to every metric, so a metric that
            # does not apply this run is omitted rather than returned as None.
            return out["loss"], {k: v for k, v in out.items() if v is not None}
        return evaluate

    def per_client_accuracy(self, params) -> List[Optional[float]]:
        """Accuracy of ``params`` on each client's own clean training shard (the fairness view):
        one forward pass over the training set, then a mean per shard; ``None`` for an empty shard."""
        model = self.build_model()
        apply_trainable_subset(model, params)
        model.eval()
        with torch.no_grad():
            correct = (model(self.train_x).argmax(1) == self.train_y).float()
        return [correct[torch.as_tensor(p, dtype=torch.long)].mean().item() if len(p) else None
                for p in self.parts]
