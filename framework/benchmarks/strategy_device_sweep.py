#!/usr/bin/env python3
"""Every aggregation strategy on CPU versus GPU, at the model scale this platform federates.

WHY THIS EXISTS
---------------
The campaign's headline compute claim is that the CPU beats the GPU at frozen-head scale. A
cell-by-cell audit found that claim rested on CPU measurements for only TWO of six strategies:
FedAvg and DeComFL, and those existed only because a separate experiment happened to need them.
FedProx, FedOpt and Robust had never been run on a CPU on any machine.

That matters most for Robust. Its coordinate-wise median and trimmed mean are a SORT over the
client dimension, not a weighted sum, and sorting has a very different parallel profile from
elementwise arithmetic. It is the single strategy most likely to break the pattern, and it was the
one with no CPU number at all.

This runs every available strategy through the SAME data, partition, seed and round budget, on CPU
and on the accelerator of the same machine, so each strategy gets a controlled A/B rather than a
cross-machine comparison.

Deliberately run at FROZEN-HEAD scale (d=1,026 by default). The claim under test is about the model
size this platform actually federates; testing at ImageNet-100 scale would answer a different
question, and the GPU is already known to win there.

    PYTHONPATH=framework/src python framework/benchmarks/strategy_device_sweep.py --out out.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fedlearn.estimators import params as P                        # noqa: E402
from fedlearn.server.strategy import FedAvg, FedProx, FedOpt       # noqa: E402
from fedlearn.server.robust_aggregation import RobustAggregator    # noqa: E402
import decomfl_vs_fedavg_dim as H                                  # noqa: E402

STRATEGIES = ("FedAvg", "FedProx", "FedOpt", "Robust", "RobustTrimmed")


def build_strategy(name, init_sd, cpr, server_lr, tau, proximal_mu=0.1):
    """Construct a strategy by name. Kept explicit so an unknown name fails loudly."""
    if name == "FedAvg":
        return FedAvg(initial_parameters=init_sd, min_fit_clients=1, clients_per_round=cpr)
    if name == "FedProx":
        # proximal_mu is a CLIENT-side term; server aggregation is identical to FedAvg by design,
        # so any CPU/GPU difference here isolates the client loop rather than the aggregator.
        return FedProx(initial_parameters=init_sd, min_fit_clients=1, clients_per_round=cpr,
                       proximal_mu=proximal_mu)
    if name == "FedOpt":
        return FedOpt(initial_parameters=init_sd, min_fit_clients=1, clients_per_round=cpr,
                      server_learning_rate=server_lr, tau=tau)
    if name == "Robust":
        return RobustAggregator(initial_parameters=init_sd, min_fit_clients=1,
                                clients_per_round=cpr, method="median")
    if name == "RobustTrimmed":
        return RobustAggregator(initial_parameters=init_sd, min_fit_clients=1,
                                clients_per_round=cpr, method="trimmed_mean", trim_ratio=0.2)
    raise SystemExit(f"unknown strategy {name!r}; expected one of {STRATEGIES}")


def run_strategy(name, *, train_x, train_y, test_x, test_y, feat_dim, n_classes, hidden,
                 clients, clients_per_round, alpha, rounds, local_epochs, lr, batch_size,
                 seed, device, server_lr=0.001, tau=1e-3, proximal_mu=0.1, eval_every=25):
    """One federated run.

    The client loop is shared by every strategy EXCEPT for FedProx's proximal term, which is a
    client-side quantity rather than an aggregation rule. An earlier version of this harness
    omitted it, so FedProx and FedAvg produced bit-identical results and were reported as two
    strategies agreeing when they were one strategy run twice.

    The term matches fedlearn.client.local_trainer.LocalTrainer.fit exactly: the exact gradient
    contribution mu*(w - w_global) is added in place to each parameter's .grad before the
    optimiser step, where w_global is the round's starting global model. mu = 0 skips it
    entirely, so FedProx at mu=0 is bitwise FedAvg.
    """
    t0 = time.time()
    torch.manual_seed(seed)
    model = H.head_model(feat_dim=feat_dim, n_classes=n_classes, hidden=hidden, seed=seed).to(device)
    d = H.model_dim(model)
    init_sd = P.trainable_state(model)
    strategy = build_strategy(name, init_sd, clients_per_round, server_lr, tau, proximal_mu)
    mu = float(proximal_mu) if name == "FedProx" else 0.0

    global_state = OrderedDict((k, v.clone()) for k, v in init_sd.items())
    parts = H.partition(train_y.numpy(), clients, alpha, seed)
    loaders = H._make_loaders(train_x, train_y, parts, batch_size, seed)
    rng = np.random.RandomState(seed + 77)
    crit = nn.CrossEntropyLoss()
    per_round, update_norms = [], []

    for r in range(1, rounds + 1):
        chosen = rng.choice(clients, size=min(clients_per_round, clients), replace=False)
        results = []
        for ci in chosen:
            local = H.head_model(feat_dim=feat_dim, n_classes=n_classes, hidden=hidden,
                                 seed=seed).to(device)
            local.load_state_dict(global_state, strict=False)
            # w_global for this round, captured BEFORE any local step (the proximal anchor).
            anchor = [p.detach().clone() for p in local.parameters() if p.requires_grad]
            opt = torch.optim.SGD([p for p in local.parameters() if p.requires_grad], lr=lr)
            local.train()
            for _ in range(local_epochs):
                for xb, yb in loaders[ci]:
                    opt.zero_grad()
                    crit(local(xb.to(device)), yb.to(device)).backward()
                    if mu > 0.0:
                        for prm, w0 in zip([q for q in local.parameters() if q.requires_grad],
                                           anchor):
                            if prm.grad is not None:
                                prm.grad.add_(prm.detach() - w0, alpha=mu)
                    opt.step()
            with torch.no_grad():
                cur = [p.detach() for p in local.parameters() if p.requires_grad]
                update_norms.append(float(sum(((c - a) ** 2).sum() for c, a in
                                              zip(cur, anchor)).sqrt()))
            results.append((P.trainable_state(local), len(loaders[ci].dataset)))

        agg = strategy.aggregate_fit(r, results)
        if agg is not None:
            global_state = agg

        if r % eval_every == 0 or r == rounds:
            model.load_state_dict(global_state, strict=False)
            per_round.append({"round": r, **H._evaluate(model, test_x.to(device), test_y.to(device))})

    return {"strategy": name, "device": device, "d": d, "hidden": hidden, "seed": seed,
            "rounds": rounds, "alpha": alpha, "proximal_mu": mu,
            "mean_update_norm": round(float(np.mean(update_norms)), 6) if update_norms else None,
            "wall_seconds": round(time.time() - t0, 3),
            "final_auc": per_round[-1]["auc"] if per_round else float("nan"),
            "best_auc": max((p["auc"] for p in per_round if np.isfinite(p["auc"])),
                            default=float("nan")),
            "peak_rss_mb": round(H.peak_rss_mb(), 1), "per_round": per_round}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--hidden", default="0")
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--clients-per-round", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--server-lr", type=float, default=0.001)
    ap.add_argument("--tau", type=float, default=1e-3)
    ap.add_argument("--proximal-mu", type=float, default=0.1,
                    help="FedProx client-side proximal strength. mu=0 reduces FedProx "
                         "to FedAvg exactly; it is the parameter that MAKES FedProx "
                         "FedProx, and was never varied before 2026-07-29.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--devices", default="cpu,auto")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    devs = []
    for dv in a.devices.split(","):
        if dv == "auto":
            dv = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else None)
        if dv:
            devs.append(dv)

    tx, ty, ex, ey = H.load_features(a.features)
    feat_dim, n_classes = tx.shape[1], int(ty.max()) + 1
    runs = []
    for h in [int(x) for x in a.hidden.split(",")]:
        for s in [int(x) for x in a.seeds.split(",")]:
            for name in a.strategies.split(","):
                for dv in devs:
                    r = run_strategy(name, train_x=tx, train_y=ty, test_x=ex, test_y=ey,
                                     feat_dim=feat_dim, n_classes=n_classes, hidden=h,
                                     clients=a.clients, clients_per_round=a.clients_per_round,
                                     alpha=a.alpha, rounds=a.rounds, local_epochs=a.local_epochs,
                                     lr=a.lr, batch_size=a.batch_size, seed=s, device=dv,
                                     server_lr=a.server_lr, tau=a.tau, proximal_mu=a.proximal_mu,
                                     eval_every=a.eval_every)
                    runs.append(r)
                    print(f"  {name:<14} d={r['d']:<8} {dv:<5} seed={s} "
                          f"auc={r['final_auc']:.4f} {r['wall_seconds']:>8.2f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"experiment": "every strategy on CPU vs GPU at frozen-head scale",
                   "meta": {"date": "2026-07-29", "host": platform.node(),
                            "platform": platform.platform(), "torch": torch.__version__,
                            "devices": devs, "features": a.features, "feat_dim": feat_dim,
                            "n_classes": n_classes, "clients": a.clients,
                            "clients_per_round": a.clients_per_round, "alpha": a.alpha,
                            "rounds": a.rounds, "local_epochs": a.local_epochs, "lr": a.lr,
                            "batch_size": a.batch_size, "server_lr": a.server_lr, "tau": a.tau},
                   "runs": runs}, f, indent=2)
    print(f"\nwrote {a.out}  ({len(runs)} runs)")


if __name__ == "__main__":
    main()
