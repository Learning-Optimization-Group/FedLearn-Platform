#!/usr/bin/env python3
"""DeComFL (zeroth-order) vs FedAvg (first-order) as a function of MODEL DIMENSION.

WHY THIS EXISTS
---------------
The platform ships two on-device training paths and they have very different portability:

  * first-order  — needs autograd on the phone. Today that means the ExecuTorch *training*
    extension, which is CPU-only ("integration with backends/delegates is still a work in
    progress") and is the reason the Adreno GPU is unreachable for training.
  * zeroth-order — needs only FORWARD passes. It runs on any inference runtime, on any
    backend, GPU included.

So if DeComFL reaches FedAvg's quality on the model the phone actually trains, the ZO path is
a portable fallback with real deployment value. If it does not, first-order is mandatory and
the runtime choice is constrained by it. That question had never been measured at the sizes
that decide it: the existing zeroth-vs-first-order record covers d=650 (logistic regression)
and d=3,466 (a toy MLP), both far below the 1,026-parameter production head and four orders
of magnitude below a full ResNet.

WHAT IS CONTROLLED
------------------
Dimension is varied by the WIDTH OF A HIDDEN LAYER on a FIXED frozen feature set. The task,
the data, the partition, the seed and the features are identical across every d — only the
number of trainable parameters changes. Changing backbone instead would confound dimension
with representation quality, which is the error this design exists to avoid.

Both arms start from a byte-identical initialisation (`init_sha` is recorded and asserted in
the tests) and are compared on two budgets, because they are not commensurable on one:
  * round-matched — the federated budget that matters when each round costs a wall-clock
    synchronisation across clients;
  * byte-matched  — the budget that matters on a metered mobile link, where DeComFL's
    dimension-free K*P scalars are its entire claim.

This uses the framework's OWN `DeComFL` strategy and `ZerothOrderEstimator`, and mirrors the
real `DeComFLClient` fit loop (Algorithm 4, lines 16-24) including the revert. `--check-rebuild`
verifies the shared-seed invariant — that a client regenerating z from the shared seed lands
on exactly the server's parameters — so a broken run cannot be reported as DeComFL's quality.

Usage:
  PYTHONPATH=framework/src python framework/benchmarks/decomfl_vs_fedavg_dim.py \
      --hidden 0,20,200 --rounds 300 --seeds 0,1,2 --device cpu \
      --out research/results/decomfl/dim_sweep.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fedlearn.estimators import params as P                              # noqa: E402
from fedlearn.estimators.zeroth_order import ZerothOrderEstimator        # noqa: E402
from fedlearn.server.decomfl_strategy import DeComFL                     # noqa: E402
from fedlearn.server.strategy import FedAvg                              # noqa: E402

FLOAT_BYTES = 4
SEED_BYTES = 4  # a uint32 seed per (k, p); the server could derive these instead, so this is
                # the pessimistic accounting for DeComFL, not the flattering one.


# ------------------------------------------------------------------------------------ model

class Head(nn.Module):
    """Linear (hidden=0) or one-hidden-layer head over frozen features.

    hidden=0 with feat_dim=512, n_classes=2 is exactly the production model: 1,026 params.
    """

    def __init__(self, feat_dim: int, n_classes: int, hidden: int = 0):
        super().__init__()
        if hidden <= 0:
            self.net = nn.Linear(feat_dim, n_classes)
        else:
            self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def head_model(*, feat_dim: int, n_classes: int, hidden: int = 0, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return Head(feat_dim, n_classes, hidden)


def model_dim(model: nn.Module) -> int:
    """Trainable dimension in the CANONICAL layout the shared-seed perturbation spans."""
    return P.num_trainable(model)


def init_sha(model: nn.Module) -> str:
    return hashlib.sha256(P.flat_params(model).cpu().numpy().tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------------------------ bytes

def decomfl_bytes_per_round(*, K: int, P_: int, d: int = 0) -> int:
    """Uplink per client-round: K*P gradient scalars (+ the K*P seeds, counted pessimistically).

    `d` is accepted and deliberately ignored — dimension-freedom is the paper's entire claim,
    and a test pins that this function does not depend on it.
    """
    return K * P_ * (FLOAT_BYTES + SEED_BYTES)


def fedavg_bytes_per_round(d: int) -> int:
    """Uplink per client-round: the full float32 parameter vector."""
    return d * FLOAT_BYTES


# ------------------------------------------------------------------------------ partitioning

def partition(labels, num_clients: int, alpha: float, seed: int, min_per_client: int = 1):
    """Dirichlet label-skew partition. Every example lands with exactly one client.

    At severe skew (alpha <= ~0.1) a Dirichlet draw can leave a client with ZERO examples, which
    makes DataLoader raise `num_samples should be a positive integer value, but got
    num_samples=0` and aborts the run. That is a harness failure rather than a meaningful
    experimental condition -- a federated round has no notion of a client with no data -- so
    empty clients are topped up to `min_per_client` by moving examples from the largest client.

    The correction is deliberately minimal: it moves the fewest examples needed, takes them from
    the most over-provisioned client, and leaves the label skew that low alpha is there to
    produce (pinned by test_partition_still_produces_label_skew_at_low_alpha).
    """
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    idx_by_class = [np.where(labels == c)[0] for c in np.unique(labels)]
    parts = [[] for _ in range(num_clients)]
    for idx in idx_by_class:
        rng.shuffle(idx)
        p = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for c, chunk in enumerate(np.split(idx, cuts)):
            parts[c].extend(chunk.tolist())

    for _ in range(num_clients * min_per_client):
        short = [c for c in range(num_clients) if len(parts[c]) < min_per_client]
        if not short:
            break
        donor = max(range(num_clients), key=lambda c: len(parts[c]))
        if len(parts[donor]) <= min_per_client:
            break                      # nothing left to give; caller has too few examples
        parts[short[0]].append(parts[donor].pop())

    return [np.array(sorted(p), dtype=np.int64) for p in parts]


# ------------------------------------------------------------------------------- evaluation

def _auc(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Binary AUC from the margin; macro one-vs-rest for k>2 (same convention as the frozen
    harness, whose binary path this reproduces exactly)."""
    k = logits.shape[1]
    if k == 2:
        return _binary_auc((logits[:, 1] - logits[:, 0]).double(), y == 1)
    vals = []
    for c in range(k):
        rest = torch.cat([logits[:, :c], logits[:, c + 1:]], dim=1).max(dim=1).values
        v = _binary_auc((logits[:, c] - rest).double(), y == c)
        if v is not None:
            vals.append(v)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _binary_auc(score: torch.Tensor, pos: torch.Tensor):
    npos = int(pos.sum())
    nneg = int((~pos).sum())
    if npos == 0 or nneg == 0:
        return None
    order = torch.argsort(score)
    ranks = torch.empty_like(order, dtype=torch.double)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.double)
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@torch.no_grad()
def _evaluate(model, x, y):
    """Metrics are computed on CPU regardless of the compute device.

    Two reasons, one of them a bug this crashed on: MPS cannot represent float64, so the AUC
    path's `.double()` raised `Cannot convert a MPS Tensor to float64` and killed the whole
    sweep. Moving to CPU also makes every metric bit-identical across cpu/mps/cuda, which a
    cross-device benchmark needs anyway &mdash; otherwise a device difference in the metric
    would be indistinguishable from a device difference in the training.
    """
    model.eval()
    logits = model(x).detach().cpu()
    y = y.detach().cpu()
    loss = float(nn.functional.cross_entropy(logits, y))
    return {"loss": loss, "auc": _auc(logits, y),
            "acc": float((logits.argmax(1) == y).double().mean())}


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


# --------------------------------------------------------------------- gradient alignment probe

def gradient_alignment(model: nn.Module, x, y, *, P_: int, mu: float, seed: int) -> float:
    """cos(g_hat, g_true) for the SAME estimator DeComFL uses.

    This is the mechanism behind whatever the training curves show: the ZO estimate's variance
    grows like d/P, so at fixed query budget its alignment with the true gradient decays like
    sqrt(P/d). Measuring it alongside the curves means a collapse in accuracy can be attributed
    rather than guessed at.
    """
    model = model.to("cpu")
    crit = nn.CrossEntropyLoss()
    model.zero_grad()
    crit(model(x), y).backward()
    g_true = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.requires_grad])

    est = ZerothOrderEstimator(smoothing_param=mu, device="cpu")
    flat = P.flat_params(model)
    d = flat.numel()
    g_hat = torch.zeros(d, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    # Single base point for the whole alignment probe, so f(x) is evaluated once.
    base_loss = est.compute_base_loss(model, flat, x, y)
    for _ in range(P_):
        s = int(rng.integers(0, 2 ** 31 - 1))
        z = est.generate_perturbation(s, d)
        g = est.compute_gradient_scalar(model, flat, z, x, y, base_loss=base_loss)
        g_hat += g * z
    g_hat /= P_
    P.set_flat_params(model, flat)

    denom = float(g_true.norm()) * float(g_hat.norm())
    if denom == 0:
        return float("nan")
    return float(torch.dot(g_true.double(), g_hat.double()) / denom)


# ---------------------------------------------------------------------------------- the arms

def _make_loaders(train_x, train_y, parts, batch_size, seed):
    out = []
    for i, idx in enumerate(parts):
        g = torch.Generator().manual_seed(seed * 1000 + i)
        ds = torch.utils.data.TensorDataset(train_x[idx], train_y[idx])
        bs = min(batch_size, max(1, len(ds)))
        out.append(torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, generator=g,
                                               drop_last=False))
    return out


def run_decomfl(*, train_x, train_y, test_x, test_y, feat_dim, n_classes, hidden, clients,
                clients_per_round, alpha, rounds, K, P_, lr, mu, batch_size, seed,
                device="cpu", check_rebuild=False, eval_every=1, align_every=0):
    """A faithful local simulation of the real DeComFL protocol.

    Server side is the framework's `DeComFL` strategy verbatim. Client side mirrors
    `DeComFLClient.fit` (Algorithm 4, lines 16-24): perturb along the shared-seed z, take the
    forward-difference scalar, step locally, and revert the whole local excursion at the end.
    """
    t0 = time.time()
    torch.manual_seed(seed)
    model = head_model(feat_dim=feat_dim, n_classes=n_classes, hidden=hidden, seed=seed).to(device)
    d = model_dim(model)
    sha = init_sha(model)

    # allow_unstable_lr: this harness EXISTS to probe the divergent regime — stability_ladder.json
    # and mu_eta_dimension_scaling.json are records of runs that blew up on purpose, and the
    # strategy's learning-rate guard (which those very results calibrate) would otherwise refuse to
    # reproduce them. The guard still logs, so an unintended sweep into that regime stays visible.
    strategy = DeComFL(initial_parameters=P.trainable_state(model), min_fit_clients=1,
                       clients_per_round=clients_per_round, num_local_steps=K,
                       num_perturbations=P_, learning_rate=lr, smoothing_param=mu, seed=seed,
                       allow_unstable_lr=True)
    strategy.device = device
    strategy.global_params_flat = strategy.global_params_flat.to(device)
    strategy.validate_participant_dim(d, "sim")

    est = ZerothOrderEstimator(smoothing_param=mu, device=device)
    parts = partition(train_y.numpy(), clients, alpha, seed)
    loaders = _make_loaders(train_x, train_y, parts, batch_size, seed)
    iters = [iter(l) for l in loaders]
    rng = np.random.RandomState(seed + 77)

    per_round, rebuild_err = [], 0.0
    bpr = decomfl_bytes_per_round(K=K, P_=P_, d=d)

    for r in range(1, rounds + 1):
        seeds = strategy.get_or_create_seeds(r)
        chosen = rng.choice(clients, size=min(clients_per_round, clients), replace=False)
        results = []
        x_global = strategy.global_params_flat.clone()

        for ci in chosen:
            x_cur = x_global.clone()
            total_pert = torch.zeros_like(x_cur)
            grads = []
            for k in range(K):
                try:
                    xb, yb = next(iters[ci])
                except StopIteration:
                    iters[ci] = iter(loaders[ci])
                    xb, yb = next(iters[ci])
                xb, yb = xb.to(device), yb.to(device)
                k_grads, delta = [], torch.zeros_like(x_cur)
                # One base loss per local step, reused across the P perturbations — mirrors
                # DeComFLClient.fit, so the measured cost is P+1 forwards per step, not 2P.
                base_loss = est.compute_base_loss(model, x_cur, xb, yb)
                for p in range(P_):
                    z = est.generate_perturbation(seeds[k][p], len(x_cur))
                    g = est.compute_gradient_scalar(
                        model, x_cur, z, xb, yb, base_loss=base_loss
                    )
                    k_grads.append(g)
                    delta += g * z
                step = (lr / P_) * delta
                x_cur -= step
                total_pert -= step
                grads.append(k_grads)
            # Algorithm 4: revert the local excursion — the client keeps only the scalars.
            x_cur -= total_pert
            results.append((f"c{ci}", grads, len(loaders[ci].dataset)))

        strategy.aggregate_fit(r, results)

        if check_rebuild:
            # Independently replay the round from the shared seeds + averaged scalars and
            # compare against the server's own vector. Any drift here means the run is not
            # DeComFL and its numbers mean nothing.
            replay = x_global.clone()
            for k in range(K):
                dl = torch.zeros_like(replay)
                for p in range(P_):
                    z = est.generate_perturbation(seeds[k][p], len(replay))
                    dl += sum(g[k][p] for _, g, _ in results) * z
                replay -= lr * dl / (len(results) * P_)
            rebuild_err = max(rebuild_err,
                              float((replay - strategy.global_params_flat).abs().max()))

        if r % eval_every == 0 or r == rounds:
            P.set_flat_params(model, strategy.global_params_flat)
            m = _evaluate(model, test_x.to(device), test_y.to(device))
            row = {"round": r, **m, "cum_bytes": bpr * len(results) * r}
            if align_every and (r % align_every == 0):
                row["cos_alignment"] = gradient_alignment(
                    head_model(feat_dim=feat_dim, n_classes=n_classes, hidden=hidden, seed=seed),
                    train_x[:256], train_y[:256], P_=P_, mu=mu, seed=seed)
            per_round.append(row)

    P.set_flat_params(model, strategy.global_params_flat)
    return _summarize("DeComFL", per_round, d=d, sha=sha, bpr=bpr, rounds=rounds, t0=t0,
                      hidden=hidden, seed=seed, rebuild_err=rebuild_err if check_rebuild else None,
                      hp={"K": K, "P": P_, "lr": lr, "mu": mu, "batch_size": batch_size})


def run_fedavg(*, train_x, train_y, test_x, test_y, feat_dim, n_classes, hidden, clients,
               clients_per_round, alpha, rounds, local_epochs, lr, batch_size, seed,
               device="cpu", eval_every=1):
    """The first-order control, on the same init, data, partition and schedule."""
    t0 = time.time()
    torch.manual_seed(seed)
    model = head_model(feat_dim=feat_dim, n_classes=n_classes, hidden=hidden, seed=seed).to(device)
    d = model_dim(model)
    sha = init_sha(model)

    strategy = FedAvg(initial_parameters=P.trainable_state(model), min_fit_clients=1)
    global_state = P.trainable_state(model)
    parts = partition(train_y.numpy(), clients, alpha, seed)
    loaders = _make_loaders(train_x, train_y, parts, batch_size, seed)
    rng = np.random.RandomState(seed + 77)
    crit = nn.CrossEntropyLoss()

    per_round = []
    bpr = fedavg_bytes_per_round(d)

    for r in range(1, rounds + 1):
        chosen = rng.choice(clients, size=min(clients_per_round, clients), replace=False)
        results = []
        for ci in chosen:
            local = head_model(feat_dim=feat_dim, n_classes=n_classes, hidden=hidden,
                               seed=seed).to(device)
            local.load_state_dict(global_state, strict=False)
            opt = torch.optim.SGD([p for p in local.parameters() if p.requires_grad], lr=lr)
            local.train()
            for _ in range(local_epochs):
                for xb, yb in loaders[ci]:
                    opt.zero_grad()
                    crit(local(xb.to(device)), yb.to(device)).backward()
                    opt.step()
            results.append((P.trainable_state(local), len(loaders[ci].dataset)))

        agg = strategy.aggregate_fit(r, results)
        if agg is not None:
            global_state = agg

        if r % eval_every == 0 or r == rounds:
            model.load_state_dict(global_state, strict=False)
            m = _evaluate(model, test_x.to(device), test_y.to(device))
            per_round.append({"round": r, **m, "cum_bytes": bpr * len(results) * r})

    return _summarize("FedAvg", per_round, d=d, sha=sha, bpr=bpr, rounds=rounds, t0=t0,
                      hidden=hidden, seed=seed, rebuild_err=None,
                      hp={"local_epochs": local_epochs, "lr": lr, "batch_size": batch_size})


def _summarize(arm, per_round, *, d, sha, bpr, rounds, t0, hidden, seed, rebuild_err, hp):
    best = max((p["auc"] for p in per_round if np.isfinite(p["auc"])), default=float("nan"))
    out = {
        "arm": arm, "d": d, "hidden": hidden, "seed": seed, "rounds": rounds,
        "init_sha": sha, "bytes_per_client_round": bpr,
        "cum_bytes": per_round[-1]["cum_bytes"] if per_round else 0,
        "final_auc": per_round[-1]["auc"] if per_round else float("nan"),
        "final_loss": per_round[-1]["loss"] if per_round else float("nan"),
        "final_acc": per_round[-1]["acc"] if per_round else float("nan"),
        "best_auc": best, "hyperparams": hp,
        "wall_seconds": round(time.time() - t0, 2), "peak_rss_mb": round(peak_rss_mb(), 1),
        "per_round": per_round,
    }
    if rebuild_err is not None:
        out["rebuild_max_abs_error"] = rebuild_err
    return out


# ------------------------------------------------------------------------------------- io

def _emit(out_dir: str, rec: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    name = f"{rec['arm']}_d{rec['d']}_h{rec.get('hidden', 0)}_seed{rec.get('seed', 0)}.json"
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return path


def auc_at_bytes(rec: dict, budget: int) -> float:
    """Best AUC reached without exceeding a cumulative uplink budget — the mobile-link metric.
    Returns nan if the arm never transmitted that little (i.e. round 1 already exceeded it)."""
    vals = [p["auc"] for p in rec["per_round"] if p["cum_bytes"] <= budget and np.isfinite(p["auc"])]
    return max(vals) if vals else float("nan")


def load_features(path: str):
    d = np.load(path)
    return (torch.from_numpy(d["train_x"]).float(), torch.from_numpy(d["train_y"]).long(),
            torch.from_numpy(d["test_x"]).float(), torch.from_numpy(d["test_y"]).long())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="npz with train_x/train_y/test_x/test_y")
    ap.add_argument("--hidden", default="0", help="comma list of hidden widths (0 = linear head)")
    ap.add_argument("--arms", default="DeComFL,FedAvg")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--decomfl-rounds", type=int, default=0,
                    help="separate (larger) budget for the ZO arm; 0 = same as --rounds")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--clients-per-round", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--lr-fedavg", type=float, default=0.05)
    ap.add_argument("--lr-decomfl", type=float, default=0.01)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--mu", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--align-every", type=int, default=0)
    ap.add_argument("--check-rebuild", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    train_x, train_y, test_x, test_y = load_features(a.features)
    feat_dim = train_x.shape[1]
    n_classes = int(train_y.max()) + 1
    hiddens = [int(h) for h in a.hidden.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    arms = a.arms.split(",")
    cell_dir = os.path.splitext(a.out)[0] + "_cells"

    runs = []
    for h in hiddens:
        for s in seeds:
            common = dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
                          feat_dim=feat_dim, n_classes=n_classes, hidden=h, clients=a.clients,
                          clients_per_round=a.clients_per_round, alpha=a.alpha,
                          batch_size=a.batch_size, seed=s, device=a.device,
                          eval_every=a.eval_every)
            if "FedAvg" in arms:
                r = run_fedavg(rounds=a.rounds, local_epochs=a.local_epochs,
                               lr=a.lr_fedavg, **common)
                _emit(cell_dir, r); runs.append(r)
                print(f"[FedAvg ] d={r['d']:<9} h={h:<5} seed={s} "
                      f"auc={r['final_auc']:.4f} bytes={r['cum_bytes']:,} {r['wall_seconds']}s",
                      flush=True)
            if "DeComFL" in arms:
                r = run_decomfl(rounds=a.decomfl_rounds or a.rounds, K=a.K, P_=a.P,
                                lr=a.lr_decomfl, mu=a.mu, check_rebuild=a.check_rebuild,
                                align_every=a.align_every, **common)
                _emit(cell_dir, r); runs.append(r)
                print(f"[DeComFL] d={r['d']:<9} h={h:<5} seed={s} "
                      f"auc={r['final_auc']:.4f} bytes={r['cum_bytes']:,} {r['wall_seconds']}s"
                      + (f" rebuild_err={r['rebuild_max_abs_error']:.2e}"
                         if "rebuild_max_abs_error" in r else ""), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({
            "experiment": "DeComFL vs FedAvg as a function of model dimension",
            "meta": {"host": platform.node(), "platform": platform.platform(),
                     "device": a.device, "torch": torch.__version__,
                     "python": platform.python_version(), "features": a.features,
                     "feat_dim": feat_dim, "n_classes": n_classes, "clients": a.clients,
                     "clients_per_round": a.clients_per_round, "alpha": a.alpha,
                     "seeds": seeds, "hidden_widths": hiddens,
                     "rounds_fedavg": a.rounds, "rounds_decomfl": a.decomfl_rounds or a.rounds,
                     "K": a.K, "P": a.P, "mu": a.mu, "lr_fedavg": a.lr_fedavg,
                     "lr_decomfl": a.lr_decomfl, "local_epochs": a.local_epochs,
                     "batch_size": a.batch_size},
            "runs": runs,
        }, f, indent=2)
    print(f"\nwrote {a.out}  ({len(runs)} runs, cells in {cell_dir})")


if __name__ == "__main__":
    main()
