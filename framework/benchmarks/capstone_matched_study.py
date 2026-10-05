"""Capstone matched study: privacy x robustness on a head over frozen chest X-ray features.

Four cells share everything except two switches (the capstone report, Table I):

    Q00  no privacy noise, clipped mean        Q10  privacy noise, clipped mean
    Q01  no privacy noise, trimmed mean        Q11  privacy noise, trimmed mean

Every cell clips each client's update to L2 norm C (common clipping), aggregates unweighted, and
calls ``fedlearn.privacy.dp_mechanism.dp_robust_aggregate``; the noise-free cells pass z = 0. Under
replace-one adjacency the noise std is z * 2C/n for the mean and z * 2C/(n - 2k) for the trimmed
mean, so one accountant z per quartet gives every private cell the same (epsilon, delta).

Matched by construction, and recorded so it can be checked:
  * clients hold DISJOINT shards from ``fedlearn.simulation.partition`` (no bootstrap overlap);
  * the partition, head initialisation, attacker identities and noise draw each have their own
    seed; local training is full-batch gradient descent and therefore deterministic;
  * ``round0_upload_digest`` hashes the uploads (after the attack, before clipping) and must be
    identical across the four cells of a quartet.

The attack is IPM (Xie et al.) at a FIXED scale, applied to the uploads BEFORE the server's common
clipping, via the same ``_apply_attack_to_round`` the breakdown harness uses. A fixed scale is
required: the per-fraction calibration used by that harness tunes the attack against FedAvg.

This uses its own head-only round loop rather than ``SimulatedFederation``: IPM is omniscient (each
round's poisoned upload is computed from the honest uploads), which the simulator's per-client
factory cannot express without changing the production coordinator.

Usage:
    python benchmarks/capstone_matched_study.py --data-dir ~/fedlearn-demo/chest_xray \\
        --clients 80 --rounds 8 --target-epsilon 1 --attack-fraction 0.2 --ipm-epsilon 4 \\
        --reps 5 --out results/capstone_quartets.json
    (or --features <cached features .pt> to skip extraction)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from benchmarks import robust_aggregation_attack as raa  # noqa: E402
from benchmarks.dp_on_head import accounted_epsilon as _accounted_epsilon  # noqa: E402
from benchmarks.dp_on_head import solve_noise_multiplier  # noqa: E402
from benchmarks.dp_on_head_xray import _binary_auc, _Head, extract_features  # noqa: E402
from fedlearn.communication.safetensors_codec import save_safetensors  # noqa: E402
from fedlearn.estimators.params import trainable_state  # noqa: E402
from fedlearn.privacy.dp_mechanism import dp_robust_aggregate, replace_one_sensitivity  # noqa: E402
from fedlearn.server.subset_federation import apply_trainable_subset, guard_client_updates  # noqa: E402
from fedlearn.simulation.partition import dirichlet_partition, iid_partition  # noqa: E402
from fedlearn.simulation.rng import torch_rng_scope  # noqa: E402

# cell -> (privacy noise on, aggregation rule)
CELLS = {
    "Q00": (False, "mean"),
    "Q10": (True, "mean"),
    "Q01": (False, "trimmed_mean"),
    "Q11": (True, "trimmed_mean"),
}


def balance_classes(x: torch.Tensor, y: torch.Tensor, seed: int):
    """Down-sample every class to the minority class's count (seeded); the minority is kept whole.

    Kermany's training split is about 1,341 NORMAL to 3,875 PNEUMONIA; balancing keeps chance
    accuracy at 1/n_classes, which the retained-lift metric assumes.
    """
    g = torch.Generator().manual_seed(seed)
    counts = torch.bincount(y)
    m = int(counts[counts > 0].min())
    keep = []
    for c in range(len(counts)):
        idx = (y == c).nonzero(as_tuple=True)[0]
        if len(idx) > m:
            idx = idx[torch.randperm(len(idx), generator=g)[:m]]
        keep.append(idx)
    keep = torch.sort(torch.cat(keep)).values
    return x[keep], y[keep]


def make_partition(labels, clients: int, seed: int, kind: str = "iid", alpha: float = 0.5):
    """Disjoint, complete, seeded client shards (index arrays) from ``fedlearn.simulation.partition``."""
    if kind == "iid":
        return iid_partition(len(labels), clients, seed)
    if kind == "dirichlet":
        return dirichlet_partition(labels, clients, alpha, seed, min_partition_size=1)
    raise ValueError(f"partition must be 'iid' or 'dirichlet', got {kind!r}")


def _digest(states) -> str:
    h = hashlib.sha256()
    for state in states:
        for k in sorted(state):
            h.update(k.encode())
            h.update(state[k].detach().float().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _wire_bytes(state) -> int:
    return len(save_safetensors([(k, v.detach().float().numpy()) for k, v in state.items()]))


def run_cell(features, *, cell, clients, rounds, local_epochs, lr, clip, trim_ratio, noise_multiplier,
             partition_seed, init_seed, attack_seed, noise_seed, attack=None, attack_fraction=0.0,
             ipm_epsilon=None, partition="iid", alpha=0.5, target_epsilon=None, delta=None,
             accounted_epsilon=None):
    """Run one cell of the matched design and return its record (see module docstring)."""
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {sorted(CELLS)}, got {cell!r}")
    if attack not in (None, "ipm"):
        raise ValueError(f"only the IPM attack is supported, got {attack!r}")
    if attack == "ipm" and ipm_epsilon is None:
        raise ValueError("IPM needs a fixed --ipm-epsilon; per-fraction calibration is not matched.")
    privacy, rule = CELLS[cell]
    trim = trim_ratio if rule == "trimmed_mean" else 0.0
    z = float(noise_multiplier) if privacy else 0.0
    sensitivity = replace_one_sensitivity(rule, clients, clip, trim)

    train_x, train_y = features["train_x"], features["train_y"]
    test_x, test_y = features["test_x"], features["test_y"]
    feat_dim, n_classes = features["feat_dim"], features["n_classes"]
    parts = make_partition(train_y.numpy(), clients, partition_seed, partition, alpha)

    with torch_rng_scope(init_seed):
        server = _Head(feat_dim, n_classes)
    net = _Head(feat_dim, n_classes)

    n_attackers = int(round(attack_fraction * clients)) if attack else 0
    attacker_ids = sorted(
        int(i) for i in np.random.default_rng(attack_seed).choice(clients, n_attackers, replace=False)
    )
    generator = torch.Generator().manual_seed(noise_seed)

    def evaluate(m):
        m.eval()
        with torch.no_grad():
            logits = m(test_x)
        acc = (logits.argmax(1) == test_y).float().mean().item()
        auc = _binary_auc(torch.softmax(logits, 1)[:, 1], test_y) if n_classes == 2 else None
        return acc, auc

    def train_head(cx, cy):
        opt = torch.optim.SGD(net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        net.train()
        for _ in range(local_epochs):
            opt.zero_grad()
            loss_fn(net(cx), cy).backward()
            opt.step()

    t0 = time.perf_counter()
    accs, aucs, comm_bytes, round0_digest = [], [], 0, None
    for rnd in range(rounds):
        global_head = trainable_state(server)
        records = {}
        for cid in range(clients):
            apply_trainable_subset(net, global_head)
            idx = torch.as_tensor(parts[cid], dtype=torch.long)
            train_head(train_x[idx], train_y[idx])
            state = trainable_state(net)
            records[cid] = {
                "n_examples": len(idx),
                "client_state": state,
                "honest_delta": {k: state[k] - global_head[k] for k in state},
            }
        overrides = raa._apply_attack_to_round(
            attack="ipm", attacker_ids=set(attacker_ids), client_records=records,
            global_params=global_head, attack_scale_signed=None, ipm_epsilon=ipm_epsilon, alie_z=None,
        ) if attacker_ids else {}
        uploads = [(f"c{cid}", overrides.get(cid, records[cid]["client_state"]), records[cid]["n_examples"])
                   for cid in range(clients)]
        guard_client_updates([u for _, u, _ in uploads], server)
        if rnd == 0:
            round0_digest = _digest(u for _, u, _ in uploads)
        comm_bytes += clients * (_wire_bytes(global_head) + _wire_bytes(uploads[0][1]))

        agg = dp_robust_aggregate(uploads, global_head, list(global_head.keys()), clip_norm=clip,
                                  noise_multiplier=z, rule=rule, trim_ratio=trim, generator=generator)
        apply_trainable_subset(server, agg)
        acc, auc = evaluate(server)
        accs.append(acc)
        aucs.append(auc)

    return {
        "cell": cell, "privacy": privacy, "rule": rule,
        "clients": clients, "rounds": rounds, "local_epochs": local_epochs, "lr": lr,
        "clip": clip, "trim_ratio": trim, "partition": partition,
        "alpha": alpha if partition == "dirichlet" else None,
        "client_sizes": [len(p) for p in parts],
        "noise_multiplier": float(noise_multiplier), "noise_applied": privacy,
        "sensitivity": sensitivity, "noise_std": z * sensitivity,
        "target_epsilon": target_epsilon, "delta": delta, "accounted_epsilon": accounted_epsilon,
        "attack": attack, "attack_fraction": attack_fraction if attack else 0.0,
        "ipm_epsilon": ipm_epsilon if attack else None, "attacker_ids": attacker_ids,
        "seeds": {"partition": partition_seed, "init": init_seed, "attack": attack_seed, "noise": noise_seed},
        "round0_upload_digest": round0_digest,
        "final_head_digest": _digest([trainable_state(server)]),
        "chance_accuracy": round(1.0 / n_classes, 4),
        "per_round_accuracy": [round(a, 6) for a in accs],
        "final_accuracy": round(accs[-1], 6),
        "final_auc": round(aucs[-1], 6) if aucs[-1] is not None else None,
        "comm_bytes_total": comm_bytes,
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }


def run_quartet(features, *, target_epsilon, delta, rounds, **kw):
    """All four cells at one shared accountant noise multiplier (no subsampling, q = 1)."""
    z = solve_noise_multiplier(target_epsilon, 1.0, rounds, delta)
    eps = _accounted_epsilon(1.0, z, rounds, delta)
    return [
        run_cell(features, cell=c, noise_multiplier=z, rounds=rounds, target_epsilon=target_epsilon,
                 delta=delta, accounted_epsilon=eps, **kw)
        for c in CELLS
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="ImageFolder root with train/ and test/ (features extracted + cached)")
    src.add_argument("--features", help="precomputed features .pt (train_x/train_y/test_x/test_y/feat_dim/n_classes)")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--no-balance", action="store_true", help="keep the raw class ratio")
    ap.add_argument("--clients", type=int, default=80)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=0.4)
    ap.add_argument("--trim-ratio", type=float, default=0.2)
    ap.add_argument("--target-epsilon", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--attack-fraction", type=float, default=0.2)
    ap.add_argument("--ipm-epsilon", type=float, default=4.0)
    ap.add_argument("--partition", choices=("iid", "dirichlet"), default="iid")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    if args.features:
        features = torch.load(args.features, weights_only=True)
        source = os.path.abspath(args.features)
    else:
        features = extract_features(args.data_dir, backbone=args.backbone)
        source = f"{os.path.abspath(args.data_dir)} ({args.backbone}, ImageNet weights)"
    if not args.no_balance:
        tx, ty = balance_classes(features["train_x"], features["train_y"], args.seed_base)
        vx, vy = balance_classes(features["test_x"], features["test_y"], args.seed_base + 1)
        features = {**features, "train_x": tx, "train_y": ty, "test_x": vx, "test_y": vy}

    records = []
    for rep in range(args.reps):
        seeds = dict(partition_seed=args.seed_base + rep, init_seed=args.seed_base + rep,
                     attack_seed=args.seed_base + 1000 + rep, noise_seed=args.seed_base + 2000 + rep)
        common = dict(clients=args.clients, rounds=args.rounds, local_epochs=args.local_epochs, lr=args.lr,
                      clip=args.clip, trim_ratio=args.trim_ratio, partition=args.partition,
                      alpha=args.alpha, target_epsilon=args.target_epsilon, delta=args.delta, **seeds)
        for attacked in (False, True):
            attack = dict(attack="ipm", attack_fraction=args.attack_fraction,
                          ipm_epsilon=args.ipm_epsilon) if attacked else {}
            for rec in run_quartet(features, **common, **attack):
                rec["rep"] = rep
                records.append(rec)
                print(f"rep {rep} {'attacked' if attacked else 'clean   '} {rec['cell']}: "
                      f"acc {rec['final_accuracy']:.4f}", flush=True)

    meta = {
        "study": "capstone matched privacy x robustness quartets",
        "harness": "framework/benchmarks/capstone_matched_study.py",
        "torch_version": torch.__version__, "platform": platform.platform(),
        "host": platform.node(), "features": source, "balanced": not args.no_balance,
        "n_train": int(features["train_x"].shape[0]), "n_test": int(features["test_x"].shape[0]),
        "feat_dim": int(features["feat_dim"]), "n_classes": int(features["n_classes"]),
        "clients": args.clients, "rounds": args.rounds, "local_epochs": args.local_epochs, "lr": args.lr,
        "clip": args.clip, "trim_ratio": args.trim_ratio, "target_epsilon": args.target_epsilon,
        "delta": args.delta, "sampling_rate_q": 1.0, "adjacency": "replace-one",
        "attack": "ipm", "attack_fraction": args.attack_fraction, "ipm_epsilon": args.ipm_epsilon,
        "attack_placement": "before common server-side clipping",
        "partition": args.partition, "alpha": args.alpha if args.partition == "dirichlet" else None,
        "seed_base": args.seed_base, "reps": args.reps,
        "total_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "records": records}, fh, indent=2)
    print(f"[*] wrote {args.out} ({len(records)} records, {meta['total_seconds']}s)")


if __name__ == "__main__":
    main()
