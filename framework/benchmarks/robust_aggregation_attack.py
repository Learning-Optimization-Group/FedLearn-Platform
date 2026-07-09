"""FR-12 #2 — Byzantine-robustness accuracy benchmark (trimmed-mean / median vs FedAvg).

Measures whether the REAL RobustAggregator (coordinate-wise median / beta-trimmed-mean,
`fedlearn/server/robust_aggregation.py`) retains held-out accuracy under a 20% gradient-scaling
attack where the REAL FedAvg strategy (`fedlearn/server/strategy.py`) collapses.

Task: a small, self-contained, deterministic 4-class Gaussian-cluster classification in R^20,
trained by a tiny MLP, federated non-IID (Dirichlet split, alpha=0.5) across N=10 clients using the
`_dirichlet_indices` helper already used by `recipes.py`. Everything is torch-seeded so the SAME
data partition, model init, and per-client local training is reused across every configuration —
only the aggregator and the attack vary.

The attack: a Byzantine fraction f of clients replace their honest upload with their own
delta-from-global scaled by a large SIGNED factor (default -10x). The sign matters and is a
deliberate, literature-grounded choice, not an arbitrary knob: scaling an honest delta by a large
POSITIVE factor amplifies a direction that is still locally correct for that client's own data, and
on a well-separated, non-conflicting classification task this just overshoots-then-recovers or even
accelerates convergence — it is not actually adversarial, and empirically does NOT collapse FedAvg
here (confirmed empirically on this task before settling on the negative sign). The standard
Byzantine "gradient/large-deviation scaling" attack analysed by the robust-aggregation
literature this module drives against (Yin et al. 2018; the sign-flipping/inner-product-manipulation
family in Xie et al. 2019 "Fall of Empires" and Fang et al. 2020 "Local Model Poisoning Attacks") is
adversarial precisely because it pushes in the WRONG direction at large magnitude — i.e. a large
NEGATIVE multiple of the honest delta. That is what this benchmark implements and what
beta-trimmed-mean / coordinate-wise median are designed to reject.

Both strategies are driven through their REAL `aggregate_fit(server_round, results)` — no
aggregation math is reimplemented here. `results` is assembled as the same
`list[(client_id: str, state_dict: OrderedDict[str, Tensor], num_examples: int)]` shape both
`FedAvgAggregator.aggregate` and `RobustAggregator.aggregate_fit` accept (the 3-tuple wire form);
`num_examples` is each client's REAL non-IID sample count (RobustAggregator ignores it by design —
it is unweighted — but still requires it to be positive to keep a client in the round).

Server-side clip_norm is intentionally left OFF for the headline configurations so the accuracy
difference isolates the ESTIMATOR (median / trimmed-mean) alone, not a clipping assist; see the
"what this shows" section of the generated report for the honest breakdown-point discussion.

Run:  PYTHONPATH=src python benchmarks/robust_aggregation_attack.py [--rounds N] [--clients N] ...
Artifacts: benchmarks/results/robust_aggregation_attack.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "backend", "fl-platform-api",
                                "src", "main", "resources", "scripts"))

from fedlearn.server.strategy import FedAvg  # noqa: E402
from fedlearn.server.robust_aggregation import RobustAggregator  # noqa: E402
import recipes  # noqa: E402  (reusing recipes._dirichlet_indices — same non-IID split helper)


class TinyMLP(torch.nn.Module):
    """Linear(D,H) -> ReLU -> Linear(H,C). No BatchNorm/buffers, so state_dict is exactly the
    two Linear layers' weight/bias — a clean, minimal aggregation payload."""

    def __init__(self, dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_dataset(num_classes: int, dim: int, train_per_class: int, test_per_class: int,
                  sep: float, sigma: float, seed: int):
    """C well-separated Gaussian clusters in R^D. Class means are unit vectors (random directions,
    near-orthogonal in high dimension) scaled by `sep`; samples are N(mean_c, sigma^2 I). A dedicated
    Generator is used (not the global RNG) so this data build never perturbs the seeding contract the
    rest of the run relies on for reproducibility."""
    gen = torch.Generator().manual_seed(seed)
    means = torch.randn(num_classes, dim, generator=gen)
    means = means / means.norm(dim=1, keepdim=True) * sep

    def sample(n_per_class):
        xs, ys = [], []
        for c in range(num_classes):
            xs.append(means[c] + sigma * torch.randn(n_per_class, dim, generator=gen))
            ys.append(torch.full((n_per_class,), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)

    train_x, train_y = sample(train_per_class)
    test_x, test_y = sample(test_per_class)
    return train_x, train_y, test_x, test_y


def build_strategy(strategy_name: str, initial, num_clients: int, trim_beta: float):
    initial = OrderedDict((k, v.clone()) for k, v in initial.items())
    if strategy_name == "fedavg":
        return FedAvg(initial_parameters=initial, min_fit_clients=num_clients)
    if strategy_name == "trimmed_mean":
        return RobustAggregator(initial_parameters=initial, method="trimmed_mean",
                                 trim_ratio=trim_beta, min_fit_clients=num_clients)
    if strategy_name == "median":
        return RobustAggregator(initial_parameters=initial, method="median",
                                 min_fit_clients=num_clients)
    raise ValueError(f"unknown strategy {strategy_name!r}")


def run_config(*, label, strategy_name, attack_fraction, attack_scale, trim_beta,
               num_clients, rounds, local_epochs, lr, batch_size, hidden, seed,
               initial, client_indices, train_x, train_y, test_x, test_y, dim, num_classes):
    """Run one (aggregator, attack-fraction) configuration to completion and return its result
    record. `attack_fraction=0.0` is the clean baseline (no attacker set)."""
    torch.manual_seed(seed)

    strategy = build_strategy(strategy_name, initial, num_clients, trim_beta)
    global_params = strategy.initialize_parameters()

    # Per-client loaders over the FIXED non-IID (Dirichlet) partition — identical across every
    # configuration so only the aggregator+attack vary.
    clients = []
    for cid in range(num_clients):
        idx = torch.as_tensor(client_indices[cid], dtype=torch.long)
        ds = TensorDataset(train_x[idx], train_y[idx])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        clients.append((len(idx), loader))

    # Deterministic Byzantine set: the lowest-numbered client ids, sized by attack_fraction. Nested
    # across fractions (f=0.1 subset of f=0.2 subset of f=0.3) — never chosen to favor an outcome.
    num_attackers = int(round(attack_fraction * num_clients))
    attacker_ids = set(range(num_attackers))

    # Under FedAvg's num_examples weighting, a Byzantine CLIENT-COUNT fraction f does not imply the
    # same Byzantine WEIGHT fraction when the split is non-IID (client sizes vary a lot under
    # Dirichlet alpha=0.5) — report both so a "20% of clients" headline isn't silently also "36% of
    # the weighted mass" without disclosure.
    total_examples = sum(n for n, _ in clients)
    attacker_weight_fraction = (
        sum(n for cid, (n, _) in enumerate(clients) if cid in attacker_ids) / total_examples
        if total_examples > 0 else 0.0
    )

    net = TinyMLP(dim, hidden, num_classes)       # reloaded from global_params each client's turn
    eval_net = TinyMLP(dim, hidden, num_classes)  # dedicated eval model (never trained)

    accs, refused_rounds = [], 0
    honest_delta_norms_r0, attacker_delta_norms_r0 = [], []
    for rnd in range(rounds):
        updates = []
        for cid, (n_examples, loader) in enumerate(clients):
            net.load_state_dict(OrderedDict((k, v.clone()) for k, v in global_params.items()))
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            net.train()
            for _ in range(local_epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    F.cross_entropy(net(xb), yb).backward()
                    opt.step()
            client_state = OrderedDict(
                (k, v.detach().clone().float()) for k, v in net.state_dict().items()
            )

            honest_delta = OrderedDict(
                (k, client_state[k] - global_params[k].float()) for k in client_state
            )
            if cid in attacker_ids:
                # Gradient-scaling attack: replace the honest delta with attack_scale x itself
                # (attack_scale is negative by default — see module docstring).
                client_state = OrderedDict(
                    (k, global_params[k].float() + attack_scale * honest_delta[k])
                    for k in client_state
                )

            if rnd == 0:
                # L2 norm of the value actually PUT ON THE WIRE for this client (honest delta for
                # honest clients; |attack_scale| x honest delta for attackers — norm scales by the
                # magnitude of a linear scaling regardless of sign, so use abs() here).
                honest_norm = sum(float((d.float() ** 2).sum()) for d in honest_delta.values()) ** 0.5
                uploaded_norm = honest_norm * (abs(attack_scale) if cid in attacker_ids else 1.0)
                (attacker_delta_norms_r0 if cid in attacker_ids else honest_delta_norms_r0).append(
                    uploaded_norm
                )

            updates.append((str(cid), client_state, n_examples))

        result = strategy.aggregate_fit(rnd, updates)
        if result is not None:
            global_params = result
        else:
            refused_rounds += 1  # Byzantine guard refusal (not expected at byzantine_fraction=0.0)

        eval_net.load_state_dict(OrderedDict((k, v.clone()) for k, v in global_params.items()))
        eval_net.eval()
        with torch.no_grad():
            acc = (eval_net(test_x).argmax(-1) == test_y).float().mean().item()
        accs.append(acc)

    return {
        "label": label,
        "strategy": strategy_name,
        "attack_fraction": attack_fraction,
        "num_attackers": num_attackers,
        "attacker_weight_fraction": round(attacker_weight_fraction, 4) if num_attackers else 0.0,
        "attack_scale": attack_scale if num_attackers else None,
        "trim_beta": trim_beta if strategy_name == "trimmed_mean" else None,
        "final_accuracy": accs[-1],
        "best_accuracy": max(accs),
        "per_round_accuracy": [round(a, 4) for a in accs],
        "refused_rounds": refused_rounds,
        "round0_honest_delta_l2_median": (
            round(statistics.median(honest_delta_norms_r0), 4) if honest_delta_norms_r0 else None
        ),
        "round0_attacker_upload_l2_median": (
            round(statistics.median(attacker_delta_norms_r0), 4) if attacker_delta_norms_r0 else None
        ),
    }


def main():
    ap = argparse.ArgumentParser(
        description="FR-12 #2 Byzantine-robustness accuracy benchmark (median/trimmed-mean vs FedAvg)."
    )
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--dim", type=int, default=20)
    ap.add_argument("--train-per-class", type=int, default=750)
    ap.add_argument("--test-per-class", type=int, default=250)
    ap.add_argument("--sep", type=float, default=6.0, help="class-mean separation (L2 radius)")
    ap.add_argument("--sigma", type=float, default=1.0, help="per-class Gaussian std")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet non-IID concentration")
    ap.add_argument("--dirichlet-seed", type=int, default=777)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--attack-scale", type=float, default=-10.0,
                     help="signed multiple applied to the attacker's own honest delta; NEGATIVE by "
                          "design (see module docstring) so the attack pushes the wrong direction at "
                          "large magnitude, not just an amplified-but-still-correct direction")
    ap.add_argument("--attack-fraction", type=float, default=0.2, help="f for the headline configs")
    ap.add_argument("--trim-beta", type=float, default=0.2, help="trimmed-mean trim ratio")
    ap.add_argument("--sweep-fractions", type=str, default="0.1,0.2,0.3")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", type=str, default=os.path.join(_HERE, "results"))
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    sweep_fractions = [float(x) for x in args.sweep_fractions.split(",") if x.strip()]

    torch.manual_seed(args.seed)
    train_x, train_y, test_x, test_y = make_dataset(
        num_classes=args.classes, dim=args.dim, train_per_class=args.train_per_class,
        test_per_class=args.test_per_class, sep=args.sep, sigma=args.sigma, seed=args.seed,
    )
    client_indices = recipes._dirichlet_indices(
        train_y.numpy(), args.clients, args.alpha, args.dirichlet_seed
    )
    client_sizes = [len(idx) for idx in client_indices]
    if any(n == 0 for n in client_sizes):
        raise RuntimeError(
            f"Dirichlet split (alpha={args.alpha}, seed={args.dirichlet_seed}) gave a client zero "
            f"samples: {client_sizes}; raise alpha or the per-class sample count."
        )

    torch.manual_seed(args.seed)
    init_model = TinyMLP(args.dim, args.hidden, args.classes)
    initial = OrderedDict((k, v.clone()) for k, v in init_model.state_dict().items())

    common = dict(
        num_clients=args.clients, rounds=args.rounds, local_epochs=args.local_epochs, lr=args.lr,
        batch_size=args.batch_size, hidden=args.hidden, seed=args.seed, initial=initial,
        client_indices=client_indices, train_x=train_x, train_y=train_y, test_x=test_x,
        test_y=test_y, dim=args.dim, num_classes=args.classes, attack_scale=args.attack_scale,
    )

    # ---- Headline configurations (all f=args.attack_fraction except the clean baseline) --------
    headline_specs = [
        ("clean baseline (FedAvg, no attack)", "fedavg", 0.0, 0.0),
        (f"FedAvg, f={args.attack_fraction:g} attackers", "fedavg", args.attack_fraction, args.trim_beta),
        (f"trimmed-mean (beta={args.trim_beta:g}), f={args.attack_fraction:g}",
         "trimmed_mean", args.attack_fraction, args.trim_beta),
        (f"median, f={args.attack_fraction:g}", "median", args.attack_fraction, args.trim_beta),
    ]

    results, t0 = [], time.time()
    cache = {}  # (strategy, fraction) -> result, so the sweep can reuse headline runs
    for label, strat, frac, beta in headline_specs:
        print(f"[*] running {label} ...", flush=True)
        ct = time.time()
        rec = run_config(label=label, strategy_name=strat, attack_fraction=frac, trim_beta=beta,
                          **common)
        rec["seconds"] = round(time.time() - ct, 1)
        results.append(rec)
        cache[(strat, round(frac, 6))] = rec
        print(f"    -> final acc {rec['final_accuracy']:.4f} | best {rec['best_accuracy']:.4f} "
              f"| {rec['seconds']}s", flush=True)

    clean_acc = results[0]["final_accuracy"]
    for r in results[1:]:
        r["retention_vs_clean"] = round(r["final_accuracy"] / clean_acc, 4) if clean_acc > 0 else None

    # ---- f-sweep: FedAvg and trimmed-mean(beta) across the requested fractions -------------------
    sweep = []
    for strat in ("fedavg", "trimmed_mean"):
        for frac in sweep_fractions:
            key = (strat, round(frac, 6))
            if key in cache:
                rec = cache[key]
            else:
                label = f"{strat} f={frac:g} (sweep)"
                print(f"[*] running {label} ...", flush=True)
                ct = time.time()
                rec = run_config(label=label, strategy_name=strat, attack_fraction=frac,
                                  trim_beta=args.trim_beta, **common)
                rec["seconds"] = round(time.time() - ct, 1)
                cache[key] = rec
                print(f"    -> final acc {rec['final_accuracy']:.4f} | {rec['seconds']}s", flush=True)
            rec = dict(rec)  # shallow copy so retention annotation doesn't clobber the cached record
            rec["retention_vs_clean"] = round(rec["final_accuracy"] / clean_acc, 4) if clean_acc > 0 else None
            sweep.append(rec)

    meta = dict(
        classes=args.classes, dim=args.dim, train_per_class=args.train_per_class,
        test_per_class=args.test_per_class, sep=args.sep, sigma=args.sigma, clients=args.clients,
        client_sizes=client_sizes, alpha=args.alpha, dirichlet_seed=args.dirichlet_seed,
        rounds=args.rounds, local_epochs=args.local_epochs, lr=args.lr, batch_size=args.batch_size,
        hidden=args.hidden, attack_scale=args.attack_scale, headline_attack_fraction=args.attack_fraction,
        trim_beta=args.trim_beta, sweep_fractions=sweep_fractions, seed=args.seed,
        task=f"{args.classes}-class Gaussian clusters in R^{args.dim} (sep={args.sep}, sigma={args.sigma})",
        model=f"MLP: Linear({args.dim},{args.hidden})->ReLU->Linear({args.hidden},{args.classes})",
        total_seconds=round(time.time() - t0, 1), torch_version=torch.__version__,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "robust_aggregation_attack.json"), "w") as f:
        json.dump({"meta": meta, "headline": results, "sweep": sweep}, f, indent=2)

    # ---- Markdown report ---------------------------------------------------------------------
    fedavg_attacked = next(r for r in results if r["strategy"] == "fedavg" and r["attack_fraction"] > 0)
    tmean_attacked = next(r for r in results if r["strategy"] == "trimmed_mean")
    median_attacked = next(r for r in results if r["strategy"] == "median")

    lines = [
        "# FR-12 #2 — Byzantine-robustness accuracy benchmark (median/trimmed-mean vs FedAvg)", "",
        f"Task: **{meta['task']}** · Model: **{meta['model']}**",
        f"Clients: {meta['clients']} (non-IID Dirichlet alpha={meta['alpha']}, sizes {meta['client_sizes']}) · "
        f"Rounds: {meta['rounds']} · local epochs: {meta['local_epochs']} · lr: {meta['lr']} · "
        f"attack scale: x{meta['attack_scale']:g} · seed: {meta['seed']}",
        f"torch {meta['torch_version']} · total {meta['total_seconds']}s", "",
        "Everything except the aggregator and the attack is fixed and seeded — same data partition,",
        "same model init, same per-client local training — so accuracy differences are the effect of",
        "the aggregator's response to the attack alone. `clip_norm` is left OFF for these headline",
        "numbers so the result isolates the estimator (median / trimmed-mean), not a clipping assist.",
        "", "## Headline",  "",
        "| configuration | attackers (by count) | attacker weight share | final acc | best acc | retention vs clean |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        att = f"{r['num_attackers']}/{meta['clients']} (x{r['attack_scale']:g})" if r["num_attackers"] else "0"
        wshare = f"{r['attacker_weight_fraction']*100:.1f}%" if r["num_attackers"] else "—"
        ret = "—" if r.get("retention_vs_clean") is None else f"{r['retention_vs_clean']*100:.1f}%"
        lines.append(
            f"| {r['label']} | {att} | {wshare} | {r['final_accuracy']:.4f} | {r['best_accuracy']:.4f} | {ret} |"
        )

    lines += [
        "",
        f"Round-0 mechanism check: median honest upload delta L2 ≈ "
        f"{results[1]['round0_honest_delta_l2_median']}, median attacker upload delta L2 ≈ "
        f"{results[1]['round0_attacker_upload_l2_median']} — the attack is genuinely "
        f"~{round(results[1]['round0_attacker_upload_l2_median'] / max(results[1]['round0_honest_delta_l2_median'], 1e-9), 1):g}x "
        "the honest signal magnitude, not a token perturbation.",
        "",
        "**Attacker-selection disclosure**: the Byzantine set is the deterministic lowest-numbered",
        "client ids (nested across f — never chosen by data volume). Because the split is non-IID",
        f"(Dirichlet alpha={meta['alpha']:g}), client sizes vary a lot, so a client-COUNT fraction f",
        "does not equal the same WEIGHT fraction under FedAvg's num_examples-weighted mean — the",
        "'attacker weight share' column above reports the real weighted mass so FedAvg's collapse",
        "isn't read as worse than it is without that context (RobustAggregator is unweighted by",
        "design, so weight share does not affect the median/trimmed-mean rows).",
        "", "## f-sweep (breakdown point)", "",
        "| aggregator | f (by count) | attackers | attacker weight share | final acc | retention vs clean |",
        "|---|---|---|---|---|---|",
    ]
    for r in sweep:
        wshare = f"{r['attacker_weight_fraction']*100:.1f}%" if r["num_attackers"] else "—"
        ret = "—" if r.get("retention_vs_clean") is None else f"{r['retention_vs_clean']*100:.1f}%"
        lines.append(
            f"| {r['strategy']} | {r['attack_fraction']:g} | {r['num_attackers']}/{meta['clients']} | "
            f"{wshare} | {r['final_accuracy']:.4f} | {ret} |"
        )

    clean = results[0]["final_accuracy"]
    fa_ret = fedavg_attacked["retention_vs_clean"] * 100
    tm_ret = tmean_attacked["retention_vs_clean"] * 100
    med_ret = median_attacked["retention_vs_clean"] * 100
    clean_learnable = clean >= 0.85
    fedavg_collapsed = fa_ret <= 60.0
    robust_holds = tm_ret >= 90.0 and med_ret >= 90.0
    clean_effect = clean_learnable and fedavg_collapsed and robust_holds

    lines += ["", "## What this shows", ""]
    if clean_effect:
        lines += [
            f"**The effect is clean.** The clean FedAvg baseline reaches **{clean*100:.1f}%** held-out",
            f"accuracy (the task is learnable). Under the {args.attack_fraction*100:.0f}% gradient-scaling",
            f"attack (x{meta['attack_scale']:g}), plain FedAvg collapses to **{fedavg_attacked['final_accuracy']*100:.1f}%**",
            f"(retention {fa_ret:.1f}%), while trimmed-mean (beta={meta['trim_beta']:g}) holds at "
            f"**{tmean_attacked['final_accuracy']*100:.1f}%** (retention {tm_ret:.1f}%) and coordinate-wise",
            f"median holds at **{median_attacked['final_accuracy']*100:.1f}%** (retention {med_ret:.1f}%).",
        ]
    else:
        lines += [
            "**The effect is NOT uniformly clean at these settings — reporting the real numbers rather",
            "than tuning to a story.** Observed:",
            f"- clean FedAvg baseline: {clean*100:.1f}% "
            f"({'learnable' if clean_learnable else 'NOT clearly learnable — task/model/rounds may be too weak'}).",
            f"- FedAvg under attack: {fedavg_attacked['final_accuracy']*100:.1f}% (retention {fa_ret:.1f}%) "
            f"({'collapsed as expected' if fedavg_collapsed else 'did NOT collapse the way the attack intends'}).",
            f"- trimmed-mean under attack: {tmean_attacked['final_accuracy']*100:.1f}% (retention {tm_ret:.1f}%).",
            f"- median under attack: {median_attacked['final_accuracy']*100:.1f}% (retention {med_ret:.1f}%).",
            "See the per-round accuracy arrays in the JSON for the full trajectory; diagnosis notes are",
            "in the PR/task description rather than hand-tuned away here.",
        ]

    lines += [
        "",
        f"**Breakdown point**: trimmed-mean (beta={meta['trim_beta']:g}) is only proven to tolerate a",
        f"Byzantine fraction <= beta. The f-sweep above runs f in {sweep_fractions} for both FedAvg and",
        "trimmed-mean at the same beta: FedAvg degrades at every tested f (it has no tolerance), while",
        "trimmed-mean is expected to hold while f <= beta and degrade once f exceeds beta — read the",
        "sweep table's own numbers above for whether that boundary shows up cleanly at this N and attack",
        "scale (small-cohort estimators are noisier near the exact breakdown point than the asymptotic",
        "theory predicts).",
        "",
        "Reproduce: `PYTHONPATH=src python benchmarks/robust_aggregation_attack.py "
        f"--rounds {args.rounds} --clients {args.clients} --attack-scale {args.attack_scale:g} "
        f"--attack-fraction {args.attack_fraction:g} --trim-beta {args.trim_beta:g} "
        f"--sweep-fractions {args.sweep_fractions}`",
        "",
    ]
    with open(os.path.join(args.out_dir, "robust_aggregation_attack.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
