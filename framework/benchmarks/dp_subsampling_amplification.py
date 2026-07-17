"""C1 x DA-14 — Poisson client subsampling (q<1) AMPLIFIES privacy on the small federated head.

The paper's C1 is a from-scratch RDP accountant for the SUBSAMPLED Gaussian Mechanism:
``compute_rdp(q, z, rounds)`` already takes a Poisson sampling rate ``q``. ``benchmarks/dp_on_head.py``
showed that a small trainable HEAD (d=99) escapes the FR-13 high-dimension collapse at FULL
participation (q=1). This benchmark exercises the SECOND privacy lever the same accountant exposes —
subsampling amplification — on that exact head-only task.

Two duals of the one amplification fact, both read straight off the REAL accountant (no amplification
math is reimplemented here — it lives inside ``compute_rdp``'s subsampled RDP):

  * FIXED z, sweep q down: the certified budget ε = get_epsilon(compute_rdp(q, z, rounds), δ) shrinks
    monotonically as q falls. A client that only participates with probability q is exposed less, so
    the SAME noise multiplier certifies a TIGHTER ε.
  * FIXED target ε, sweep q down: the solved noise multiplier
    z = required_noise_multiplier(ε, q, rounds, δ) shrinks monotonically as q falls. Equivalently,
    subsampling lets you inject LESS noise for the same budget — which is what buys back utility.

At q=1 both levers collapse to the non-subsampled Gaussian, i.e. exactly ``dp_on_head``'s
full-participation baseline (``accounted_epsilon(1.0, …)`` / ``solve_noise_multiplier(…, 1.0, …)``).

Optionally it also RUNS the head-only DP-FedAvg task with per-round Poisson participation at rate q
(at the fixed z), reusing ``dp_on_head``'s frozen-backbone synthetic task + the real
``dp_mechanism.dp_aggregate``, and reports the achieved accuracy and mean participant count. That
utility column is the honest cost side of the FIXED-z table: fewer participants per round means a
larger per-round noise std (z·S/n) even as the certified ε improves — the dual (solve a smaller z at
fixed ε) is the lever that turns the amplification back into utility.

HONEST CAVEATS (for the paper):
  * The DP mechanism, the RDP accountant, the sampling rate q fed to ``compute_rdp``, every solved z,
    every certified ε, and the head dimension d are all REAL and measured — the amplification is the
    accountant's, not a hand-tuned curve.
  * The UTILITY target is a SEEDED SYNTHETIC balanced separable classification task (the same one
    ``dp_on_head`` / ``frozen_backbone_fl`` use), chosen so a linear head over the frozen backbone can
    fit it and the no-DP baseline is ~perfect by construction. It isolates the DP noise's effect; it
    is NOT a production accuracy. The Poisson mask and the DP noise draw from dedicated seeded
    generators, so the whole sweep is byte-reproducible.
  * Certified ε here is the accountant's classic-Mironov bound (intentionally conservative vs Opacus'
    tighter Balle bound — see ``dp_accountant``); the amplification TREND is bound-independent.

Reproduce:  cd framework && PYTHONPATH=src python benchmarks/dp_subsampling_amplification.py
Artifacts:  benchmarks/results/dp_subsampling_amplification.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from fedlearn.estimators.params import trainable_state
from fedlearn.privacy.dp_accountant import compute_rdp, get_epsilon, required_noise_multiplier
from fedlearn.privacy.dp_mechanism import dp_aggregate
from fedlearn.server.subset_federation import apply_trainable_subset, guard_client_updates

# Reuse the EXACT frozen-backbone head task shape (do not fork it): the derived model from
# frozen_backbone_fl and the seeded synthetic balanced task from dp_on_head.
from benchmarks.dp_on_head import _make_task
from benchmarks.frozen_backbone_fl import _Derived, _build

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def certified_epsilon(q: float, z: float, rounds: int, delta: float) -> float:
    """The ε the from-scratch RDP accountant certifies for a FIXED noise multiplier z at sampling
    rate ``q`` over ``rounds`` compositions. This is the amplification lever: at fixed z it decreases
    monotonically as q shrinks. Identical to ``dp_on_head.accounted_epsilon`` (which fixes q=1)."""
    return float(get_epsilon(compute_rdp(q, z, rounds), delta)[0])


def solved_noise_multiplier(target_epsilon: float, q: float, rounds: int, delta: float) -> float:
    """The smallest noise multiplier z the accountant needs to certify ``target_epsilon`` at sampling
    rate ``q`` over ``rounds`` rounds — the dual lever: at fixed target ε it decreases monotonically as
    q shrinks (less noise for the same budget)."""
    return float(required_noise_multiplier(target_epsilon, q, rounds, delta))


def run_utility(*, q, z, rounds, clients, d_in, d_hidden, n_classes, clip, delta, lr,
                local_epochs, sep, seed, dp_seed, sample_seed):
    """Head-only DP-FedAvg with PER-ROUND Poisson participation at rate ``q`` and a FIXED noise
    multiplier ``z``; returns the achieved held-out accuracy and participation stats.

    Each round, every enrolled client is independently included with probability ``q`` (Poisson
    subsampling, from a dedicated ``sample_seed`` generator). The participating subset trains its head
    locally, and the head aggregate is privatised by the REAL ``dp_aggregate`` (per-client L2 clip ->
    uniform average over the participants -> N(0, (z·S/n)^2) noise, n = that round's participant
    count). An empty round (no client sampled) simply leaves the global unchanged — the honest Poisson
    behaviour. Data, backbone, and local training are identical to ``dp_on_head``'s task so the only
    variable is participation + noise.
    """
    torch.manual_seed(seed)
    base = _build(d_in, d_hidden, n_classes, seed=seed)             # shared frozen backbone
    parts, Xte, yte = _make_task(clients, d_in, n_classes, sep, seed)

    def peer():
        m = _Derived(d_in, d_hidden, n_classes)
        m.load_state_dict(base.state_dict())                        # every peer shares the exact backbone
        for p in m.backbone.parameters():
            p.requires_grad_(False)
        return m

    server = peer()
    nets = [peer() for _ in range(clients)]
    backbone0 = server.backbone.weight.detach().clone()

    def accuracy(m):
        m.eval()
        with torch.no_grad():
            return (m(Xte).argmax(1) == yte).float().mean().item()

    def train_head(m, cx, cy):
        opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        m.train()
        for _ in range(local_epochs):
            opt.zero_grad()
            loss_fn(m(cx), cy).backward()
            opt.step()

    sampler = torch.Generator().manual_seed(sample_seed)            # Poisson participation RNG
    dp_gen = torch.Generator().manual_seed(dp_seed)                 # DP noise RNG (dedicated)

    initial_acc = accuracy(server)
    wire_head_only = True
    accs, participants = [], []
    for _rnd in range(rounds):
        global_head = trainable_state(server)
        # Poisson subsampling: include client i iff Bernoulli(q). torch.rand is in [0, 1), so q=1.0
        # selects everyone deterministically (the full-participation baseline).
        mask = torch.rand(clients, generator=sampler) < q
        idx = mask.nonzero(as_tuple=False).flatten().tolist()
        participants.append(len(idx))
        if not idx:
            accs.append(accuracy(server))                          # empty sample -> global unchanged
            continue
        updates = []
        for i in idx:
            m, (cx, cy) = nets[i], parts[i]
            apply_trainable_subset(m, global_head)
            train_head(m, cx, cy)
            u = trainable_state(m)
            if set(u.keys()) != {"head.weight", "head.bias"}:
                wire_head_only = False
            updates.append((f"c{i}", u, cx.shape[0]))
        guard_client_updates([u for _, u, _ in updates], server)
        agg = dp_aggregate(
            updates, global_head, list(global_head.keys()),
            clip_norm=clip, noise_multiplier=z, generator=dp_gen,
        )
        apply_trainable_subset(server, agg)
        accs.append(accuracy(server))

    d = sum(v.numel() for v in trainable_state(server).values())
    return {
        "aggregatable_coords_d": d,
        "initial_accuracy": round(initial_acc, 4),
        "final_accuracy": round(accs[-1], 4),
        "best_accuracy": round(max(accs), 4),
        "per_round_accuracy": [round(a, 4) for a in accs],
        "per_round_participants": participants,
        "mean_participants": round(statistics.mean(participants), 4),
        "expected_participants": round(q * clients, 4),
        "noise_std_per_coord_full": round(z * clip / clients, 6),  # std at n=clients (q=1)
        "backbone_unchanged": bool(torch.equal(server.backbone.weight.detach(), backbone0)),
        "wire_is_head_only": wire_head_only,
    }


def run_sweep(*, q_values, fixed_z, target_epsilon, rounds, clients, d_in=256, d_hidden=32,
              n_classes=3, clip=0.4, delta=1e-5, lr=0.5, local_epochs=5, sep=2.0, seed=1234,
              dp_seed=777, sample_seed=99, with_utility=True):
    """One row per sampling rate ``q``. Each row carries BOTH accountant levers — the certified ε at
    the FIXED ``fixed_z`` and the solved z at the FIXED ``target_epsilon`` — plus, when
    ``with_utility``, the head-only DP-FedAvg accuracy under per-round Poisson participation at rate q
    and the fixed z. Returns ``{"meta", "results"}``, mirroring ``dp_on_head``'s output shape.

    q_values are used as given (the report reads best when they descend, e.g. [1.0, 0.5, 0.25, 0.1]).
    """
    q_values = [float(q) for q in q_values]
    results = []
    for q in q_values:
        cert = certified_epsilon(q, fixed_z, rounds, delta)
        z_star = solved_noise_multiplier(target_epsilon, q, rounds, delta)
        dual_eps = certified_epsilon(q, z_star, rounds, delta)     # round-trip check for the dual
        rec = {
            "sampling_rate_q": q,
            "certified_epsilon_fixed_z": cert,
            "solved_z_fixed_epsilon": z_star,
            "dual_accounted_epsilon": dual_eps,
        }
        if with_utility:
            u = run_utility(
                q=q, z=fixed_z, rounds=rounds, clients=clients, d_in=d_in, d_hidden=d_hidden,
                n_classes=n_classes, clip=clip, delta=delta, lr=lr, local_epochs=local_epochs,
                sep=sep, seed=seed, dp_seed=dp_seed, sample_seed=sample_seed,
            )
            rec.update(u)
        results.append(rec)

    # Headline factors, measured from the endpoints of the sweep (q descending assumed for naming).
    cert_max = max(r["certified_epsilon_fixed_z"] for r in results)
    cert_min = min(r["certified_epsilon_fixed_z"] for r in results)
    z_max = max(r["solved_z_fixed_epsilon"] for r in results)
    z_min = min(r["solved_z_fixed_epsilon"] for r in results)
    amplification_factor = round(cert_max / cert_min, 3) if cert_min > 0 else None
    noise_reduction_factor = round(z_max / z_min, 3) if z_min > 0 else None

    d = results[0].get("aggregatable_coords_d") if with_utility else None
    meta = dict(
        q_values=q_values, fixed_z=fixed_z, target_epsilon=target_epsilon, rounds=rounds,
        clients=clients, d_in=d_in, d_hidden=d_hidden, n_classes=n_classes, clip_norm_S=clip,
        delta=delta, lr=lr, local_epochs=local_epochs, sep=sep, seed=seed, dp_seed=dp_seed,
        sample_seed=sample_seed, with_utility=with_utility, head_d=d,
        amplification_factor=amplification_factor, noise_reduction_factor=noise_reduction_factor,
        model="frozen Linear backbone + trainable Linear head (DA-11 derived-model shape)",
        task="seeded SYNTHETIC balanced Gaussian-blob classification through a frozen backbone "
             "(real DP + subsampled RDP accountant; synthetic accuracy)",
        torch_version=torch.__version__,
    )
    return {"meta": meta, "results": results}


def _render_md(meta, results):
    fz = meta["fixed_z"]
    te = meta["target_epsilon"]
    amp = meta["amplification_factor"]
    nrf = meta["noise_reduction_factor"]
    util = meta["with_utility"]
    lines = [
        "# C1 x DA-14 — subsampling amplification on the small federated head", "",
        f"Task: **{meta['task']}**",
        f"Model: **{meta['model']}**",
        f"Rounds: {meta['rounds']} · Enrolled clients: {meta['clients']} · Clip S: {meta['clip_norm_S']} · "
        f"δ: {meta['delta']} · sep: {meta['sep']} · seed: {meta['seed']} "
        f"(dp_seed {meta['dp_seed']}, sample_seed {meta['sample_seed']})",
        f"torch {meta['torch_version']}", "",
        "The from-scratch RDP accountant's `compute_rdp(q, z, rounds)` takes the Poisson sampling rate "
        "`q` directly, so subsampling amplification is the accountant's own — nothing below "
        "reimplements it.", "",
        "## Lever 1 — FIXED noise multiplier z, sweep q (certified ε amplifies down)", "",
        f"At a fixed **z = {fz}**, a client sampled with probability `q` is exposed less, so the "
        f"accountant certifies a **tighter** ε as `q` shrinks: `certified ε = "
        f"get_epsilon(compute_rdp(q, {fz}, {meta['rounds']}), {meta['delta']})`."
        + (" `mean part.` is the measured mean participant count / round under Poisson(q); "
           "`final acc` is the head-only DP-FedAvg accuracy at that fixed z (honest cost side — "
           "fewer participants means a larger per-round noise std z·S/n)." if util else ""),
        "",
    ]
    if util:
        lines += [
            "| q | certified ε (z fixed) | mean part./round | expected qN | final acc | best acc |",
            "|---|---|---|---|---|---|",
        ]
        for r in results:
            lines.append(
                f"| {r['sampling_rate_q']:g} | {r['certified_epsilon_fixed_z']:.4f} | "
                f"{r['mean_participants']:.2f} | {r['expected_participants']:.2f} | "
                f"{r['final_accuracy']:.4f} | {r['best_accuracy']:.4f} |"
            )
    else:
        lines += ["| q | certified ε (z fixed) |", "|---|---|"]
        for r in results:
            lines.append(f"| {r['sampling_rate_q']:g} | {r['certified_epsilon_fixed_z']:.4f} |")
    lines += [
        "",
        f"**Amplification factor** (certified ε at the largest q ÷ at the smallest q, fixed z) = "
        f"**{amp}×** — the same z certifies a {amp}× tighter budget once clients are subsampled.",
        "",
        "## Lever 2 (dual) — FIXED target ε, sweep q (solved z shrinks — less noise)", "",
        f"Hold the budget at **ε = {te:g}** and solve the noise multiplier per q: "
        f"`z = required_noise_multiplier({te:g}, q, {meta['rounds']}, {meta['delta']})`. Smaller q "
        "needs **less** noise for the same budget — the utility-preserving reading of amplification. "
        "`accounted ε` re-certifies the solved z back to the target (round-trip check).", "",
        "| q | solved z (ε fixed) | accounted ε |",
        "|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['sampling_rate_q']:g} | {r['solved_z_fixed_epsilon']:.4f} | "
            f"{r['dual_accounted_epsilon']:.4f} |"
        )
    lines += [
        "",
        f"**Noise-reduction factor** (solved z at the largest q ÷ at the smallest q, fixed ε) = "
        f"**{nrf}×** — subsampling to the smallest q lets you cut the noise multiplier {nrf}×.",
        "",
        "## What this shows", "",
        "Subsampling is a second, orthogonal privacy lever alongside the small-d head of DA-14: at a "
        "fixed noise level it tightens the certified budget, and at a fixed budget it lets you use "
        "less noise (hence keep more utility). Both readings are the SAME accountant fact "
        "(`compute_rdp` with q<1), reported as its two duals. At **q=1 both collapse to the "
        "non-subsampled Gaussian** — exactly `dp_on_head`'s full-participation baseline.",
        "",
        "*Caveat:* the utility target is a seeded synthetic balanced separable task (so no-DP is "
        "~perfect by construction); the DP mechanism, the subsampled RDP accountant, the sampling rate "
        "q, every solved z, and every certified ε are all real and measured. Certified ε uses the "
        "accountant's conservative classic-Mironov bound; the amplification trend is bound-independent.",
        "",
        f"Reproduce: `PYTHONPATH=src python benchmarks/dp_subsampling_amplification.py "
        f"--rounds {meta['rounds']} --clients {meta['clients']} --fixed-z {fz} "
        f"--target-epsilon {te:g} --clip {meta['clip_norm_S']} "
        f"--q-values {','.join(format(r['sampling_rate_q'], 'g') for r in results)}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="C1 x DA-14 subsampling-amplification benchmark on the frozen-backbone head.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--q-values", type=str, default="1.0,0.5,0.25,0.1",
                    help="comma-separated Poisson sampling rates (descending reads best)")
    ap.add_argument("--fixed-z", type=float, default=2.0, help="fixed noise multiplier for lever 1")
    ap.add_argument("--target-epsilon", type=float, default=4.0, help="fixed target ε for lever 2")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--clients", type=int, default=32, help="enrolled clients (Poisson population)")
    ap.add_argument("--clip", type=float, default=0.4, help="DP L2 clip norm S")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--d-in", type=int, default=256, help="frozen backbone input dim")
    ap.add_argument("--d-hidden", type=int, default=32, help="frozen backbone feature dim (drives head d)")
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--sep", type=float, default=2.0, help="Gaussian-blob class separation")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dp-seed", type=int, default=777)
    ap.add_argument("--sample-seed", type=int, default=99, help="Poisson participation RNG seed")
    ap.add_argument("--no-utility", action="store_true", help="accountant levers only (skip the FL run)")
    ap.add_argument("--out-dir", type=str, default=RESULTS_DIR)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    q_values = [float(x) for x in args.q_values.split(",") if x.strip()]

    t0 = time.time()
    out = run_sweep(
        q_values=q_values, fixed_z=args.fixed_z, target_epsilon=args.target_epsilon,
        rounds=args.rounds, clients=args.clients, d_in=args.d_in, d_hidden=args.d_hidden,
        n_classes=args.n_classes, clip=args.clip, delta=args.delta, lr=args.lr,
        local_epochs=args.local_epochs, sep=args.sep, seed=args.seed, dp_seed=args.dp_seed,
        sample_seed=args.sample_seed, with_utility=not args.no_utility,
    )
    out["meta"]["total_seconds"] = round(time.time() - t0, 1)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "dp_subsampling_amplification.json"), "w") as f:
        json.dump(out, f, indent=2)
    md = _render_md(out["meta"], out["results"])
    with open(os.path.join(args.out_dir, "dp_subsampling_amplification.md"), "w") as f:
        f.write(md)

    print(md)


if __name__ == "__main__":
    main()
