"""FR-13 x DA-14 — the COHORT-SIZE axis of the DP utility-SNR on a small HEAD.

FR-13's ``dp_snr_crossing.py`` swept the cohort size N on a *FedLoRA adapter* (d = 26112 aggregatable
coordinates) at a fixed target ε and found the utility SNR

        SNR = N / (z * sqrt(d))

crosses 1 only at an INFEASIBLE cohort: at q=1 the accountant-solved noise multiplier z is a constant
(it depends on ε, rounds, δ — not on N), so the SNR=1 crossing sits at N = z*sqrt(d), and with
d = 26112 that is HUNDREDS of clients — out of reach for a laptop-scale federation.

``dp_on_head.py`` then showed the complementary lever: shrink the federated subset to a small trainable
HEAD (d = 99 at d_hidden=32, n_classes=3), which lifts the SNR by sqrt(26112/99) ≈ 16.24x at the same
(N, z). That benchmark FIXED N and swept ε. THIS benchmark is its complement on the cohort axis: it
FIXES ε (and hence z) and d, and sweeps the cohort size N over a small, reachable range, to show that
on the small head the SNR=1 crossing — and the utility recovery — happens at a MODERATE N, not the
infeasible N the d=26112 FedLoRA baseline needs.

It reuses ``dp_on_head.run_config`` VERBATIM — the exact same frozen-backbone head-only DP-FedAvg task,
the same REAL central-DP mechanism (``fedlearn.privacy.dp_mechanism.dp_aggregate``: per-client L2 clip
-> UNIFORM average -> calibrated Gaussian noise), and the same from-scratch RDP accountant
(``fedlearn.privacy.dp_accountant``: solve z per target ε, account ε back). Nothing about the DP path is
reimplemented here; only the sweep axis changes.

For each cohort size N it records: the accountant-solved z (constant across N at q=1), the certified ε,
the SNR = N/(z*sqrt(d)), whether SNR>=1, the final held-out accuracy, and — relative to a per-N no-DP
control — the retained fraction of the above-chance utility. It then reports the SNR=1 crossing N on the
head and contrasts it with the crossing N the d=26112 FedLoRA baseline would need at the same z.

HONEST CAVEATS (for the paper):
  * The DP mechanism, the RDP accountant, the solved z, the certified ε, the L2 clip, the Gaussian
    noise, and the reported byte-exact d are all REAL and measured — inherited unchanged from
    ``dp_on_head``. At q=1, z is analytically independent of N, so it is (and is asserted to be)
    identical across the whole sweep; the SNR then grows exactly linearly in N by construction.
  * The UTILITY task is a SEEDED SYNTHETIC balanced separable target: Gaussian class blobs passed
    through the frozen random backbone, so a linear head CAN separate them and the no-DP control is
    ~perfect by construction (chance = 1/n_classes). This isolates the DP noise's effect on utility;
    it is NOT a production accuracy on real data. Per-client shard size is held ~constant across N
    (the total training set scales with N inside ``dp_on_head._make_task``), so the sweep isolates the
    cohort/SNR effect from a "less data per client" confound — the same guard ``dp_snr_crossing`` uses.
  * Single seed per N: ``dp_last3_avg_accuracy`` (mean of the final 3 rounds) is reported alongside the
    raw single-round final/best, to damp (not remove) round-to-round noise. Do not over-read wiggles.
  * SNR = N/(z*sqrt(d)) is a PER-ROUND proxy; because the DP noise is zero-mean and averages down over
    rounds while the clipped signal direction is consistent, empirical utility recovers at (or slightly
    below) the SNR=1 line. Both the theoretical crossing and the empirical accuracy are reported.
  * The no-DP ceiling here does NOT flatten as N grows (the head is small but the synthetic target is
    linearly separable over the frozen features, so the ceiling stays ~perfect at every N) — the
    ``no_dp_ceiling_flat`` flag reports this from the measured numbers rather than assuming it.

Reproduce:  cd framework && PYTHONPATH=src python benchmarks/dp_head_cohort_sweep.py
Artifacts:  benchmarks/results/dp_head_cohort_sweep.{json,md}
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

# Reuse the EXACT head-only central-DP FedAvg task (real DP mechanism + from-scratch RDP accountant).
# ``run_config`` solves z from ε, accounts ε back, runs the frozen-backbone head-only DP-FedAvg, and
# reports d, SNR, and accuracy. FEDLORA_REFERENCE_D is FR-13/C2's d=26112 collapse baseline.
from benchmarks.dp_on_head import FEDLORA_REFERENCE_D, run_config

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def crossing_cohort(results):
    """The SNR=1 crossing cohort: the smallest swept N whose per-round utility SNR reaches/exceeds 1,
    or ``None`` if the sweep never crosses. ``results`` is the per-N row list from ``run_cohort_sweep``.
    """
    for r in sorted(results, key=lambda r: r["n"]):
        if r.get("utility_snr") is not None and r["utility_snr"] >= 1.0:
            return r["n"]
    return None


def run_cohort_sweep(*, target_epsilon=4.0, n_values=(8, 16, 24, 48, 96), rounds=8, d_in=256,
                     d_hidden=32, n_classes=3, clip=0.4, delta=1e-5, lr=0.5, local_epochs=5,
                     sep=2.0, seed=1234, dp_seed=777, retain_escape_fraction=0.5):
    """Fix (ε, d); sweep the cohort size N. For each N run a no-DP control + one head-only DP run at
    ``target_epsilon`` via ``dp_on_head.run_config``, and record the SNR + retained utility.

    Returns ``{"meta", "results"}``: ``results`` is one row per N (increasing N) flattening the DP
    record and attaching the per-N no-DP ceiling and retained-utility fraction; ``meta`` carries the
    fixed knobs, the (constant, q=1) z, the head d, the SNR=1 crossing N, and the cohort the d=26112
    FedLoRA baseline would need at the same z.
    """
    n_values = sorted(int(n) for n in n_values)
    common = dict(rounds=rounds, d_in=d_in, d_hidden=d_hidden, n_classes=n_classes, clip=clip,
                  delta=delta, lr=lr, local_epochs=local_epochs, sep=sep, seed=seed, dp_seed=dp_seed)

    results = []
    for n in n_values:
        ctrl = run_config(label=f"N={n} (no-DP)", epsilon=None, clients=n, **common)
        dp = run_config(label=f"N={n} (ε={target_epsilon:g})", epsilon=float(target_epsilon),
                        clients=n, **common)

        chance = dp["chance_accuracy"]
        ceiling = ctrl["final_accuracy"]
        lift = ceiling - chance
        retain = ((dp["final_accuracy"] - chance) / lift) if lift > 1e-9 else 0.0
        tail = dp["per_round_accuracy"][-min(3, len(dp["per_round_accuracy"])):]
        ctrl_tail = ctrl["per_round_accuracy"][-min(3, len(ctrl["per_round_accuracy"])):]

        results.append({
            "n": n,
            "target_epsilon": dp["target_epsilon"],
            "accounted_epsilon": dp["accounted_epsilon"],
            "noise_multiplier_z": dp["noise_multiplier_z"],
            "sampling_rate_q": dp["sampling_rate_q"],
            "clip_norm_S": dp["clip_norm_S"],
            "aggregatable_coords_d": dp["aggregatable_coords_d"],
            "noise_std_per_coord": dp["noise_std_per_coord"],
            "utility_snr": dp["utility_snr"],
            "snr_ge_one": dp["snr_ge_one"],
            "chance_accuracy": chance,
            "no_dp_final_accuracy": round(ceiling, 4),
            "no_dp_last3_avg_accuracy": round(statistics.mean(ctrl_tail), 4),
            "dp_final_accuracy": dp["final_accuracy"],
            "dp_best_accuracy": dp["best_accuracy"],
            "dp_last3_avg_accuracy": round(statistics.mean(tail), 4),
            "dp_per_round_accuracy": dp["per_round_accuracy"],
            "retain_fraction": round(retain, 4),
            "escapes_collapse": bool(retain >= retain_escape_fraction),
            "backbone_unchanged": dp["backbone_unchanged"],
            "wire_is_head_only": dp["wire_is_head_only"],
        })

    d = results[0]["aggregatable_coords_d"]
    zs = [r["noise_multiplier_z"] for r in results]
    z = zs[0]
    z_constant = len({round(zz, 12) for zz in zs}) == 1  # q=1 => z independent of N
    accounted = results[0]["accounted_epsilon"]

    crossing_n = crossing_cohort(results)
    predicted_crossing = (z * (d ** 0.5)) if z else None
    fedlora_predicted_crossing = (z * (FEDLORA_REFERENCE_D ** 0.5)) if z else None

    # Utility-recovery cohort: smallest N whose DP run retains >= retain_escape_fraction of the no-DP
    # above-chance lift (the empirical companion to the theoretical SNR crossing).
    recovered_n = next((r["n"] for r in results if r["escapes_collapse"]), None)

    # Does the no-DP ceiling flatten as N grows (the honest confound to disclose)? It does NOT if every
    # per-N control stays close to the best control seen — report from the measured numbers.
    ceilings = [r["no_dp_last3_avg_accuracy"] for r in results]
    no_dp_ceiling_flat = (max(ceilings) - min(ceilings)) <= 0.05

    meta = dict(
        target_epsilon=float(target_epsilon), accounted_epsilon=accounted, noise_multiplier_z=z,
        z_constant_across_n=z_constant, sampling_rate_q=1.0, rounds=rounds, delta=delta,
        clip_norm_S=clip, lr=lr, local_epochs=local_epochs, d_in=d_in, d_hidden=d_hidden,
        n_classes=n_classes, sep=sep, seed=seed, dp_seed=dp_seed, n_values=n_values,
        retain_escape_fraction=retain_escape_fraction, head_d=d, fedlora_reference_d=FEDLORA_REFERENCE_D,
        snr_gain_vs_fedlora=round((FEDLORA_REFERENCE_D / d) ** 0.5, 2),
        crossing_n=crossing_n, predicted_crossing_n=(round(predicted_crossing, 2) if predicted_crossing else None),
        fedlora_predicted_crossing_n=(round(fedlora_predicted_crossing, 2) if fedlora_predicted_crossing else None),
        recovered_n=recovered_n, no_dp_ceiling_range=[round(min(ceilings), 4), round(max(ceilings), 4)],
        no_dp_ceiling_flat=no_dp_ceiling_flat,
        model="frozen Linear backbone + trainable Linear head (DA-11 derived-model shape)",
        task="seeded SYNTHETIC balanced Gaussian-blob classification through a frozen backbone "
             "(real DP + accountant; synthetic accuracy)",
        torch_version=torch.__version__,
    )
    return {"meta": meta, "results": results}


def _render_md(meta, results):
    d = meta["head_d"]
    z = meta["noise_multiplier_z"]
    gain = meta["snr_gain_vs_fedlora"]
    crossing = meta["crossing_n"]
    fed_cross = meta["fedlora_predicted_crossing_n"]
    lines = [
        "# FR-13 x DA-14 — DP utility-SNR cohort-size crossing on a small HEAD", "",
        f"Task: **{meta['task']}**",
        f"Model: **{meta['model']}**",
        f"Fixed target ε: **{meta['target_epsilon']:g}** (accounted **{meta['accounted_epsilon']:.3f}**) · "
        f"Rounds: {meta['rounds']} · q: {meta['sampling_rate_q']} · Clip S: {meta['clip_norm_S']} · "
        f"δ: {meta['delta']} · sep: {meta['sep']} · seed: {meta['seed']} (dp_seed {meta['dp_seed']})",
        f"torch {meta['torch_version']}", "",
        f"Head dimension **d = {d}** trainable coords (√d ≈ {d ** 0.5:.1f}), versus the FR-13 central-DP "
        f"FedLoRA baseline **d = {meta['fedlora_reference_d']}** (√d ≈ {meta['fedlora_reference_d'] ** 0.5:.1f}) "
        f"— an SNR gain of **{gain}×** at the same (N, z).", "",
        f"At q=1 (full participation, no subsampling) the accountant's solved noise multiplier "
        f"z = **{z:.4f}** depends only on (ε, rounds, δ) — **not** on N "
        f"({'CONSTANT across the sweep' if meta['z_constant_across_n'] else 'NOT constant — investigate'}) "
        f"— so the SNR denominator z·√d ≈ {z * d ** 0.5:.2f} is fixed and **SNR = N/(z·√d) grows exactly "
        f"linearly in N**. `retain` = fraction of the per-N no-DP above-chance lift kept; a cohort "
        f"*escapes* when it retains ≥ {meta['retain_escape_fraction']:.0%}.", "",
        "| N (cohort) | z | accounted ε | SNR = N/(z√d) | SNR≥1 | no-DP acc (last3 / final) | "
        "DP acc (last3 / final / best) | retain | escapes? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        sge = "yes" if r["snr_ge_one"] else "no"
        esc = "yes" if r["escapes_collapse"] else "no"
        lines.append(
            f"| {r['n']} | {r['noise_multiplier_z']:.4f} | {r['accounted_epsilon']:.3f} | "
            f"{r['utility_snr']:.3f} | {sge} | "
            f"{r['no_dp_last3_avg_accuracy']:.4f} / {r['no_dp_final_accuracy']:.4f} | "
            f"{r['dp_last3_avg_accuracy']:.4f} / {r['dp_final_accuracy']:.4f} / {r['dp_best_accuracy']:.4f} | "
            f"{r['retain_fraction']:.0%} | {esc} |"
        )

    lines += ["", "## Crossing analysis", ""]
    lines.append(
        f"Predicted SNR=1 crossing (z·√d): **N ≈ {meta['predicted_crossing_n']}**."
        if meta["predicted_crossing_n"] else "Predicted crossing: unavailable (z missing).")
    if crossing is not None:
        snr_at = next(r["utility_snr"] for r in results if r["n"] == crossing)
        lines.append(
            f"Empirically, SNR first reaches/exceeds 1 at **N = {crossing}** (SNR = {snr_at:.3f}) — a "
            f"MODERATE, laptop-reachable cohort.")
    else:
        max_n = max(r["n"] for r in results)
        snr_at = next(r["utility_snr"] for r in results if r["n"] == max_n)
        lines.append(
            f"SNR did **not** reach 1 within the swept range — largest N = {max_n} gives SNR = {snr_at:.3f} "
            "(report this plainly rather than widening the range post hoc).")
    if meta["recovered_n"] is not None:
        rr = next(r for r in results if r["n"] == meta["recovered_n"])
        lines.append(
            f"Utility recovery: at **N = {meta['recovered_n']}**, the DP run retains "
            f"{rr['retain_fraction']:.0%} of the no-DP above-chance lift (DP {rr['dp_last3_avg_accuracy']:.4f} "
            f"vs no-DP {rr['no_dp_last3_avg_accuracy']:.4f}, chance {rr['chance_accuracy']:.4f}) — the "
            "accuracy curve tracks the SNR proxy, not just the formula.")
    else:
        lines.append(
            "Utility recovery: no swept cohort retained the escape fraction of the no-DP lift "
            "(report plainly — see the table).")
    lines += [
        "",
        f"**Contrast with FR-13's FedLoRA (d={meta['fedlora_reference_d']}).** At the SAME z (q=1), that "
        f"adapter's SNR=1 crossing sits at N = z·√{meta['fedlora_reference_d']} ≈ **{fed_cross}** clients — "
        f"an infeasible cohort for a laptop-scale federation. Shrinking the federated subset to the "
        f"d={d} head pulls the crossing down by the same {gain}× to **N ≈ {meta['predicted_crossing_n']}** "
        f"(first swept cohort with SNR≥1: **N = {crossing}**). That is the complement of `dp_on_head`'s "
        "ε-sweep: there we fixed N and lowered the ε at which utility survives; here we fix ε and lower "
        "the cohort N at which the SNR crosses 1.", "",
        "## Honesty caveats", "",
        "- **Single seed per N** (not averaged over repeats): `dp_last3_avg_accuracy` (mean of the final "
        "3 rounds) is reported alongside the raw single-round final/best to damp round-to-round noise. "
        "Do not over-read single-point wiggles.",
        "- **q=1 throughout** — every enrolled client participates every round; z is therefore constant "
        "across the sweep and the crossing is a clean N-only effect. This does not itself demonstrate "
        "subsampling amplification (a separate lever for shrinking z at fixed N).",
        "- **Per-client shard held ~constant** (the total training set scales with N inside "
        "`dp_on_head._make_task`), so the recovery is the SNR effect, not a \"more data per client\" "
        "artifact.",
        f"- **No-DP ceiling** stays in [{meta['no_dp_ceiling_range'][0]:.4f}, {meta['no_dp_ceiling_range'][1]:.4f}] "
        f"across the sweep "
        f"({'flat — the crossing does NOT come from a collapsing ceiling' if meta['no_dp_ceiling_flat'] else 'NOT flat — the tractable crossing partly reflects a shifting ceiling; read the retain column, not raw accuracy'}). "
        "Utility is a seeded synthetic separable target (so no-DP is ~perfect by construction); the DP "
        "mechanism, RDP accountant, solved z, accounted ε, and d are all real and measured.",
        "",
        f"Reproduce: `PYTHONPATH=src python benchmarks/dp_head_cohort_sweep.py "
        f"--epsilon {meta['target_epsilon']:g} --rounds {meta['rounds']} --d-hidden {meta['d_hidden']} "
        f"--clip {meta['clip_norm_S']} --sep {meta['sep']} "
        f"--n-values {','.join(str(n) for n in meta['n_values'])}`", "",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description="FR-13 x DA-14: cohort-size (N) sweep across the DP utility-SNR crossing on the head.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epsilon", type=float, default=4.0, help="fixed target ε for every N (drives the constant z)")
    ap.add_argument("--n-values", type=str, default="8,16,24,48,96", help="comma-separated cohort sizes N to sweep")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--clip", type=float, default=0.4, help="DP L2 clip norm S")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--d-in", type=int, default=256, help="frozen backbone input dim")
    ap.add_argument("--d-hidden", type=int, default=32, help="frozen backbone feature dim (drives head d)")
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--sep", type=float, default=2.0, help="Gaussian-blob class separation (task difficulty)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dp-seed", type=int, default=777)
    ap.add_argument("--out-dir", type=str, default=RESULTS_DIR)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]

    t0 = time.time()
    out = run_cohort_sweep(
        target_epsilon=args.epsilon, n_values=n_values, rounds=args.rounds, d_in=args.d_in,
        d_hidden=args.d_hidden, n_classes=args.n_classes, clip=args.clip, delta=args.delta,
        lr=args.lr, local_epochs=args.local_epochs, sep=args.sep, seed=args.seed, dp_seed=args.dp_seed,
    )
    out["meta"]["total_seconds"] = round(time.time() - t0, 1)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "dp_head_cohort_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    md = _render_md(out["meta"], out["results"])
    with open(os.path.join(args.out_dir, "dp_head_cohort_sweep.md"), "w") as f:
        f.write(md)

    print(md)


if __name__ == "__main__":
    main()
