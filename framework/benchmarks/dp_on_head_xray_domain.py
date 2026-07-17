"""DA-11 — does a DOMAIN backbone beat frozen ImageNet for DP-on-head, on REAL chest X-ray?

Every caveat in `dp_on_head_xray.py` / `_cohort.py` is the same one: the frozen backbone is ImageNet
(out-of-domain), stated as a conservative floor. The DA-11 design's headline is that DOMAIN-fit dominates
the freeze knob (a domain backbone + linear head matches/beats full fine-tune and is far more
data-efficient) — but that was a borrowed citation, never measured in this repo. This benchmark measures
it: for each backbone it extracts BOTH a frozen-ImageNet feature set AND a DOMAIN-adapted one (ImageNet-init
fine-tuned on the X-ray train split, then frozen), runs the SAME real central-DP ε-sweep over each
(imported `dp_on_head_xray.run_sweep`, unchanged), and compares.

Two questions, both controlled (same head d, same DP mechanism/accountant/z — only the backbone's feature
quality changes, so any DP difference is attributable to features, not to d or the accountant):
  1. Does the domain backbone RAISE the no-DP ceiling (accuracy/AUC)?  [the DA-11 thesis]
  2. At the same d, does it TIGHTEN the DP escape boundary (escape at a smaller ε)?  [the controlled test
     of the ε-sweep's own finding that feature quality co-determines DP robustness at fixed d]

HONEST CAVEATS (for the paper):
  * The domain backbone is fine-tuned on the SAME chest_xray train split the linear probe then federates
    over — an IN-DISTRIBUTION UPPER BOUND on the domain-fit benefit. A true domain backbone (DA-11 §4.1:
    own-trained on a SEPARATE corpus, NIH ChestX-ray14) would not have seen this exact train set; it is
    still deferred. The TEST split is held out from BOTH the backbone fine-tune AND the head, so test
    accuracy is honest; but read the lift as "how much can better features help here", not a
    generalization estimate.
  * DP mechanism, RDP accountant, solved z, byte-exact d, and the head are all real and unchanged from
    `dp_on_head_xray`. DP averaged over the same noise seeds. One balanced binary set / clip / seed.

Reproduce:  cd framework && PYTHONPATH=src python benchmarks/dp_on_head_xray_domain.py \
              --backbones resnet18,resnet50 --domain-epochs 5
Artifacts:  benchmarks/results/dp_on_head_xray_domain.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from benchmarks.dp_on_head_xray import (
    DEFAULT_DATA_DIR, RESULTS_DIR, extract_domain_features, extract_features, run_sweep)


def tightest_escaping_epsilon(sweep):
    """Smallest target ε whose DP run escapes (mean retention >= the fraction), or None if none escape."""
    epss = [r["target_epsilon"] for r in sweep["results"]
            if r.get("target_epsilon") is not None and r.get("escapes_collapse")]
    return min(epss) if epss else None


def _fmt_eps(e):
    return "—" if e is None else f"{e:g}"


def _render_md(pairs, meta):
    """`pairs` = list of {backbone, imagenet: sweep, domain: sweep}."""
    lines = [
        "# DA-11 — DOMAIN backbone vs frozen ImageNet for DP-on-head (REAL chest X-ray)", "",
        f"Fixed: rounds {meta['rounds']} · N {meta['clients']} · q 1.0 · clip S {meta['clip']} · "
        f"δ {meta['delta']} · {meta['n_dp_seeds']} DP seeds · domain fine-tune {meta['domain_epochs']} epochs "
        f"(lr {meta['domain_lr']}) · seed {meta['seed']} · torch {torch.__version__}",
        f"Data: {meta['n_train']} train / {meta['n_test']} test, chance {1.0/meta['n_classes']:.3f}.", "",
        "Controlled comparison: the domain backbone changes ONLY feature quality — head d, DP mechanism, "
        "accountant, and solved z are identical to the ImageNet run — so any difference is attributable to "
        "the features, not to d or the privacy accounting.", "",
        "## Q1 — does the domain backbone raise the no-DP ceiling? (the DA-11 thesis)",
        "",
        "| backbone | variant | head d | no-DP acc | no-DP AUC |",
        "|---|---|---|---|---|",
    ]
    for p in pairs:
        for tag, s in (("ImageNet", p["imagenet"]), ("domain", p["domain"])):
            m = s["meta"]
            auc = "—" if m["no_dp_auc"] is None else f"{m['no_dp_auc']:.3f}"
            lines.append(f"| {p['backbone']} | {m['variant']} | {m['head_d']} | "
                         f"{m['no_dp_accuracy']:.3f} | {auc} |")
    lines += ["", "## Q2 — at the SAME d, does the domain backbone tighten the DP escape boundary?", "",
              "| backbone | variant | tightest escaping ε | per-ε retain (8 / 4 / 1 / 0.5 / 0.1) |",
              "|---|---|---|---|"]
    for p in pairs:
        for tag, s in (("ImageNet", p["imagenet"]), ("domain", p["domain"])):
            m = s["meta"]
            te = tightest_escaping_epsilon(s)
            rets = {r["target_epsilon"]: r["accuracy_retention"] for r in s["results"]
                    if r["target_epsilon"] is not None}
            row = " / ".join(f"{int(rets[e]*100)}%" if e in rets and rets[e] is not None else "—"
                             for e in (8, 4, 1, 0.5, 0.1))
            lines.append(f"| {p['backbone']} | {m['variant']} | **ε={_fmt_eps(te)}** | {row} |")

    # Data-driven verdict per backbone.
    lines += ["", "## What this shows", ""]
    for p in pairs:
        mi, md = p["imagenet"]["meta"], p["domain"]["meta"]
        ti, td = tightest_escaping_epsilon(p["imagenet"]), tightest_escaping_epsilon(p["domain"])
        ceil_up = md["no_dp_accuracy"] - mi["no_dp_accuracy"]
        tighter = (td is not None and (ti is None or td < ti))
        lines.append(
            f"- **{p['backbone']}:** no-DP {mi['no_dp_accuracy']:.3f} → {md['no_dp_accuracy']:.3f} "
            f"({'+' if ceil_up >= 0 else ''}{ceil_up:.3f} with domain-fit; AUC "
            f"{mi['no_dp_auc']:.3f} → {md['no_dp_auc']:.3f}). Tightest escaping ε: "
            f"ImageNet ε={_fmt_eps(ti)} → domain ε={_fmt_eps(td)} — "
            + ("the domain backbone **tightens** the DP escape at the same head d, confirming (by a "
               "controlled intervention) that feature quality co-determines DP robustness — the ε-sweep's "
               "cross-backbone observation, now caused on purpose." if tighter else
               "the DP escape boundary did **not** tighten here (see the retain columns) — the no-DP "
               "ceiling moved but the escape ε was already at the grid edge or the lift was too small to "
               "cross a coarser ε.")
        )
    lines += [
        "",
        "*Honest caveats:* the domain backbone is fine-tuned on the SAME chest_xray train split the head "
        "then federates over — an IN-DISTRIBUTION UPPER BOUND on the domain-fit benefit; the own-trained "
        "separate-corpus (NIH) backbone of DA-11 §4.1 is still deferred. Test held out from backbone AND "
        "head (test accuracy honest). Same head d / DP mechanism / accountant / z as the ImageNet run; DP "
        f"averaged over {meta['n_dp_seeds']} seeds; one balanced binary set / clip / seed.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="DA-11 domain vs ImageNet backbone for DP-on-head (real X-ray).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--backbones", type=str, default="resnet18,resnet50")
    ap.add_argument("--domain-epochs", type=int, default=5)
    ap.add_argument("--domain-lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--epsilons", type=str, default="8,4,1,0.5,0.1")
    ap.add_argument("--clip", type=float, default=0.4)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dp-seeds", type=str, default="777,778,779,780,781")
    ap.add_argument("--out-dir", type=str, default=RESULTS_DIR)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    epsilons = [float(x) for x in args.epsilons.split(",") if x.strip()]
    dp_seeds = [int(x) for x in args.dp_seeds.split(",") if x.strip()]
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]

    def sweep(feats):
        return run_sweep(features=feats, epsilons=epsilons, rounds=args.rounds, clients=args.clients,
                         clip=args.clip, delta=args.delta, lr=args.lr, local_epochs=args.local_epochs,
                         seed=args.seed, dp_seeds=dp_seeds)

    t0 = time.time()
    pairs = []
    for bb in backbones:
        print(f"[*] {bb}: frozen ImageNet features ...", flush=True)
        fi = extract_features(args.data_dir, backbone=bb, pretrained=True, img_size=args.img_size,
                              device=device, backbone_seed=args.seed, subset=args.subset)
        print(f"[*] {bb}: DOMAIN-adapting ({args.domain_epochs} ep) then extracting ...", flush=True)
        fd = extract_domain_features(args.data_dir, backbone=bb, epochs=args.domain_epochs,
                                     lr=args.domain_lr, img_size=args.img_size, device=device,
                                     seed=args.seed, pretrained=True, subset=args.subset)
        si, sd = sweep(fi), sweep(fd)
        print(f"    {bb}: no-DP ImageNet {si['meta']['no_dp_accuracy']:.3f} -> domain "
              f"{sd['meta']['no_dp_accuracy']:.3f}; tightest escape "
              f"ε={_fmt_eps(tightest_escaping_epsilon(si))} -> ε={_fmt_eps(tightest_escaping_epsilon(sd))}",
              flush=True)
        pairs.append({"backbone": bb, "imagenet": si, "domain": sd})

    m0 = pairs[0]["imagenet"]["meta"]
    meta = dict(rounds=args.rounds, clients=args.clients, clip=args.clip, delta=args.delta,
                n_dp_seeds=len(dp_seeds), domain_epochs=args.domain_epochs, domain_lr=args.domain_lr,
                seed=args.seed, n_train=m0["n_train"], n_test=m0["n_test"], n_classes=m0["n_classes"],
                device=device, total_seconds=round(time.time() - t0, 1))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "dp_on_head_xray_domain.json"), "w") as f:
        json.dump({"meta": meta, "pairs": pairs}, f, indent=2, default=str)
    md = _render_md(pairs, meta)
    with open(os.path.join(args.out_dir, "dp_on_head_xray_domain.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
