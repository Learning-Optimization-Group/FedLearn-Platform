#!/usr/bin/env python3
"""Generate the measured frozen-vs-full trade-off the project picker shows.

The picker must not carry hand-written numbers. Every figure a user sees when choosing a training
arm is derived here from the campaign's own verdict record, so the UI is a *rendering of the
record* and cannot drift from it. This mirrors the rule the benchmark reports already follow.

    python scripts/build_arm_tradeoff.py            # regenerate, fail if the record is missing
    python scripts/build_arm_tradeoff.py --check    # verify the committed file is current

Input  (untracked, local): research/results/frozen-backbone/VERDICT_frozen_vs_full.json
Output (tracked):          fl-runtime/arm_tradeoff.json

The input is gitignored and the output is committed, which is deliberate: the frontend needs these
numbers at runtime and cannot read the research tree. The committed file therefore records the
sha256 of the record it came from, so a later divergence is detectable rather than invisible.

WHICH CONTRAST IS QUOTED, AND WHY IT IS THE CONSERVATIVE ONE
------------------------------------------------------------
The verdict contains several frozen-vs-full contrasts and they do not all point the same way. This
script quotes the *controlled* one — same backbone (timm resnet50_gn.a1h_in1k), same protocol,
varying only the arm — in which FULL FINE-TUNING WINS on accuracy, sign-consistent across 6/6 seeds
at both shard sizes.

That is the contrast least favourable to the frozen arm, and quoting it is the point. The
cross-backbone comparison (frozen ResNet-18/BN vs full ResNet-50/GN) shows frozen *tying* the best
full arm, but it varies backbone, depth, normalisation and training recipe at once, so a picker
quoting it would be advertising a confound as a feature. The verdict says so explicitly: "Read
+0.0033 as a tie, never as a frozen win."
"""

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(REPO, "research", "results", "frozen-backbone", "VERDICT_frozen_vs_full.json")
OUT = os.path.join(REPO, "fl-runtime", "arm_tradeoff.json")


def build(record_path=RECORD):
    with open(record_path, "rb") as fh:
        raw = fh.read()
    v = json.loads(raw)
    q = v["QUALIFICATION_2026_08_10"]
    ctrl = q["controlled_contrast_same_backbone"]          # shard 10, same backbone
    rep = q["REPLICATED_AT_SHARD70_2026_08_11"]            # shard 70, same contrast
    comm = q["legs_untouched"]["communication_like_for_like_resnet50gn_400r"]
    dev = v["ondevice_step_cost_vivo_sdm845"]

    full_mb = comm["full_bytes"] / 1e6
    frozen_mb = comm["frozen_bytes"] / 1e6

    return {
        "$schema_version": 1,
        "generated_by": "scripts/build_arm_tradeoff.py",
        "source": "research/results/frozen-backbone/VERDICT_frozen_vs_full.json",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_date": v["date"],

        # The one-line trade-off, taken from the verdict's own reading of the controlled contrast.
        "headline": (
            f"Full fine-tuning buys +{ctrl['full_minus_frozen']:.4f} AUC for "
            f"{comm['ratio']:,}x the communication. Defensible in a datacenter, not on a phone."),

        "measured_on": {
            "task": "chest X-ray pneumonia, binary AUC",
            "backbone": "timm resnet50_gn.a1h_in1k (identical for both arms)",
            "protocol": "400 rounds, 3 seeds, alpha=1.0, 20 clients, 10/round, 3 local epochs",
            "accuracy_hardware": "RTX 4060",
            "ondevice_hardware": "vivo 1805 (SDM845), ExecuTorch portable",
        },

        "arms": {
            "FULL": {
                "accuracy_auc": round(ctrl["full_auc_mean"], 4),
                "accuracy_delta_vs_frozen": round(ctrl["full_minus_frozen"], 4),
                "accuracy_sign_consistent": bool(ctrl["sign_consistent"]) and bool(rep["sign_consistent"]),
                "accuracy_seeds_agreeing": "6/6 (3 at shard 10, 3 at shard 70)",
                "comm_total_mb_400r": round(full_mb, 1),
                "ondevice_ms_per_step": dev["full_executorch_portable_ms"],
                "ondevice_feasible": False,
                "summary": "Higher accuracy. Needs a datacenter — a single on-device step costs "
                           f"{dev['full_executorch_portable_ms'] / 1000:.1f} s, so a federated "
                           "round is not feasible on a phone.",
            },
            "FROZEN_HEAD": {
                "accuracy_auc": round(ctrl["frozen_auc_mean"], 4),
                "accuracy_delta_vs_frozen": 0.0,
                "accuracy_sign_consistent": True,
                "accuracy_seeds_agreeing": "6/6 (3 at shard 10, 3 at shard 70)",
                "comm_total_mb_400r": round(frozen_mb, 1),
                "ondevice_ms_per_step": dev["frozen_executorch_portable_ms"],
                "ondevice_feasible": True,
                "summary": f"{comm['ratio']:,}x less communication and the only arm that runs "
                           f"on-device ({dev['frozen_executorch_portable_ms']} ms/step). Costs "
                           f"{ctrl['full_minus_frozen']:.4f} AUC against a full fine-tune.",
            },
        },

        "comm_ratio": comm["ratio"],
        # The record's OWN like-for-like on-device ratio (LiteRT both sides), not a ratio computed
        # here. Dividing the two ExecuTorch-portable numbers instead would yield ~401,800x — a
        # figure that is arithmetically real but dominated by ExecuTorch portable being a poor
        # kernel for full backprop, and one the verdict deliberately did not headline. Quoting it
        # would be picking the most extreme available framing.
        "ondevice_ratio": round(dev["ratio_like_for_like_litert"]),
        "ondevice_ratio_basis": "LiteRT Flex on both arms (the record's like-for-like comparison)",

        # Shown in the UI, not buried. A number without its caveat is a claim the record does not
        # support, and every one of these is a limit a reviewer would raise.
        "caveats": [
            "One task (chest X-ray), one alpha (1.0), one architecture family, three seeds per cell.",
            "The communication ratio is ROUND-BUDGET DEPENDENT: it grows with the number of rounds "
            "because the frozen arm's one-shot backbone delivery is amortised over them. Quoted at "
            "400 rounds.",
            "Accuracy and on-device latency were measured on DIFFERENT hardware (RTX 4060 and an "
            "SDM845 handset). The joint claim composes two measurements; no single cell measures both.",
            "On-device numbers are single-handset.",
            "No differential privacy evaluated. DP noise scales with parameter count and should "
            "favour the frozen arm further, but that is an expectation, not a measurement.",
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the committed file matches the record; do not write")
    ap.add_argument("--record", default=RECORD)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if not os.path.exists(args.record):
        sys.exit(f"record not found: {args.record}\nThe research tree is untracked and local-only; "
                 f"regenerating requires it. Nothing was written.")

    built = build(args.record)
    if args.check:
        if not os.path.exists(args.out):
            sys.exit(f"{args.out} does not exist; run: python scripts/build_arm_tradeoff.py")
        with open(args.out) as fh:
            have = json.load(fh)
        if have != built:
            sys.exit(f"{args.out} is STALE relative to the record. "
                     f"Regenerate: python scripts/build_arm_tradeoff.py")
        print(f"{args.out} is current (source sha256 {built['source_sha256'][:12]})")
        return

    with open(args.out, "w") as fh:
        json.dump(built, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.out}  (source sha256 {built['source_sha256'][:12]})")
    print(f"  {built['headline']}")


if __name__ == "__main__":
    main()
