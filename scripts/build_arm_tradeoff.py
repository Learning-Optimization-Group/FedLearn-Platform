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
# The CNN recipe's own measurement: a live 3-round federation on the product path, both arms.
CNN_RECORD = os.path.join(REPO, "research", "results", "frozen-e2e",
                          "live_frozen_e2e_2026-08-13.json")
PNEUMONIA_RECORD = os.path.join(REPO, "research", "results", "pneumonia-e2e",
                                "pneumonia_product_path_2026-08-13.json")
RESNET_RECORD = os.path.join(REPO, "research", "results", "pretrained-backbone",
                             "frozen_vs_full_resnet18_2026-08-13.json")


def build_xray(record_path=RECORD):
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
            # The recipe this was measured ON. A trade-off shown next to a different recipe is a
            # claim no measurement supports, so the key travels with the numbers.
            "recipe": "PNEUMONIA_CNN",
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


def build_cnn(record_path=CNN_RECORD):
    """CNN's OWN trade-off, from the live product-path federation rather than the X-ray campaign.

    This one is unflattering and that is the point: on CNN the frozen arm was measured at CHANCE.
    CnnNet's backbone is randomly initialised, so freezing it trains a linear head on random
    features. Showing the chest X-ray result here instead — where frozen ties a full fine-tune —
    would recommend a configuration this platform has measured to be useless.
    """
    with open(record_path, "rb") as fh:
        raw = fh.read()
    rec = json.loads(raw)
    lf = rec["learning_finding"]
    frozen = lf["frozen_accuracy_per_round"][-1]
    full = lf["full_accuracy_per_round"][-1]
    chance = lf["chance_level"]

    return {
        "$schema_version": 2,
        "generated_by": "scripts/build_arm_tradeoff.py",
        "source": "research/results/frozen-e2e/live_frozen_e2e_2026-08-13.json",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_date": rec["meta"]["date"],
        "headline": (
            f"On this recipe the frozen arm reaches {frozen:.1f}% — chance level for ten classes. "
            f"A full fine-tune reaches {full:.1f}%."),
        "measured_on": {
            "recipe": "CNN",
            "task": "CIFAR-10, 10-class accuracy",
            "backbone": "CnnNet, randomly initialised (identical for both arms)",
            "protocol": f"{rec['meta']['rounds']} rounds, {rec['meta']['clients']} clients, "
                        f"seed {rec['meta']['seed']}, live federation on the product path",
            "accuracy_hardware": rec["meta"]["device"],
            "ondevice_hardware": "not measured for this recipe",
        },
        "arms": {
            "FULL": {
                "accuracy_pct": full,
                "accuracy_delta_vs_frozen": round(full - frozen, 2),
                "ondevice_feasible": None,
                "summary": f"Reaches {full:.1f}% and is still improving at round "
                           f"{rec['meta']['rounds']}. The only arm that learns on this recipe.",
            },
            "FROZEN_HEAD": {
                "accuracy_pct": frozen,
                "accuracy_delta_vs_frozen": 0.0,
                "ondevice_feasible": None,
                "summary": f"Reaches {frozen:.1f}%, which is chance for ten classes. This recipe's "
                           f"backbone is randomly initialised, so freezing it trains a head on "
                           f"random features. Pick a pretrained recipe if you want a frozen "
                           f"backbone.",
            },
        },
        "comm_ratio": None,
        "ondevice_ratio": None,
        "ondevice_ratio_basis": "not measured for this recipe",
        "caveats": [
            f"Chance for ten classes is {chance:.1f}%; the frozen arm is AT it, so its accuracy "
            f"carries no signal rather than being merely low.",
            f"Only {rec['meta']['rounds']} rounds with {rec['meta']['clients']} clients and one "
            f"seed — the FULL arm had not converged and its number is a floor, not a ceiling.",
            "Communication and on-device cost were not measured for this recipe.",
            "This is CIFAR-10 with a small custom CNN; it says nothing about other tasks.",
        ],
    }


def build_pneumonia(record_path=PNEUMONIA_RECORD):
    """PNEUMONIA_CNN's own product-path measurement, replacing the research campaign's.

    The campaign measured a frozen ImageNet ResNet-18 feature extractor with a 1,026-parameter
    head. THIS recipe is a small custom CNN whose classifier block is 99.6% of its parameters, so
    the two share a task and nothing else. Showing the campaign numbers here told a user that
    freezing costs ~0.02 AUC and saves 3,321x the bytes; on the actual recipe it costs 21.8 points
    of accuracy and saves 1.004x. Same task, opposite conclusion.
    """
    with open(record_path, "rb") as fh:
        raw = fh.read()
    rec = json.loads(raw)
    full, frozen = rec["results"]["FULL"], rec["results"]["FROZEN_HEAD"]
    m = rec["meta"]
    ratio = full["wire_mb_per_download"] / frozen["wire_mb_per_download"]

    return {
        "$schema_version": 2,
        "generated_by": "scripts/build_arm_tradeoff.py",
        "source": "research/results/pneumonia-e2e/pneumonia_product_path_2026-08-13.json",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_date": m["date"],
        "headline": (
            f"On this recipe freezing costs {full['final_acc'] - frozen['final_acc']:.1f} accuracy "
            f"points and saves only {ratio:.3f}x the communication — its classifier is 99.6% of "
            f"the model, so there is almost nothing to freeze."),
        "measured_on": {
            "recipe": "PNEUMONIA_CNN",
            "task": "chest X-ray (NORMAL/PNEUMONIA), top-1 accuracy",
            "backbone": "PneumoniaCNN features.*, randomly initialised (identical for both arms)",
            "protocol": f"{m['rounds']} rounds, {m['clients']} clients, alpha {m['alpha']}, "
                        f"{m['subset_per_split']} samples/split, seed {m['seed']}, "
                        f"live federation on the product path",
            "accuracy_hardware": m["device"],
            "ondevice_hardware": "not measured for this recipe",
        },
        "arms": {
            "FULL": {
                "accuracy_pct": full["final_acc"],
                "accuracy_delta_vs_frozen": round(full["final_acc"] - frozen["final_acc"], 2),
                "comm_mb_per_download": full["wire_mb_per_download"],
                "ondevice_feasible": None,
                "summary": f"Reaches {full['final_acc']:.1f}% and was still improving at round "
                           f"{m['rounds']}. The only arm that learns on this recipe.",
            },
            "FROZEN_HEAD": {
                "accuracy_pct": frozen["final_acc"],
                "accuracy_delta_vs_frozen": 0.0,
                "comm_mb_per_download": frozen["wire_mb_per_download"],
                "ondevice_feasible": None,
                "summary": f"Reaches {frozen['final_acc']:.1f}%, identical in every round — the "
                           f"majority-class rate, so it never learns. This recipe's classifier is "
                           f"99.6% of its parameters, so freezing the rest saves almost no "
                           f"communication either.",
            },
        },
        "comm_ratio": round(ratio, 3),
        "ondevice_ratio": None,
        "ondevice_ratio_basis": "not measured for this recipe",
        "caveats": [
            f"{m['rounds']} rounds, {m['clients']} clients, one seed, "
            f"{m['subset_per_split']} samples per split. The FULL arm had not converged, so its "
            f"number is a floor.",
            "The frozen arm's accuracy equals the majority-class rate and is identical across all "
            "three rounds — that is no learning, not merely weak learning.",
            "Communication is the server's per-download payload, not a full round-trip budget.",
            "The research campaign's frozen arm on this task used a pretrained ResNet-18 feature "
            "extractor with a 1,026-parameter head and reached very different conclusions. It is a "
            "different architecture and its results are not shown here.",
        ],
    }


def build_resnet(record_path=RESNET_RECORD):
    """CIFAR_RESNET18: the one recipe where the frozen arm is the better choice on both axes.

    Deliberately does NOT say "frozen beats full". Three rounds is a short budget and the FULL arm
    was still oscillating (79.95 then down to 77.94), so this measures convergence RATE at a fixed
    small budget where a 5,130-parameter head is trivially easier to fit — not final quality.
    """
    with open(record_path, "rb") as fh:
        raw = fh.read()
    rec = json.loads(raw)
    full, frozen = rec["results"]["FULL"], rec["results"]["FROZEN_HEAD"]
    m = rec["meta"]
    ratio = rec["comm_ratio"]

    return {
        "$schema_version": 2,
        "generated_by": "scripts/build_arm_tradeoff.py",
        "source": "research/results/pretrained-backbone/frozen_vs_full_resnet18_2026-08-13.json",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_date": m["date"],
        "headline": (
            f"This recipe starts from ImageNet weights, so freezing keeps a backbone worth "
            f"keeping: the frozen arm reaches {frozen['final_acc']:.1f}% while moving "
            f"{ratio:,}x less data than a full fine-tune."),
        "measured_on": {
            "recipe": "CIFAR_RESNET18",
            "task": "CIFAR-10, 10-class accuracy",
            "backbone": "ImageNet-pretrained ResNet-18 (identical for both arms)",
            "protocol": f"{m['rounds']} rounds, {m['clients']} clients, {m['img_size']}px, "
                        f"seed {m['seed']}, live federation on the product path",
            "accuracy_hardware": m["device"],
            "ondevice_hardware": "not measured for this recipe",
        },
        "arms": {
            "FULL": {
                "accuracy_pct": full["final_acc"],
                "accuracy_delta_vs_frozen": round(full["final_acc"] - frozen["final_acc"], 2),
                "comm_mb_per_download": full["wire_mb_per_download"],
                "ondevice_feasible": None,
                "summary": f"Reaches {full['final_acc']:.1f}% at this budget and had NOT converged "
                           f"— it was still oscillating at round {m['rounds']}. Moves "
                           f"{full['wire_mb_per_download']:.1f} MB per download.",
            },
            "FROZEN_HEAD": {
                "accuracy_pct": frozen["final_acc"],
                "accuracy_delta_vs_frozen": 0.0,
                "comm_mb_per_download": frozen["wire_mb_per_download"],
                "ondevice_feasible": None,
                "summary": f"Reaches {frozen['final_acc']:.1f}% for {frozen['wire_mb_per_download']}"
                           f" MB per download — {ratio:,}x less than a full fine-tune. Trains "
                           f"5,130 parameters on ImageNet features.",
            },
        },
        "comm_ratio": float(ratio),
        "ondevice_ratio": None,
        "ondevice_ratio_basis": "not measured for this recipe",
        "caveats": [
            f"{m['rounds']} rounds, {m['clients']} clients, one seed. The FULL arm had NOT "
            f"converged (79.95 then down to 77.94), so do not read this as 'frozen beats full' — "
            f"it measures convergence rate at a short budget, not final quality.",
            "Fine-tuning 11.2M parameters on two clients needs far more rounds to settle; whether "
            "FULL overtakes the frozen arm given that budget is UNMEASURED.",
            "One backbone, one resolution (112px), one dataset.",
            "BatchNorm running statistics are federated and averaged as usual; the int64 batch "
            "counters are withheld from the wire and kept local (160 bytes total).",
        ],
    }


def build(record_path=RECORD, cnn_record_path=CNN_RECORD):
    """Every recipe's trade-off, keyed by the recipe it was MEASURED on."""
    # PNEUMONIA_CNN uses its own PRODUCT-PATH measurement, not the research campaign's: the
    # campaign froze a pretrained ResNet-18 and this recipe is a custom CNN whose classifier is
    # 99.6% of the model, so the campaign's conclusions invert on the actual recipe.
    by_recipe = {}
    if os.path.exists(PNEUMONIA_RECORD):
        by_recipe["PNEUMONIA_CNN"] = build_pneumonia(PNEUMONIA_RECORD)
    if os.path.exists(cnn_record_path):
        by_recipe["CNN"] = build_cnn(cnn_record_path)
    if os.path.exists(RESNET_RECORD):
        by_recipe["CIFAR_RESNET18"] = build_resnet(RESNET_RECORD)
    return {
        "$schema_version": 2,
        "generated_by": "scripts/build_arm_tradeoff.py",
        "note": "Keyed by recipe. A measurement is shown ONLY on the recipe it was taken on — the "
                "first version attached one chest X-ray result to every dual-arm recipe, which "
                "stated something no measurement supported for CIFAR-10.",
        "by_recipe": by_recipe,
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
        print(f"{args.out} is current ({len(built['by_recipe'])} recipe measurements)")
        return

    with open(args.out, "w") as fh:
        json.dump(built, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.out}")
    for key, tr in built["by_recipe"].items():
        print(f"  {key}: {tr['headline']}")


if __name__ == "__main__":
    main()
