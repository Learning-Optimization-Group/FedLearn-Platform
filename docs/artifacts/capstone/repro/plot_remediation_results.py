"""Plot only public pilot/finalist analysis summaries; never fit or load a bundle."""
import argparse
import copy
import hashlib
from importlib.metadata import version
import json
import math
from numbers import Real
from pathlib import Path
import statistics

METHODS = ("mean512", "projected128", "trim16", "center_quarter")
LABELS = ("Mean\n512", "Projected mean\n128", "Trimmed mean\n16", "Centered clip\nC/4")
COLORS = ("#246a86", "#b64f38")


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("finite numeric plot value required")
    return float(value)


def _close(left, right):
    if not math.isclose(_finite(left), _finite(right), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("summary mean, SD, or paired effect disagrees with per-seed evidence")


def _select(summary, n, **identity):
    found = [arm for arm in summary["arms"] if all(arm.get(k) == v for k, v in identity.items())]
    if len(found) != 1:
        raise ValueError("missing or ambiguous requested plot arm")
    arm = found[0]
    if (arm.get("status") != "complete" or arm.get("n") != n or arm.get("planned_n") != n
            or not isinstance(arm.get("per_seed"), dict) or len(arm["per_seed"]) != n):
        raise ValueError("complete declared seed budget required for plot arm")
    per_seed = {str(seed): _finite(metrics["auroc"]) for seed, metrics in arm["per_seed"].items()}
    if any(not 0 <= value <= 1 for value in per_seed.values()):
        raise ValueError("AUROC must be between zero and one")
    values = list(per_seed.values())
    mean, sd = statistics.mean(values), statistics.stdev(values)
    _close(arm["metrics"]["auroc"]["mean"], mean)
    _close(arm["metrics"]["auroc"]["sample_sd"], sd)
    return dict(identity, arm_id=arm["id"], n=n, mean=mean, sample_sd=sd, per_seed=per_seed)


def _paired(rows):
    seed_sets = {tuple(sorted(row["per_seed"])) for row in rows}
    if len(seed_sets) != 1:
        raise ValueError("plotted arms must share paired training seed sets")


def collect_data(pilot, finalists):
    """Validate selections and expose exact public numbers used in every figure."""
    for value, schema in ((pilot, "capstone.effusion_pilot_analysis.v1"),
                          (finalists, "capstone.effusion_finalist_analysis.v1")):
        if value.get("schema") != schema or value.get("status") != "complete":
            raise ValueError("complete public analysis summary required")
    projection = [_select(pilot, 2, tag="mean" if width == 512 else f"mean_k{width}",
                          backend="byzfl", private=private, attack="clean", width=width,
                          rule="mean", kind="fl")
                  for private in (False, True) for width in (64, 128, 256, 512)]
    buckets = [_select(pilot, 2, tag=f"median_b{bucket}_{partition}_{suffix}", backend="byzfl",
                       private=False, attack=attack, bucket=bucket, alpha=alpha,
                       fraction=0. if attack == "clean" else .2, rule="median", kind="fl")
               for partition, alpha in (("iid", None), ("a0p1", .1)) for bucket in (1, 2, 4)
               for attack, suffix in (("clean", "clean"), ("owned_pgd", "pgd_f16"))]
    final = [_select(finalists, 12, method=method, private=private, attack=attack)
             for private in (False, True) for method in METHODS for attack in ("clean", "owned_pgd")]
    for rows in (projection, buckets, final):
        _paired(rows)
    primary = copy.deepcopy(finalists.get("primary"))
    if (not isinstance(primary, dict) or primary.get("contrast") !=
            "alternative_minus_mean:projected128:private:clean"):
        raise ValueError("declared primary contrast required")
    _close(primary["technical_margin"], .01)
    for key in ("seed_interval", "patient_interval"):
        interval = primary.get(key)
        if not isinstance(interval, dict) or not interval.get("scope"):
            raise ValueError("both separately scoped primary intervals required")
        for field in ("point", "ci_low", "ci_high"):
            _finite(interval[field])
        if interval["ci_low"] > interval["ci_high"]:
            raise ValueError("primary interval endpoints are reversed")
    seed = primary["seed_interval"]
    if seed.get("n") != 12 or len(seed.get("seed_effects", {})) != 12:
        raise ValueError("twelve primary paired seed effects required")
    left = next(r for r in final if r["method"] == "projected128" and r["private"] and r["attack"] == "clean")
    right = next(r for r in final if r["method"] == "mean512" and r["private"] and r["attack"] == "clean")
    effects = {str(s): _finite(v) for s, v in seed["seed_effects"].items()}
    if set(effects) != set(left["per_seed"]):
        raise ValueError("primary paired seed identities disagree with arms")
    for s, value in effects.items():
        _close(value, left["per_seed"][s] - right["per_seed"][s])
    _close(seed["point"], statistics.mean(effects.values()))
    _close(primary["patient_interval"]["point"], seed["point"])
    # Keep only public plotting fields, not arbitrary summary provenance payloads.
    primary = dict(contrast=primary["contrast"], technical_margin=.01,
                   seed_interval={k: seed[k] for k in ("n", "point", "ci_low", "ci_high", "scope")}
                                 | {"seed_effects": effects},
                   patient_interval={k: primary["patient_interval"][k]
                                     for k in ("point", "ci_low", "ci_high", "scope")})
    return dict(schema="capstone.remediation_plot_data.v1", projection=projection, buckets=buckets,
                finalists=final, primary=primary,
                descriptive_error_bars="mean plus/minus one sample SD across paired training seeds; not a patient CI",
                interval_scope="Separate fixed-cohort training-seed and fixed-head patient intervals; no joint uncertainty.")


def draw_figures(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
                         "axes.labelsize": 11, "legend.fontsize": 10, "svg.fonttype": "none",
                         "svg.hashsalt": "capstone-remediation-v1",
                         "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=220, facecolor="white", metadata={"Software": ""})
        fig.savefig(output / f"{name}.svg", facecolor="white", metadata={"Creator": "", "Date": None})
        svg = output / f"{name}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)

    def axes_style(ax):
        ax.grid(axis="y", color="#e0e4e7", linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_ylabel("Development-selection AUROC")

    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    for private, color in zip((False, True), COLORS):
        rows = [r for r in data["projection"] if r["private"] == private]
        ax.errorbar(np.arange(4), [r["mean"] for r in rows], yerr=[r["sample_sd"] for r in rows],
                    marker="o", markersize=6, capsize=4, color=color,
                    label="Central GDP, epsilon=4" if private else "Zero noise")
    ax.set_xticks(np.arange(4), ["64", "128", "256", "512\n(unprojected)"])
    ax.set_xlabel("Head input width (fixed public projection below 512)")
    ax.set_title("Projection pilot: clean ByzFL mean\nTwo paired seeds; whiskers = +/-1 seed SD (descriptive)")
    axes_style(ax)
    ax.legend(loc="best", frameon=False)
    save(fig, "pilot-projection")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True, layout="constrained")
    for ax, alpha, title in zip(axes, (None, .1), ("Patient IID", "Label-skew, alpha=0.1")):
        for attack, color in zip(("clean", "owned_pgd"), COLORS):
            rows = [r for r in data["buckets"] if r["alpha"] == alpha and r["attack"] == attack]
            ax.errorbar(np.arange(3), [r["mean"] for r in rows], yerr=[r["sample_sd"] for r in rows],
                        marker="o", capsize=4, color=color,
                        label="Clean" if attack == "clean" else "Owned PGD, 16/80 clients")
        ax.set_xticks(np.arange(3), ["1", "2", "4"])
        ax.set_xlabel("Bucket size")
        ax.set_title(title)
        axes_style(ax)
        ax.legend(frameon=False, loc="lower left")
    axes[1].set_ylabel("")
    fig.suptitle("Median bucket pilot, zero noise\nTwo paired seeds; whiskers = +/-1 seed SD (descriptive)", fontsize=13)
    save(fig, "pilot-buckets")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True, layout="constrained")
    for ax, private in zip(axes, (False, True)):
        for index, method in enumerate(METHODS):
            paired = [next(r for r in data["finalists"] if r["private"] == private
                           and r["method"] == method and r["attack"] == attack)
                      for attack in ("clean", "owned_pgd")]
            for seed in sorted(paired[0]["per_seed"]):
                ax.plot([index - .13, index + .13], [r["per_seed"][seed] for r in paired],
                        color="#8c969e", alpha=.22, linewidth=.7, zorder=1)
            for offset, row, color in zip((-.13, .13), paired, COLORS):
                ax.errorbar(index + offset, row["mean"], yerr=row["sample_sd"], marker="o",
                            capsize=4, color=color, zorder=3,
                            label=("Clean" if row["attack"] == "clean" else "Owned PGD, 16/80 clients")
                            if index == 0 else None)
        ax.set_xticks(np.arange(4), LABELS)
        ax.set_title("Central GDP, epsilon=4" if private else "Zero noise")
        axes_style(ax)
        ax.legend(frameon=False, loc="best")
    axes[1].set_ylabel("")
    fig.suptitle("Finalist policies: 12 paired training seeds per condition\nWhiskers = +/-1 across-seed SD, descriptive; not a patient CI", fontsize=13)
    save(fig, "finalist-methods")

    primary = data["primary"]
    seed_interval, patient_interval = primary["seed_interval"], primary["patient_interval"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained", gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    effects = sorted(seed_interval["seed_effects"].items(), key=lambda item: int(item[0]))
    ax.scatter(np.arange(len(effects)), [v for _, v in effects], color=COLORS[0], s=35)
    ax.axhline(seed_interval["point"], color=COLORS[0], label="Mean paired gain")
    ax.axhline(.01, color=COLORS[1], linestyle="--", label="Technical margin: 0.01")
    ax.axhline(0, color="#7c858a", linewidth=.8)
    ax.set_xticks(np.arange(len(effects)), [s for s, _ in effects], rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Fresh training seed")
    ax.set_ylabel("Paired AUROC gain: projected 128 minus mean 512")
    ax.set_title("Primary paired gains: clean, central GDP epsilon=4")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.2)
    ax = axes[1]
    for y, interval, color in zip((1, 0), (seed_interval, patient_interval), COLORS):
        ax.hlines(y, interval["ci_low"], interval["ci_high"], color=color, linewidth=2.2)
        ax.scatter([interval["point"]], [y], color=color, s=35)
        ax.text(interval["point"], y + .14,
                f'{interval["point"]:.4f} [{interval["ci_low"]:.4f}, {interval["ci_high"]:.4f}]',
                ha="center", va="bottom", fontsize=9)
    ax.axvline(.01, color=COLORS[1], linestyle="--", linewidth=1)
    ax.axvline(0, color="#7c858a", linewidth=.8)
    ax.set_yticks([1, 0], ["Training-seed t\nfixed cohort", "Patient bootstrap\nfixed realized heads"])
    ax.set_ylim(-.55, 1.6)
    ax.margins(x=.15)
    ax.set_xlabel("Paired AUROC gain")
    ax.set_title("Separate 95% intervals\nDifferent targets; no joint uncertainty")
    ax.grid(axis="x", alpha=.2)
    save(fig, "primary-paired-gain")


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def main(pilot_summary, finalist_summary, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("use a fresh figure output directory")
    # Read exactly these two supplied public JSON files; there is no bundle loader.
    raw = {"pilot": Path(pilot_summary).read_bytes(), "finalists": Path(finalist_summary).read_bytes()}
    data = collect_data(json.loads(raw["pilot"]), json.loads(raw["finalists"]))
    output.mkdir(parents=True, exist_ok=False)
    (output / "plot-data.json").write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    draw_figures(data, output)
    receipt = dict(schema="capstone.remediation_plot_receipt.v1",
                   input_sha256={name: _hash(content) for name, content in raw.items()},
                   renderer_sha256=_hash(Path(__file__).read_bytes()),
                   dependencies={name: version(name) for name in ("matplotlib", "numpy")},
                   output_sha256={p.name: _hash(p.read_bytes()) for p in sorted(output.iterdir())},
                   scope="Public aggregate figures on a repeatedly used development-selection cohort; descriptive SD bars and separate interval targets.")
    (output / "plot-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--finalist-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(main(args.pilot_summary, args.finalist_summary, args.output), sort_keys=True))
