"""Render a selection-only scalar reanalysis as a reproducible local addendum."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


CASES = [
    ("A11", "auroc|local|eps1", "Local DP epsilon 1"),
    ("A11", "local-central|eps8", "Local minus central, epsilon 8"),
    ("A12", "drop|eps8|mean|sign_flip", "Local mean sign-flip damage, epsilon 8"),
    ("A12", "drop|eps8|mean|label_flip", "Local mean label-flip damage, epsilon 8"),
    ("A19", "nonIidMinusIid|a0p1|dp4|mean|clean", "Strong-skew private AUROC change"),
    ("A22", "minusR0|R2|eps8", "DINOv2 minus R0, epsilon 8"),
    ("A22", "minusR0|R2|eps4", "DINOv2 minus R0, epsilon 4"),
    ("A22", "dpCostMinusR0dpCost|R2|eps4", "DINOv2 minus R0 privacy cost, epsilon 4"),
    ("A23", "distMinusRdp|b16|eps4", "Distributed b16 minus central RDP, epsilon 4"),
    ("A23", "distMinusRdp|b16|eps1", "Distributed b16 minus central RDP, epsilon 1"),
    ("A23", "damageDistMinusRdp|signflip4", "Extra distributed damage, four attackers"),
]


def interval(pair):
    return f"[{pair[0]:+.4f}, {pair[1]:+.4f}]"


def render(analysis, output_dir):
    analysis, output_dir = Path(analysis), Path(output_dir)
    data = json.loads(analysis.read_text())
    if data.get("schema") != "capstone.selection_seed_reanalysis.v1" or set(data["studies"]) != {"A11", "A12", "A19", "A22", "A23"}:
        raise ValueError("unsupported reanalysis schema or study coverage")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "selection-seed-reanalysis-2026-10-24"
    public_json, public_csv = output_dir / f"{stem}.json", output_dir / f"{stem}.csv"
    for target in (public_json, public_csv, output_dir / f"{stem}.md", output_dir / f"{stem}.html"):
        if target.exists():
            raise FileExistsError(f"addendum already exists: {target.name}")
    shutil.copyfile(analysis, public_json)
    shutil.copyfile(analysis.with_name("contrasts.csv"), public_csv)
    lines = ["# Selection-only paired-seed reanalysis", "", "24 October 2026. Retrospective correction of uncertainty interpretation; no new model fits.", "",
             "Eight pinned aggregate summaries were hash-verified. All 173 original contrast points in A11, A12, A19, A22 and A23 were reproduced from saved per-run AUROC scalars to tolerance 1e-10. No patient manifests, per-image scores or raw holdout/test inputs were read. Original artifacts and conditional patient intervals remain unchanged.", "",
             "## What these intervals measure", "",
             "The original patient-cluster intervals condition on five realized heads. The new paired seed-mean t intervals describe variation across those five development training replicates on the fixed selection cohort. These are separate uncertainty targets: the new intervals can be wider or narrower, and neither is a joint patient-and-training interval.", "",
             "The t intervals are approximate, retrospective and unadjusted for multiple comparisons. Normality is unverified. Five seeds, including seeds used during recipe development, do not establish independent confirmation, equivalence, population generalization or privacy of the research exports.", "",
             "## Complete coverage", "", "| Study | Original contrasts reproduced | Effect intervals excluding zero: patient | Effect intervals excluding zero: seed |", "| --- | ---: | ---: | ---: |"]
    for study, contrasts in data["studies"].items():
        effects = [v for k, v in contrasts.items() if not k.startswith("auroc|")]
        lines.append(f"| {study} | {len(contrasts)} | {sum(v['conditional_interval_excludes_zero'] for v in effects)} | {sum(v['seed_interval_excludes_zero'] for v in effects)} |")
    lines += ["", "Interval classifications are descriptive, not multiplicity-adjusted discovery counts. Absolute-AUROC contrasts are excluded from the zero-comparison columns.", "",
              "## Selected decision-relevant contrasts", "", "| Study / contrast | Point | Original conditional patient 95% interval | New paired-seed 95% interval |", "| --- | ---: | --- | --- |"]
    for study, key, label in CASES:
        result = data["studies"][study][key]
        lines.append(f"| {study}: {label} | {result['seed']['point']:+.4f} | {interval(result['conditional_patient_interval95'])} | {interval(result['seed']['interval95'])} |")
    lines += ["", "## Interpretation", "",
              "- A11's epsilon-1 local-DP interval is much wider over seeds than over patients. Its observed point remains weak, while the local-minus-central epsilon-8 gap remains negative under the seed view. This is not a conversion between their different trust guarantees.",
              "- A12's large attack damage survives the seed analysis in the highlighted mean cells. This supports a failure of the tested local protocol/rules, not a general impossibility for all local-DP defences.",
              "- A19's strong-skew private utility decline remains negative. The additional privacy-cost contrast below separates that decline from the clean nonprivate decline.",
              "- A22's epsilon-8 encoder gain has a positive seed interval while its original patient interval includes zero. At epsilon 4 the gain remains unresolved; all DP-cost differences remain unresolved. This illustrates the different inference targets and does not justify either equivalence or a confirmed population-wide encoder upgrade.",
              "- A23's b16-minus-central intervals permit material gains and losses. Close calibrated noise and a seed interval crossing zero do not establish utility equivalence. The large additional damage from four clip-skipping clients remains clear across seeds.", "",
              "## Additional exploratory A19 privacy-cost contrasts", "",
              "These four-term contrasts were added retrospectively: (non-IID clean minus private) minus (IID clean minus private). They are not original preregistered endpoints.", "",
              "| Skew | Additional privacy cost | Paired-seed 95% interval |", "| --- | ---: | --- |"]
    for key, result in data["additional_A19_incremental_cost"].items():
        lines.append(f"| {key.split('|')[1]} | {result['point']:+.5f} | {interval(result['interval95'])} |")
    lines += ["", "The strong-skew value is 0.03253 from saved run scalars; subtraction of the rounded published table entries gives approximately 0.0326. It is not the full 0.03836 private-AUROC decline.", "",
              "## Claim boundaries", "",
              "A14 raw-honest oracle attacked cells retain only nominal aggregate-input noise calibration; the reused one-upload argument does not establish their client-data epsilon. A23's mu/N noise allocation does not establish protection against its coded-share collusion threshold. Those mechanism issues are independent of this statistical reanalysis. Public/shared preprocessing, deterministic research randomness, selection/tuning and diagnostic export limitations remain.", "",
              "## Reproduction and provenance", "",
              f"[All derived contrasts]({public_json.name}) and [CSV table]({public_csv.name}) contain only aggregate metrics and nonidentifying source provenance. SHA-256 of the JSON: `{hashlib.sha256(public_json.read_bytes()).hexdigest()}`.", "",
              "Run `repro/reanalyze_selection_seeds.py` with the declared selection evidence root and a fresh output directory, then render this addendum with `repro/render_selection_seed_addendum.py`. The analysis script accepts only its eight pinned summary inputs and reconciles complete seed/contrast coverage before writing results.", "",
              "The analysis and contrast-specification tests pass 30 synthetic checks. No full framework gate was repeated for this analysis-only phase; the prior ByzFL gate remains historical evidence.", ""]
    markdown = "\n".join(lines)
    source = output_dir / f"{stem}.md"
    source.write_text(markdown)
    renderer_path = output_dir / "review-2026-10-23" / "render_review.py"
    spec = importlib.util.spec_from_file_location("review_renderer", renderer_path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    header = (output_dir / "review-notes-2026-10-23.html").read_text().split("<main>", 1)[0]
    header = header.replace("Capstone adversarial review — 23 October 2026", "Selection-only seed reanalysis — 24 October 2026")
    source.with_suffix(".html").write_text(header + "<main>" + renderer.render(markdown.splitlines()) + "</main></body></html>\n")
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.analysis, args.output_dir).name)
