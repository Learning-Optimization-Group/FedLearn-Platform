"""Public aggregate plotting checks with wholly synthetic summaries."""
import copy
import importlib
import json
import statistics

import pytest


def module():
    return importlib.import_module("plot_remediation_results")


def arm(tag, values, **identity):
    return dict(id=tag, tag=tag, status="complete", n=len(values), planned_n=len(values),
                per_seed={str(i + 1): {"auroc": value} for i, value in enumerate(values)},
                metrics={"auroc": dict(n=len(values), mean=statistics.mean(values),
                                        sample_sd=statistics.stdev(values))}, **identity)


def summaries():
    pilot = dict(schema="capstone.effusion_pilot_analysis.v1", status="complete", arms=[])
    for private in (False, True):
        for width in (64, 128, 256, 512):
            tag = "mean" if width == 512 else f"mean_k{width}"
            pilot["arms"].append(arm(tag, [.6, .7], backend="byzfl", private=private,
                attack="clean", width=width, rule="mean", kind="fl", fraction=0., bucket=1, alpha=None))
    for partition, alpha in (("iid", None), ("a0p1", .1)):
        for bucket in (1, 2, 4):
            for attack, suffix, fraction in (("clean", "clean", 0.), ("owned_pgd", "pgd_f16", .2)):
                tag = f"median_b{bucket}_{partition}_{suffix}"
                pilot["arms"].append(arm(tag, [.55, .75], backend="byzfl", private=False,
                    attack=attack, width=512, rule="median", kind="fl", bucket=bucket,
                    alpha=alpha, fraction=fraction))
    final = dict(schema="capstone.effusion_finalist_analysis.v1", status="complete", arms=[])
    for method in ("mean512", "projected128", "trim16", "center_quarter"):
        for private in (False, True):
            for attack in ("clean", "owned_pgd"):
                values = [.6 + i * .001 + (.02 if method == "projected128" else 0.) for i in range(12)]
                final["arms"].append(arm(f"{method}:{private}:{attack}", values, method=method,
                                          private=private, attack=attack))
    final["primary"] = dict(technical_margin=.01,
        contrast="alternative_minus_mean:projected128:private:clean",
        seed_interval=dict(n=12, point=.02, ci_low=.01, ci_high=.03,
                           seed_effects={str(i + 1): .02 for i in range(12)}, scope="fixed cohort"),
        patient_interval=dict(point=.02, ci_low=.015, ci_high=.025, scope="fixed realized heads"))
    return pilot, final


def test_plot_data_selects_exact_conditions_preserves_seed_values_and_sd():
    pilot, final = summaries()
    pilot["arms"].append(arm("native-decoy", [.1, .2], backend="native", private=False,
                              attack="clean", width=512, rule="mean", kind="fl"))
    data = module().collect_data(pilot, final)
    assert len(data["projection"]) == 8 and len(data["buckets"]) == 12
    assert len(data["finalists"]) == 16
    first = data["projection"][0]
    assert first["mean"] == pytest.approx(.65)
    assert first["sample_sd"] == pytest.approx(.1 / 2 ** .5)
    assert first["per_seed"] == {"1": .6, "2": .7}
    assert data["primary"]["technical_margin"] == .01
    assert data["primary"]["seed_interval"]["point"] == .02
    assert data["primary"]["patient_interval"]["point"] == .02


@pytest.mark.parametrize("mutation", ["incomplete", "missing", "duplicate", "mean", "sd", "nonfinite", "unpaired", "primary"])
def test_refuses_incomplete_ambiguous_or_inconsistent_plot_evidence(mutation):
    pilot, final = summaries()
    if mutation == "incomplete": final["status"] = "incomplete"
    elif mutation == "missing": pilot["arms"].pop()
    elif mutation == "duplicate": pilot["arms"].append(copy.deepcopy(pilot["arms"][0]))
    elif mutation == "mean": final["arms"][0]["metrics"]["auroc"]["mean"] = .9
    elif mutation == "sd": final["arms"][0]["metrics"]["auroc"]["sample_sd"] = .9
    elif mutation == "nonfinite": pilot["arms"][0]["per_seed"]["1"]["auroc"] = float("nan")
    elif mutation == "unpaired": final["arms"][0]["per_seed"]["99"] = final["arms"][0]["per_seed"].pop("1")
    else: final["primary"]["patient_interval"]["point"] = .9
    with pytest.raises(ValueError):
        module().collect_data(pilot, final)


def test_synthetic_run_writes_all_formats_data_hashes_and_refuses_overwrite(tmp_path):
    pilot, final = summaries()
    left, right = tmp_path / "pilot.json", tmp_path / "final.json"
    left.write_text(json.dumps(pilot))
    right.write_text(json.dumps(final))
    output = tmp_path / "figures"
    receipt = module().main(left, right, output)
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.svg"))) == 4
    assert len(receipt["output_sha256"]) == 9
    assert "plot-data.json" in receipt["output_sha256"]
    assert len(receipt["input_sha256"]["pilot"]) == len(receipt["renderer_sha256"]) == 64
    assert str(tmp_path) not in (output / "plot-data.json").read_text()
    repeated = module().main(left, right, tmp_path / "repeated-figures")
    assert repeated["output_sha256"] == receipt["output_sha256"]
    with pytest.raises(FileExistsError):
        module().main(left, right, output)


def test_missing_future_finalist_summary_creates_no_output(tmp_path):
    left = tmp_path / "pilot.json"
    left.write_text(json.dumps(summaries()[0]))
    with pytest.raises(FileNotFoundError):
        module().main(left, tmp_path / "future-final.json", tmp_path / "figures")
    assert not (tmp_path / "figures").exists()
