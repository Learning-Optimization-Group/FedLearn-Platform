"""Whole-patient NIH quarantine and immutable original evaluation membership."""
import csv
import hashlib
import json
import subprocess
import sys

from PIL import Image
import pytest

from tests.test_capstone_nih_data import nih_tree
from benchmarks.nih_data import prepare, verify_preparation
from benchmarks.nih_image_audit import audit_images
from benchmarks import nih_cohort as cohort


def _rows(directory, name):
    with (directory / f"{name}.csv").open(newline="") as file:
        return list(csv.DictReader(file))


def _snapshot(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir()}


def _copy(root, source, destination):
    (root / destination["image_path"]).write_bytes((root / source["image_path"]).read_bytes())


@pytest.fixture
def original(nih_tree, tmp_path):
    preparation = tmp_path / "original"
    prepare(nih_tree, preparation, num_clients=2)
    return nih_tree, preparation


def _audit(root, preparation, tmp_path):
    output = tmp_path / "audit"
    audit_images(root, preparation, output)
    return output


def test_clean_cohort_retains_every_original_row_and_verifies(original, tmp_path):
    root, preparation = original
    audit = _audit(root, preparation, tmp_path)
    before = _snapshot(preparation), _snapshot(audit)
    output = tmp_path / "cohort"
    report = cohort.prepare_cohort(root, preparation, audit, output)
    assert report["policy_version"] == "nih-development-quarantine-v1"
    assert report["development_content_qualified"] is True
    assert report["independent_split_rgb_overlap_groups"] == 0
    assert report["quarantined"] == {"images": 0, "patients": 0, "positive_images": 0, "negative_images": 0}
    for split in ("train", "validation", "test", "client-0000", "client-0001"):
        assert (output / f"{split}.csv").read_bytes() == (preparation / f"{split}.csv").read_bytes()
    assert cohort.verify_cohort(root, preparation, audit, output) == report
    assert before == (_snapshot(preparation), _snapshot(audit))
    other = tmp_path / "second"
    cohort.prepare_cohort(root, preparation, audit, other)
    assert _snapshot(other) == _snapshot(output)
    with pytest.raises(ValueError):
        verify_preparation(root, output)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_cross_test_collision_removes_whole_development_patient_only(original, tmp_path, split):
    root, preparation = original
    source = _rows(preparation, split)[0]
    _copy(root, source, _rows(preparation, "test")[0])
    audit = _audit(root, preparation, tmp_path)
    assert json.loads((audit / "report.json").read_text())["qualified"] is False
    output = tmp_path / "cohort"
    report = cohort.prepare_cohort(root, preparation, audit, output)
    assert (output / "test.csv").read_bytes() == (preparation / "test.csv").read_bytes()
    expected = [r for r in _rows(preparation, split) if r["patient_id"] != source["patient_id"]]
    assert _rows(output, split) == expected
    excluded = _rows(output, "quarantined")
    assert len(excluded) == 2 and {r["patient_id"] for r in excluded} == {source["patient_id"]}
    assert report["quarantined"]["images"] == 2
    assert report["quarantined"]["patients"] == 1
    assert report["development_content_qualified"] is True
    reasons = json.loads((output / "reasons.json").read_text())["patients"]
    assert reasons[0]["patient_id"] == source["patient_id"]
    assert "cross_patient_rgb_identity" in reasons[0]["causes"]
    group = json.loads((audit / "duplicates.json").read_text())["rgb"][0]
    assert reasons[0]["rgb_group_sha256s"] == [group["sha256"]]
    clients = [_rows(output, f"client-{cid:04d}") for cid in range(2)]
    assert {r["image_index"] for client in clients for r in client} == {r["image_index"] for r in _rows(output, "train")}
    assert not {r["patient_id"] for r in clients[0]} & {r["patient_id"] for r in clients[1]}


def test_one_group_across_all_three_splits_quarantines_both_development_patients(original, tmp_path):
    root, prep = original
    source, validation, test = [_rows(prep, split)[0] for split in ("train", "validation", "test")]
    _copy(root, source, validation)
    _copy(root, source, test)
    audit = _audit(root, prep, tmp_path)
    output = tmp_path / "cohort"
    report = cohort.prepare_cohort(root, prep, audit, output)
    assert report["quarantined"]["patients"] == 2
    assert report["quarantined"]["images"] == 4
    assert {r["patient_id"] for r in _rows(output, "quarantined")} == {source["patient_id"], validation["patient_id"]}
    assert report["independent_split_rgb_overlap_groups"] == 0
    assert (output / "test.csv").read_bytes() == (prep / "test.csv").read_bytes()


def test_within_training_cross_patient_collision_removes_both_patients(original, tmp_path):
    root, prep = original
    source = _rows(prep, "train")[0]
    destination = next(r for r in _rows(prep, "train") if r["patient_id"] != source["patient_id"])
    _copy(root, source, destination)
    audit = _audit(root, prep, tmp_path)
    assert json.loads((audit / "report.json").read_text())["qualified"] is True
    report = cohort.prepare_cohort(root, prep, audit, tmp_path / "cohort")
    assert report["quarantined"]["patients"] == 2
    assert report["quarantined"]["images"] == 4


def test_same_patient_contradictory_pixels_quarantine_without_relabeling(original, tmp_path):
    root, prep = original
    split = next(s for s in ("train", "validation") if any(r["patient_id"] == "3" for r in _rows(prep, s)))
    rows = [r for r in _rows(prep, split) if r["patient_id"] == "3"]
    assert {r["label"] for r in rows} == {"0", "1"}
    _copy(root, *rows)
    audit = _audit(root, prep, tmp_path)
    output = tmp_path / "cohort"
    report = cohort.prepare_cohort(root, prep, audit, output)
    assert report["quarantined"]["patients"] == 1
    assert {r["label"] for r in _rows(output, "quarantined")} == {"0", "1"}
    reason = json.loads((output / "reasons.json").read_text())["patients"][0]
    assert reason["causes"] == ["contradictory_rgb_labels"]


def test_same_patient_same_label_duplicates_do_not_trigger_quarantine(original, tmp_path):
    root, prep = original
    source = next(r for r in _rows(prep, "train") if r["patient_id"] != "3")
    destination = next(r for r in _rows(prep, "train") if r["patient_id"] == source["patient_id"] and r != source)
    _copy(root, source, destination)
    audit = _audit(root, prep, tmp_path)
    report = cohort.prepare_cohort(root, prep, audit, tmp_path / "cohort")
    assert report["quarantined"]["images"] == 0


def test_ambiguous_official_test_is_retained_and_disclosed(original, tmp_path):
    root, prep = original
    rows = _rows(prep, "test")
    _copy(root, rows[0], rows[-1])
    audit = _audit(root, prep, tmp_path)
    output = tmp_path / "cohort"
    report = cohort.prepare_cohort(root, prep, audit, output)
    assert report["quarantined"]["images"] == 0
    assert report["retained_official_test_ambiguous_rgb_groups"] == 1
    assert report["development_content_qualified"] is True
    assert (output / "test.csv").read_bytes() == (prep / "test.csv").read_bytes()


@pytest.mark.parametrize("problem", ["corruption", "empty_train", "empty_validation", "too_many_clients"])
def test_unresolved_or_infeasible_data_refuses_before_publication(nih_tree, tmp_path, problem):
    root, prep = nih_tree, tmp_path / "original"
    prepare(root, prep, num_clients=8 if problem == "too_many_clients" else 2)
    target = _rows(prep, "test")[0]
    if problem == "corruption":
        (root / _rows(prep, "train")[0]["image_path"]).write_bytes(b"broken")
    else:
        split = "validation" if problem == "empty_validation" else "train"
        rows = _rows(prep, split)
        if problem == "too_many_clients":
            rows = rows[:1]
        for row in rows:
            _copy(root, target, row)
    audit = _audit(root, prep, tmp_path)
    output = tmp_path / "cohort"
    with pytest.raises(ValueError):
        cohort.prepare_cohort(root, prep, audit, output)
    assert not output.exists()


@pytest.mark.parametrize("location", ["existing", "symlink", "source", "original", "audit"])
def test_unsafe_output_never_overwrites_inputs(original, tmp_path, location):
    root, prep = original
    audit = _audit(root, prep, tmp_path)
    output = tmp_path / "cohort"
    if location == "existing":
        output.mkdir()
        (output / "sentinel").write_text("preserve")
    elif location == "symlink":
        output.symlink_to(tmp_path / "missing")
    else:
        output = {"source": root, "original": prep, "audit": audit}[location] / "cohort"
    with pytest.raises((ValueError, FileExistsError)):
        cohort.prepare_cohort(root, prep, audit, output)
    if location == "existing":
        assert (output / "sentinel").read_text() == "preserve"
    elif location != "symlink":
        assert not output.exists()


@pytest.mark.parametrize("mutation", ["exclusions", "reasons", "test", "clients", "order", "settings", "report_format", "extra", "missing"])
def test_reconstruction_rejects_forged_rehashed_or_incomplete_cohort(original, tmp_path, mutation):
    root, prep = original
    _copy(root, _rows(prep, "train")[0], _rows(prep, "test")[0])
    audit = _audit(root, prep, tmp_path)
    output = tmp_path / "cohort"
    cohort.prepare_cohort(root, prep, audit, output)
    report = json.loads((output / "report.json").read_text())
    changed = None
    if mutation == "exclusions":
        changed = "quarantined.csv"
        with (output / changed).open("a") as file:
            file.write((output / "train.csv").read_text().splitlines()[1] + "\n")
    elif mutation == "reasons":
        changed = "reasons.json"
        value = json.loads((output / changed).read_text())
        value["patients"][0]["causes"] = []
        (output / changed).write_text(json.dumps(value))
    elif mutation in {"test", "clients", "order"}:
        changed = {"test": "test.csv", "clients": "client-0000.csv", "order": "train.csv"}[mutation]
        lines = (output / changed).read_text().splitlines()
        (output / changed).write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
    elif mutation == "settings":
        report["seed"] += 1
    elif mutation == "report_format":
        (output / "report.json").write_text(json.dumps(report, separators=(",", ":")))
    elif mutation == "extra":
        (output / "extra").touch()
    else:
        (output / "reasons.json").unlink()
    if changed:
        report["payload_hashes"][changed] = hashlib.sha256((output / changed).read_bytes()).hexdigest()
    if mutation != "report_format":
        (output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="cohort"):
        cohort.verify_cohort(root, prep, audit, output)


@pytest.mark.parametrize("changed", ["source", "original", "audit"])
def test_changed_source_evidence_refuses_curation(original, tmp_path, changed):
    root, prep = original
    audit = _audit(root, prep, tmp_path)
    if changed == "source":
        Image.new("RGB", (4, 4), "red").save(root / _rows(prep, "train")[0]["image_path"])
    elif changed == "original":
        (prep / "train.csv").write_text("forged")
    else:
        (audit / "duplicates.json").write_text('{"file":[],"rgb":[]}')
    with pytest.raises(ValueError):
        cohort.prepare_cohort(root, prep, audit, tmp_path / "cohort")


def test_interrupted_completion_never_verifies(original, tmp_path, monkeypatch):
    root, prep = original
    audit = _audit(root, prep, tmp_path)
    output = tmp_path / "cohort"
    def interrupt(*args):
        raise OSError("interrupted")
    monkeypatch.setattr(cohort.os, "replace", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        cohort.prepare_cohort(root, prep, audit, output)
    assert not (output / "report.json").exists()
    with pytest.raises(ValueError, match="cohort"):
        cohort.verify_cohort(root, prep, audit, output)


def test_cli_preparation_and_separate_verification(original, tmp_path):
    root, prep = original
    audit = _audit(root, prep, tmp_path)
    output = tmp_path / "cohort"
    command = [sys.executable, "-m", "benchmarks.nih_cohort", "--data-root", str(root),
               "--original-preparation", str(prep), "--image-audit", str(audit), "--output", str(output)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["development_content_qualified"] is True
    verified = subprocess.run([*command, "--verify-only"], text=True, capture_output=True)
    assert verified.returncode == 0, verified.stderr
    assert result.stdout == verified.stdout
