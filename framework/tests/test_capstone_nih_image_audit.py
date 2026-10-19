"""Image identity and refusal gates before NIH feature extraction."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, PngImagePlugin
import pytest

from tests.test_capstone_nih_data import nih_tree
from benchmarks.nih_data import prepare
from benchmarks import nih_image_audit as audit


@pytest.fixture
def prepared(nih_tree, tmp_path):
    directory = tmp_path / "manifests"
    prepare(nih_tree, directory, num_clients=2)
    return nih_tree, directory


def _rows(directory):
    with (directory / "images.csv").open(newline="") as file:
        return list(csv.DictReader(file))


def _image(root, name):
    return root / "images_001" / "images" / name


def _split_names(preparation, split):
    with (preparation / f"{split}.csv").open(newline="") as file:
        return [row["image_index"] for row in csv.DictReader(file)]


def test_complete_audit_decodes_every_prepared_image(prepared, tmp_path):
    root, prep = prepared
    output = tmp_path / "audit"
    report = audit.audit_images(root, prep, output)
    assert report["images"] == 24
    assert report["failed_images"] == 0
    assert report["qualified"] is True
    assert report["cross_split_duplicate_groups"] == 0
    assert report["preparation_sha256"] == hashlib.sha256((prep / "report.json").read_bytes()).hexdigest()
    assert report["source_hashes"] == json.loads((prep / "report.json").read_text())["source_hashes"]
    rows = _rows(output)
    assert len(rows) == 24
    assert [r["image_index"] for r in rows] == sorted(r["image_index"] for r in rows)
    for row in rows:
        raw = (root / row["image_path"]).read_bytes()
        assert int(row["file_bytes"]) == len(raw)
        assert row["file_sha256"] == hashlib.sha256(raw).hexdigest()
        assert row["width"] == row["height"] == "4"
        rgb = Image.open(io.BytesIO(raw)).convert("RGB").tobytes()
        expected = hashlib.sha256(b"capstone-nih-rgb-v1\0" + b"4x4\0" + rgb).hexdigest()
        assert row["rgb_sha256"] == expected and row["error"] == ""
    assert audit.verify_image_audit(root, prep, output) == report
    other = tmp_path / "other"
    audit.audit_images(root, prep, other)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == {p.name: p.read_bytes() for p in other.iterdir()}


@pytest.mark.parametrize("reencode", [False, True])
def test_cross_split_duplicates_refuse_qualification_without_dropping_rows(prepared, tmp_path, reencode):
    root, prep = prepared
    source = _image(root, _split_names(prep, "train")[0])
    destination = _image(root, _split_names(prep, "test")[0])
    if reencode:
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("note", "same pixels, different PNG bytes")
        with Image.open(source) as image:
            image.save(destination, pnginfo=metadata)
        assert source.read_bytes() != destination.read_bytes()
    else:
        destination.write_bytes(source.read_bytes())
    output = tmp_path / "audit"
    report = audit.audit_images(root, prep, output)
    assert report["qualified"] is False and report["images"] == 24
    assert report["cross_split_duplicate_groups"] >= 1
    groups = json.loads((output / "duplicates.json").read_text())
    group = next(g for g in groups["rgb"] if destination.name in g["image_indices"])
    assert set(group["splits"]) == {"train", "test"}
    assert group["cross_patient"] is True
    assert len(groups["file"]) == int(not reencode)
    assert audit.verify_image_audit(root, prep, output) == report


def test_grayscale_and_rgb_encoding_share_canonical_pixel_identity(prepared, tmp_path):
    root, prep = prepared
    left, right = _split_names(prep, "test")[:2]
    Image.new("L", (4, 4), 19).save(_image(root, left))
    Image.new("RGB", (4, 4), (19, 19, 19)).save(_image(root, right))
    report = audit.audit_images(root, prep, tmp_path / "audit")
    assert report["qualified"] is True
    assert report["within_split_duplicate_groups"] == 1
    rows = {r["image_index"]: r for r in _rows(tmp_path / "audit")}
    assert rows[left]["rgb_sha256"] == rows[right]["rgb_sha256"]
    assert rows[left]["file_sha256"] != rows[right]["file_sha256"]


def test_same_split_duplicates_disclose_patient_and_label_conflicts(prepared, tmp_path):
    root, prep = prepared
    names = _split_names(prep, "test")
    source, destination = _image(root, names[0]), _image(root, names[-1])
    destination.write_bytes(source.read_bytes())
    output = tmp_path / "audit"
    report = audit.audit_images(root, prep, output)
    assert report["qualified"] is True
    assert report["within_split_duplicate_groups"] == 2  # raw and RGB identities
    group = json.loads((output / "duplicates.json").read_text())["rgb"][0]
    assert group["cross_patient"] and group["contradictory_labels"]
    assert group["splits"] == ["test"]


@pytest.mark.parametrize("problem,category", [("corrupt", "decode_error"), ("jpeg", "not_single_frame_png"),
                                              ("animated", "not_single_frame_png"), ("large_file", "file_too_large"),
                                              ("large_pixels", "pixel_limit"), ("pillow_bomb", "decode_error")])
def test_image_failures_are_retained_and_block_qualification(prepared, tmp_path, monkeypatch, problem, category):
    root, prep = prepared
    path = _image(root, _split_names(prep, "train")[0])
    if problem == "corrupt":
        path.write_bytes(b"not an image")
    elif problem == "jpeg":
        Image.new("RGB", (4, 4)).save(path, format="JPEG")
    elif problem == "animated":
        Image.new("RGB", (4, 4)).save(path, save_all=True, append_images=[Image.new("RGB", (4, 4), "red")])
    elif problem == "large_file":
        monkeypatch.setattr(audit, "MAX_FILE_BYTES", 1)
    elif problem == "large_pixels":
        Image.new("RGB", (5, 4)).save(path)
        monkeypatch.setattr(audit, "MAX_PIXELS", 16)
    else:
        Image.new("RGB", (5, 4)).save(path)
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 17)
    output = tmp_path / "audit"
    report = audit.audit_images(root, prep, output)
    assert report["qualified"] is False
    rows = {r["image_index"]: r for r in _rows(output)}
    assert rows[path.name]["error"] == category
    assert rows[path.name]["rgb_sha256"] == ""
    assert len(rows) == 24
    assert report["failed_images"] == (24 if problem == "large_file" else 1)


@pytest.mark.parametrize("location", ["existing", "symlink", "source", "preparation"])
def test_unsafe_output_refuses_before_writing(prepared, tmp_path, location):
    root, prep = prepared
    output = tmp_path / "audit"
    if location == "existing":
        output.mkdir()
        (output / "sentinel").write_text("preserve")
    elif location == "symlink":
        output.symlink_to(tmp_path / "missing")
    elif location == "source":
        output = root / "audit"
    else:
        output = prep / "audit"
    with pytest.raises((FileExistsError, ValueError)):
        audit.audit_images(root, prep, output)
    if location == "existing":
        assert (output / "sentinel").read_text() == "preserve"
    elif location != "symlink":
        assert not output.exists()


@pytest.mark.parametrize("mutation", ["source", "payload", "report", "rehash_forgery", "extra", "missing"])
def test_full_verification_rejects_changed_or_forged_evidence(prepared, tmp_path, mutation):
    root, prep = prepared
    output = tmp_path / "audit"
    audit.audit_images(root, prep, output)
    if mutation == "source":
        Image.new("RGB", (4, 4), "orange").save(_image(root, _split_names(prep, "train")[0]))
    elif mutation in {"payload", "rehash_forgery"}:
        path = output / "images.csv"
        path.write_text(path.read_text().replace(",4,4,", ",3,4,", 1))
        if mutation == "rehash_forgery":
            report = json.loads((output / "report.json").read_text())
            report["payload_hashes"]["images.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (output / "report.json").write_text(json.dumps(report))
    elif mutation == "report":
        report = json.loads((output / "report.json").read_text())
        report["images"] = 999
        (output / "report.json").write_text(json.dumps(report))
    elif mutation == "extra":
        (output / "unexpected").write_text("x")
    else:
        (output / "duplicates.json").unlink()
    with pytest.raises(ValueError, match="audit"):
        audit.verify_image_audit(root, prep, output)


def test_interrupted_publication_never_verifies(prepared, tmp_path, monkeypatch):
    root, prep = prepared
    output = tmp_path / "audit"
    def interrupted(*args):
        raise OSError("interrupted before completion")
    monkeypatch.setattr(audit.os, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        audit.audit_images(root, prep, output)
    assert not (output / "report.json").exists()
    with pytest.raises(ValueError, match="audit"):
        audit.verify_image_audit(root, prep, output)


def test_cli_failure_is_nonzero_and_preserves_completed_evidence(prepared, tmp_path):
    root, prep = prepared
    _image(root, _split_names(prep, "train")[0]).write_bytes(b"broken")
    output = tmp_path / "audit"
    command = [sys.executable, "-m", "benchmarks.nih_image_audit", "--data-root", str(root),
               "--preparation", str(prep), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stdout)["qualified"] is False
    assert (output / "report.json").exists()
    verified = subprocess.run([*command, "--verify-only"], capture_output=True, text=True)
    assert verified.returncode == 2
    assert json.loads(verified.stdout)["qualified"] is False


def test_bounded_parallel_decoding_has_identical_audit_bytes(prepared, tmp_path):
    root, prep = prepared
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    audit.audit_images(root, prep, serial)
    audit.audit_images(root, prep, parallel, workers=2)
    assert {p.name: p.read_bytes() for p in serial.iterdir()} == {p.name: p.read_bytes() for p in parallel.iterdir()}
    assert audit.verify_image_audit(root, prep, parallel, workers=2)["qualified"] is True


@pytest.mark.parametrize("workers", [True, False, 0, -1, 17, 1.5, "2"])
def test_invalid_worker_count_refuses_without_output(prepared, tmp_path, workers):
    root, prep = prepared
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="workers"):
        audit.audit_images(root, prep, output, workers=workers)
    assert not output.exists()


def test_parallel_reader_survives_unrelated_benchmarks_module_on_import_path(prepared, tmp_path, monkeypatch):
    root, prep = prepared
    shadow = tmp_path / "unrelated"
    shadow.mkdir()
    (shadow / "benchmarks.py").write_text('"""Unrelated application module."""\n')
    monkeypatch.syspath_prepend(str(shadow))
    assert audit.audit_images(root, prep, tmp_path / "audit", workers=2)["qualified"] is True
