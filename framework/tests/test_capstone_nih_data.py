"""NIH labels, source identities and patient ownership for capstone preparation."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from PIL import Image

from benchmarks.nih_data import load_source, prepare, split_patients, verify_preparation


HEADERS = ["Image Index", "Finding Labels", "Patient ID"]


def _write_metadata(root, rows, headers=HEADERS):
    with (root / "Data_Entry_2017.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _metadata(root):
    with (root / "Data_Entry_2017.csv").open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


@pytest.fixture
def nih_tree(tmp_path):
    root = tmp_path / "nih"
    images = root / "images_001" / "images"
    images.mkdir(parents=True)
    findings = {1: "No Finding", 2: "Mass", 3: "Effusion|Pneumonia", 4: "No Finding",
                5: "Mass", 6: "No Finding", 7: "Edema", 8: "No Finding", 9: "Hernia",
                10: "No Finding", 11: "No Finding", 12: "Pneumothorax"}
    rows, train, test = [], [], []
    for patient, label in findings.items():
        for frame in range(2):
            name = f"{patient:08d}_{frame:03d}.png"
            image_label = "No Finding" if patient == 3 and frame == 1 else label
            rows.append({"Image Index": name, "Finding Labels": image_label, "Patient ID": str(patient)})
            Image.new("RGB", (4, 4), (patient, frame, 7)).save(images / name)
            (test if patient >= 11 else train).append(name)
    _write_metadata(root, rows)
    (root / "train_val_list.txt").write_text("\n".join(train) + "\n")
    (root / "test_list.txt").write_text("\n".join(test) + "\n")
    return root


def test_any_finding_is_positive_even_when_pneumonia_is_absent(nih_tree):
    source = load_source(nih_tree)
    rows = {row.image_index: row for row in source.rows}
    assert rows["00000001_000.png"].label == 0
    assert rows["00000002_000.png"].label == 1
    assert rows["00000003_000.png"].label == 1
    assert rows["00000003_001.png"].label == 0
    assert rows["00000003_000.png"].findings == ("Effusion", "Pneumonia")
    assert rows["00000001_000.png"].patient_id == "1"
    assert rows["00000001_000.png"].image_path == "images_001/images/00000001_000.png"
    assert len(source.rows) == 24 and len(source.official_train) == 20 and len(source.official_test) == 4
    assert tuple(r.image_index for r in source.rows) == tuple(sorted(rows))
    for name in ("Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt"):
        assert source.source_hashes[name] == hashlib.sha256((nih_tree / name).read_bytes()).hexdigest()


@pytest.mark.parametrize("label", ["", "Unknown", "No Finding|Mass", "Mass|Mass",
                                        "No Finding|No Finding", "Mass|", "|Mass", " Mass"])
def test_invalid_findings_refuse_instead_of_becoming_negative(nih_tree, label):
    rows = _metadata(nih_tree)
    rows[0]["Finding Labels"] = label
    _write_metadata(nih_tree, rows)
    with pytest.raises(ValueError, match="finding"):
        load_source(nih_tree)


@pytest.mark.parametrize("patient", ["", "0", "-1", "1.0", "2", "abc"])
def test_bad_or_inconsistent_patient_identity_refuses(nih_tree, patient):
    rows = _metadata(nih_tree)
    rows[0]["Patient ID"] = patient
    _write_metadata(nih_tree, rows)
    with pytest.raises(ValueError, match="patient"):
        load_source(nih_tree)


def test_duplicate_metadata_and_required_headers_refuse(nih_tree):
    rows = _metadata(nih_tree)
    _write_metadata(nih_tree, rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        load_source(nih_tree)
    _write_metadata(nih_tree, rows, [*HEADERS, "Patient ID"])
    with pytest.raises(ValueError, match="column"):
        load_source(nih_tree)


def test_missing_required_header_and_truncated_row_refuse(nih_tree):
    path = nih_tree / "Data_Entry_2017.csv"
    path.write_text("Image Index,Finding Labels\n00000001_000.png,No Finding\n")
    with pytest.raises(ValueError, match="column"):
        load_source(nih_tree)
    path.write_text("Image Index,Finding Labels,Patient ID\n00000001_000.png\n")
    with pytest.raises(ValueError):
        load_source(nih_tree)


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate", "directory", "escaping_symlink"])
def test_image_inventory_must_match_safe_files_exactly(nih_tree, tmp_path, mode):
    path = nih_tree / "images_001/images/00000001_000.png"
    if mode == "missing":
        path.unlink()
    elif mode == "extra":
        Image.new("RGB", (4, 4)).save(path.with_name("99999999_000.png"))
    elif mode == "duplicate":
        duplicate = nih_tree / "images_002/images"
        duplicate.mkdir(parents=True)
        (duplicate / path.name).write_bytes(path.read_bytes())
    elif mode == "directory":
        path.unlink()
        path.mkdir()
    else:
        external = tmp_path / "external.png"
        path.rename(external)
        path.symlink_to(external)
    with pytest.raises(ValueError, match="image"):
        load_source(nih_tree)


@pytest.mark.parametrize("mode", ["duplicate", "missing", "overlap", "unknown", "malformed", "empty"])
def test_official_lists_require_exact_disjoint_coverage(nih_tree, mode):
    path = nih_tree / "train_val_list.txt"
    names = path.read_text().splitlines()
    if mode == "duplicate":
        names.append(names[0])
    elif mode == "missing":
        names.pop()
    elif mode == "overlap":
        names.append("00000011_000.png")
    elif mode == "unknown":
        names.append("99999999_000.png")
    elif mode == "malformed":
        names[0] = "../00000001_000.png"
    else:
        names = []
    path.write_text("\n".join(names) + "\n")
    with pytest.raises(ValueError, match="split"):
        load_source(nih_tree)


def test_official_patient_overlap_refuses_even_when_image_ids_do_not_overlap(nih_tree):
    train = nih_tree / "train_val_list.txt"
    test = nih_tree / "test_list.txt"
    moved = train.read_text().splitlines().pop(0)
    train.write_text("\n".join(n for n in train.read_text().splitlines() if n != moved) + "\n")
    test.write_text(test.read_text() + moved + "\n")
    with pytest.raises(ValueError, match="patient overlap"):
        load_source(nih_tree)


def test_input_order_and_leading_zero_patient_spelling_do_not_change_identities(nih_tree):
    first = load_source(nih_tree)
    rows = _metadata(nih_tree)
    rows[0]["Patient ID"] = "0001"
    _write_metadata(nih_tree, list(reversed(rows)))
    for name in ("train_val_list.txt", "test_list.txt"):
        path = nih_tree / name
        path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    second = load_source(nih_tree)
    assert first.rows == second.rows
    assert first.official_train == second.official_train and first.official_test == second.official_test
    assert first.source_hashes != second.source_hashes


def test_empty_metadata_refuses(nih_tree):
    _write_metadata(nih_tree, [])
    with pytest.raises(ValueError, match="metadata"):
        load_source(nih_tree)


def test_unterminated_csv_quotation_refuses_even_when_values_look_valid(nih_tree):
    path = nih_tree / "Data_Entry_2017.csv"
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    lines[-1] = '00000012_001.png,Pneumothorax,"12'
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="CSV"):
        load_source(nih_tree)


def test_patient_splits_preserve_official_test_and_mixed_patient_ownership(nih_tree):
    source = load_source(nih_tree)
    splits = split_patients(source, validation_fraction=0.2, seed=42)
    patients = {key: {r.patient_id for r in rows} for key, rows in splits.items()}
    assert not patients["train"] & patients["validation"]
    assert not patients["train"] & patients["test"]
    assert not patients["validation"] & patients["test"]
    assert {r.image_index for r in splits["test"]} == source.official_test
    assert {r.image_index for r in splits["train"] + splits["validation"]} == source.official_train
    assert sorted(map(len, splits.values())) == [4, 4, 16]
    mixed = [row for rows in splits.values() for row in rows if row.patient_id == "3"]
    assert {r.label for r in mixed} == {0, 1} and len(mixed) == 2
    assert sum("3" in members for members in patients.values()) == 1


def test_split_is_reproducible_and_patient_sampling_uses_the_seed(nih_tree):
    source = load_source(nih_tree)
    first = split_patients(source, validation_fraction=0.2, seed=42)
    assert first == split_patients(source, validation_fraction=0.2, seed=42)
    memberships = {tuple(r.image_index for r in split_patients(
        source, validation_fraction=0.2, seed=seed)["validation"]) for seed in range(5)}
    assert len(memberships) > 1


def test_preparation_preserves_all_rows_and_each_patient_has_one_client(nih_tree, tmp_path):
    output = tmp_path / "prepared"
    report = prepare(nih_tree, output, num_clients=3)
    assert report["image_content_verified"] is False
    assert report["label_definition"].startswith("No Finding")
    assert report["splits"]["train"]["images"] == 16
    assert report["splits"]["validation"]["images"] == 4
    assert report["splits"]["test"]["images"] == 4
    assert report["splits"]["train"]["patients"] == 8
    assert report["splits"]["train"]["positive_images"] + report["splits"]["train"]["negative_images"] == 16
    assert verify_preparation(nih_tree, output) == report
    assert json.loads((output / "report.json").read_text()) == report
    owners, client_images = {}, []
    for client in range(3):
        with (output / f"client-{client:04d}.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        assert rows
        for row in rows:
            assert owners.setdefault(row["patient_id"], client) == client
            assert not Path(row["image_path"]).is_absolute()
            assert (nih_tree / row["image_path"]).is_file()
            client_images.append(row["image_index"])
    with (output / "train.csv").open(newline="") as file:
        train = list(csv.DictReader(file))
    assert sorted(client_images) == [r["image_index"] for r in train]
    assert len(set(client_images)) == len(client_images)
    for name, digest in report["manifest_hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert str(nih_tree) not in (output / "report.json").read_text()


def test_same_inputs_produce_byte_identical_outputs(nih_tree, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    prepare(nih_tree, a, num_clients=2)
    prepare(nih_tree, b, num_clients=2)
    assert {p.name: p.read_bytes() for p in a.iterdir()} == {p.name: p.read_bytes() for p in b.iterdir()}


def test_reordered_inputs_preserve_manifest_membership(nih_tree, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = prepare(nih_tree, a, num_clients=2)
    _write_metadata(nih_tree, list(reversed(_metadata(nih_tree))))
    for name in ("train_val_list.txt", "test_list.txt"):
        path = nih_tree / name
        path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    second = prepare(nih_tree, b, num_clients=2)
    assert first["source_hashes"] != second["source_hashes"]
    assert first["assignment_sha256"] == second["assignment_sha256"]
    assert first["manifest_hashes"] == second["manifest_hashes"]
    with pytest.raises(ValueError, match="source|provenance"):
        verify_preparation(nih_tree, a)


@pytest.mark.parametrize("settings", [
    {"num_clients": 0}, {"num_clients": True}, {"num_clients": 1.5}, {"num_clients": 9},
    {"seed": True}, {"seed": -1}, {"seed": 2.5}, {"validation_fraction": True},
    {"validation_fraction": 0}, {"validation_fraction": 1}, {"validation_fraction": -0.1},
    {"validation_fraction": float("nan")}, {"validation_fraction": float("inf")},
])
def test_invalid_settings_refuse_before_creating_output(nih_tree, tmp_path, settings):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        prepare(nih_tree, output, **{"num_clients": 2, **settings})
    assert not output.exists()


@pytest.mark.parametrize("kind", ["directory", "file", "broken_symlink"])
def test_existing_outputs_are_never_overwritten(nih_tree, tmp_path, kind):
    output = tmp_path / "existing"
    if kind == "directory":
        output.mkdir()
        (output / "keep.txt").write_text("keep")
    elif kind == "file":
        output.write_text("keep")
    else:
        output.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        prepare(nih_tree, output, num_clients=2)
    if kind == "directory":
        assert (output / "keep.txt").read_text() == "keep"
    elif kind == "file":
        assert output.read_text() == "keep"
    else:
        assert output.is_symlink()


def test_output_inside_source_refuses_without_changing_source(nih_tree):
    with pytest.raises(ValueError, match="source"):
        prepare(nih_tree, nih_tree / "generated", num_clients=2)
    assert not (nih_tree / "generated").exists()


@pytest.mark.parametrize("change", ["manifest", "source", "report", "missing_report", "extra_file"])
def test_verifier_rejects_tampered_or_incomplete_preparation(nih_tree, tmp_path, change):
    output = tmp_path / "prepared"
    prepare(nih_tree, output, num_clients=2)
    if change == "manifest":
        path = output / "train.csv"
        path.write_bytes(path.read_bytes().replace(b",0,No Finding", b",1,No Finding", 1))
    elif change == "source":
        path = nih_tree / "Data_Entry_2017.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "report":
        path = output / "report.json"
        report = json.loads(path.read_text())
        report["assignment_sha256"] = "0" * 64
        path.write_text(json.dumps(report))
    elif change == "missing_report":
        (output / "report.json").unlink()
    else:
        (output / "client-9999.csv").write_text("unexpected")
    with pytest.raises(ValueError):
        verify_preparation(nih_tree, output)


def test_fully_rehashed_wrong_client_ownership_still_refuses(nih_tree, tmp_path):
    output = tmp_path / "prepared"
    prepare(nih_tree, output, num_clients=2)
    a, b = output / "client-0000.csv", output / "client-0001.csv"
    first, second = a.read_bytes(), b.read_bytes()
    a.write_bytes(second)
    b.write_bytes(first)
    path = output / "report.json"
    report = json.loads(path.read_text())
    for file in [a, b]:
        report["manifest_hashes"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_preparation(nih_tree, output)


@pytest.mark.parametrize("field,value", [("schema_version", 99), ("schema_version", True),
                                         ("partition_algorithm", "unknown"), ("seed", True),
                                         ("num_clients", 2.0)])
def test_unsupported_or_malformed_report_refuses(nih_tree, tmp_path, field, value):
    output = tmp_path / "prepared"
    prepare(nih_tree, output, num_clients=2)
    path = output / "report.json"
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_preparation(nih_tree, output)


def test_partial_write_is_not_a_completed_preparation(nih_tree, tmp_path, monkeypatch):
    output = tmp_path / "interrupted"
    original = Path.open

    def fail_validation(self, *args, **kwargs):
        if self == output / "validation.csv":
            raise OSError("interrupted disk write")
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_validation)
        with pytest.raises(OSError, match="interrupted"):
            prepare(nih_tree, output, num_clients=2)
    assert not (output / "report.json").exists()
    with pytest.raises(ValueError, match="report"):
        verify_preparation(nih_tree, output)
    assert len(load_source(nih_tree).rows) == 24


def test_cli_prepares_and_verifies_without_modifying_files(nih_tree, tmp_path):
    output = tmp_path / "cli"
    command = [sys.executable, "-m", "benchmarks.nih_data", "--data-root", str(nih_tree),
               "--output", str(output), "--clients", "2"]
    created = subprocess.run(command, capture_output=True, text=True)
    assert created.returncode == 0, created.stderr
    assert json.loads(created.stdout)["splits"]["test"]["images"] == 4
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    checked = subprocess.run(command + ["--verify-only"], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    (output / "train.csv").write_text("broken")
    refused = subprocess.run(command + ["--verify-only"], capture_output=True, text=True)
    assert refused.returncode != 0 and "error" in refused.stderr.lower()
