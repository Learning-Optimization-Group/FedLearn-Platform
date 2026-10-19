"""NIH labels, source identities and patient ownership for capstone preparation."""
import csv
import hashlib
import pytest
from PIL import Image

from benchmarks.nih_data import load_source


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
