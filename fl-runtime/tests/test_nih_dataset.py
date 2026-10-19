"""NIH preparation must keep all images of a patient together and preserve labels."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


def source(tmp_path):
    root = tmp_path / "nih"
    images = root / "images_001" / "images"
    images.mkdir(parents=True)
    rows, train, test = [], [], []
    for patient in range(1, 21):
        for followup in (0, 1):
            name = f"{patient:08d}_{followup:03d}.png"
            finding = "Effusion|Pneumonia" if patient % 2 and followup else "No Finding"
            rows.append([name, finding, followup, patient])
            (images / name).write_bytes(b"original")
            (train if patient <= 16 else test).append(name)
    with (root / "Data_Entry_2017.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Index", "Finding Labels", "Follow-up #", "Patient ID"])
        writer.writerows(rows)
    (root / "train_val_list.txt").write_text("\n".join(train) + "\n")
    (root / "test_list.txt").write_text("\n".join(test) + "\n")
    return root


def read(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def prepare(root, out, seed=42):
    from nih_dataset import prepare_dataset
    return prepare_dataset(root, out, num_clients=2, validation_fraction=0.25, seed=seed)


def test_preparation_is_exhaustive_patient_separated_and_reproducible(tmp_path):
    root = source(tmp_path)
    out = tmp_path / "prepared"
    prepare(root, out)
    partitions = [read(out / name) for name in ("client-0.csv", "client-1.csv", "validation.csv", "test.csv")]
    patients = [{row["patient_id"] for row in rows} for rows in partitions]
    assert sum(map(len, partitions)) == 40
    for i in range(4):
        for j in range(i):
            assert not patients[i] & patients[j]
    assert patients[-1] == {"17", "18", "19", "20"}
    assert len(patients[2]) == 4
    for rows in partitions:
        assert {r["label"] for r in rows} == {"PNEUMONIA_ABSENT", "PNEUMONIA_PRESENT"}
        for row in rows:
            assert (root / row["image_path"]).read_bytes() == b"original"
            assert (row["label"] == "PNEUMONIA_PRESENT") == ("Pneumonia" in row["findings"].split("|"))
    report = json.loads((out / "report.json").read_text())
    assert report["splits"]["test"]["images"] == 8
    assert report["splits"]["test"]["positive_images"] == 2
    assert report["patient_overlap"] == 0
    other = tmp_path / "repeated"
    prepare(root, other)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == {p.name: p.read_bytes() for p in other.iterdir()}
    changed = tmp_path / "changed"
    prepare(root, changed, seed=99)
    assert (out / "client-0.csv").read_bytes() != (changed / "client-0.csv").read_bytes()
    assert (out / "test.csv").read_bytes() == (changed / "test.csv").read_bytes()


@pytest.mark.parametrize("damage", ["missing_image", "directory_image", "duplicate_image", "split_overlap", "patient_overlap",
                                   "uncovered", "unknown_label", "bad_patient", "unsafe_name"])
def test_invalid_source_leaves_no_output(tmp_path, damage):
    root = source(tmp_path)
    metadata = root / "Data_Entry_2017.csv"
    train, test = root / "train_val_list.txt", root / "test_list.txt"
    image = root / "images_001/images/00000001_000.png"
    if damage == "missing_image":
        image.unlink()
    elif damage == "directory_image":
        image.unlink(); image.mkdir()
    elif damage == "duplicate_image":
        duplicate = root / "images_002/images"; duplicate.mkdir(parents=True)
        (duplicate / image.name).write_bytes(b"original")
    elif damage == "split_overlap":
        test.write_text(test.read_text() + image.name + "\n")
    elif damage == "patient_overlap":
        train.write_text(train.read_text().replace(image.name + "\n", ""))
        test.write_text(test.read_text() + image.name + "\n")
    elif damage == "uncovered":
        train.write_text(train.read_text().replace(image.name + "\n", ""))
    elif damage == "unknown_label":
        metadata.write_text(metadata.read_text().replace("Effusion|Pneumonia", "Pneumonia|Unknown"))
    elif damage == "bad_patient":
        metadata.write_text(metadata.read_text().replace(",0,1\n", ",0,invalid\n"))
    else:
        metadata.write_text(metadata.read_text().replace(image.name, "../" + image.name))
    out = tmp_path / "output"
    with pytest.raises(ValueError):
        prepare(root, out)
    assert not out.exists()


def test_preparation_never_overwrites_existing_output(tmp_path):
    root = source(tmp_path)
    out = tmp_path / "output"; out.mkdir()
    marker = out / "important"; marker.write_text("keep")
    with pytest.raises(FileExistsError):
        prepare(root, out)
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("clients,fraction", [(0, .2), (2, 0), (2, 1), (100, .25)])
def test_invalid_or_empty_partition_request_is_refused(tmp_path, clients, fraction):
    from nih_dataset import prepare_dataset
    with pytest.raises(ValueError):
        prepare_dataset(source(tmp_path), tmp_path / "out", num_clients=clients,
                        validation_fraction=fraction, seed=42)


def test_cli_prepares_real_manifests_and_refuses_second_write(tmp_path):
    root = source(tmp_path)
    output = tmp_path / "prepared"
    script = Path(__file__).resolve().parents[2] / "scripts/prepare_nih.py"
    command = [sys.executable, str(script), "--source", str(root), "--out", str(output),
               "--validation-fraction", "0.25"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(read(output / "test.csv")) == 8
    before = (output / "report.json").read_bytes()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert "refusing to overwrite" in result.stderr
    assert (output / "report.json").read_bytes() == before
