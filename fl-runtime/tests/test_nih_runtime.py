"""NIH runtime must preserve source labels, patient ownership and held-out intent."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from nih_dataset import CLASSES, prepare_dataset


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "source";images = source / "images_001/images";images.mkdir(parents=True)
    rows, train, test = [], [], []
    for patient in range(1, 21):
        for followup in (0, 1):
            name = f"{patient:08d}_{followup:03d}.png"
            finding = "Pneumonia|Effusion" if patient % 2 and followup else "No Finding"
            rows.append([name, finding, followup, patient])
            pixels = (np.arange(81).reshape(9, 9) + patient + followup).astype(np.uint8)
            Image.fromarray(pixels).save(images / name)
            (train if patient <= 16 else test).append(name)
    with (source / "Data_Entry_2017.csv").open("w", newline="") as file:
        writer = csv.writer(file);writer.writerow(["Image Index", "Finding Labels", "Follow-up #", "Patient ID"])
        writer.writerows(rows)
    (source / "train_val_list.txt").write_text("\n".join(train) + "\n")
    (source / "test_list.txt").write_text("\n".join(test) + "\n")
    out = tmp_path / "prepared"
    prepare_dataset(source, out, num_clients=2, validation_fraction=.25, seed=42)
    return source, out


def test_verified_manifest_preserves_every_clients_patient_and_raw_labels(prepared):
    from nih_runtime import load_manifest
    source, out = prepared
    a, _ = load_manifest(source, out, "client-0.csv")
    b, _ = load_manifest(source, out, "client-1.csv")
    val, _ = load_manifest(source, out, "validation.csv")
    test, _ = load_manifest(source, out, "test.csv")
    groups = [a, b, val, test]
    assert sum(map(len, groups)) == 40
    for i, rows in enumerate(groups):
        assert not {r["patient_id"] for r in rows} & {r["patient_id"] for other in groups[:i] for r in other}
        assert all((r["label"] == CLASSES[1]) == ("Pneumonia" in r["findings"].split("|")) for r in rows)


@pytest.mark.parametrize("damage", ["label", "path", "patient", "test_leak", "wrong_client", "hash", "source"])
def test_changed_manifest_cannot_be_legitimized_by_updating_its_report(prepared, damage):
    from nih_runtime import load_manifest
    source, out = prepared
    path = out / "client-0.csv"
    with path.open() as file:
        reader = csv.DictReader(file);fields = reader.fieldnames;rows = list(reader)
    if damage == "label": rows[0]["label"] = CLASSES[1] if rows[0]["label"] == CLASSES[0] else CLASSES[0]
    elif damage == "path": rows[0]["image_path"] = "../escape.png"
    elif damage == "patient": rows[0]["patient_id"] = "999"
    elif damage in ("test_leak", "wrong_client"):
        with (out / ("test.csv" if damage == "test_leak" else "client-1.csv")).open() as file:
            rows[0] = next(csv.DictReader(file))
    elif damage == "source":
        p = source / "Data_Entry_2017.csv";p.write_text(p.read_text().replace("Pneumonia|Effusion", "Effusion"))
    elif damage == "hash": rows[0]["label"] = "changed-without-updating-hash"
    if damage != "source":
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fields, lineterminator="\n");writer.writeheader();writer.writerows(rows)
        if damage != "hash":
            p = out / "report.json";report = json.loads(p.read_text())
            report["manifest_hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            p.write_text(json.dumps(report))
    with pytest.raises(ValueError): load_manifest(source, out, "client-0.csv")


def test_lazy_dataset_matches_existing_grayscale_transform_and_detects_changed_images(prepared):
    from nih_runtime import NihDataset, load_manifest
    import recipes
    source, out = prepared
    rows, provenance = load_manifest(source, out, "client-0.csv")
    dataset = NihDataset(source, rows, provenance)
    assert len(dataset) == len(rows)
    actual, label = dataset[0]
    with Image.open(source / rows[0]["image_path"]) as image:
        expected = recipes.pneumonia_transform()(image)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert label == CLASSES.index(rows[0]["label"])
    Image.new("L", (9, 9), 0).save(source / rows[0]["image_path"])
    with pytest.raises(ValueError, match="changed"):
        dataset[0]


def test_explicit_balanced_sampling_is_reproducible_patient_unique_and_records_counts(prepared):
    from nih_runtime import load_manifest, select_rows
    source, out = prepared
    rows, _ = load_manifest(source, out, "train.csv")
    selected, provenance = select_rows(rows, count=4, seed=7, balanced=True, patient_unique=True)
    assert len(selected) == 4 and len({r["patient_id"] for r in selected}) == 4
    assert sum(r["label"] == CLASSES[1] for r in selected) == 2
    assert provenance["record_count"] == 4 and provenance["positive_images"] == 2
    assert select_rows(rows, count=4, seed=7, balanced=True, patient_unique=True)[0] == selected
    with pytest.raises(ValueError): select_rows(rows, count=100, seed=7, balanced=True)


def test_unique_sampling_preserves_negative_capacity_when_positive_patients_have_both_labels():
    from nih_runtime import select_rows
    rows = [dict(image_index="00000001_000.png", patient_id="1", label=CLASSES[1]),
            dict(image_index="00000001_001.png", patient_id="1", label=CLASSES[0]),
            dict(image_index="00000002_000.png", patient_id="2", label=CLASSES[1])]
    selected, _ = select_rows(rows, count=2, balanced=True, patient_unique=True, seed=10)
    assert {r["image_index"] for r in selected} == {"00000001_001.png", "00000002_000.png"}


def test_round_loader_uses_validation_only_and_records_proportional_sampling(prepared):
    from nih_runtime import validation_loader
    source, out = prepared
    loader = validation_loader(source, out, max_images=4, seed=42, batch_size=3)
    assert len(loader.dataset) == 4
    assert loader.dataset.provenance["split"] == "validation"
    assert loader.dataset.provenance["total_split_images"] == 8
    assert loader.dataset.provenance["record_count"] == 4
    assert {r["patient_id"] for r in loader.dataset.rows}.isdisjoint({"17", "18", "19", "20"})
    assert [len(y) for _, y in loader] == [3, 1]


def test_binary_metrics_expose_majority_class_failure_and_ranking_quality():
    from nih_runtime import binary_metrics
    metrics = binary_metrics([0, 0, 0, 1], [.1, .2, .3, .4])
    assert metrics["accuracy"] == .75
    assert metrics["balanced_accuracy"] == .5
    assert metrics["auroc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["majority_accuracy"] == .75
    assert metrics["true_positive"] == 0 and metrics["false_negative"] == 1
    single = binary_metrics([0, 0], [.1, .2])
    assert single["auroc"] is None and single["balanced_accuracy"] is None


def test_seeded_lazy_loader_keeps_all_rows_in_exact_round_order(prepared):
    from nih_runtime import NihDataset, load_manifest
    from local_snapshot import SeededSnapshotLoader
    from fedlearn.contract.batch_order import seeded_permutation
    source, out = prepared;rows, provenance = load_manifest(source, out, "client-0.csv")
    dataset = NihDataset(source, rows, provenance)
    loader = SeededSnapshotLoader(dataset, batch_size=5, seed=42, drop_last=False)
    loader.set_round(2);actual = list(loader)
    order = seeded_permutation(len(dataset), 42, 2, 0)
    expected = torch.stack([dataset[i][0] for i in order])
    assert torch.equal(torch.cat([x for x, _ in actual]), expected)
    assert sum(len(y) for _, y in actual) == len(dataset)


def test_client_manifest_binding_requires_own_training_partition_source_and_schema(prepared):
    from nih_runtime import client_manifest_snapshot
    from execution_plan import resolve_model_training
    source, out = prepared
    data = resolve_model_training("NIH_PNEUMONIA_CNN", "FedAvg", "FULL", data_source="LOCAL_SNAPSHOT").data
    snapshot = client_manifest_snapshot(out / "client-0.csv", source, 0, data)
    assert len(snapshot.dataset) == 12 and len(snapshot.snapshot_id) == 64
    for name in ("validation.csv", "test.csv", "train.csv", "client-1.csv"):
        with pytest.raises(ValueError, match="partition"):
            client_manifest_snapshot(out / name, source, 0, data)
    with pytest.raises(ValueError, match="source"):
        client_manifest_snapshot(out / "client-0.csv", None, 0, data)
    data.label_schema_id = "labels-sha256:" + "0" * 64
    with pytest.raises(ValueError, match="schema"):
        client_manifest_snapshot(out / "client-0.csv", source, 0, data)


def test_package_cli_preserves_exact_tensors_provenance_and_explicit_heldout_intent(prepared, tmp_path):
    from execution_plan import resolve_model_training
    from local_snapshot import load_local_snapshot
    from nih_runtime import load_manifest, NihDataset, select_rows
    source, out = prepared
    script = Path(__file__).resolve().parents[2] / "scripts/make_nih_image_package.py"
    package = tmp_path / "phone.zip"
    command = [sys.executable, str(script), "--source", str(source), "--prepared", str(out),
               "--manifest", "client-0.csv", "--out", str(package), "--count", "4", "--seed", "7"]
    subprocess.run(command, check=True, capture_output=True)
    provenance = json.loads(package.with_suffix(".json").read_text())
    assert provenance["split"] == "client-0" and provenance["purpose"] == "train"
    assert provenance["record_count"] == 4 and len(provenance["image_sha256"]) == 4
    data = resolve_model_training("NIH_PNEUMONIA_CNN", "FedAvg", "FULL", data_source="LOCAL_SNAPSHOT").data
    actual = load_local_snapshot(package, data, CLASSES)
    rows, origin = load_manifest(source, out, "client-0.csv")
    selected, _ = select_rows(rows, count=4, seed=7)
    expected = NihDataset(source, selected, origin)
    assert torch.equal(actual.dataset.tensors[0], torch.stack([expected[i][0] for i in range(4)]))
    assert subprocess.run(command, capture_output=True).returncode != 0
    command[command.index("client-0.csv")] = "test.csv"
    command[command.index(str(package))] = str(tmp_path / "heldout.zip")
    assert subprocess.run(command, capture_output=True).returncode != 0
    subprocess.run(command + ["--purpose", "eval"], check=True, capture_output=True)
