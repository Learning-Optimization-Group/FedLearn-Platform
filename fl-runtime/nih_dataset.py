"""Prepare patient-separated NIH pneumonia-label manifests without moving images."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

CLASSES = ["PNEUMONIA_ABSENT", "PNEUMONIA_PRESENT"]
FINDINGS = {"Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"}
FIELDS = ["image_path", "image_index", "patient_id", "label", "findings"]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split(path):
    entries = path.read_text(encoding="utf-8").splitlines()
    if not entries or any(not re.fullmatch(r"\d{8}_\d{3}\.png", name) for name in entries):
        raise ValueError("invalid or empty official split")
    if len(set(entries)) != len(entries):
        raise ValueError("duplicate official split entry")
    return set(entries)


def _source(root):
    images = {}
    for path in sorted(root.glob("images_*/images/*.png")):
        if path.name in images or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("duplicate image filename or image outside dataset root")
        images[path.name] = path.relative_to(root).as_posix()
    rows = {}
    with (root / "Data_Entry_2017.csv").open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for field in ("Image Index", "Finding Labels", "Patient ID"):
            if (reader.fieldnames or []).count(field) != 1:
                raise ValueError("missing or duplicate NIH metadata column")
        for row in reader:
            name, finding, patient = row["Image Index"], row["Finding Labels"], row["Patient ID"]
            if (not isinstance(name, str) or not re.fullmatch(r"\d{8}_\d{3}\.png", name)
                    or name in rows or name not in images):
                raise ValueError("invalid, duplicate or missing metadata image")
            if (not isinstance(patient, str) or not re.fullmatch(r"[0-9]+", patient)
                    or int(patient) <= 0 or int(name[:8]) != int(patient)):
                raise ValueError("invalid or inconsistent patient ID")
            labels = finding.split("|") if isinstance(finding, str) else []
            if (not labels or len(set(labels)) != len(labels)
                    or (labels != ["No Finding"] and not set(labels) <= FINDINGS)):
                raise ValueError("unknown or invalid NIH finding labels")
            rows[name] = dict(image_path=images[name], image_index=name, patient_id=str(int(patient)),
                              label=CLASSES[int("Pneumonia" in labels)], findings=finding)
    if not rows or set(images) != set(rows):
        raise ValueError("images and metadata do not match exactly")
    train, test = _split(root / "train_val_list.txt"), _split(root / "test_list.txt")
    if train & test or train | test != set(rows):
        raise ValueError("official splits overlap or do not cover metadata")
    train_patients = {rows[name]["patient_id"] for name in train}
    test_patients = {rows[name]["patient_id"] for name in test}
    if train_patients & test_patients:
        raise ValueError("official splits have patient overlap")
    return rows, train, test


def prepare_dataset(root, output, *, num_clients=2, validation_fraction=0.2, seed=42):
    """Write complete CSV splits/client manifests and a reproducibility report.

    Patient groups are stratified by any pneumonia finding, ordered by seeded
    SHA-256, then divided into validation and round-robin client groups. This
    balances patient strata, not image counts; no image is dropped or duplicated.
    """
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite prepared dataset")
    if (type(num_clients) is not int or num_clients < 1 or type(seed) is not int
            or not isinstance(validation_fraction, (float, int)) or isinstance(validation_fraction, bool)
            or not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1):
        raise ValueError("invalid client count, validation fraction or seed")
    rows, train_names, test_names = _source(root)
    patient_positive = {}
    for name in train_names:
        row = rows[name]
        patient_positive[row["patient_id"]] = (patient_positive.get(row["patient_id"], False)
                                                or row["label"] == CLASSES[1])
    validation, clients = set(), [set() for _ in range(num_clients)]
    for positive in (False, True):
        ordered = sorted((p for p, v in patient_positive.items() if v == positive),
                         key=lambda p: (hashlib.sha256(f"nih-patient-v1:{seed}:{p}".encode()).digest(), p))
        n_val = max(1, int(len(ordered) * validation_fraction))
        if n_val >= len(ordered) or len(ordered) - n_val < num_clients:
            raise ValueError("not enough patients in each stratum for nonempty validation and clients")
        validation.update(ordered[:n_val])
        for i, patient in enumerate(ordered[n_val:]):
            clients[i % num_clients].add(patient)
    split_rows = {
        "train": [rows[n] for n in sorted(train_names) if rows[n]["patient_id"] not in validation],
        "validation": [rows[n] for n in sorted(train_names) if rows[n]["patient_id"] in validation],
        "test": [rows[n] for n in sorted(test_names)],
    }
    for index, patients in enumerate(clients):
        split_rows[f"client-{index}"] = [r for r in split_rows["train"] if r["patient_id"] in patients]
    groups = [validation, {rows[n]["patient_id"] for n in test_names}, *clients]
    if any(a & b for i, a in enumerate(groups) for b in groups[:i]):
        raise ValueError("prepared splits have patient overlap")
    if sum(len(split_rows[f"client-{i}"]) for i in range(num_clients)) != len(split_rows["train"]):
        raise ValueError("client manifests do not cover training images")
    report = dict(schema_version=1, dataset="NIH_ChestXray14", classes=CLASSES,
                  label_definition="Pneumonia present in report-derived Finding Labels versus absent; absent is not normal",
                  seed=seed, num_clients=num_clients, validation_fraction=validation_fraction,
                  partition_algorithm="nih-patient-v1: SHA256 seeded patient order within any-pneumonia strata",
                  source_hashes={n: _sha(root / n) for n in
                                 ("Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt")},
                  patient_overlap=0, image_overlap=0, image_content_verified=False,
                  splits={name: dict(images=len(records), patients=len({r["patient_id"] for r in records}),
                                     positive_images=sum(r["label"] == CLASSES[1] for r in records))
                          for name, records in split_rows.items()}, manifest_hashes={})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # exclusive creation; never replace an existing output
    try:
        for name, records in split_rows.items():
            path = output / f"{name}.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(records)
            report["manifest_hashes"][path.name] = _sha(path)
        (output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(output)
        raise
    return report
