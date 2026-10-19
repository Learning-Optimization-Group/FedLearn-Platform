"""Verified NIH manifests, immutable lazy images and binary evaluation metrics."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from local_snapshot import LocalSnapshot, _read, _unique_fields, canonical, digest
from nih_dataset import CLASSES, FIELDS, _source, partition_rows


def load_manifest(source, prepared, name):
    """Recompute source labels and patient ownership, rather than trusting a mutable report."""
    source, prepared = Path(source), Path(prepared)
    if not isinstance(name, str) or not re.fullmatch(r"(?:train|validation|test|client-[0-9]+)\.csv", name):
        raise ValueError("invalid NIH manifest name")
    report_bytes = _read(prepared / "report.json", 64 * 1024)
    report = json.loads(report_bytes, object_pairs_hook=_unique_fields)
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int or report.get("schema_version") != 1
            or report.get("dataset") != "NIH_ChestXray14" or report.get("classes") != CLASSES
            or type(report.get("seed")) is not int or type(report.get("num_clients")) is not int
            or not 1 <= report["num_clients"] <= 10000
            or not isinstance(report.get("source_hashes"), dict)
            or not isinstance(report.get("manifest_hashes"), dict)
            or report.get("label_definition") != "Pneumonia present in report-derived Finding Labels versus absent; absent is not normal"):
        raise ValueError("invalid NIH preparation report")
    fraction = report.get("validation_fraction")
    if (not isinstance(fraction, (int, float)) or isinstance(fraction, bool)
            or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError("invalid NIH validation fraction")
    for filename in ("Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt"):
        if digest(_read(source / filename)) != report.get("source_hashes", {}).get(filename):
            raise ValueError("NIH source metadata changed from preparation")
    raw_rows, train, test = _source(source)
    expected = partition_rows(raw_rows, train, test, num_clients=report["num_clients"],
                              validation_fraction=fraction, seed=report["seed"])
    split = name.removesuffix(".csv")
    if split not in expected:
        raise ValueError("unknown NIH client partition")
    data = _read(prepared / name)
    if digest(data) != report.get("manifest_hashes", {}).get(name):
        raise ValueError("NIH manifest hash mismatch")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
    rows = list(reader)
    if reader.fieldnames != FIELDS or rows != expected[split] or not rows:
        raise ValueError("NIH manifest differs from source labels or patient partition")
    provenance = dict(dataset="NIH_ChestXray14", classes=CLASSES, split=split,
        label_definition=report["label_definition"], manifest_sha256=digest(data),
        preparation_sha256=digest(report_bytes), preparation_seed=report["seed"],
        total_split_images=len(rows), source_metadata_hashes=report["source_hashes"])
    return rows, provenance


def client_manifest_snapshot(path, source, partition_id, requirement):
    """Bind only the participant's complete training manifest to the NIH preparation."""
    from execution_plan import resolve_model_training
    path = Path(path)
    if type(partition_id) is not int or partition_id < 0 or path.name != f"client-{partition_id}.csv":
        raise ValueError("NIH training requires this client's partition manifest")
    if source is None:
        raise ValueError("NIH manifest training requires --nih-source")
    expected = resolve_model_training("NIH_PNEUMONIA_CNN", "FedAvg", "FULL", data_source="LOCAL_SNAPSHOT").data
    if requirement != expected:
        raise ValueError("NIH manifest schema or preparation differs from the contract")
    rows, provenance = load_manifest(source, path.parent, path.name)
    dataset = NihDataset(source, rows, provenance)
    return LocalSnapshot(dataset, digest(canonical(dict(dataset.provenance,
        preparation=expected.SerializeToString(deterministic=True).hex()))))


def select_rows(rows, *, count=None, seed=42, balanced=False, patient_unique=False):
    """Explicit, reproducible image sampling; no balancing is implicit in training."""
    count = len(rows) if count is None else count
    if (type(count) is not int or not 1 <= count <= len(rows) or type(seed) is not int
            or type(balanced) is not bool or type(patient_unique) is not bool):
        raise ValueError("invalid NIH sampling request")
    positive = sum(r["label"] == CLASSES[1] for r in rows)
    n_pos = count // 2 if balanced else int(count * positive / len(rows) + .5)
    if not balanced and count >= 2 and 0 < positive < len(rows):
        n_pos = max(1, min(count - 1, n_pos))
    selected, patients = [], set()
    negative_patients = {r["patient_id"] for r in rows if r["label"] == CLASSES[0]}
    for label, budget in ((CLASSES[1], n_pos), (CLASSES[0], count - n_pos)):
        ordered = sorted((r for r in rows if r["label"] == label),
            key=lambda r: (patient_unique and label == CLASSES[1] and r["patient_id"] in negative_patients,
                hashlib.sha256(f"nih-image-v1:{seed}:{r['image_index']}".encode()).digest(), r["image_index"]))
        taken = 0
        for row in ordered:
            if taken == budget:
                break
            if patient_unique and row["patient_id"] in patients:
                continue
            selected.append(row)
            patients.add(row["patient_id"])
            taken += 1
        if taken != budget:
            raise ValueError("insufficient NIH images/patients for explicit sample")
    selected.sort(key=lambda r: r["image_index"])
    return selected, dict(record_count=len(selected), positive_images=n_pos, negative_images=count - n_pos,
        selection_seed=seed, balanced=balanced, patient_unique=patient_unique,
        selected_rows_sha256=digest(canonical(selected)))


class NihDataset(Dataset):
    """Read prepared tensors lazily; an image changed after binding refuses the fit."""
    def __init__(self, source, rows, provenance):
        import recipes
        self.source, self.rows = Path(source).resolve(), list(rows)
        self.transform = recipes.pneumonia_transform()
        self.image_hashes = []
        for row in self.rows:
            path = (self.source / row["image_path"]).resolve()
            if not path.is_relative_to(self.source) or not path.is_file():
                raise ValueError("NIH image is not a local file inside its source root")
            self.image_hashes.append(digest(_read(path, 32 * 1024 * 1024)))
        self.provenance = dict(provenance, record_count=len(rows),
            positive_images=sum(r["label"] == CLASSES[1] for r in rows),
            negative_images=sum(r["label"] == CLASSES[0] for r in rows),
            image_content_sha256=digest(canonical(self.image_hashes)))

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.source / row["image_path"]).resolve()
        if not path.is_relative_to(self.source): raise ValueError("NIH image path changed")
        data = _read(path, 32 * 1024 * 1024)
        if digest(data) != self.image_hashes[index]: raise ValueError("NIH image content changed after binding")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or not all(1 <= side <= 4096 for side in image.size):
                raise ValueError("invalid NIH PNG or image dimensions")
            tensor = self.transform(image)
        if tensor.shape != (1, 224, 224) or not torch.isfinite(tensor).all():
            raise ValueError("invalid prepared NIH tensor")
        return tensor, CLASSES.index(row["label"])


def validation_loader(source, prepared, *, max_images=None, seed=42, batch_size=32):
    rows, provenance = load_manifest(source, prepared, "validation.csv")
    selected, sample = select_rows(rows, count=min(max_images, len(rows)) if max_images is not None else None,
                                   seed=seed)
    dataset = NihDataset(source, selected, dict(provenance, **sample))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def binary_metrics(labels, scores):
    """Scores are probabilities for Pneumonia PRESENT; ties predict class zero as argmax does."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or scores.shape != labels.shape or not len(labels)
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all()
            or (scores < 0).any() or (scores > 1).any()):
        raise ValueError("invalid binary labels or probabilities")
    pred, positive = scores > .5, labels == 1
    tp, fn = int((pred & positive).sum()), int((~pred & positive).sum())
    tn, fp = int((~pred & ~positive).sum()), int((pred & ~positive).sum())
    both = tp + fn > 0 and tn + fp > 0
    return dict(accuracy=(tp + tn) / len(labels), balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2 if both else None,
        auroc=float(roc_auc_score(labels, scores)) if both else None,
        average_precision=float(average_precision_score(labels, scores)) if both else None,
        majority_accuracy=max(tp + fn, tn + fp) / len(labels),
        true_positive=tp, false_negative=fn, true_negative=tn, false_positive=fp,
        positive_images=tp + fn, negative_images=tn + fp, record_count=len(labels))
