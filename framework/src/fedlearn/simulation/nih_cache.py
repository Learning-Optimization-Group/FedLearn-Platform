"""Verify and load a fixed, source-bound NIH frozen-feature cache."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np


SPLITS = ("train", "validation", "test")
CACHE_VERSION = "capstone-nih-frozen-features-v1"
CSV_FIELDS = ("image_index", "patient_id", "image_path", "label", "findings")
PAYLOADS = frozenset({"cohort-report.json", *(f"{s}{suffix}" for s in SPLITS
                                            for suffix in (".csv", "-x.npy", "-y.npy"))})
TRANSFORM = {"color": "RGB", "resize": [224, 224], "interpolation": "bilinear",
             "antialias": True, "scale": "uint8/255",
             "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_manifest(path):
    """Read canonical identities; row order is meaningful and is never repaired."""
    reader = csv.DictReader(io.StringIO(Path(path).read_text(), newline=""), strict=True)
    if reader.fieldnames != list(CSV_FIELDS):
        raise ValueError("cache manifest columns differ from NIH schema")
    rows = list(reader)
    if not rows:
        raise ValueError("cache split is empty")
    seen = set()
    for row in rows:
        image, patient = row["image_index"], row["patient_id"]
        if (None in row or not isinstance(image, str)
                or not re.fullmatch(r"[0-9]{8}_[0-9]{3}\.png", image)
                or not isinstance(patient, str) or not re.fullmatch(r"[1-9][0-9]*", patient)
                or int(image[:8]) != int(patient) or image in seen
                or not isinstance(row["image_path"], str)
                or not re.fullmatch(r"images_[0-9]+/images/" + re.escape(image), row["image_path"])
                or row["label"] not in ("0", "1") or not row["findings"]
                or (row["findings"] == "No Finding") != (row["label"] == "0")):
            raise ValueError("cache manifest has invalid or duplicate NIH identity/label")
        seen.add(image)
    return rows


def _canonical_report(path):
    raw = path.read_bytes()
    report = json.loads(raw)
    if not isinstance(report, dict) or raw != json_bytes(report):
        raise ValueError("cache report is not canonical JSON")
    return report


def _load_array(path):
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError("cache payload must be one NPY array, not an archive")
    return array


def _validate_provenance(report, cohort):
    if (type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or report.get("cache_version") != CACHE_VERSION
            or type(report.get("feat_dim")) is not int or report["feat_dim"] != 512
            or type(report.get("n_classes")) is not int or report["n_classes"] != 2
            or report.get("classes") != ["NO_FINDING", "FINDING"]
            or report.get("backbone") != "resnet18" or report.get("transform") != TRANSFORM
            or report.get("feature_dtype") != "float32" or report.get("label_dtype") != "int64"
            or type(report.get("pretrained")) is not bool
            or type(report.get("backbone_seed")) is not int or not 0 <= report["backbone_seed"] < 2**63
            or report.get("device") not in ("cpu", "mps", "cuda")
            or not _digest(report.get("model_state_sha256"))):
        raise ValueError("cache feature provenance is unsupported")
    identity = ("ResNet18_Weights.IMAGENET1K_V1" if report["pretrained"]
                else f"random-seed:{report['backbone_seed']}")
    if report.get("weight_identity") != identity:
        raise ValueError("cache weight identity is inconsistent")
    for field in ("batch_size", "audit_workers", "image_workers", "torch_num_threads"):
        upper = 256 if field == "batch_size" else 16 if field.endswith("workers") else 2**31
        if type(report.get(field)) is not int or not 1 <= report[field] <= upper:
            raise ValueError("cache execution settings are invalid")
    if (type(cohort.get("schema_version")) is not int or cohort["schema_version"] != 1
            or cohort.get("cohort_version") != "capstone-nih-curated-cohort-v1"
            or cohort.get("policy_version") != "nih-development-quarantine-v1"
            or cohort.get("development_content_qualified") is not True
            or cohort.get("official_test_unchanged") is not True
            or type(cohort.get("independent_split_rgb_overlap_groups")) is not int
            or cohort["independent_split_rgb_overlap_groups"] != 0):
        raise ValueError("cache cohort content gate is not qualified")
    for field in ("source_hashes", "original_preparation_sha256", "image_audit_sha256",
                  "dataset", "classes", "label_definition"):
        if report.get(field) != cohort.get(field) or field not in report or field not in cohort:
            raise ValueError(f"cache cohort provenance differs: {field}")
    hashes = report["source_hashes"]
    if (not isinstance(hashes, dict)
            or set(hashes) != {"Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt"}
            or not all(_digest(value) for value in hashes.values())
            or not all(_digest(report.get(field)) for field in
                       ("original_preparation_sha256", "image_audit_sha256", "curated_cohort_sha256"))):
        raise ValueError("cache source identity is invalid")


def load_nih_cache(output: Path, *, expected_report_sha256: str | None = None) -> dict:
    """Load read-only arrays after integrity and alignment checks.

    A trusted expected digest binds this cache to a caller's selected experiment.
    Hashes alone do not authenticate a replacement report or recompute features.
    """
    output = Path(output)
    try:
        if expected_report_sha256 is not None and not _digest(expected_report_sha256):
            raise ValueError("invalid expected cache report digest")
        if output.is_symlink() or not output.is_dir():
            raise ValueError("cache directory is missing or a symlink")
        if {p.name for p in output.iterdir()} != PAYLOADS | {"report.json"}:
            raise ValueError("cache has missing or unexpected files")
        for name in PAYLOADS | {"report.json"}:
            if (output / name).is_symlink() or not (output / name).is_file():
                raise ValueError("cache payload is not a regular file")
        report_sha = sha256_file(output / "report.json")
        if expected_report_sha256 is not None and report_sha != expected_report_sha256:
            raise ValueError("cache report identity differs from expected digest")
        report = _canonical_report(output / "report.json")
        hashes = report.get("payload_hashes")
        if not isinstance(hashes, dict) or set(hashes) != PAYLOADS:
            raise ValueError("cache payload hash allowlist mismatch")
        for name, expected in hashes.items():
            if not _digest(expected) or sha256_file(output / name) != expected:
                raise ValueError(f"cache payload hash mismatch: {name}")
        cohort = _canonical_report(output / "cohort-report.json")
        _validate_provenance(report, cohort)
        if report["curated_cohort_sha256"] != hashes["cohort-report.json"]:
            raise ValueError("cache curated report identity mismatch")
        splits, all_patients, all_images = {}, set(), set()
        for split in SPLITS:
            name = f"{split}.csv"
            if (hashes[name] != cohort["payload_hashes"][name]
                    or report["selected_manifest_hashes"][name] != hashes[name]):
                raise ValueError("cache manifest differs from bound curated cohort")
            rows = read_manifest(output / name)
            patients, images = {r["patient_id"] for r in rows}, {r["image_index"] for r in rows}
            if patients & all_patients or images & all_images:
                raise ValueError("cache independent splits overlap")
            all_patients.update(patients)
            all_images.update(images)
            labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
            counts = {"images": len(rows), "patients": len(patients),
                      "positive_images": int(labels.sum()), "negative_images": int(len(rows) - labels.sum())}
            if report["splits"][split] != counts or cohort["selected_splits"][split] != counts:
                raise ValueError("cache split summary disagrees with identities")
            x = _load_array(output / f"{split}-x.npy")
            y = _load_array(output / f"{split}-y.npy")
            if (x.dtype != np.dtype("float32") or x.shape != (len(rows), 512)
                    or y.dtype != np.dtype("int64") or y.shape != (len(rows),)):
                raise ValueError("cache array dtype or shape mismatch")
            if not np.array_equal(y, labels):
                raise ValueError("cache label array differs from manifest rows")
            for start in range(0, len(rows), 4096):
                if not np.isfinite(x[start:start + 4096]).all():
                    raise ValueError("cache features contain nonfinite values")
            splits[split] = {"x": x, "y": y, "image_ids": [r["image_index"] for r in rows],
                             "patient_ids": [r["patient_id"] for r in rows]}
        return {"splits": splits, "provenance": report, "cohort_provenance": cohort,
                "report_sha256": report_sha}
    except (OSError, ValueError, TypeError, KeyError, csv.Error, OverflowError) as error:
        raise ValueError(f"NIH cache verification failed: {error}") from error
