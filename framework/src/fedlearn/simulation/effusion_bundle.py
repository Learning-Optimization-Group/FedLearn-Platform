"""Hash-pinned Effusion R0 bundle: standardised train + development-selection features, no holdout, no test.

Written once by the capstone preparation script from verified inputs; loaded by the Simulation Lab's Effusion
workload. The loader re-verifies every payload hash, shape, label and patient-disjointness before use, so a
bundle cannot silently carry holdout rows, changed arrays or a different label rule.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile

import numpy as np

BUNDLE_VERSION = "capstone-nih-effusion-r0-selection-v1"
FEAT_DIM = 512
SPLITS = (("train", "train"), ("selection", "evaluation"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_split(name, split):
    x, y = np.asarray(split["x"]), np.asarray(split["y"])
    if x.ndim != 2 or x.shape[1] != FEAT_DIM or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError(f"{name}: features must be finite float32 with {FEAT_DIM} columns")
    if y.dtype != np.int64 or y.shape != (len(x),) or not set(np.unique(y)) <= {0, 1}:
        raise ValueError(f"{name}: labels must be int64 binary, one per row")
    if len(split["patient_ids"]) != len(x) or len(split["image_ids"]) != len(x):
        raise ValueError(f"{name}: identity rows must match the feature rows")
    if not all(isinstance(p, str) and re.fullmatch(r"[1-9][0-9]*", p) for p in split["patient_ids"]):
        raise ValueError(f"{name}: patient IDs must be canonical positive decimal strings")
    if len(set(split["image_ids"])) != len(x):
        raise ValueError(f"{name}: duplicate image IDs")


def write_effusion_bundle(output, train: dict, selection: dict, *, provenance: dict) -> str:
    """Write the bundle once (refusing an existing path); return the report SHA-256."""
    output = Path(output)
    if output.exists():
        raise ValueError(f"output exists: {output}")
    for name, split in (("train", train), ("selection", selection)):
        _check_split(name, split)
    if set(train["patient_ids"]) & set(selection["patient_ids"]):
        raise ValueError("train/selection patient overlap")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".bundle-", dir=output.parent))
    hashes, counts = {}, {}
    for name, split in (("train", train), ("selection", selection)):
        for kind in ("x", "y"):
            path = staging / f"{name}-{kind}.npy"
            np.save(path, np.asarray(split[kind]))
            hashes[path.name] = _sha256(path.read_bytes())
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["image_index", "patient_id"])
        writer.writerows(zip(split["image_ids"], split["patient_ids"]))
        (staging / f"{name}.csv").write_bytes(stream.getvalue().encode())
        hashes[f"{name}.csv"] = _sha256(stream.getvalue().encode())
        y = np.asarray(split["y"])
        counts[name] = {"images": int(len(y)), "positives": int(y.sum()), "patients": len(set(split["patient_ids"]))}
    report = {"bundle_version": BUNDLE_VERSION, "dataset": "NIH_ChestXray14", "target": "Effusion",
              "feat_dim": FEAT_DIM, "n_classes": 2, "classes": ["NOT_EFFUSION", "EFFUSION"], "standardised": True,
              "pretrained": True, "holdoutIncluded": False, "testIncluded": False, "counts": counts,
              "payload_hashes": hashes, "provenance": provenance}
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    (staging / "report.json").write_bytes(raw)
    staging.rename(output)
    return _sha256(raw)


def is_effusion_bundle(path) -> bool:
    try:
        return json.loads((Path(path) / "report.json").read_bytes()).get("bundle_version") == BUNDLE_VERSION
    except (OSError, ValueError):
        return False


def load_effusion_bundle(path, *, expected_report_sha256: str) -> dict:
    """Verify and load a bundle; returns train and evaluation (selection) splits plus the report."""
    path = Path(path)
    raw = (path / "report.json").read_bytes()
    if _sha256(raw) != expected_report_sha256:
        raise ValueError("bundle report hash differs from the expected pin")
    report = json.loads(raw)
    if report.get("bundle_version") != BUNDLE_VERSION or report.get("holdoutIncluded") is not False \
            or report.get("testIncluded") is not False or report.get("feat_dim") != FEAT_DIM:
        raise ValueError("unsupported or unsafe bundle report")
    for name, digest in report["payload_hashes"].items():
        if _sha256((path / name).read_bytes()) != digest:
            raise ValueError(f"bundle payload hash mismatch: {name}")
    out = {"report": report, "report_sha256": expected_report_sha256}
    for name, key in SPLITS:
        rows = list(csv.reader(io.StringIO((path / f"{name}.csv").read_text(), newline="")))
        if rows[0] != ["image_index", "patient_id"]:
            raise ValueError(f"{name}: identity schema")
        split = {"x": np.load(path / f"{name}-x.npy"), "y": np.load(path / f"{name}-y.npy"),
                 "image_ids": [r[0] for r in rows[1:]], "patient_ids": [r[1] for r in rows[1:]]}
        _check_split(name, split)
        if report["counts"][name] != {"images": int(len(split["y"])), "positives": int(split["y"].sum()),
                                      "patients": len(set(split["patient_ids"]))}:
            raise ValueError(f"{name}: counts differ from the report")
        out[key] = split
    if set(out["train"]["patient_ids"]) & set(out["evaluation"]["patient_ids"]):
        raise ValueError("train/selection patient overlap")
    return out
