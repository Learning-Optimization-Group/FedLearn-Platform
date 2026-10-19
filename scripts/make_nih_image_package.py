#!/usr/bin/env python3
"""Package a verified NIH client partition for the binary pneumonia recipe."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "fl-runtime"), str(REPO / "framework/src")]

from fedlearn.contract.image_package import write_package
from local_snapshot import MAX_BYTES, _read, digest
from make_image_package import prepare
from nih_dataset import CLASSES
from nih_runtime import load_manifest, select_rows, NihDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--patient-unique", action="store_true")
    parser.add_argument("--purpose", choices=["train", "eval"], default="train")
    args = parser.parse_args()
    if args.purpose == "train" and not re.fullmatch(r"client-[0-9]+\.csv", args.manifest):
        parser.error("training packages require a client partition; held-out data requires --purpose eval")
    if args.purpose == "eval" and args.manifest not in ("validation.csv", "test.csv"):
        parser.error("evaluation packages require validation.csv or test.csv")
    if not 1 <= args.count <= MAX_BYTES // (224 * 224 * 4 + 8):
        parser.error("package count exceeds the prepared tensor size limit")
    sidecar = args.out.with_suffix(".json")
    if args.out == sidecar or args.out.exists() or sidecar.exists():
        parser.error("package or provenance output already exists")
    rows, provenance = load_manifest(args.source, args.prepared, args.manifest)
    selected, sample = select_rows(rows, count=args.count, seed=args.seed,
                                  balanced=args.balanced, patient_unique=args.patient_unique)
    bound = NihDataset(args.source, selected, dict(provenance, **sample))
    images = []
    for row, expected_hash in zip(selected, bound.image_hashes):
        data = _read(bound.source / row["image_path"], 32 * 1024 * 1024)
        if digest(data) != expected_hash:
            raise ValueError("NIH image changed while packaging")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or not all(1 <= side <= 4096 for side in image.size):
                raise ValueError("invalid NIH PNG or image dimensions")
            images.append(prepare(image, "NIH_PNEUMONIA_CNN"))
    labels = [row["label"] for row in selected]
    report = dict(bound.provenance, purpose=args.purpose, recipe="NIH_PNEUMONIA_CNN",
        image_shape=[224, 224, 1], image_sha256=bound.image_hashes,
        records=selected, preparation="PIL grayscale + bilinear resize; contract float32 unit tensor and mean/std 0.5")
    created = []
    try:
        with args.out.open("xb") as file:
            created.append(args.out)
            write_package(file, np.stack(images), labels, class_names=CLASSES)
        report["package_sha256"] = digest(args.out.read_bytes())
        with sidecar.open("x") as file:
            created.append(sidecar)
            file.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    print(f"Wrote {len(selected)} NIH {args.purpose} images; provenance: {sidecar}")


if __name__ == "__main__":
    main()
