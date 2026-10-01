#!/usr/bin/env python3
"""Make an image dataset package a phone can import for a run on images (Stage 4 S6).

Samples CIFAR-10 images (raw 8-bit RGB, 32 x 32) from one split with a fixed seed and writes them as a package:
dataset.json + records.jsonl in a zip (fedlearn.contract.image_package). A sidecar <out>.json records the split, seed
and the sampled indices, so a replay can rebuild exactly what the phone trained.

Take the phone's images from the training split: the CNN's FL server evaluates on the test split, so training images
drawn from it would make the server's accuracy optimistic.

    PYTHONPATH=framework/src .venv/bin/python scripts/make_image_package.py --out cifar_300.zip --count 300 --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "fl-runtime"))

from fedlearn.contract.image_package import write_package  # noqa: E402


def cifar10(split: str):
    """(images [n, 32, 32, 3] uint8, labels as class names, class names in label order) of a CIFAR-10 split."""
    import datasets as hf_datasets
    import recipes
    data = hf_datasets.load_dataset("cifar10")[split]
    names = data.features["label"].names
    if names != recipes.get_recipe("CNN").classes:
        raise SystemExit(f"CIFAR-10 label names {names} are not the CNN recipe's classes")
    return data, names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--count", required=True, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    a = ap.parse_args()
    data, names = cifar10(a.split)
    indices = sorted(np.random.default_rng(a.seed).choice(len(data), size=a.count, replace=False).tolist())
    rows = data.select(indices)
    images = np.stack([np.asarray(img.convert("RGB"), dtype=np.uint8) for img in rows["img"]])
    labels = [names[i] for i in rows["label"]]
    write_package(a.out, images, labels, class_names=names)
    sidecar = {
        "dataset": "cifar10", "split": a.split, "seed": a.seed, "count": a.count, "indices": indices,
        "classNames": names, "labelCounts": {n: labels.count(n) for n in names},
        "packageSha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
    }
    a.out.with_suffix(".json").write_text(json.dumps(sidecar, indent=1) + "\n")
    print(f"wrote {a.out} ({a.count} images, {a.out.stat().st_size} bytes); provenance in {a.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
