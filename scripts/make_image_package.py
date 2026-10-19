#!/usr/bin/env python3
"""Make an image dataset package a phone can import for a run on images (Stage 4 S6).

Samples CIFAR-10 images (raw 8-bit RGB, 32 x 32), or chest X-rays for PNEUMONIA_CNN, from one split with a fixed seed and writes them as a package:
dataset.json + records.jsonl in a zip (fedlearn.contract.image_package). A sidecar <out>.json records the split, seed
and the sampled indices, so a replay can rebuild exactly what the phone trained.

Take the phone's images from the training split: the CNN's FL server evaluates on the test split, so training images
drawn from it would make the server's accuracy optimistic.

    PYTHONPATH=framework/src .venv/bin/python scripts/make_image_package.py --out cifar_300.zip --count 300 --seed 7

--recipe CIFAR_RESNET18 packages the 112 x 112 images that recipe trains on, resized exactly as its laptops resize them.
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


def prepare(img, recipe_key: str) -> np.ndarray:
    """The 8-bit pixels (H x W x C) a device imports for ``recipe_key``: the image through the recipe's own steps before
    ToTensor, exactly as its laptops apply them. Those steps (CIFAR_RESNET18's 32 -> 112 Resize; PNEUMONIA_CNN's
    Grayscale and 224 x 224 Resize) are PIL operations that yield 8-bit pixels, so the contract's ImageToUnitTensor and
    NormalizeChannels then reproduce the laptop's tensors exactly. A recipe without a Grayscale step trains RGB."""
    import torchvision.transforms as T
    import recipes
    transforms = {"CNN": recipes._cnn_transform, "CIFAR_RESNET18": recipes._resnet18_transform,
                  "PNEUMONIA_CNN": recipes.pneumonia_transform,
                  "NIH_PNEUMONIA_CNN": recipes.pneumonia_transform}[recipe_key]().transforms
    if not any(isinstance(t, T.Grayscale) for t in transforms):
        img = img.convert("RGB")
    for t in transforms:
        if isinstance(t, T.ToTensor):
            break
        if not isinstance(t, (T.Grayscale, T.Resize)):
            raise ValueError(f"{recipe_key}'s step {t} before ToTensor is not one a package can carry exactly")
        img = t(img)
    pixels = np.asarray(img, dtype=np.uint8)
    return pixels[..., None] if pixels.ndim == 2 else pixels


def cifar10(split: str):
    """(images [n, 32, 32, 3] uint8, labels as class names, class names in label order) of a CIFAR-10 split."""
    import datasets as hf_datasets
    import recipes
    data = hf_datasets.load_dataset("cifar10")[split]
    names = data.features["label"].names
    if names != recipes.get_recipe("CNN").classes:
        raise SystemExit(f"CIFAR-10 label names {names} are not the CNN recipe's classes")
    return data, names


def chest_xray(split: str):
    """(raw images, labels as class names, class names) of the pneumonia recipe's own source split, read as its laptops
    read it (recipes._full_dataset: FEDLEARN_PNEUMONIA_DIR or the HuggingFace chest X-ray set), labels NORMAL=0,
    PNEUMONIA=1."""
    import recipes
    base = recipes._full_dataset(split)
    names = recipes.PNEUMONIA_CLASSES

    class Rows:
        def __len__(self):
            return len(base.indices)

        def select(self, indices):
            return [(_raw_xray(base, i), names[base.labels()[i]]) for i in indices]

    return Rows(), names


def _raw_xray(base, i):
    """The untransformed image at position i of a recipes pneumonia dataset wrapper."""
    from PIL import Image
    if hasattr(base, "_folder"):                       # a local ImageFolder
        path, _ = base._folder.samples[base.indices[i]]
        return Image.open(path)
    img = base._ds[base.indices[i]][base._img_col]     # HuggingFace
    return img if hasattr(img, "mode") else Image.open(img) if isinstance(img, str) else Image.fromarray(img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--count", required=True, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--recipe", default="CNN", choices=["CNN", "CIFAR_RESNET18", "PNEUMONIA_CNN"],
                    help="prepare the images as this recipe's laptops do (CIFAR_RESNET18 resizes CIFAR-10 to 112 x 112; "
                         "PNEUMONIA_CNN takes chest X-rays to 224 x 224 grayscale)")
    a = ap.parse_args()
    if a.recipe == "PNEUMONIA_CNN":
        data, names = chest_xray(a.split)
        indices = sorted(np.random.default_rng(a.seed).choice(len(data), size=a.count, replace=False).tolist())
        pairs = data.select(indices)
        images = np.stack([prepare(img, a.recipe) for img, _ in pairs])
        labels = [label for _, label in pairs]
    else:
        data, names = cifar10(a.split)
        indices = sorted(np.random.default_rng(a.seed).choice(len(data), size=a.count, replace=False).tolist())
        rows = data.select(indices)
        images = np.stack([prepare(img, a.recipe) for img in rows["img"]])
        labels = [names[i] for i in rows["label"]]
    write_package(a.out, images, labels, class_names=names)
    sidecar = {
        "dataset": "chest-xray" if a.recipe == "PNEUMONIA_CNN" else "cifar10",
        "split": a.split, "seed": a.seed, "count": a.count, "recipe": a.recipe,
        "image_shape": list(images.shape[1:]), "indices": indices,
        "classNames": names, "labelCounts": {n: labels.count(n) for n in names},
        "packageSha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
    }
    a.out.with_suffix(".json").write_text(json.dumps(sidecar, indent=1) + "\n")
    print(f"wrote {a.out} ({a.count} images, {a.out.stat().st_size} bytes); provenance in {a.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
