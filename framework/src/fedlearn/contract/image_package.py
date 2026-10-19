"""Image dataset packages: the zip a participant imports on a phone for a run on images (Stage 4 S6).

A package is a zip of two files:

- ``dataset.json``: ``{"schemaVersion": 1, "modality": "image", "height", "width", "channels", "classNames",
  "recordCount"}``;
- ``records.jsonl``: one ``{"label": <class name>, "pixels": <base64 of the H*W*C uint8 values, HWC>}`` per image.

The pixels are raw, so a device prepares them with the contract's ImageToUnitTensor and NormalizeChannels exactly; no
image decoder is involved. Writing refuses what a device's importer would refuse. The output is deterministic (fixed
zip timestamps), so a package's bytes depend only on its content.
"""
from __future__ import annotations

import base64
import json
import os
import zipfile
from typing import BinaryIO, Sequence, Union

import numpy as np

_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def write_package(target: Union[str, "os.PathLike[str]", BinaryIO], images: np.ndarray, labels: Sequence[str],
                  class_names: Sequence[str]) -> None:
    """Write images [n, height, width, channels] (uint8) with their class-name labels as a package."""
    if images.dtype != np.uint8 or images.ndim != 4:
        raise ValueError(f"expected uint8 images [n, height, width, channels], not {images.dtype} {images.shape}")
    n, h, w, c = images.shape
    if c not in (1, 3):
        raise ValueError(f"an image has 1 or 3 channels, not {c}")
    if len(labels) != n:
        raise ValueError(f"{len(labels)} labels for {n} images")
    unknown = sorted(set(labels) - set(class_names))
    if unknown:
        raise ValueError(f"labels {unknown[:5]} are not among the classes")
    meta = {"schemaVersion": 1, "modality": "image", "height": h, "width": w, "channels": c,
            "classNames": list(class_names), "recordCount": n}
    records = "".join(json.dumps({"label": label, "pixels": base64.b64encode(image.tobytes()).decode("ascii")},
                                 separators=(",", ":")) + "\n" for image, label in zip(images, labels))
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, text in (("dataset.json", json.dumps(meta, separators=(",", ":"))), ("records.jsonl", records)):
            info = zipfile.ZipInfo(name, date_time=_FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, text)


def read_package(source) -> tuple[np.ndarray, list[str], list[str]]:
    """(images [n, h, w, c] uint8, labels, class names) of a package."""
    with zipfile.ZipFile(source) as z:
        meta = json.loads(z.read("dataset.json"))
        lines = z.read("records.jsonl").decode("utf-8").splitlines()
    shape = (meta["height"], meta["width"], meta["channels"])
    records = [json.loads(line) for line in lines if line.strip()]
    images = np.stack([np.frombuffer(base64.b64decode(r["pixels"]), dtype=np.uint8).reshape(shape)
                       for r in records])
    return images, [r["label"] for r in records], list(meta["classNames"])
