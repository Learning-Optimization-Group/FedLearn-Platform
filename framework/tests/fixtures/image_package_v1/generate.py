"""Write image_package_v1: a tiny image package and the tensors the contract's transforms make of it.

tiny.zip holds 3 RGB images of 4 x 5 pixels (classes cat, dog). tiny_inputs.f32 is what ImageToUnitTensor then
NormalizeChannels(0.5, 0.5) make of them (CHW float32, little-endian), and tiny_targets.i64 their class indices. The
Kotlin importer's test imports tiny.zip and must write exactly these bytes.

    PYTHONPATH=framework/src .venv/bin/python framework/tests/fixtures/image_package_v1/generate.py
"""
from __future__ import annotations

import io
import os

import numpy as np

from fedlearn.contract.image_package import write_package
from fedlearn.contract.image_transforms import image_to_unit_tensor, normalize_channels

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["cat", "dog"]
LABELS = ["dog", "cat", "dog"]


def build() -> tuple[bytes, bytes, bytes]:
    images = np.random.default_rng(23).integers(0, 256, size=(3, 4, 5, 3), dtype=np.uint8)
    buf = io.BytesIO()
    write_package(buf, images, LABELS, CLASSES)
    tensors = np.stack([normalize_channels(image_to_unit_tensor(p), [0.5] * 3, [0.5] * 3) for p in images])
    targets = np.array([CLASSES.index(label) for label in LABELS], dtype="<i8")
    return buf.getvalue(), tensors.astype("<f4").tobytes(), targets.tobytes()


def main() -> None:
    for name, data in zip(("tiny.zip", "tiny_inputs.f32", "tiny_targets.i64"), build()):
        with open(os.path.join(HERE, name), "wb") as fh:
            fh.write(data)


if __name__ == "__main__":
    main()
