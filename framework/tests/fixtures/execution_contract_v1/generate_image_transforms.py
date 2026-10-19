"""Write image_transforms_v1.golden: the tensors the contract's image transforms must produce, to the bit.

Each line is ``case name h w c mean=m,... std=s,... : pixels hex : float32 words hex``: an 8-bit HWC image, the
optional NormalizeChannels settings (empty for none), and the CHW float32 result of ImageToUnitTensor (then
NormalizeChannels). The Kotlin importer's test reproduces every line; test_image_transforms.py checks that this file is
what the reference computes, and the reference is checked against torchvision.

    PYTHONPATH=framework/src .venv/bin/python framework/tests/fixtures/execution_contract_v1/generate_image_transforms.py
"""
from __future__ import annotations

import os

import numpy as np

from fedlearn.contract.image_transforms import image_to_unit_tensor, normalize_channels

HERE = os.path.dirname(os.path.abspath(__file__))

HALF3, HALF1 = ([0.5] * 3, [0.5] * 3), ([0.5], [0.5])
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _random(h, w, c, seed):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, c), dtype=np.uint8)


CASES = [
    ("every_byte_gray", np.arange(256, dtype=np.uint8).reshape(16, 16, 1), None),
    ("every_byte_gray_half", np.arange(256, dtype=np.uint8).reshape(16, 16, 1), HALF1),
    ("one_pixel_rgb", _random(1, 1, 3, 1), None),
    ("rgb_7x5_half", _random(7, 5, 3, 2), HALF3),
    ("rgb_7x5_imagenet", _random(7, 5, 3, 3), IMAGENET),
    ("rgb_4x6_imagenet_extremes", np.array([0, 255] * 36, dtype=np.uint8).reshape(4, 6, 3), IMAGENET),
]


def _floats(values) -> str:
    return ",".join(repr(float(np.float32(v))) for v in values)


def golden_text() -> str:
    lines = ["# case name h w c mean=... std=... : pixels (HWC uint8) hex : result (CHW float32 bits) hex "
             "(ImageToUnitTensor, NormalizeChannels; generate_image_transforms.py)"]
    for name, pixels, norm in CASES:
        h, w, c = pixels.shape
        out = image_to_unit_tensor(pixels)
        mean, std = norm if norm else ([], [])
        if norm:
            out = normalize_channels(out, mean, std)
        words = " ".join(f"{b:08x}" for b in np.ascontiguousarray(out, dtype=np.float32).view(np.uint32).ravel())
        lines.append(f"case {name} {h} {w} {c} mean={_floats(mean)} std={_floats(std)} : "
                     f"{pixels.tobytes().hex()} : {words}")
    return "\n".join(lines) + "\n"


def main() -> None:
    with open(os.path.join(HERE, "image_transforms_v1.golden"), "w") as fh:
        fh.write(golden_text())


if __name__ == "__main__":
    main()
