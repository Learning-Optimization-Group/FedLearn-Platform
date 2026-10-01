"""The execution contract's image transforms, exactly as the contract specifies them.

``ImageToUnitTensor`` turns an 8-bit HWC image into a float32 CHW tensor, each value ``float32(p) / float32(255)``;
``NormalizeChannels`` then computes ``(v - mean[c]) / std[c]`` per channel with float32 mean and std. Each step is one
IEEE-754 binary32 operation rounded to nearest-even, so this reference, torchvision's ToTensor and Normalize, and a
device's importer produce identical bits. Pinned by image_transforms_v1.golden.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_255 = np.float32(255)


def image_to_unit_tensor(pixels: np.ndarray) -> np.ndarray:
    """An 8-bit image [height, width, channels] as float32 [channels, height, width], each value p / 255."""
    if pixels.dtype != np.uint8 or pixels.ndim != 3:
        raise ValueError(f"expected uint8 pixels [height, width, channels], not {pixels.dtype} {pixels.shape}")
    return np.ascontiguousarray((pixels.astype(np.float32) / _255).transpose(2, 0, 1))


def normalize_channels(tensor: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """(v - mean[c]) / std[c] for a float32 [channels, height, width] tensor, in float32 throughout."""
    channels = tensor.shape[0]
    if len(mean) != channels or len(std) != channels:
        raise ValueError(f"{len(mean)} means and {len(std)} stds for {channels} channels")
    m = np.asarray(mean, dtype=np.float32)[:, None, None]
    s = np.asarray(std, dtype=np.float32)[:, None, None]
    return (tensor.astype(np.float32, copy=False) - m) / s


def apply_transforms(pixels: np.ndarray, transforms) -> np.ndarray:
    """A contract's image transforms (ImageToUnitTensor, then optionally NormalizeChannels) applied to one image."""
    out = None
    for transform in transforms:
        operation = transform.WhichOneof("operation")
        if operation == "image_to_unit_tensor":
            image = transform.image_to_unit_tensor
            if pixels.shape != (image.height, image.width, image.channels):
                raise ValueError(f"pixels {pixels.shape} are not the stated image "
                                 f"{(image.height, image.width, image.channels)}")
            out = image_to_unit_tensor(pixels)
        elif operation == "normalize_channels" and out is not None:
            norm = transform.normalize_channels
            out = normalize_channels(out, list(norm.mean), list(norm.std))
        else:
            raise ValueError(f"{operation} cannot apply to an image here")
    if out is None:
        raise ValueError("no image conversion stated")
    return out
