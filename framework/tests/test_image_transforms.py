"""The execution contract's image transforms (Stage 4 S5) are torchvision's, to the bit.

A phone imports a user's images as raw 8-bit pixels and converts them with ImageToUnitTensor and NormalizeChannels. A
laptop trains the same images through torchvision's ToTensor and Normalize. For a phone's update to stay replayable,
the tensors must be identical, not close: fedlearn.contract.image_transforms is the reference both are checked
against, and image_transforms_v1.golden carries its output to the Kotlin importer.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF

from fedlearn.communication.generated import execution_contract_pb2 as pb
from fedlearn.contract import image_transforms as it

GOLDEN = os.path.join(os.path.dirname(__file__), "fixtures", "execution_contract_v1", "image_transforms_v1.golden")

SHAPES = [(1, 1, 1), (1, 1, 3), (7, 5, 3), (16, 16, 1), (32, 32, 3), (3, 224, 1)]
NORMS = {
    "half_rgb": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "half_gray": ([0.5], [0.5]),
}


def _pixels(h, w, c, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, c), dtype=np.uint8)


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_unit_tensor_is_torchvisions_to_tensor(shape):
    pixels = _pixels(*shape)
    ours = it.image_to_unit_tensor(pixels)
    theirs = TF.to_tensor(pixels).numpy()
    assert ours.shape == (shape[2], shape[0], shape[1]) and ours.dtype == np.float32
    assert np.array_equal(_bits(ours), _bits(theirs))


def test_every_byte_value_converts_as_torchvision_does():
    pixels = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
    assert np.array_equal(_bits(it.image_to_unit_tensor(pixels)), _bits(TF.to_tensor(pixels).numpy()))


@pytest.mark.parametrize("name", NORMS)
def test_normalisation_is_torchvisions_normalize(name):
    mean, std = NORMS[name]
    pixels = _pixels(32, 32, len(mean), seed=3)
    ours = it.normalize_channels(it.image_to_unit_tensor(pixels), mean, std)
    theirs = TF.normalize(TF.to_tensor(pixels), mean, std).numpy()
    assert np.array_equal(_bits(ours), _bits(theirs))


def test_the_contracts_transforms_apply_in_order():
    transforms = [pb.Transform(image_to_unit_tensor=pb.ImageToUnitTensor(height=7, width=5, channels=3)),
                  pb.Transform(normalize_channels=pb.NormalizeChannels(mean=[0.5] * 3, std=[0.5] * 3))]
    pixels = _pixels(7, 5, 3, seed=9)
    expected = TF.normalize(TF.to_tensor(pixels), [0.5] * 3, [0.5] * 3).numpy()
    assert np.array_equal(_bits(it.apply_transforms(pixels, transforms)), _bits(expected))


def test_the_stated_float32_mean_is_used_not_a_float64_one():
    # The contract carries mean and std as float32: 0.485 is stored as float32(0.485), and torchvision rounds its
    # Python floats to the tensor's float32 the same way. Normalising with the float64 value lands elsewhere.
    pixels = _pixels(32, 32, 3, seed=5)
    x = it.image_to_unit_tensor(pixels)
    mean, std = NORMS["imagenet"]
    stated = it.normalize_channels(x, mean, std)
    float64_mean = ((x.astype(np.float64) - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]).astype(
        np.float32)
    assert not np.array_equal(_bits(stated), _bits(float64_mean))


def test_pixels_that_are_not_the_stated_image_are_refused():
    transforms = [pb.Transform(image_to_unit_tensor=pb.ImageToUnitTensor(height=7, width=5, channels=3))]
    with pytest.raises(ValueError):
        it.apply_transforms(_pixels(5, 7, 3), transforms)
    with pytest.raises(ValueError):
        it.apply_transforms(_pixels(7, 5, 3).astype(np.int16), transforms)


def test_the_golden_is_what_the_reference_computes():
    """image_transforms_v1.golden, which the Kotlin importer must reproduce, is regenerated here."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_image_transforms", os.path.join(
        os.path.dirname(GOLDEN), "generate_image_transforms.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with open(GOLDEN) as fh:
        assert fh.read() == module.golden_text()
