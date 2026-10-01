"""Image dataset packages (Stage 4 S6): the zipped dataset.json + records.jsonl a phone imports.

fedlearn.contract.image_package writes and reads them. A committed package (fixtures/image_package_v1) and the tensors
the contract's transforms make of it are regenerated here; the Kotlin importer's test imports the same zip and must
produce the same bytes, so the format is pinned on both sides.
"""
from __future__ import annotations

import io
import json
import os
import zipfile

import numpy as np
import pytest

from fedlearn.contract import image_package as ip

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "image_package_v1")


def _images(n=3, h=4, w=5, c=3, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(n, h, w, c), dtype=np.uint8)


def test_a_written_package_reads_back_exactly(tmp_path):
    images, labels = _images(), ["cat", "dog", "cat"]
    path = tmp_path / "p.zip"
    ip.write_package(path, images, labels, class_names=["cat", "dog"])
    back, back_labels, classes = ip.read_package(path)
    assert np.array_equal(back, images) and back_labels == labels and classes == ["cat", "dog"]


def test_the_package_is_the_format_the_device_reads(tmp_path):
    path = tmp_path / "p.zip"
    ip.write_package(path, _images(n=2), ["dog", "cat"], class_names=["cat", "dog"])
    with zipfile.ZipFile(path) as z:
        assert sorted(z.namelist()) == ["dataset.json", "records.jsonl"]
        meta = json.loads(z.read("dataset.json"))
        records = [json.loads(line) for line in z.read("records.jsonl").decode().splitlines()]
    assert meta == {"schemaVersion": 1, "modality": "image", "height": 4, "width": 5, "channels": 3,
                    "classNames": ["cat", "dog"], "recordCount": 2}
    assert [r["label"] for r in records] == ["dog", "cat"] and set(records[0]) == {"label", "pixels"}


@pytest.mark.parametrize("bad", [
    lambda: ip.write_package(io.BytesIO(), _images().astype(np.int16), ["cat"] * 3, class_names=["cat"]),
    lambda: ip.write_package(io.BytesIO(), _images(), ["cat", "dog"], class_names=["cat", "dog"]),
    lambda: ip.write_package(io.BytesIO(), _images(), ["cat", "cow", "cat"], class_names=["cat", "dog"]),
    lambda: ip.write_package(io.BytesIO(), _images(c=2), ["cat"] * 3, class_names=["cat"]),
])
def test_what_the_device_would_refuse_is_not_written(bad):
    with pytest.raises(ValueError):
        bad()


def test_the_committed_package_and_its_tensors_are_what_the_reference_writes():
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_image_package", os.path.join(FIXTURE, "generate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected_zip, expected_inputs, expected_targets = module.build()
    with open(os.path.join(FIXTURE, "tiny.zip"), "rb") as fh:
        assert fh.read() == expected_zip
    with open(os.path.join(FIXTURE, "tiny_inputs.f32"), "rb") as fh:
        assert fh.read() == expected_inputs
    with open(os.path.join(FIXTURE, "tiny_targets.i64"), "rb") as fh:
        assert fh.read() == expected_targets
