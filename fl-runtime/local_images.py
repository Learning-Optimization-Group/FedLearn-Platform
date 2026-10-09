"""Contract-bound raw image packages and already prepared Android image snapshots."""
from __future__ import annotations

import base64
from contextlib import contextmanager
import json
import math
from pathlib import Path
import struct
import zipfile

import numpy as np
import torch
from torch.utils.data import TensorDataset

from fedlearn.communication.generated import execution_contract_pb2 as pb
from fedlearn.contract.image_transforms import apply_transforms
from local_snapshot import (MAX_BYTES, MAX_RECORDS, LocalSnapshot, _unique_fields, canonical,
                            digest, load_prepared_snapshot, snapshot_content)


def _preparation(requirement, classes):
    shape = list(requirement.input_shape)
    schema = "labels-sha256:" + digest(json.dumps(classes, separators=(",", ":")).encode())
    steps = requirement.transforms
    if (requirement.task != pb.TASK_IMAGE_CLASSIFICATION or len(shape) != 3
            or shape[0] not in (1, 3) or not all(1 <= d <= 4096 for d in shape[1:])
            or requirement.input_dtype != pb.DTYPE_F32 or requirement.class_count != len(classes)
            or requirement.label_schema_id != schema or len(steps) not in (1, 2)
            or steps[0].WhichOneof("operation") != "image_to_unit_tensor"):
        raise ValueError("unsupported local image schema or preparation")
    image = steps[0].image_to_unit_tensor
    c, h, w = shape
    if [image.channels, image.height, image.width] != shape:
        raise ValueError("image transform differs from required shape")
    preparation = f"image:{h}x{w}x{c}"
    if len(steps) == 2:
        if steps[1].WhichOneof("operation") != "normalize_channels":
            raise ValueError("unsupported local image normalization")
        norm = steps[1].normalize_channels
        mean, std = np.asarray(norm.mean, dtype="<f4"), np.asarray(norm.std, dtype="<f4")
        if len(mean) != c or len(std) != c or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("invalid image normalization")
        preparation += ";mean=" + ",".join(f"{bits:08x}" for bits in mean.view("<u4"))
        preparation += ";std=" + ",".join(f"{bits:08x}" for bits in std.view("<u4"))
    return shape, schema, preparation


def _check_zip_directory(file):
    """Bound ZIP's metadata before ZipFile materializes its central-directory entries."""
    size = file.seek(0, 2)
    if size > MAX_BYTES:
        raise ValueError("local image package exceeds the size limit")
    file.seek(max(0, size - 65557))  # EOCD plus the maximum ZIP comment
    tail = file.read(65557)
    end = tail.rfind(b"PK\x05\x06")
    if end < 0 or end + 22 > len(tail):
        raise ValueError("image package has no valid ZIP directory footer")
    absolute_end = size - len(tail) + end
    if absolute_end >= 20:
        file.seek(absolute_end - 20)
        if file.read(4) == b"PK\x06\x07":
            raise ValueError("ZIP64 directory is unsupported for bounded two-file image packages")
    _, disk, directory_disk, disk_entries, entries, directory_size, offset, comment = struct.unpack_from(
        "<4s4H2IH", tail, end)
    if (disk != 0 or directory_disk != 0 or entries != 2 or disk_entries != 2
            or not 1 <= directory_size <= 64 * 1024 or offset == 0xffffffff
            or offset + directory_size > size - len(tail) + end or end + 22 + comment != len(tail)):
        raise ValueError("unsupported or oversized ZIP directory")


@contextmanager
def _open_package(path):
    with path.open("rb") as file:
        _check_zip_directory(file)
        # Reuse the checked file descriptor even if the source path is replaced.
        with zipfile.ZipFile(file) as package:
            yield package


def _package(path, requirement, shape, classes, schema, preparation):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("local image package exceeds the size limit")
    try:
        with _open_package(path) as z:
            infos = z.infolist()
            if len(infos) != 2 or {i.filename for i in infos} != {"dataset.json", "records.jsonl"}:
                raise ValueError("image package must contain exactly dataset.json and records.jsonl")
            if z.getinfo("dataset.json").file_size > 64 * 1024 or z.getinfo("records.jsonl").file_size > MAX_BYTES:
                raise ValueError("image package expands beyond size limit")
            with z.open("dataset.json") as file:
                raw = file.read(64 * 1024 + 1)
                if len(raw) > 64 * 1024:
                    raise ValueError("image metadata exceeds size limit")
            meta = json.loads(raw, object_pairs_hook=_unique_fields)
            keys = {"schemaVersion", "modality", "height", "width", "channels", "classNames", "recordCount"}
            if not isinstance(meta, dict) or set(meta) != keys:
                raise ValueError("invalid image package metadata")
            c, h, w = shape
            if (any(type(meta[k]) is not int for k in ("schemaVersion", "height", "width", "channels", "recordCount"))
                    or meta["schemaVersion"] != 1 or meta["modality"] != "image"
                    or [meta["channels"], meta["height"], meta["width"]] != shape
                    or meta["classNames"] != classes or not 1 <= meta["recordCount"] <= MAX_RECORDS):
                raise ValueError("image package schema differs from the run")
            n, elements = meta["recordCount"], math.prod(shape)
            if n * (elements * 4 + 8) > MAX_BYTES:
                raise ValueError("prepared local images exceed size limit")
            x, y = np.empty((n, *shape), dtype="<f4"), np.empty(n, dtype="<i8")
            count, consumed = 0, 0
            line_limit = ((elements + 2) // 3) * 4 + 4096
            with z.open("records.jsonl") as file:
                while raw := file.readline(line_limit + 1):
                    consumed += len(raw)
                    if len(raw) > line_limit or consumed > MAX_BYTES:
                        raise ValueError("image record exceeds size limit")
                    if not raw.strip():
                        continue
                    if count >= n:
                        raise ValueError("image recordCount mismatch")
                    record = json.loads(raw, object_pairs_hook=_unique_fields)
                    if (not isinstance(record, dict) or set(record) != {"label", "pixels"}
                            or record["label"] not in classes or not isinstance(record["pixels"], str)):
                        raise ValueError("invalid image label or record")
                    pixels = base64.b64decode(record["pixels"], validate=True)
                    if len(pixels) != elements:
                        raise ValueError("image pixel length differs from required shape")
                    pixels = np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, c)
                    x[count] = apply_transforms(pixels, requirement.transforms)
                    y[count] = classes.index(record["label"])
                    count += 1
            if count != n:
                raise ValueError("image recordCount mismatch")
    except (zipfile.BadZipFile, UnicodeError, RuntimeError, NotImplementedError) as exc:
        raise ValueError("unreadable or unsupported image package") from exc
    content = snapshot_content(shape, classes, schema, x.tobytes(), y.tobytes(), n,
                               modality="image", transforms=preparation)
    return x, y, content


def load_image_snapshot(path, requirement, classes) -> LocalSnapshot:
    """Prepare package pixels once, or verify a snapshot's original preparation; never reapply it."""
    classes = list(classes)
    shape, schema, preparation = _preparation(requirement, classes)
    path = Path(path)
    if path.is_dir():
        x, y, content = load_prepared_snapshot(path, shape, classes, schema,
                                              modality="image", transforms=preparation)
    else:
        x, y, content = _package(path, requirement, shape, classes, schema, preparation)
    if not np.isfinite(x).all() or (y < 0).any() or (y >= len(classes)).any():
        raise ValueError("local image data contains non-finite inputs or out-of-range labels")
    return LocalSnapshot(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), digest(canonical(content)))
