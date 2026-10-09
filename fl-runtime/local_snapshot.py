"""Validated local data and the contract's reproducible minibatch order.

Accepts a UTF-8 CSV with one named label column, or the immutable snapshot
directory used by Android. All rows belong to this client; no shared-data
partitioner, cached split, normalization or implicit holdout is applied.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from fedlearn.communication.generated import execution_contract_pb2 as pb
from fedlearn.contract.batch_order import batches, seeded_permutation

MAX_BYTES = 256 * 1024 * 1024
MAX_RECORDS = 1_000_000


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class LocalSnapshot:
    dataset: TensorDataset
    snapshot_id: str


def _read(path: Path, limit=MAX_BYTES) -> bytes:
    if path.stat().st_size > limit:
        raise ValueError("local dataset exceeds the size limit")
    with path.open("rb") as file:
        data = file.read(limit + 1)
    if len(data) > limit:
        raise ValueError("local dataset exceeds the size limit")
    return data


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate snapshot metadata field")
        result[key] = value
    return result


def snapshot_content(shape, classes, schema, xbytes, ybytes, n, *, modality="vector", transforms=None):
    content = dict(schemaVersion=1, modality=modality, inputShape=shape, inputDtype="f32",
                   classNames=classes, labelSchemaId=schema, recordCount=n,
                   inputsSha256=digest(xbytes), targetsSha256=digest(ybytes))
    if transforms is not None:
        content["transforms"] = transforms
    return content


def load_prepared_snapshot(path, shape, classes, schema, *, modality="vector", transforms=None):
    """Read Android's immutable tensors only after schema, preparation and hashes agree."""
    meta = json.loads(_read(path / "snapshot.json", 64 * 1024), object_pairs_hook=_unique_fields)
    keys = {"schemaVersion", "modality", "inputShape", "inputDtype", "classNames", "labelSchemaId",
            "recordCount", "inputsSha256", "targetsSha256", "snapshotId"}
    if transforms is not None:
        keys.add("transforms")
    if not isinstance(meta, dict) or set(meta) != keys:
        raise ValueError("snapshot metadata fields differ from schema v1")
    if (type(meta["schemaVersion"]) is not int or meta["schemaVersion"] != 1
            or meta["modality"] != modality or meta["inputShape"] != shape
            or any(type(d) is not int for d in meta["inputShape"])
            or meta["inputDtype"] != "f32" or meta["classNames"] != classes
            or meta["labelSchemaId"] != schema or type(meta["recordCount"]) is not int
            or not 1 <= meta["recordCount"] <= MAX_RECORDS
            or meta.get("transforms") != transforms):
        raise ValueError("snapshot schema or preparation differs from the run's data requirement")
    content = {k: v for k, v in meta.items() if k != "snapshotId"}
    if digest(canonical(content)) != meta["snapshotId"]:
        raise ValueError("snapshot metadata digest mismatch")
    n = meta["recordCount"]
    if n * (math.prod(shape) * 4 + 8) > MAX_BYTES:
        raise ValueError("prepared local dataset exceeds the size limit")
    xbytes, ybytes = _read(path / "inputs.f32"), _read(path / "targets.i64")
    if digest(xbytes) != meta["inputsSha256"] or digest(ybytes) != meta["targetsSha256"]:
        raise ValueError("snapshot content digest mismatch")
    if len(xbytes) != n * math.prod(shape) * 4 or len(ybytes) != n * 8:
        raise ValueError("snapshot tensor sizes differ from recordCount")
    x = np.frombuffer(xbytes, dtype="<f4").reshape(n, *shape).copy()
    y = np.frombuffer(ybytes, dtype="<i8").copy()
    return x, y, content


def load_local_snapshot(path, requirement, classes) -> LocalSnapshot:
    if requirement.task == pb.TASK_IMAGE_CLASSIFICATION:
        from local_images import load_image_snapshot
        return load_image_snapshot(path, requirement, classes)
    return load_vector_snapshot(path, requirement, classes)


def load_vector_snapshot(path, requirement, classes) -> LocalSnapshot:
    """Read once into owned tensors, refusing malformed data or a different schema."""
    classes = list(classes)
    schema = "labels-sha256:" + digest(json.dumps(classes, separators=(",", ":")).encode())
    shape = list(requirement.input_shape)
    if (requirement.task != pb.TASK_VECTOR_CLASSIFICATION or len(shape) != 1 or shape[0] < 1
            or requirement.input_dtype != pb.DTYPE_F32 or requirement.class_count != len(classes)
            or requirement.label_schema_id != schema or len(requirement.transforms) != 1
            or requirement.transforms[0].WhichOneof("operation") != "identity_vector"
            or requirement.transforms[0].identity_vector.width != shape[0]):
        raise ValueError("local dataset requires a supported vector schema and identity transform")
    width = shape[0]
    path = Path(path)
    if path.is_dir():
        x, y, content = load_prepared_snapshot(path, shape, classes, schema)
    else:
        lines = _read(path).decode("utf-8").splitlines()
        if not lines:
            raise ValueError("local dataset is empty")
        header = [s.strip() for s in lines[0].split(",")]
        if header.count("label") != 1 or len(header) != width + 1 or len(set(header)) != len(header):
            raise ValueError("CSV needs one label column and the run's feature width")
        label_at = header.index("label")
        rows, targets = [], []
        for line in lines[1:]:
            if not line.strip():
                continue
            fields = [s.strip() for s in line.split(",")]
            if '"' in line or len(fields) != len(header) or len(rows) >= MAX_RECORDS:
                raise ValueError("invalid local CSV row or record limit exceeded")
            targets.append(classes.index(fields[label_at]))
            rows.append([float(v) for i, v in enumerate(fields) if i != label_at])
        if not rows:
            raise ValueError("local dataset is empty")
        with np.errstate(over="ignore"):
            x = np.asarray(rows, dtype="<f4")
        y = np.asarray(targets, dtype="<i8")
        content = dict(schemaVersion=1, modality="vector", inputShape=shape, inputDtype="f32",
                       classNames=classes, labelSchemaId=schema, recordCount=len(y),
                       inputsSha256=digest(x.tobytes()), targetsSha256=digest(y.tobytes()))
    if not np.isfinite(x).all() or (y < 0).any() or (y >= len(classes)).any():
        raise ValueError("local data contains non-finite features or out-of-range labels")
    return LocalSnapshot(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), digest(canonical(content)))


class SeededSnapshotLoader:
    """One permutation per epoch, reset to the authoritative server round before fit."""
    def __init__(self, dataset, *, batch_size, seed, drop_last):
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        self.dataset, self.batch_size, self.seed, self.drop_last = dataset, batch_size, seed, drop_last
        self.round = None
        self.epoch = 0

    def set_round(self, round_):
        if type(round_) is not int or not 1 <= round_ < 2**64:
            raise ValueError("local snapshot training requires the actual server round")
        self.round, self.epoch = round_, 0

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.round is None:
            raise ValueError("local snapshot training requires the actual server round")
        order = seeded_permutation(len(self.dataset), self.seed, self.round, self.epoch)
        self.epoch += 1
        x, y = self.dataset.tensors
        for indices in batches(order, self.batch_size, self.drop_last):
            yield x[indices], y[indices]
