"""Training-only Effusion feature packs and the frozen CPU linear-head recipe.

The JSON manifest and its two sibling NumPy files are pinned before loading. Features
are already projected by preparation; clients only slice shared host storage. Tensor
storage is host-owned and treated as read-only throughout training.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
from threading import Lock
import time
from types import MappingProxyType

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fedlearn.simulation.rng import ClientRng

_SCHEMA_KEYS = {
    "schema",
    "dataset",
    "target",
    "training_only",
    "selection_included",
    "holdout_included",
    "test_included",
    "feat_dim",
    "original_feat_dim",
    "projection_seed",
    "projection_sha256",
    "width",
    "run_id",
    "host_label",
    "num_clients",
    "seed",
    "clients",
    "payload_sha256",
    "settings",
}
_SETTINGS = dict(
    lr=0.03, local_epochs=7, batch_size=256, clip_norm=0.05, optimizer="sgd", device="cpu", rounds=5
)
_HEAD_KEYS = ("head.weight", "head.bias")


class FitCancelled(RuntimeError):
    """A requested stop discarded an unfinished local fit."""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def _settings(settings):
    if not isinstance(settings, Mapping) or set(settings) != set(_SETTINGS):
        raise ValueError("settings differ from the frozen recipe schema")
    for key, expected in _SETTINGS.items():
        value = settings[key]
        if key == "rounds":
            if type(value) is not int or value not in (5, 20):
                raise ValueError("rounds must be 5 or 20")
        elif key in ("local_epochs", "batch_size"):
            if type(value) is not int or value != expected:
                raise ValueError(f"{key} differs from the frozen recipe")
        elif isinstance(expected, float):
            if type(value) not in (int, float) or not math.isfinite(value) or value != expected:
                raise ValueError(f"{key} differs from the frozen recipe")
        elif type(value) is not str or value != expected:
            raise ValueError(f"{key} differs from the frozen recipe")
    return dict(settings)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _verified_bytes(path, expected):
    _sha256(expected)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input must be a regular non-symlink file: {path.name}")
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError(f"input hash mismatch: {path.name}")
    return contents


@dataclass(frozen=True)
class HostData:
    """One verified training pack; all clients share these CPU tensors without copying."""

    x: torch.Tensor
    y: torch.Tensor
    manifest: Mapping
    clients: Mapping
    settings: Mapping


def load_host_data(path: Path, expected_sha256: str) -> HostData:
    """Load a pinned host JSON manifest and only its assigned ``x.npy``/``y.npy`` rows."""
    path = Path(path)
    raw = _verified_bytes(path, expected_sha256)
    try:
        manifest = json.loads(raw, object_pairs_hook=_unique_object)
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid host manifest JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != _SCHEMA_KEYS:
        raise ValueError("host manifest differs from the training-only schema")
    expected = dict(
        schema="fedlearn.effusion_host_data.v1",
        dataset="NIH_ChestXray14",
        target="Effusion",
        training_only=True,
        selection_included=False,
        holdout_included=False,
        test_included=False,
    )
    for key, value in expected.items():
        if type(manifest[key]) is not type(value) or manifest[key] != value:
            raise ValueError(f"invalid training-only manifest field: {key}")
    width = _integer(manifest["width"], "width", 1)
    if (
        width not in (128, 512)
        or type(manifest["feat_dim"]) is not int
        or manifest["feat_dim"] != width
    ):
        raise ValueError("feature width must agree and be 128 or 512")
    for key, value in (("original_feat_dim", 512), ("projection_seed", 20261025)):
        if type(manifest[key]) is not int or manifest[key] != value:
            raise ValueError(f"invalid {key}")
    _sha256(manifest["projection_sha256"])
    _integer(manifest["seed"], "seed")
    total_clients = _integer(manifest["num_clients"], "num_clients", 1)
    run_id = manifest["run_id"]
    if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 200:
        raise ValueError("run_id must be a nonempty string of at most 200 characters")
    if manifest["host_label"] not in ("mac", "g14", "orin"):
        raise ValueError("unknown host_label")
    settings = _settings(manifest["settings"])
    payloads = manifest["payload_sha256"]
    if not isinstance(payloads, dict) or set(payloads) != {"x.npy", "y.npy"}:
        raise ValueError("only assigned training feature/label payloads are permitted")
    for entry in path.parent.iterdir():
        if entry.name not in payloads and entry.suffix.lower() in (".npy", ".npz", ".pt", ".pth"):
            raise ValueError("unexpected array payload in training-only host pack")
    arrays = {}
    for name, expected_hash in payloads.items():
        try:
            source = BytesIO(_verified_bytes(path.parent / name, expected_hash))
            array = np.load(source, allow_pickle=False)
            if not isinstance(array, np.ndarray) or source.tell() != len(source.getbuffer()):
                raise ValueError("payload must contain exactly one NumPy array")
            arrays[name] = array
        except (OSError, TypeError) as exc:
            raise ValueError(f"invalid training payload: {name}") from exc
    x, y = arrays["x.npy"], arrays["y.npy"]
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != width or not np.isfinite(x).all():
        raise ValueError("features must be a finite float32 matrix of the declared width")
    if y.dtype != np.int64 or y.ndim != 1 or len(y) != len(x) or not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be an int64 binary vector matching the feature rows")
    if not x.flags.c_contiguous or not y.flags.c_contiguous:
        raise ValueError("training arrays must be C contiguous")
    members = manifest["clients"]
    if not isinstance(members, list) or not members:
        raise ValueError("at least one assigned logical client is required")
    clients, cursor = {}, 0
    for member in members:
        if not isinstance(member, dict) or set(member) != {
            "client_id",
            "start",
            "stop",
            "identity",
        }:
            raise ValueError("invalid assigned client schema")
        cid = _integer(member["client_id"], "client_id")
        start = _integer(member["start"], "start")
        stop = _integer(member["stop"], "stop", 1)
        if cid >= total_clients or cid in clients:
            raise ValueError("assigned client IDs must be unique and in the global cohort")
        if start != cursor or stop <= start or stop > len(y):
            raise ValueError("assigned ranges must partition all rows contiguously without overlap")
        if member["identity"] != f"{run_id}:client-{cid:05d}":
            raise ValueError("logical identity does not match run and client ID")
        clients[cid], cursor = _freeze(member), stop
    if cursor != len(y):
        raise ValueError("assigned ranges leave missing training rows")
    return HostData(
        torch.from_numpy(x),
        torch.from_numpy(y),
        _freeze(manifest),
        MappingProxyType(clients),
        _freeze(settings),
    )


class _Head(nn.Module):
    """A binary linear head whose construction never consumes a global RNG draw."""

    def __init__(self, width):
        super().__init__()
        self.head = nn.Module()
        self.head.weight = nn.Parameter(torch.zeros((2, width), dtype=torch.float32))
        self.head.bias = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(self, x):
        return F.linear(x, self.head.weight, self.head.bias)


def _parameters(model):
    return OrderedDict((key, value.detach().clone()) for key, value in model.named_parameters())


class FeatureState:
    """One logical client's model with shared data and a fresh SGD optimizer per fit.

    ``settings`` must exactly match the verified pack. Rounds are one-based and
    bounded by its declared 5/20-round budget. ``get_parameters`` returns owned copies.
    """

    def __init__(self, data: HostData, client_id: int, settings: dict):
        _integer(client_id, "client_id")
        if client_id not in data.clients:
            raise ValueError("client is not assigned to this host")
        settings = _settings(settings)
        if settings != dict(data.settings):
            raise ValueError("client settings differ from the verified host pack")
        self.data, self.client_id, self.settings = data, client_id, _freeze(settings)
        member = data.clients[client_id]
        self.x = data.x[member["start"] : member["stop"]]
        self.y = data.y[member["start"] : member["stop"]]
        self._model = _Head(data.x.shape[1])
        self._lock = Lock()

    def get_parameters(self) -> OrderedDict:
        with self._lock:
            return _parameters(self._model)

    def fit(
        self, parameters, round_number: int, should_stop: Callable
    ) -> tuple[OrderedDict, int, dict]:
        _integer(round_number, "round_number", 1)
        if round_number > self.settings["rounds"]:
            raise ValueError("round exceeds the frozen round budget")
        if not isinstance(parameters, Mapping) or set(parameters) != set(_HEAD_KEYS):
            raise ValueError("global head keys differ")
        width = self.x.shape[1]
        for key, shape in zip(_HEAD_KEYS, ((2, width), (2,))):
            value = parameters[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.device.type != "cpu"
                or tuple(value.shape) != shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"global head layout/dtype/finiteness differs: {key}")
        with self._lock:
            started = time.perf_counter()
            model = _Head(width)
            model.load_state_dict(parameters, strict=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.settings["lr"])
            generator = ClientRng(self.data.manifest["seed"], self.client_id).torch_generator(
                round_number
            )
            # HeadClient constructs nn.Linear inside the round RNG scope before
            # applying the global head. Consume its initialization draws locally.
            nn.init.kaiming_uniform_(torch.empty((2, width)), a=math.sqrt(5), generator=generator)
            torch.empty(2).uniform_(
                -1 / math.sqrt(width), 1 / math.sqrt(width), generator=generator
            )
            batch_count, loss_sum, observations = 0, 0.0, 0
            for _ in range(self.settings["local_epochs"]):
                order = torch.randperm(len(self.y), generator=generator)
                for indices in order.split(self.settings["batch_size"]):
                    if should_stop():
                        raise FitCancelled("local fit cancelled before completion")
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(self.x[indices]), self.y[indices])
                    if not torch.isfinite(loss):
                        raise ValueError("local training produced nonfinite loss")
                    loss.backward()
                    optimizer.step()
                    batch_count += 1
                    loss_sum += float(loss.detach()) * len(indices)
                    observations += len(indices)
            if should_stop():
                raise FitCancelled("local fit cancelled before publication")
            result = _parameters(model)
            if any(not torch.isfinite(value).all() for value in result.values()):
                raise ValueError("local training produced nonfinite parameters")
            self._model = model
            return (
                result,
                len(self.y),
                {
                    "loss": loss_sum / observations,
                    "batch_count": batch_count,
                    "fit_elapsed_s": time.perf_counter() - started,
                    "client_id": self.client_id,
                },
            )
