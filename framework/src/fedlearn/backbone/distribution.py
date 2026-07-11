"""DA-11 §4.5b: client-side frozen-backbone distribution.

A frozen backbone (the DA-11 ``BASE_REF`` artifact) is serialized to deterministic, content-addressed
bytes, fetched by a client through an injected seam, sha256-verified, content-addressed-cached, and
reconstructed onto a model whose head is trained locally. The fetch source is a ``Callable[[], bytes]``
so this framework contract is independent of HOW the bytes arrive (Phase 2B wires it to the Java
``BASE_REF`` endpoint). Fail-loud throughout: a hash mismatch or a key mismatch is rejected, never
silently loaded (the TINYNET_GOLDEN ``model_dim`` class of bug).
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from fedlearn.communication.safetensors_codec import load_safetensors, save_safetensors
from fedlearn.estimators.params import frozen_state


def serialize_backbone(model: nn.Module) -> bytes:
    """Deterministic safetensors blob of the model's :func:`frozen_state` (F32, named order)."""
    # .cpu() so a backbone trained/held on MPS/CUDA serializes without a device-transfer crash
    # (no-op for a CPU tensor); the wire is CPU float32 (safetensors_codec).
    tensors = [(name, t.cpu().numpy()) for name, t in frozen_state(model).items()]
    return save_safetensors(tensors)


def backbone_sha256(blob: bytes) -> str:
    """The content address of a backbone blob: lowercase-hex sha256."""
    return hashlib.sha256(blob).hexdigest()


class BackboneIntegrityError(ValueError):
    """Fetched backbone bytes' sha256 does not match the requested content address."""


class BackboneCache:
    """Content-addressed on-disk cache for frozen-backbone blobs. A blob is fetched once (via an
    injected ``fetch`` callable), sha256-verified against the requested key, and stored at
    ``cache_dir/<sha256>``. Subsequent requests for the same key are served from disk without
    fetching. A cache file whose bytes no longer hash to its name is treated as a miss and re-fetched
    (self-healing). Writes are atomic (temp file + ``os.replace``) so a crash mid-write never leaves a
    half-written blob under its final content-addressed name.
    """

    def __init__(self, cache_dir: "os.PathLike[str] | str") -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, sha256: str) -> Path:
        return self._dir / sha256

    def get_or_fetch(self, sha256: str, fetch: Callable[[], bytes]) -> Path:
        target = self.path_for(sha256)
        if target.exists() and backbone_sha256(target.read_bytes()) == sha256:
            return target  # cache hit
        blob = fetch()
        actual = backbone_sha256(blob)
        if actual != sha256:
            raise BackboneIntegrityError(
                f"backbone integrity check failed: requested {sha256} but fetched bytes hash to "
                f"{actual} — refusing to cache (possible corruption or wrong artifact)."
            )
        self._atomic_write(target, blob)
        return target

    def _atomic_write(self, target: Path, blob: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise


class BackboneKeyMismatch(ValueError):
    """A fetched backbone blob's key set does not match the model's declared frozen layout."""


def reconstruct_frozen_backbone(model: nn.Module, backbone_bytes: bytes) -> nn.Module:
    """Load a fetched frozen-backbone blob onto ``model`` (non-strict), re-freeze the loaded
    parameters, and return the same model. The head is never touched — after this call the model's
    only trainable (federated) subset is its head.

    Fail-loud: the blob's key set MUST equal ``frozen_state(model)``'s keys. An unexpected key (a blob
    that carries something the model does not declare frozen) or a missing key (a truncated blob)
    raises :class:`BackboneKeyMismatch` rather than silently loading a partial/misaligned backbone.
    """
    tensors, _meta = load_safetensors(backbone_bytes)
    blob_keys = [name for name, _ in tensors]
    expected_keys = list(frozen_state(model).keys())
    if set(blob_keys) != set(expected_keys):
        raise BackboneKeyMismatch(
            f"backbone key mismatch: blob has {sorted(blob_keys)} but the model declares frozen "
            f"layout {sorted(expected_keys)}. Send serialize_backbone(model) for the same recipe."
        )
    state = OrderedDict((name, torch.from_numpy(arr)) for name, arr in tensors)
    model.load_state_dict(state, strict=False)  # head keys are 'missing' (trained locally) — expected
    param_names = {name for name, _ in model.named_parameters()}
    for name in blob_keys:
        if name in param_names:
            model.get_parameter(name).requires_grad_(False)
    return model
