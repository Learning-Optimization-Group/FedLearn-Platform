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

import torch.nn as nn

from fedlearn.communication.safetensors_codec import save_safetensors
from fedlearn.estimators.params import frozen_state


def serialize_backbone(model: nn.Module) -> bytes:
    """Deterministic safetensors blob of the model's :func:`frozen_state` (F32, named order)."""
    tensors = [(name, t.numpy()) for name, t in frozen_state(model).items()]
    return save_safetensors(tensors)


def backbone_sha256(blob: bytes) -> str:
    """The content address of a backbone blob: lowercase-hex sha256."""
    return hashlib.sha256(blob).hexdigest()
