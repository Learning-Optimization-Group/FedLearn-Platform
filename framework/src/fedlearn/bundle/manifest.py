"""Versioned adapter-bundle manifest (DA-9) — the unit of specialization.

A bundle packages a specialized model for delivery / serving / on-device training:
  * a manifest (this module) describing identity (``artifact_sha256`` — the same content hash the
    registry stores, DA-1/DA-3), kind, frozen base, LoRA config, license, an eval-card reference,
    and a per-file sha256 list;
  * adapter weights as **safetensors** (the hardened wire format — never torch.save/pickle); full
    checkpoints stay as the imaging air-gap export format.

The manifest validates against ``adapter_bundle.schema.json`` (committed next to this module, since
``docs/`` is gitignored). See ``BUNDLE_FORMAT.md`` for the human-readable spec and the fixture-MVP
boundary where real export is not yet wired into the mobile bundle path.
"""
from __future__ import annotations

import hashlib
import json
from importlib import resources
from typing import Dict, List, Optional, Tuple

import numpy as np

from fedlearn.communication.safetensors_codec import load_safetensors, save_safetensors

SCHEMA_VERSION = "1.0"
_SCHEMA_RESOURCE = "adapter_bundle.schema.json"
_VALID_KINDS = ("LORA_ADAPTER", "FULL_CHECKPOINT")


def sha256_hex(data: bytes) -> str:
    """Lowercase-hex sha256 of ``data`` — the content-address that aligns a bundle with its registry row."""
    return hashlib.sha256(data).hexdigest()


def load_schema() -> Dict[str, object]:
    """Load the committed bundle JSON schema (packaged next to this module)."""
    with resources.files("fedlearn.bundle").joinpath(_SCHEMA_RESOURCE).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def adapter_to_safetensors(state_dict: Dict[str, object],
                           metadata: Optional[Dict[str, str]] = None) -> bytes:
    """Serialize a state_dict (adapter or checkpoint) to safetensors bytes, in a deterministic order.

    Accepts numpy arrays or torch tensors (anything with ``.detach()``); values are stored float32.
    """
    tensors: List[Tuple[str, np.ndarray]] = []
    for name, value in state_dict.items():
        arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        tensors.append((name, arr))
    return save_safetensors(tensors, metadata)


def safetensors_to_state_dict(blob: bytes) -> Dict[str, np.ndarray]:
    """Inverse of :func:`adapter_to_safetensors` — recovers the named tensors as a dict."""
    tensors, _metadata = load_safetensors(blob)
    return {name: arr for name, arr in tensors}


def build_manifest(*, artifact_sha256: str, kind: str, recipe_key: str,
                   base_model_ref: Optional[str], license_tag: Optional[str],
                   lora: Optional[Dict[str, object]], eval_card_ref: Optional[str],
                   files: List[Dict[str, str]],
                   provenance: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Build a bundle manifest. A LORA_ADAPTER must name its frozen base and carry a LoRA config
    (the same invariant the registry enforces for the ADAPTER_OF edge, DA-3)."""
    if kind not in _VALID_KINDS:
        raise ValueError(f"unknown bundle kind {kind!r}; expected one of {_VALID_KINDS}")
    if kind == "LORA_ADAPTER":
        if not base_model_ref:
            raise ValueError("a LORA_ADAPTER bundle requires base_model_ref")
        if not lora:
            raise ValueError("a LORA_ADAPTER bundle requires a lora config")

    manifest: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_sha256": artifact_sha256,
        "kind": kind,
        "recipe_key": recipe_key,
        "base_model_ref": base_model_ref,
        "license_tag": license_tag,
        "lora": lora,
        "eval_card_ref": eval_card_ref,
        "files": list(files),
    }
    if provenance is not None:
        manifest["provenance"] = provenance
    return manifest
