"""DA-9 — the versioned adapter-bundle format (the unit of specialization).

Pins the format contract: a manifest validates against the committed JSON schema and carries the
base + LoRA config for an adapter; adapters serialize as safetensors and round-trip; a LORA_ADAPTER
manifest without a base is rejected. The manifest's artifact_sha256 is the content hash that aligns
a bundle with its registry row (DA-1/DA-3).
"""
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from fedlearn.bundle.manifest import (
    SCHEMA_VERSION,
    adapter_to_safetensors,
    build_manifest,
    load_schema,
    safetensors_to_state_dict,
    sha256_hex,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "src/fedlearn/bundle/adapter_bundle.schema.json"


def _lora_config():
    # Matches recipes.py LLM_LORA: r=8, alpha=16, dropout=0.05, target_modules=[q_proj, v_proj].
    return {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}


def test_committed_schema_is_present_and_loadable():
    assert SCHEMA_PATH.exists(), "the bundle schema must be committed next to the code (docs/ is gitignored)"
    assert load_schema() == json.loads(SCHEMA_PATH.read_text())


def test_lora_bundle_manifest_validates_and_lists_base_and_lora():
    weights = adapter_to_safetensors({
        "base_model.q_proj.lora_A": np.zeros((8, 4), dtype="f4"),
        "base_model.q_proj.lora_B": np.ones((4, 8), dtype="f4"),
    })
    sha = sha256_hex(weights)
    manifest = build_manifest(
        artifact_sha256=sha, kind="LORA_ADAPTER", recipe_key="LLM_LORA",
        base_model_ref="qwen2.5-0.5b", license_tag="Apache-2.0", lora=_lora_config(),
        eval_card_ref=None, files=[{"name": "adapter_model.safetensors", "sha256": sha}],
    )

    jsonschema.validate(manifest, load_schema())  # validates against the committed schema
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["base_model_ref"] == "qwen2.5-0.5b"
    assert manifest["lora"]["r"] == 8 and manifest["lora"]["target_modules"] == ["q_proj", "v_proj"]
    assert manifest["artifact_sha256"] == sha  # aligns the bundle with its registry row


def test_adapter_safetensors_roundtrip_reproduces_state_dict_keys_and_values():
    rng = np.random.default_rng(0)
    state_dict = {
        "lora_A": rng.standard_normal((8, 4)).astype("f4"),
        "lora_B": rng.standard_normal((4, 8)).astype("f4"),
    }
    blob = adapter_to_safetensors(state_dict)
    recovered = safetensors_to_state_dict(blob)

    assert set(recovered.keys()) == set(state_dict.keys())
    for key in state_dict:
        assert np.allclose(recovered[key], state_dict[key])


def test_a_lora_adapter_bundle_without_a_base_is_rejected():
    with pytest.raises(ValueError):
        build_manifest(
            artifact_sha256="a" * 64, kind="LORA_ADAPTER", recipe_key="LLM_LORA",
            base_model_ref=None, license_tag="Apache-2.0", lora=_lora_config(),
            eval_card_ref=None, files=[],
        )


def test_full_checkpoint_bundle_manifest_validates_without_a_base():
    manifest = build_manifest(
        artifact_sha256="b" * 64, kind="FULL_CHECKPOINT", recipe_key="PNEUMONIA_CNN",
        base_model_ref=None, license_tag=None, lora=None, eval_card_ref=None,
        files=[{"name": "model.safetensors", "sha256": "c" * 64}],
    )
    jsonschema.validate(manifest, load_schema())
    assert manifest["kind"] == "FULL_CHECKPOINT"
