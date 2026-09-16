# SCRIPTS/tests/test_recipes_base_models_buildable.py
"""The recipe catalog must only advertise base models init_model can build (DA-10).

`recipes.py --describe` feeds the project-creation picker, so every `base_models`
entry is a user-facing promise that init_model.get_model can instantiate that
architecture. Two contracts keep the catalog honest:

1. Every advertised (recipe key, base_model) pair instantiates without raising.
   Pretrained-checkpoint downloads are stubbed with tiny from_config models, so
   the test proves the *name-dispatch path* reaches a real constructor — it never
   hits the network.
2. A recipe advertising MORE THAN ONE base model must actually dispatch on
   model_name: a bogus name must raise. If get_model ignores the name, every
   advertised entry silently yields the same architecture, so any entry beyond
   the first is a lie the first contract cannot see.
"""
import os
import sys

import pytest
import torch

# recipes.py / init_model.py live in the scripts dir; make them importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402
import init_model  # noqa: E402


def _tiny_seqcls_model(pretrained_id, num_labels):
    """Map a known HF checkpoint id to a tiny same-family model built from config."""
    from transformers import (AutoModelForSequenceClassification, LlamaConfig,
                              OPTConfig, Qwen2Config)
    common = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                  vocab_size=256, max_position_embeddings=128, num_labels=num_labels)
    if pretrained_id == "facebook/opt-125m":
        cfg = OPTConfig(ffn_dim=128, word_embed_proj_dim=64, **common)
    elif pretrained_id == "Qwen/Qwen2.5-0.5B":
        cfg = Qwen2Config(intermediate_size=128, num_key_value_heads=2, **common)
    elif pretrained_id == "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T":
        cfg = LlamaConfig(intermediate_size=128, num_key_value_heads=2, **common)
    else:
        pytest.fail(f"advertised base model resolved to unexpected checkpoint {pretrained_id!r}")
    return AutoModelForSequenceClassification.from_config(cfg)


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0


@pytest.fixture
def stub_pretrained_downloads(monkeypatch):
    """Swap HF checkpoint downloads for tiny local models; keep name resolution real."""
    import transformers

    def fake_model_from_pretrained(pretrained_id, *args, **kwargs):
        return _tiny_seqcls_model(pretrained_id, kwargs.get("num_labels", 2))

    monkeypatch.setattr(transformers.AutoModelForSequenceClassification,
                        "from_pretrained", fake_model_from_pretrained)
    monkeypatch.setattr(transformers.AutoTokenizer,
                        "from_pretrained", lambda *args, **kwargs: _FakeTokenizer())
    # The offline override would bypass the name->checkpoint resolution under test.
    monkeypatch.delenv("FEDLEARN_LLM_LORA_BASE", raising=False)


_CATALOG = [(r["key"], name) for r in recipes.RECIPE_METADATA for name in r["base_models"]]
_MULTI_MODEL_KEYS = [r["key"] for r in recipes.RECIPE_METADATA if len(r["base_models"]) > 1]


@pytest.mark.parametrize("key,name", _CATALOG, ids=[f"{k}/{n}" for k, n in _CATALOG])
def test_every_advertised_base_model_instantiates(stub_pretrained_downloads, key, name):
    model = init_model.get_model(key, name, "cpu")
    assert isinstance(model, torch.nn.Module), (
        f"get_model({key!r}, {name!r}) returned {type(model)!r}, not a torch module"
    )


@pytest.mark.parametrize("key", _MULTI_MODEL_KEYS)
def test_multi_model_recipes_dispatch_on_model_name(stub_pretrained_downloads, key):
    with pytest.raises(ValueError):
        init_model.get_model(key, "no-such-model-xyz", "cpu")
