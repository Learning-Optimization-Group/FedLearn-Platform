# SCRIPTS/tests/test_llm_lora_recipe.py
import os, sys
import warnings
import pytest

# recipes.py lives in the scripts dir; make it importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402


def _tiny_seqcls():
    """A tiny Qwen2 SeqCls model built from config — no download."""
    from transformers import Qwen2Config, AutoModelForSequenceClassification
    c = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
                    max_position_embeddings=128, num_labels=2)
    return AutoModelForSequenceClassification.from_config(c)


LORA = {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}


def test_ffa_freezes_lora_a_keeps_b_trainable():
    model = recipes.apply_lora(_tiny_seqcls(), LORA, "FFA_LORA")
    a = [p.requires_grad for n, p in model.named_parameters() if "lora_A" in n]
    b = [p.requires_grad for n, p in model.named_parameters() if "lora_B" in n]
    assert a and all(req is False for req in a), "FFA must freeze every lora_A"
    assert b and any(b), "lora_B must stay trainable"


def test_fedit_keeps_both_trainable():
    model = recipes.apply_lora(_tiny_seqcls(), LORA, "FEDIT")
    a = [p.requires_grad for n, p in model.named_parameters() if "lora_A" in n]
    b = [p.requires_grad for n, p in model.named_parameters() if "lora_B" in n]
    assert a and all(a), "FEDIT must keep lora_A trainable"
    assert b and all(b), "FEDIT must keep lora_B trainable"


def test_adapter_keys_ffa_excludes_a_includes_b_and_head():
    model = recipes.apply_lora(_tiny_seqcls(), LORA, "FFA_LORA")
    keys = recipes.llm_lora_adapter_keys(model, "FFA_LORA")
    assert keys, "adapter_keys must be non-empty"
    assert not any("lora_A" in k for k in keys), "FFA upload must exclude lora_A"
    assert any("lora_B" in k for k in keys)
    assert any("score" in k or "modules_to_save" in k for k in keys), "head must be uploaded"


def test_adapter_keys_fedit_includes_a():
    model = recipes.apply_lora(_tiny_seqcls(), LORA, "FEDIT")
    keys = recipes.llm_lora_adapter_keys(model, "FEDIT")
    assert any("lora_A" in k for k in keys)
    assert any("lora_B" in k for k in keys)


def test_get_recipe_is_functional_and_text():
    r = recipes.get_recipe("LLM_LORA")
    assert r.is_functional is True
    assert r.input_kind == "text"
    assert r.task_type == "SEQ_CLASSIFICATION"
    assert r.classes == ["negative", "positive"]
    assert r.lora == {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}
    assert recipes._METADATA_BY_KEY["LLM_LORA"]["aggregation"] == "FFA_LORA"


def test_adapter_keys_emits_no_warning():
    model = recipes.apply_lora(_tiny_seqcls(), LORA, "FFA_LORA")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning, module="peft")
        keys = recipes.llm_lora_adapter_keys(model, "FFA_LORA")
    assert keys


@pytest.mark.slow
def test_build_model_smoke(tmp_path):
    """Exercise the real build_model runtime path (catches NameError-class bugs)."""
    from transformers import AutoTokenizer
    d = str(tmp_path / "tiny")
    _tiny_seqcls().save_pretrained(d)
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B").save_pretrained(d)
    os.environ["FEDLEARN_LLM_LORA_BASE"] = d
    try:
        model = recipes.get_recipe("LLM_LORA").build_model("cpu", model_name="qwen2.5-0.5b", aggregation="FFA_LORA")
        frozen_a = [p.requires_grad for n, p in model.named_parameters() if "lora_A" in n]
        assert frozen_a and not any(frozen_a), "FFA build_model must freeze lora_A"
    finally:
        os.environ.pop("FEDLEARN_LLM_LORA_BASE", None)
