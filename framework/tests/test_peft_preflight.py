# framework/tests/test_peft_preflight.py
import os
import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

BASE = os.environ.get("FEDLEARN_LLM_LORA_BASE", "Qwen/Qwen2.5-0.5B")

@pytest.mark.slow
def test_seqcls_head_is_score():
    m = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)
    assert hasattr(m, "score"), "SeqCls head attr must be 'score' for modules_to_save"

@pytest.mark.slow
def test_get_peft_state_dict_is_base_free_and_roundtrips():
    base = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                     task_type="SEQ_CLS", target_modules=["q_proj", "v_proj"],
                     modules_to_save=["score"])
    model = get_peft_model(base, cfg)
    sd = get_peft_model_state_dict(model)
    assert sd, "adapter state dict must be non-empty"
    # base-free: no plain base weights (every key is a lora_* or modules_to_save/head key)
    assert all(("lora_" in k) or ("modules_to_save" in k) or ("score" in k) for k in sd), \
        f"unexpected base keys leaked: {[k for k in sd if 'lora_' not in k and 'score' not in k and 'modules_to_save' not in k][:5]}"
    out = set_peft_model_state_dict(model, sd)
    assert list(out.unexpected_keys) == [], f"round-trip leaked unexpected_keys: {out.unexpected_keys}"
