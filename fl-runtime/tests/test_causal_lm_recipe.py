import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402

LORA = {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}


def _tiny_causal():
    from transformers import Qwen2Config, AutoModelForCausalLM
    c = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
                    max_position_embeddings=128)
    return AutoModelForCausalLM.from_config(c)


def test_causal_apply_lora_has_no_head_and_freezes_a():
    model = recipes.apply_lora(_tiny_causal(), LORA, "FFA_LORA", task_type="CAUSAL_LM")
    names = [n for n, _ in model.named_parameters()]
    assert not any("score" in n or "modules_to_save" in n for n in names), "CAUSAL_LM has no classification head"
    a = [p.requires_grad for n, p in model.named_parameters() if "lora_A" in n]
    b = [p.requires_grad for n, p in model.named_parameters() if "lora_B" in n]
    assert a and all(r is False for r in a) and b and any(b)


def test_causal_adapter_keys_exclude_head():
    model = recipes.apply_lora(_tiny_causal(), LORA, "FFA_LORA", task_type="CAUSAL_LM")
    keys = recipes.llm_lora_adapter_keys(model, "FFA_LORA")
    assert keys and not any("score" in k for k in keys)
    assert not any("lora_A" in k for k in keys)  # FFA excludes A
    assert any("lora_B" in k for k in keys)


def test_seq_cls_apply_lora_unchanged():
    from transformers import Qwen2Config, AutoModelForSequenceClassification
    c = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=256, max_position_embeddings=128, num_labels=2)
    model = recipes.apply_lora(AutoModelForSequenceClassification.from_config(c), LORA, "FFA_LORA")
    assert any("score" in n or "modules_to_save" in n for n in (n for n, _ in model.named_parameters()))


def test_dolly_loader_renders_and_labels(monkeypatch):
    # network-free: feed a tiny in-memory dolly-shaped dataset
    import datasets
    tiny = datasets.Dataset.from_list([
        {"instruction": "Say hi", "context": "", "response": "hi", "category": "open_qa"},
        {"instruction": "Sum", "context": "1 and 2", "response": "3", "category": "closed_qa"},
    ] * 4)
    monkeypatch.setattr(recipes, "_load_llm_tokenizer",
                        lambda model_name=None: __import__("transformers").AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"))
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: tiny)
    import recipes as r
    monkeypatch.setattr(r, "load_dataset", datasets.load_dataset, raising=False)
    train, _ = recipes.load_dolly_client_data(0, 2, batch_size=2)
    batch = next(iter(train))
    assert set(["input_ids", "attention_mask", "labels"]).issubset(batch.keys())
    # labels == input_ids except -100 at pad positions
    il, lab, am = batch["input_ids"][0], batch["labels"][0], batch["attention_mask"][0]
    for i in range(len(il)):
        assert lab[i] == (il[i] if am[i] == 1 else -100)
