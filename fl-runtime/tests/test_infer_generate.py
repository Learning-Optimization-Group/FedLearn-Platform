import os, sys
import pytest

SC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SC)


@pytest.mark.slow
def test_generate_text_completion_only(tmp_path):
    import torch  # noqa
    from transformers import Qwen2Config, AutoModelForCausalLM, AutoTokenizer
    import infer, recipes  # noqa

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    cfg = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=len(tok),
                      max_position_embeddings=512)
    base = AutoModelForCausalLM.from_config(cfg)
    from recipes import apply_lora, get_recipe
    net = apply_lora(base, get_recipe("LLM_LORA").lora, "FFA_LORA", task_type="CAUSAL_LM")
    net.config.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    res = infer.generate_text(net, tok, "Say hi.", max_new_tokens=8, temperature=0.0)
    assert res["ok"] is True and res["modelType"] == "LLM_LORA"
    assert isinstance(res["generatedText"], str)
    # The prompt / template must NOT be echoed back in the completion.
    assert "### Instruction:" not in res["generatedText"]
    assert "Say hi." not in res["generatedText"]
    assert res["tokenCount"] >= 0 and res["finishReason"] in ("stop", "length")


def test_build_model_generation_kind(monkeypatch):
    # LLM_LORA + CAUSAL_LM yields input_kind 'generation'; default stays 'text'. (No network: stub the recipe.)
    import infer, recipes

    class _Stub:
        classes = ["x"]
        def build_model(self, device, model_name=None, aggregation="FFA_LORA", task_type="SEQ_CLASSIFICATION"):
            return object()
        def input_transform(self, model_name=None):
            return object()
    monkeypatch.setattr(recipes, "get_recipe", lambda k: _Stub())
    _, _, kind_gen, _ = infer.build_model("LLM_LORA", "qwen2.5-0.5b", "CAUSAL_LM")
    _, _, kind_cls, _ = infer.build_model("LLM_LORA", "qwen2.5-0.5b", "SEQ_CLASSIFICATION")
    assert kind_gen == "generation" and kind_cls == "text"
