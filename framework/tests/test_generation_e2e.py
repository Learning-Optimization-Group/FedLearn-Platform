import os, sys
import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..",
                       "backend", "fl-platform-api", "src", "main", "resources", "scripts")
sys.path.insert(0, SCRIPTS)


@pytest.mark.slow
def test_generation_produces_completion(tmp_path):
    from transformers import Qwen2Config, AutoModelForCausalLM, AutoTokenizer
    import infer
    from recipes import apply_lora, get_recipe

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    cfg = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=len(tok),
                      max_position_embeddings=512)
    net = apply_lora(AutoModelForCausalLM.from_config(cfg), get_recipe("LLM_LORA").lora,
                     "FFA_LORA", task_type="CAUSAL_LM")
    net.config.pad_token_id = tok.eos_token_id

    res = infer.generate_text(net, tok, "Write one sentence about dogs.", max_new_tokens=16, temperature=0.0)
    assert res["ok"] and isinstance(res["generatedText"], str)
    assert res["finishReason"] in ("stop", "length")
