import os, sys, subprocess, tempfile
import numpy as np
import pytest

HERE = os.path.dirname(__file__)
SCRIPTS = os.path.join(HERE, "..")


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    from transformers import Qwen2Config, AutoModelForSequenceClassification, AutoTokenizer
    d = str(tmp_path_factory.mktemp("tiny_qwen"))
    c = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
                    max_position_embeddings=128, num_labels=2)
    AutoModelForSequenceClassification.from_config(c).save_pretrained(d)
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B").save_pretrained(d)  # cached after first run
    return d


@pytest.mark.slow
def test_init_model_saves_adapter_only(tiny_base, tmp_path):
    out = str(tmp_path / "adapter.npz")
    env = {**os.environ, "FEDLEARN_LLM_LORA_BASE": tiny_base}
    subprocess.run(
        [sys.executable, "init_model.py", "--model-type", "LLM_LORA", "--model-name",
         "qwen2.5-0.5b", "--optimizer", "AdamW", "--out", out, "--aggregation", "FFA_LORA"],
        cwd=SCRIPTS, env=env, check=True)
    keys = list(np.load(out).keys())
    assert keys, "npz must be non-empty"
    # base-free: every saved key is a lora_* or head key (decode __DOT__ first)
    dec = [k.replace("__DOT__", ".") for k in keys]
    assert all(("lora_" in k) or ("score" in k) or ("modules_to_save" in k) for k in dec), dec
    assert any("lora_A" in k for k in dec), "init save must include A (full adapter)"
