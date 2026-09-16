import os, sys, json, subprocess
import numpy as np
import pytest

HERE = os.path.dirname(__file__)
SCRIPTS = os.path.join(HERE, "..")
sys.path.insert(0, SCRIPTS)


@pytest.mark.slow
def test_infer_llm_lora_text(tmp_path):
    from transformers import Qwen2Config, AutoModelForSequenceClassification, AutoTokenizer
    from peft import get_peft_model_state_dict
    # tiny base whose vocab matches the tokenizer (real SST-2-style token ids are valid)
    base_dir = str(tmp_path / "tiny")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    cfg = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=len(tok),
                      max_position_embeddings=512, num_labels=2)
    AutoModelForSequenceClassification.from_config(cfg).save_pretrained(base_dir)
    tok.save_pretrained(base_dir)
    os.environ["FEDLEARN_LLM_LORA_BASE"] = base_dir
    import importlib, recipes
    importlib.reload(recipes)
    # build the adapter via the recipe + save it as the project .npz (full adapter A+B+head)
    model = recipes.get_recipe("LLM_LORA").build_model("cpu", model_name="qwen2.5-0.5b", aggregation="FFA_LORA")
    adapter = get_peft_model_state_dict(model, save_embedding_layers=False)
    npz = str(tmp_path / "adapter.npz")
    np.savez(npz, **{k.replace(".", "__DOT__"): v.detach().cpu().numpy() for k, v in adapter.items()})

    in_json = str(tmp_path / "in.json"); out_json = str(tmp_path / "out.json")
    json.dump({"kind": "text", "text": "a wonderful, moving film"}, open(in_json, "w"))
    env = {**os.environ, "FEDLEARN_LLM_LORA_BASE": base_dir}
    subprocess.run([sys.executable, "infer.py", "--model-type", "LLM_LORA", "--model-name", "qwen2.5-0.5b",
                    "--model-path", npz, "--in", in_json, "--out", out_json],
                   cwd=SCRIPTS, env=env, check=True)
    res = json.load(open(out_json))
    assert res["ok"] is True, res
    assert res["predictedLabel"] in ["negative", "positive"]
    assert len(res["probabilities"]) == 2 and abs(sum(res["probabilities"]) - 1.0) < 1e-3


def test_transformer_classes_are_defined():
    # The recipe metadata must declare the 3 CB labels (not empty) so predictedLabel is meaningful.
    import importlib, recipes
    importlib.reload(recipes)
    t = next(r for r in recipes.RECIPE_METADATA if r["key"] == "TRANSFORMER")
    assert t["classes"] == ["entailment", "contradiction", "neutral"]
