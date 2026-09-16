# framework/tests/test_causal_lm_e2e.py
import os, sys, math
import pytest
import torch
from collections import OrderedDict

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..",
                       "backend", "fl-platform-api", "src", "main", "resources", "scripts")
sys.path.insert(0, SCRIPTS)


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    from transformers import Qwen2Config, AutoModelForCausalLM, AutoTokenizer
    d = str(tmp_path_factory.mktemp("tiny_causal"))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    c = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=len(tok), max_position_embeddings=512)
    AutoModelForCausalLM.from_config(c).save_pretrained(d)
    tok.save_pretrained(d)
    return d


@pytest.mark.slow
def test_causal_lm_e2e_ffa(tiny_base, monkeypatch):
    monkeypatch.setenv("FEDLEARN_LLM_LORA_BASE", tiny_base)
    monkeypatch.setenv("FEDLEARN_LLM_LORA_SUBSET", "40")
    import importlib, recipes
    importlib.reload(recipes)
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from fedlearn.server.strategy import FedLoRA

    recipe = recipes.get_recipe("LLM_LORA")
    TT, AGG, N = "CAUSAL_LM", "FFA_LORA", 2

    init = recipe.build_model("cpu", model_name="qwen2.5-0.5b", aggregation=AGG, task_type=TT)
    initial = OrderedDict(get_peft_model_state_dict(init, save_embedding_layers=False))
    assert any("lora_A" in k for k in initial) and not any("score" in k for k in initial)

    strategy = FedLoRA(initial_parameters=initial, aggregation=AGG, min_fit_clients=N)
    global_params = strategy.initialize_parameters()

    for rnd in range(2):
        updates = []
        for cid in range(N):
            net = recipe.build_model("cpu", model_name="qwen2.5-0.5b", aggregation=AGG, task_type=TT)
            out = set_peft_model_state_dict(net, OrderedDict(global_params))
            assert list(out.unexpected_keys) == []
            train, _ = recipe.load_client_data(cid, N, task_type=TT, batch_size=2)
            opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
            net.train()
            for batch in train:
                opt.zero_grad()
                loss = net(**{k: v for k, v in batch.items()}).loss
                loss.backward(); opt.step()
            keys = recipe.adapter_keys(net, AGG)
            upload = OrderedDict((k, v) for k, v in get_peft_model_state_dict(net, save_embedding_layers=False).items() if k in keys)
            assert upload and not any("score" in k for k in upload) and not any("lora_A" in k for k in upload)
            updates.append((upload, len(train.dataset)))
        global_params = strategy.aggregate_fit(rnd, updates)
        assert any("lora_A" in k for k in global_params)

        ev = recipe.build_model("cpu", model_name="qwen2.5-0.5b", aggregation=AGG, task_type=TT)
        set_peft_model_state_dict(ev, OrderedDict(global_params)); ev.eval()
        loader = recipe.load_server_test_data(task_type=TT, batch_size=4)
        tot, nb = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                tot += ev(**{k: v for k, v in batch.items()}).loss.item(); nb += 1
        ppl = math.exp(tot / nb)
        assert ppl > 0 and math.isfinite(ppl)
