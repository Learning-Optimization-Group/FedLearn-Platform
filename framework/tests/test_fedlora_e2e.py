# framework/tests/test_fedlora_e2e.py
"""End-to-end federated-LoRA acceptance test.

Runs an in-process federated-LoRA loop (no gRPC) over 2 rounds with 2 clients,
for both FFA_LORA and FEDIT aggregation modes.

Acceptance gates (from spec §10):
- set_peft_model_state_dict round-trip: unexpected_keys == []
- client uploads are adapter-only (base-free): every key contains lora_ / score / modules_to_save
- FFA_LORA uploads exclude lora_A
- after aggregate: global still carries lora_A (FFA re-attaches; FedIT averages it in)
- server eval produces numeric accuracy in [0, 1] each round
- loop runs to completion for both modes (no convergence assertion)
"""

import os
import sys
import pytest
import torch
from collections import OrderedDict

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..",
                       "backend", "fl-platform-api", "src", "main", "resources", "scripts")
sys.path.insert(0, SCRIPTS)


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    """Build a tiny Qwen2 model with the REAL tokenizer vocab so SST-2 token IDs are in range."""
    from transformers import AutoTokenizer, Qwen2Config, AutoModelForSequenceClassification
    d = str(tmp_path_factory.mktemp("tiny_qwen_e2e"))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    cfg = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=len(tok),          # real vocab so SST-2 token IDs are valid
        max_position_embeddings=512,
        num_labels=2,
    )
    AutoModelForSequenceClassification.from_config(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return d


def _run(aggregation, tiny_base, monkeypatch):
    monkeypatch.setenv("FEDLEARN_LLM_LORA_BASE", tiny_base)
    monkeypatch.setenv("FEDLEARN_LLM_LORA_SUBSET", "32")

    import importlib
    import recipes
    importlib.reload(recipes)

    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from fedlearn.server.strategy import FedLoRA

    recipe = recipes.get_recipe("LLM_LORA")
    NUM_CLIENTS = 2

    # Initial global adapter = full (A+B+head) from a freshly-built model.
    init_model = recipe.build_model("cpu", model_name="qwen2.5-0.5b", aggregation=aggregation)
    initial = OrderedDict(get_peft_model_state_dict(init_model, save_embedding_layers=False))
    assert any("lora_A" in k for k in initial), "initial global must carry lora_A"

    strategy = FedLoRA(
        initial_parameters=initial,
        aggregation=aggregation,
        min_fit_clients=NUM_CLIENTS,
    )
    global_params = strategy.initialize_parameters()

    accuracies = []
    for rnd in range(2):
        updates = []
        for cid in range(NUM_CLIENTS):
            net = recipe.build_model("cpu", model_name="qwen2.5-0.5b", aggregation=aggregation)
            # set_peft_model_state_dict mutates its input (deletes keys while remapping
            # modules_to_save entries), so pass a copy to preserve global_params for
            # subsequent clients and rounds.
            out = set_peft_model_state_dict(net, OrderedDict(global_params))
            assert list(out.unexpected_keys) == [], (
                f"round {rnd} client {cid}: unexpected_keys after set_peft_model_state_dict: "
                f"{out.unexpected_keys}"
            )

            train, _ = recipe.load_client_data(cid, NUM_CLIENTS, batch_size=8)
            opt = torch.optim.AdamW(
                [p for p in net.parameters() if p.requires_grad], lr=1e-3
            )
            net.train()
            for batch in train:
                opt.zero_grad()
                loss = net(**{k: v for k, v in batch.items()}).loss
                loss.backward()
                opt.step()

            adapter_keys = recipe.adapter_keys(net, aggregation)
            full_sd = get_peft_model_state_dict(net, save_embedding_layers=False)
            upload = OrderedDict(
                (k, v) for k, v in full_sd.items() if k in adapter_keys
            )

            # Non-empty check.
            assert upload, f"round {rnd} client {cid}: upload dict is empty"

            # Base-free: every key must contain lora_ or score or modules_to_save.
            bad_keys = [
                k for k in upload
                if "lora_" not in k and "score" not in k and "modules_to_save" not in k
            ]
            assert not bad_keys, (
                f"round {rnd} client {cid}: base keys leaked into upload: {bad_keys[:5]}"
            )

            # FFA_LORA: client must not upload lora_A (it is frozen+shared on the server).
            if aggregation == "FFA_LORA":
                a_keys = [k for k in upload if "lora_A" in k]
                assert not a_keys, (
                    f"round {rnd} client {cid}: FFA upload must exclude lora_A, found: {a_keys}"
                )

            updates.append((upload, len(train.dataset)))

        global_params = strategy.aggregate_fit(rnd, updates)

        # After aggregation the global must still carry lora_A (re-attached for FFA; averaged for FEDIT).
        assert any("lora_A" in k for k in global_params), (
            f"round {rnd}: global_params missing lora_A after aggregate_fit"
        )

        # Server evaluation: rebuild model, load adapter, forward on validation shard.
        ev = recipe.build_model("cpu", model_name="qwen2.5-0.5b", aggregation=aggregation)
        set_peft_model_state_dict(ev, OrderedDict(global_params))
        ev.eval()
        loader = recipe.load_server_test_data(batch_size=16)
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                logits = ev(**{k: v for k, v in batch.items() if k != "labels"}).logits
                correct += (logits.argmax(-1) == batch["labels"]).sum().item()
                total += batch["labels"].numel()
        acc = correct / max(total, 1)
        accuracies.append(acc)

    assert len(accuracies) == 2, "expected accuracy for each of 2 rounds"
    assert all(0.0 <= a <= 1.0 for a in accuracies), (
        f"accuracy out of [0,1]: {accuracies}"
    )


@pytest.mark.slow
def test_fedlora_e2e_ffa(tiny_base, monkeypatch):
    _run("FFA_LORA", tiny_base, monkeypatch)


@pytest.mark.slow
def test_fedlora_e2e_fedit(tiny_base, monkeypatch):
    _run("FEDIT", tiny_base, monkeypatch)
