"""DA-14 Ph3.3b — the production client federates a derived model HEAD-ONLY.

Mirrors the LLM_LORA adapter-subset path exactly: a client whose model_type is a derived
(frozen-backbone) recipe uploads only its trainable subset (the head) via get_parameters — the
frozen backbone never rides the wire. This pins the production wire payload for the derived path;
the full spawnable derived recipe (data + fit + server eval) is a later additive step.
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402


def test_derived_client_uploads_head_only(monkeypatch):
    import client  # noqa: E402
    # Engage the derived path the way __main__ does (mirror the LLM_LORA test's flag flips).
    monkeypatch.setattr(client, "USE_LLM_LORA", False, raising=False)
    monkeypatch.setattr(client, "USE_LLM", False, raising=False)
    monkeypatch.setattr(client, "USE_MLP", False, raising=False)
    monkeypatch.setattr(client, "USE_PNEUMONIA", False, raising=False)
    monkeypatch.setattr(client, "USE_DERIVED", True, raising=False)
    # The recipe must be named explicitly. USE_DERIVED used to imply FROZEN_DEMO, and the client
    # selected that recipe's synthetic vector shard from it — the conflation that handed a frozen
    # CIFAR run a 256-dim vector batch. Data now follows the RECIPE, so a test exercising the
    # FROZEN_DEMO path has to say so.
    monkeypatch.setattr(client, "MODEL_TYPE", "FROZEN_DEMO", raising=False)

    net = recipes.get_recipe("FROZEN_DEMO").build_model("cpu")
    params = client.ZOSLClient.get_parameters(types.SimpleNamespace(net=net))

    # Head-only wire — only the trainable head, never the frozen backbone (mirrors adapter-only).
    assert set(params.keys()) == {"head.weight", "head.bias"}
    assert not any("backbone" in k for k in params)


def test_derived_client_runs_fit_end_to_end_head_only(monkeypatch):
    """DA-14 Ph3.3c: a derived ZOSLClient builds the frozen-backbone model, loads its self-contained
    shard, runs a real fit() round, and returns HEAD-ONLY — the head trains, the frozen backbone
    never moves. Mirrors test_client_llm_lora's fit round-trip for the derived path."""
    from collections import OrderedDict
    import torch
    import client  # noqa: E402
    monkeypatch.setattr(client, "USE_LLM", False, raising=False)
    monkeypatch.setattr(client, "USE_MLP", False, raising=False)
    monkeypatch.setattr(client, "USE_PNEUMONIA", False, raising=False)
    monkeypatch.setattr(client, "USE_LLM_LORA", False, raising=False)
    monkeypatch.setattr(client, "USE_DERIVED", True, raising=False)
    # The recipe must be named explicitly. USE_DERIVED used to imply FROZEN_DEMO, and the client
    # selected that recipe's synthetic vector shard from it — the conflation that handed a frozen
    # CIFAR run a 256-dim vector batch. Data now follows the RECIPE, so a test exercising the
    # FROZEN_DEMO path has to say so.
    monkeypatch.setattr(client, "MODEL_TYPE", "FROZEN_DEMO", raising=False)

    c = client.ZOSLClient(partition_id=0, dataset_name="frozen_demo", num_clients=2)
    backbone_before = c.net.backbone.weight.detach().clone()

    initial = c.get_parameters()
    assert set(initial.keys()) == {"head.weight", "head.bias"}

    new_params, n = c.fit(OrderedDict(initial), {"server_round": 1, "local_epochs": 1})

    assert n > 0
    assert set(new_params.keys()) == {"head.weight", "head.bias"}                  # head-only wire
    assert not torch.equal(new_params["head.weight"], initial["head.weight"])      # the head trained
    assert torch.equal(c.net.backbone.weight.detach(), backbone_before)            # backbone unmoved
