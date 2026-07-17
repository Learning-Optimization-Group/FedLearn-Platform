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

    net = recipes.get_recipe("FROZEN_DEMO").build_model("cpu")
    params = client.ZOSLClient.get_parameters(types.SimpleNamespace(net=net))

    # Head-only wire — only the trainable head, never the frozen backbone (mirrors adapter-only).
    assert set(params.keys()) == {"head.weight", "head.bias"}
    assert not any("backbone" in k for k in params)
