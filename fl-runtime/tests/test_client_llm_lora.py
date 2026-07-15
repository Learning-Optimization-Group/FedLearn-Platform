import os
import sys
import importlib
import pytest
from collections import OrderedDict

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    from transformers import Qwen2Config, AutoModelForSequenceClassification, AutoTokenizer
    d = str(tmp_path_factory.mktemp("tiny_qwen_c"))
    c = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
                    max_position_embeddings=128, num_labels=2)
    AutoModelForSequenceClassification.from_config(c).save_pretrained(d)
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B").save_pretrained(d)
    return d


@pytest.mark.slow
def test_client_get_parameters_is_adapter_only(tiny_base, monkeypatch):
    monkeypatch.setenv("FEDLEARN_LLM_LORA_BASE", tiny_base)
    monkeypatch.setenv("FEDLEARN_LLM_LORA_SUBSET", "16")
    import client
    importlib.reload(client)
    # Engage the LLM_LORA path the way __main__ does:
    client.USE_LLM = False
    client.USE_MLP = False
    client.USE_PNEUMONIA = False
    client.USE_LLM_LORA = True
    client.LLM_LORA_AGGREGATION = "FFA_LORA"
    client.LLM_LORA_MODEL_NAME = "qwen2.5-0.5b"
    c = client.ZOSLClient(partition_id=0, dataset_name="sst2", num_clients=2)
    keys = list(c.get_parameters().keys())
    assert keys and not any("lora_A" in k for k in keys), "FFA upload excludes lora_A"
    assert all(("lora_" in k) or ("score" in k) or ("modules_to_save" in k) for k in keys), keys


@pytest.mark.slow
@pytest.mark.parametrize("agg", ["FFA_LORA", "FEDIT"])
def test_client_fit_round_trip(tiny_base, monkeypatch, agg):
    """Prove Fix 1: fit() completes without KeyError for LLM_LORA in both aggregation modes.

    The bug was in the post-train debug loop which indexed net.state_dict() with keys
    from the compacted peft upload form — mismatched namespaces raised KeyError on round 1.

    The tiny_base model has vocab_size=256 so we inject a synthetic data loader that
    produces input_ids within [0, 255] rather than using the real SST-2 + Qwen tokenizer
    (which would have token ids >> 256, causing IndexError in the embedding layer).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    monkeypatch.setenv("FEDLEARN_LLM_LORA_BASE", tiny_base)
    monkeypatch.setenv("FEDLEARN_LLM_LORA_SUBSET", "16")
    import client
    importlib.reload(client)

    client.USE_LLM = False
    client.USE_MLP = False
    client.USE_PNEUMONIA = False
    client.USE_LLM_LORA = True
    client.LLM_LORA_AGGREGATION = agg
    client.LLM_LORA_MODEL_NAME = "qwen2.5-0.5b"

    c = client.ZOSLClient(partition_id=0, dataset_name="sst2", num_clients=2)

    # Replace trainloader with a synthetic one that is safe for vocab_size=256.
    # 8 samples, seq_len=16, all token ids in [0, 255].
    n_samples, seq_len, vocab_size = 8, 16, 256
    input_ids = torch.randint(0, vocab_size, (n_samples, seq_len))
    attention_mask = torch.ones(n_samples, seq_len, dtype=torch.long)
    labels = torch.randint(0, 2, (n_samples,))

    class _DictLoader:
        """Yields dict batches for vocab_size=256 synthetic data.

        Exposes .batch_size and .dataset so client.train() and fit() can
        introspect it the same way they would a real DataLoader.
        """
        def __init__(self, ids, mask, lbls, batch_size=8):
            self._ds = TensorDataset(ids, mask, lbls)
            self._bs = batch_size
            self.batch_size = batch_size  # read by train() banner

        def __iter__(self):
            loader = DataLoader(self._ds, batch_size=self._bs, shuffle=False)
            for ids_b, mask_b, lbl_b in loader:
                yield {"input_ids": ids_b, "attention_mask": mask_b, "labels": lbl_b}

        def __len__(self):
            return (len(self._ds) + self._bs - 1) // self._bs

        @property
        def dataset(self):
            return self._ds

    c.trainloader = _DictLoader(input_ids, attention_mask, labels)

    # Simulate incoming global parameters from the server (round 1 send-back)
    initial = c.get_parameters()
    assert initial, "get_parameters() must return non-empty adapter dict"

    # fit() signature: (self, parameters: OrderedDict, config: dict) -> (OrderedDict, int)
    # grpc_client is None by default — no gRPC dependency in fit() body
    new_params, n = c.fit(OrderedDict(initial), {"server_round": 1, "local_epochs": 1})

    # Must not KeyError (the bug) and must return something sensible
    assert n > 0, "fit() must return positive num_examples"
    assert new_params, "fit() must return non-empty parameter dict"

    # Returned keys must all be adapter-only (no full model weights)
    for k in new_params:
        assert ("lora_" in k) or ("score" in k) or ("modules_to_save" in k), (
            f"Unexpected non-adapter key in fit() return: {k!r}"
        )

    # FFA_LORA: lora_A is frozen and NOT uploaded by the client
    if agg == "FFA_LORA":
        assert not any("lora_A" in k for k in new_params), (
            "FFA_LORA fit() must not return lora_A keys"
        )
