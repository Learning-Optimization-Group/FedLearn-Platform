"""DA-14 Phase 1 — CNN (CIFAR-10) data loading collapses onto the recipe registry.

Determinism traps this pins (all three were called out in the refactor research):
  1. The client partitioner is a FIXED 10 shards (flwr IidPartitioner), NOT num_clients — routing
     it through num_clients (or the Dirichlet _dirichlet_indices path the pneumonia recipe uses)
     would silently change every shard.
  2. The shard pipeline is shuffle(seed=42) -> shard(10, pid, contiguous) -> split(0.2, seed=42);
     any change to a constant reshuffles the partition.
  3. Source asymmetry: the CLIENT shard comes from HuggingFace 'cifar10' via flwr_datasets; the
     SERVER test set comes from torchvision CIFAR10 — two different loaders, preserved as-is.

Offline: a synthetic HF dataset of PIL images (no 170MB CIFAR download). The real-CIFAR
end-to-end byte check is a separate @pytest.mark.slow test.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402


def _synthetic_cifar_base(n=60):
    """A tiny HF dataset shaped like flwr's cifar10 split: an Image 'img' column + int 'label'."""
    import numpy as np
    import datasets
    from datasets import Features, Image as HFImage, Value
    from PIL import Image
    imgs = [Image.fromarray(np.full((32, 32, 3), (i * 7) % 256, dtype=np.uint8)) for i in range(n)]
    feats = Features({"img": HFImage(), "label": Value("int64")})
    return datasets.Dataset.from_dict({"img": imgs, "label": [i % 10 for i in range(n)]}, features=feats)


def _install_fake_fds(monkeypatch, base):
    """Replace flwr_datasets.FederatedDataset with a fake backed by `base`, recording the
    partitioner count it was constructed with. recipes imports it lazily, so this patch takes."""
    seen = {}

    class _FakeFDS:
        def __init__(self, dataset, partitioners):
            seen["dataset"] = dataset
            seen["num_partitions"] = partitioners["train"]

        def load_partition(self, pid):
            return base.shard(num_shards=seen["num_partitions"], index=pid, contiguous=True)

    monkeypatch.setattr("flwr_datasets.FederatedDataset", _FakeFDS)
    return seen


def test_cnn_constants_match_legacy_call_sites():
    import client
    assert recipes.CNN_NUM_PARTITIONS == client.NUM_PARTITIONS == 10
    assert recipes.CNN_BATCH_SIZE == client.BATCH_SIZE == 32
    assert recipes.CNN_SERVER_TEST_BATCH == 128
    import torchvision.transforms as T
    tf = recipes._cnn_transform()
    assert isinstance(tf.transforms[0], T.ToTensor)
    assert isinstance(tf.transforms[1], T.Normalize)
    assert tuple(tf.transforms[1].mean) == (0.5, 0.5, 0.5)
    assert tuple(tf.transforms[1].std) == (0.5, 0.5, 0.5)


def test_cnn_client_uses_fixed_10_partitions_ignoring_num_clients(monkeypatch):
    """THE critical guard: the shard count is a fixed 10, never num_clients."""
    base = _synthetic_cifar_base()
    seen = _install_fake_fds(monkeypatch, base)
    recipes.get_recipe("CNN").load_client_data(partition_id=0, num_clients=7)
    assert seen["dataset"] == "cifar10"
    assert seen["num_partitions"] == 10  # fixed CNN_NUM_PARTITIONS, NOT num_clients=7


def test_cnn_client_partition_content_matches_reference_pipeline(monkeypatch):
    """The recipe reproduces shard(10, pid, contiguous) -> split(0.2, seed=42) -> normalize.
    Compared against an inline reference built from the same synthetic base; the val loader is
    unshuffled, so the first batch is deterministic."""
    base = _synthetic_cifar_base()
    _install_fake_fds(monkeypatch, base)

    _, recipe_val = recipes.get_recipe("CNN").load_client_data(partition_id=1, num_clients=99)

    # Reference: exactly what the legacy client.py CNN branch does, computed here independently.
    from torch.utils.data import DataLoader
    ref = base.shard(num_shards=10, index=1, contiguous=True).train_test_split(test_size=0.2, seed=42)
    tf = recipes._cnn_transform()

    def _apply(b):
        b["img"] = [tf(im) for im in b["img"]]
        return b

    ref_val = DataLoader(ref["test"].with_transform(_apply), batch_size=32, num_workers=0)

    rb = next(iter(recipe_val))
    xb = next(iter(ref_val))
    assert torch.equal(rb["img"], xb["img"])
    assert torch.equal(torch.as_tensor(rb["label"]), torch.as_tensor(xb["label"]))


def test_cnn_server_uses_torchvision_test_split_batch_128(monkeypatch):
    """Server test set comes from TORCHVISION CIFAR10 (train=False), batched at 128, unshuffled —
    the source asymmetry with the client's flwr/HF shard is intentional and preserved."""
    from torch.utils.data import SequentialSampler, TensorDataset
    seen = {}

    def _fake_cifar10(root, train, download, transform):
        seen["root"], seen["train"], seen["transform"] = root, train, transform
        return TensorDataset(torch.zeros(10, 3, 32, 32), torch.zeros(10, dtype=torch.long))

    monkeypatch.setattr("torchvision.datasets.CIFAR10", _fake_cifar10)
    loader = recipes.get_recipe("CNN").load_server_test_data()
    assert seen["train"] is False               # test split, not train
    assert loader.batch_size == 128
    assert isinstance(loader.sampler, SequentialSampler)  # shuffle=False
    import torchvision.transforms as T
    assert isinstance(seen["transform"].transforms[1], T.Normalize)
