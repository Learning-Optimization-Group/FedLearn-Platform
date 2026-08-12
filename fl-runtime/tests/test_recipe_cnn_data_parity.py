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


def _install_fake_dataset(monkeypatch, base):
    """Back `datasets.load_dataset` with `base`, recording the dataset name it was asked for.

    P0-2b: this used to patch `flwr_datasets.FederatedDataset`. Patching one level lower is a
    strictly stronger test — the old fake stood in for the whole of FederatedDataset and so
    silently skipped its shuffle(seed=42), leaving that step unpinned. The shard pipeline is
    now exercised end to end against the real `datasets` API, which is all flwr was wrapping.
    """
    seen = {}

    def _fake_load_dataset(name, *a, **kw):
        seen["dataset"] = name
        return {"train": base}

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)
    return seen


def _reference_shard(base, pid, num_shards=10, seed=42):
    """The shard pipeline, written out independently of the implementation.

    Byte-identical to what flwr_datasets produced: FederatedDataset shuffles each split with
    seed 42 *before* partitioning, and IidPartitioner.load_partition(i) is exactly
    shard(num_shards=N, index=i, contiguous=True). Verified per-partition against the real
    flwr by research/benchmarks/verify_flwr_shard_equivalence.py.
    """
    return base.shuffle(seed=seed).shard(num_shards=num_shards, index=pid, contiguous=True)


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
    base = _synthetic_cifar_base(n=100)
    seen = _install_fake_dataset(monkeypatch, base)
    recipes.get_recipe("CNN").load_client_data(partition_id=0, num_clients=7)
    assert seen["dataset"] == "cifar10"
    assert recipes.CNN_NUM_PARTITIONS == 10  # fixed, NOT num_clients=7

    # The shard must be 1/10th of the base, whatever num_clients said.
    train, _ = recipes.get_recipe("CNN").load_client_data(partition_id=0, num_clients=7)
    assert len(train.dataset) + len(_.dataset) == len(base) // 10


def test_cnn_client_shard_is_the_shuffled_contiguous_shard(monkeypatch):
    """Pins the shuffle(seed=42) step that the previous FederatedDataset-level fake hid.

    Without this, dropping flwr could have silently removed the shuffle: every partition would
    still be a valid 1/10th split, the suite would stay green, and every CIFAR-10 result before
    and after would quietly stop being comparable.
    """
    base = _synthetic_cifar_base(n=100)
    _install_fake_dataset(monkeypatch, base)

    shard = recipes._cnn_iid_shard(partition_id=3)
    ref = _reference_shard(base, pid=3)
    assert list(shard["label"]) == list(ref["label"])
    # ...and specifically NOT the unshuffled shard, which is what a dropped shuffle would give.
    unshuffled = base.shard(num_shards=10, index=3, contiguous=True)
    assert list(shard["label"]) != list(unshuffled["label"])


def test_cnn_client_partition_content_matches_reference_pipeline(monkeypatch):
    """The recipe reproduces shuffle(42) -> shard(10, pid, contiguous) -> split(0.2, seed=42)
    -> normalize. Compared against an inline reference built from the same synthetic base; the
    val loader is unshuffled, so the first batch is deterministic."""
    base = _synthetic_cifar_base(n=100)
    _install_fake_dataset(monkeypatch, base)

    _, recipe_val = recipes.get_recipe("CNN").load_client_data(partition_id=1, num_clients=99)

    # Reference: exactly what the legacy client.py CNN branch does, computed here independently.
    from torch.utils.data import DataLoader
    ref = _reference_shard(base, pid=1).train_test_split(test_size=0.2, seed=42)
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
